"""SparseGPT: one-shot pruning with second-order weight reconstruction.

Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately
Pruned in One-Shot" (ICML 2023).

Magnitude, Wanda and OBS-saliency pruning only *choose* which weights to
drop. SparseGPT also *updates the surviving weights* so the layer output on
calibration data changes as little as possible. For a layer ``y = W x``
with calibration inputs ``X`` (rows ``x_n``), every output row ``w`` of ``W``
solves the same layer-wise problem

    min_{w'} || (w - w') X^T ||^2   s.t.  w' has the chosen sparsity,

whose Hessian is ``H = X^T X`` (shared across rows). Columns are processed
left to right. When column ``j`` is pruned in a row, the Optimal Brain
Surgeon update spreads its error onto the not-yet-processed columns:

    err = w_j / [U]_jj,      w_{j:} -= err * U_{j, j:}

where ``U`` is the upper Cholesky factor of ``H^{-1}`` (``H^{-1} = U^T U``),
which encodes the inverse Hessians of every trailing sub-problem at once.
The mask is chosen *adaptively* per block of ``block_size`` columns, using
the OBS saliency ``w^2 / [U]_jj^2`` of the already-updated weights.
Choices are made across the whole block (unstructured) or as ``n`` of every
``m`` consecutive weights in a row (``nm=(2, 4)`` semi-structured).
``H`` is dampened by ``percdamp * mean(diag H)`` for numerical stability,
as in the reference implementation.

Weights are flat C-contiguous buffers of shape ``(out, in)`` or
``(out, in, kh, kw)``. Calibration ``inputs`` are rows of length
``in * kh * kw`` (raw features, or im2col patches for conv layers), the
same convention as :mod:`prune_kit.thinet`. Pure Python, ``O(d^3)`` in the
row width ``d``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .layers import LayerSpec

__all__ = [
    "SparseGPTResult",
    "layer_hessian",
    "sparsegpt_prune_layer",
    "sparsegpt_prune_model",
    "sparsegpt_compare",
    "output_reconstruction_error",
]


def _product(shape: Sequence[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def _validate(weights, shape, inputs) -> Tuple[Tuple[int, ...], int, int, List[List[float]]]:
    shape = tuple(int(d) for d in shape)
    if len(shape) < 2 or any(d < 1 for d in shape):
        raise ValueError("shape must be (out, in) or (out, in, kh, kw) with positive dims")
    if len(weights) != _product(shape):
        raise ValueError(f"weights has {len(weights)} entries, shape {shape} needs {_product(shape)}")
    width = _product(shape[1:])
    rows = [[float(v) for v in row] for row in inputs]
    if not rows:
        raise ValueError("inputs must contain at least one calibration row")
    if any(len(r) != width for r in rows):
        raise ValueError(f"every calibration row must have length {width}")
    return shape, shape[0], width, rows


def layer_hessian(inputs: Sequence[Sequence[float]], percdamp: float = 0.01) -> List[List[float]]:
    """``H = X^T X / n`` plus ``percdamp * mean(diag H)`` on the diagonal."""
    rows = [[float(v) for v in r] for r in inputs]
    if not rows:
        raise ValueError("inputs must contain at least one calibration row")
    d = len(rows[0])
    n = len(rows)
    H = [[0.0] * d for _ in range(d)]
    for r in rows:
        for i in range(d):
            ri = r[i]
            if ri == 0.0:
                continue
            Hi = H[i]
            for j in range(i, d):
                Hi[j] += ri * r[j]
    for i in range(d):
        for j in range(i, d):
            H[i][j] /= n
            H[j][i] = H[i][j]
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    mean_diag = sum(H[i][i] for i in range(d)) / d
    damp = percdamp * mean_diag if mean_diag > 0 else percdamp
    for i in range(d):
        if H[i][i] == 0.0:
            H[i][i] = 1.0  # dead input: any weight on it is free to prune
        H[i][i] += damp
    return H


def _cholesky_lower(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - math.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("Hessian is not positive definite; increase percdamp")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _inverse_spd(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    L = _cholesky_lower(A)
    # Invert L (lower triangular), then A^{-1} = L^{-T} L^{-1}.
    Linv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Linv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            Linv[i][j] = -math.fsum(L[i][k] * Linv[k][j] for k in range(j, i)) / L[i][i]
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = math.fsum(Linv[k][i] * Linv[k][j] for k in range(j, n))
            inv[i][j] = inv[j][i] = v
    return inv


def _upper_cholesky_of_inverse(H: List[List[float]]) -> List[List[float]]:
    """Upper-triangular ``U`` with ``H^{-1} = U^T U``."""
    Hinv = _inverse_spd(H)
    L = _cholesky_lower(Hinv)
    n = len(L)
    return [[L[j][i] for j in range(n)] for i in range(n)]  # U = L^T


def output_reconstruction_error(
    original: Sequence[float],
    pruned: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """``sum_n ||W x_n - W' x_n||^2`` and its ratio to ``sum_n ||W x_n||^2``."""
    shape, out, width, rows = _validate(original, shape, inputs)
    if len(pruned) != len(original):
        raise ValueError("pruned and original weights differ in size")
    err = 0.0
    total = 0.0
    for o in range(out):
        w = original[o * width:(o + 1) * width]
        p = pruned[o * width:(o + 1) * width]
        for r in rows:
            y = math.fsum(a * b for a, b in zip(w, r))
            yp = math.fsum(a * b for a, b in zip(p, r))
            err += (y - yp) ** 2
            total += y * y
    return {"squared_error": err, "relative_error": err / total if total > 0 else 0.0}


@dataclass(frozen=True)
class SparseGPTResult:
    """Outcome of :func:`sparsegpt_prune_layer`."""

    weights: List[float]
    mask: List[int]
    density: float
    squared_error: float
    relative_error: float
    pattern: str

    def as_dict(self) -> Dict[str, object]:
        return {"density": self.density, "squared_error": self.squared_error,
                "relative_error": self.relative_error, "pattern": self.pattern}


def sparsegpt_prune_layer(
    weights: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
    density: float = 0.5,
    *,
    nm: Optional[Tuple[int, int]] = None,
    block_size: int = 128,
    percdamp: float = 0.01,
    update: bool = True,
) -> SparseGPTResult:
    """One-shot SparseGPT pruning of one layer.

    ``density`` is the fraction of weights to keep (unstructured; ignored
    when ``nm=(n, m)`` is given, which keeps ``n`` of every ``m``
    consecutive weights in each row). With ``update=False`` the OBS weight
    update is skipped, leaving SparseGPT's mask selection on the original
    weights (useful as an ablation).
    """
    shape, out, width, rows = _validate(weights, shape, inputs)
    if nm is not None:
        n_keep, m = int(nm[0]), int(nm[1])
        if not (0 < n_keep < m):
            raise ValueError("nm must satisfy 0 < n < m")
        if width % m != 0:
            raise ValueError(f"row width {width} is not divisible by m={m}")
        pattern = f"{n_keep}:{m}"
    else:
        if not 0.0 < density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        pattern = "unstructured"
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if nm is not None and block_size % nm[1] != 0:
        block_size = max(nm[1], (block_size // nm[1]) * nm[1])
    H = layer_hessian(rows, percdamp=percdamp)
    U = _upper_cholesky_of_inverse(H)
    W = [[float(weights[o * width + j]) for j in range(width)] for o in range(out)]
    keep = [[1] * width for _ in range(out)]
    for b0 in range(0, width, block_size):
        b1 = min(b0 + block_size, width)
        # Adaptive mask selection for this block on the current weights.
        if nm is None:
            sal = []
            for o in range(out):
                for j in range(b0, b1):
                    sal.append((W[o][j] ** 2 / U[j][j] ** 2, o, j))
            n_prune = int(round((1.0 - density) * len(sal)))
            sal.sort(key=lambda t: (t[0], t[1], t[2]))
            for _, o, j in sal[:n_prune]:
                keep[o][j] = 0
        else:
            n_keep, m = nm
            for o in range(out):
                for g0 in range(b0, b1, m):
                    group = sorted(range(g0, g0 + m), key=lambda j: (W[o][j] ** 2 / U[j][j] ** 2, j))
                    for j in group[: m - n_keep]:
                        keep[o][j] = 0
        # Column-by-column OBS update; errors propagate to all later columns.
        for j in range(b0, b1):
            d = U[j][j]
            Uj = U[j]
            for o in range(out):
                if keep[o][j]:
                    continue
                w = W[o][j]
                W[o][j] = 0.0
                if not update or w == 0.0:
                    continue
                err = w / d
                Wo = W[o]
                for c in range(j + 1, width):
                    Wo[c] -= err * Uj[c]
    flat = [W[o][j] if keep[o][j] else 0.0 for o in range(out) for j in range(width)]
    mask = [keep[o][j] for o in range(out) for j in range(width)]
    err = output_reconstruction_error(weights, flat, shape, rows)
    return SparseGPTResult(weights=flat, mask=mask, density=sum(mask) / len(mask),
                           squared_error=err["squared_error"], relative_error=err["relative_error"],
                           pattern=pattern)


def sparsegpt_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    inputs: Dict[str, Sequence[Sequence[float]]],
    density: float = 0.5,
    **kwargs,
) -> Dict[str, SparseGPTResult]:
    """Apply :func:`sparsegpt_prune_layer` to every layer with calibration inputs.

    Layers without inputs are left untouched and omitted from the result.
    In a sequential network, collect each layer's inputs from the already
    pruned preceding layers, as in the paper.
    """
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(inputs) - set(by_name))
    if unknown:
        raise ValueError(f"inputs given for unknown layers: {unknown}")
    results: Dict[str, SparseGPTResult] = {}
    for spec in specs:
        if spec.name not in inputs:
            continue
        if spec.name not in model:
            raise KeyError(f"model is missing weights for layer {spec.name!r}")
        results[spec.name] = sparsegpt_prune_layer(model[spec.name], spec.shape, inputs[spec.name],
                                                   density, **kwargs)
    return results


def sparsegpt_compare(
    weights: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
    density: float = 0.5,
    *,
    nm: Optional[Tuple[int, int]] = None,
    percdamp: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Relative output error of magnitude, Wanda, SparseGPT mask-only, and SparseGPT."""
    shape, out, width, rows = _validate(weights, shape, inputs)
    col_norm = [math.sqrt(math.fsum(r[j] ** 2 for r in rows)) for j in range(width)]

    def score_prune(score):
        pruned = [0.0] * len(weights)
        if nm is None:
            order = sorted(range(len(weights)), key=lambda i: (-score(i), i))
            n_keep = int(round(density * len(weights)))
            for i in order[:n_keep]:
                pruned[i] = float(weights[i])
        else:
            n_keep, m = nm
            for o in range(out):
                for g0 in range(0, width, m):
                    idx = [o * width + j for j in range(g0, g0 + m)]
                    for i in sorted(idx, key=lambda i: (-score(i), i))[:n_keep]:
                        pruned[i] = float(weights[i])
        return pruned

    variants = {
        "magnitude": score_prune(lambda i: abs(weights[i])),
        "wanda": score_prune(lambda i: abs(weights[i]) * col_norm[i % width]),
        "sparsegpt_mask_only": sparsegpt_prune_layer(weights, shape, rows, density, nm=nm,
                                                     percdamp=percdamp, update=False).weights,
        "sparsegpt": sparsegpt_prune_layer(weights, shape, rows, density, nm=nm, percdamp=percdamp).weights,
    }
    return {name: output_reconstruction_error(weights, w, shape, rows) for name, w in variants.items()}
