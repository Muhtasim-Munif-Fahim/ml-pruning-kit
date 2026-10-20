"""ThiNet: data-driven input-channel pruning with least-squares reconstruction.

Luo, Wu & Lin, "ThiNet: A Filter Level Pruning Method for Deep Neural
Network Compression" (ICCV 2017); the reconstruction step follows He,
Zhang & Sun, "Channel Pruning for Accelerating Very Deep Neural Networks"
(ICCV 2017).

Norm-based channel pruning ranks channels by the size of their *weights*.
ThiNet instead looks at what each input channel actually contributes to
the layer's output on calibration data. For a layer ``y = W x`` with input
channels ``c`` (one column block of ``W`` per channel), the output is the
sum of per-channel contributions ``z_c = W[:, c] x[c]``. Removing a set
``T`` of channels changes the output by ``sum_{c in T} z_c``, so the
reconstruction error over the calibration samples is

    E(T) = sum_n || sum_{c in T} z_{n,c} ||^2 = sum_{c,d in T} G[c, d]

with the channel Gram matrix ``G[c, d] = sum_n <z_{n,c}, z_{n,d}>``.
:func:`thinet_select_channels` grows ``T`` greedily, each time adding the
channel that increases ``E`` the least, until only ``density`` of the
channels remain. Because the cost is the norm of the *summed* removed
contributions, later picks account for interactions with channels already
removed (a channel whose contribution partly cancels the accumulated error
becomes cheap), which per-channel scores cannot see. The search is
greedy, so it is not guaranteed to find the optimal subset.

:func:`least_squares_reconstruct` then refits the kept weights so the
pruned layer reproduces the original outputs as well as possible,
``W' = argmin || W X - W'_S X_S ||^2`` (ridge-regularised), which often
recovers most of the error -- exactly so when a removed channel is a
linear combination of kept ones.

Weights are flat C-contiguous buffers of shape ``(out, in)`` (dense) or
``(out, in, kh, kw)`` (conv). Calibration ``inputs`` are rows of length
``in * kh * kw`` laid out channel-major like the weight columns (for a
conv layer these are im2col patches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec
from .structured import channel_norms

__all__ = [
    "ThiNetResult",
    "channel_contribution_gram",
    "thinet_select_channels",
    "least_squares_reconstruct",
    "reconstruction_error",
    "thinet_prune_layer",
    "thinet_prune_model",
    "thinet_prune_summary",
]


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate(weights: Sequence[float], shape: Sequence[int], inputs: Sequence[Sequence[float]]):
    shape = tuple(int(d) for d in shape)
    if len(shape) < 2 or any(d <= 0 for d in shape):
        raise ValueError("shape must have at least 2 positive dimensions")
    if len(weights) != _product(shape):
        raise ValueError(f"weights has {len(weights)} values, shape {shape} needs {_product(shape)}")
    if not all(math.isfinite(float(w)) for w in weights):
        raise ValueError("weights must be finite")
    n_in = shape[1]
    inner = _product(shape[2:]) if len(shape) > 2 else 1
    width = n_in * inner
    if not inputs:
        raise ValueError("inputs must contain at least one calibration row")
    rows = []
    for row in inputs:
        values = [float(v) for v in row]
        if len(values) != width:
            raise ValueError(f"each input row must have {width} values (in * kh * kw)")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("inputs must be finite")
        rows.append(values)
    return shape, n_in, inner, rows


def _validate_density(density: float) -> float:
    if not 0.0 < float(density) <= 1.0:
        raise ValueError("density must be in (0, 1]")
    return float(density)


def _weight_matrix(weights: Sequence[float], shape: Tuple[int, ...]) -> List[List[float]]:
    width = _product(shape[1:])
    return [[float(weights[o * width + j]) for j in range(width)] for o in range(shape[0])]


def _contributions(W, row, n_in, inner):
    """``z_c = W[:, block c] @ x[block c]`` for every channel ``c``."""

    out = []
    for c in range(n_in):
        lo = c * inner
        block = row[lo : lo + inner]
        out.append([math.fsum(wr[lo + k] * block[k] for k in range(inner)) for wr in W])
    return out


def channel_contribution_gram(
    weights: Sequence[float], shape: Sequence[int], inputs: Sequence[Sequence[float]]
) -> List[List[float]]:
    """``G[c][d] = sum_n <z_{n,c}, z_{n,d}>`` over the calibration rows."""

    shape, n_in, inner, rows = _validate(weights, shape, inputs)
    W = _weight_matrix(weights, shape)
    G = [[0.0] * n_in for _ in range(n_in)]
    for row in rows:
        z = _contributions(W, row, n_in, inner)
        for c in range(n_in):
            zc = z[c]
            for d in range(c, n_in):
                value = math.fsum(a * b for a, b in zip(zc, z[d]))
                G[c][d] += value
                if d != c:
                    G[d][c] += value
    return G


def _keep_count(n: int, density: float, min_channels: int) -> int:
    return min(n, max(int(min_channels), int(round(n * density))))


def thinet_select_channels(
    weights: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
    density: float,
    *,
    min_channels: int = 1,
) -> Dict[str, object]:
    """Greedy ThiNet channel selection.

    Returns ``keep`` (sorted kept channel indices), ``removed`` (in the order
    they were removed), and ``errors``, the cumulative squared reconstruction
    error ``E(T)`` after each removal.
    """

    density = _validate_density(density)
    if int(min_channels) < 1:
        raise ValueError("min_channels must be at least 1")
    G = channel_contribution_gram(weights, shape, inputs)
    n_in = len(G)
    n_keep = _keep_count(n_in, density, min_channels)
    removed: List[int] = []
    cross = [0.0] * n_in  # sum_{t in T} G[t][c]
    error = 0.0
    errors: List[float] = []
    remaining = set(range(n_in))
    while len(remaining) > n_keep:
        best = min(remaining, key=lambda c: (2.0 * cross[c] + G[c][c], c))
        error += 2.0 * cross[best] + G[best][best]
        remaining.discard(best)
        removed.append(best)
        errors.append(max(error, 0.0))
        for c in range(n_in):
            cross[c] += G[best][c]
    return {"keep": sorted(remaining), "removed": removed, "errors": errors}


def _solve(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Solve ``A X = B`` (``A`` square) by Gauss-Jordan with partial pivoting."""

    n = len(A)
    m = len(B[0]) if B else 0
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-300:
            raise ValueError("normal equations are singular; use a positive ridge")
        M[col], M[pivot] = M[pivot], M[col]
        inv = 1.0 / M[col][col]
        M[col] = [v * inv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                M[r] = [a - factor * b for a, b in zip(M[r], M[col])]
    return [row[n : n + m] for row in M]


def least_squares_reconstruct(
    weights: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
    keep: Sequence[int],
    *,
    ridge: float = 1e-8,
) -> List[float]:
    """Refit the kept channels' weights to reproduce the original outputs.

    Solves ``min_{W'} sum_n ||W x_n - W'_S x_{n,S}||^2 + lam ||W'_S||^2``
    with ``lam = ridge * trace(X_S^T X_S) / dim`` (a scale-free ridge that
    keeps the system well-posed when kept inputs are collinear). Removed
    channels are zero in the returned flat buffer.
    """

    shape, n_in, inner, rows = _validate(weights, shape, inputs)
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    keep = sorted({int(c) for c in keep})
    if not keep or keep[0] < 0 or keep[-1] >= n_in:
        raise ValueError("keep must list valid input-channel indices")
    W = _weight_matrix(weights, shape)
    cols = [c * inner + k for c in keep for k in range(inner)]
    dim = len(cols)
    A = [[0.0] * dim for _ in range(dim)]
    out = shape[0]
    B = [[0.0] * out for _ in range(dim)]
    for row in rows:
        xs = [row[j] for j in cols]
        y = [math.fsum(w * x for w, x in zip(wr, row)) for wr in W]
        for i in range(dim):
            xi = xs[i]
            if xi == 0.0:
                continue
            Ai = A[i]
            for j in range(dim):
                Ai[j] += xi * xs[j]
            Bi = B[i]
            for o in range(out):
                Bi[o] += xi * y[o]
    trace = sum(A[i][i] for i in range(dim))
    lam = ridge * (trace / dim if trace > 0 else 1.0)
    for i in range(dim):
        A[i][i] += lam
    X = _solve(A, B)  # dim x out
    width = n_in * inner
    new = [0.0] * len(weights)
    for i, j in enumerate(cols):
        for o in range(out):
            new[o * width + j] = X[i][o]
    return new


def reconstruction_error(
    original: Sequence[float],
    pruned: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """``sum_n ||W x_n - W' x_n||^2`` and its ratio to ``sum_n ||W x_n||^2``."""

    shape, _, _, rows = _validate(original, shape, inputs)
    if len(pruned) != len(original):
        raise ValueError("pruned and original weights differ in size")
    W = _weight_matrix(original, shape)
    P = _weight_matrix(pruned, shape)
    err = 0.0
    total = 0.0
    for row in rows:
        for wr, pr in zip(W, P):
            y = math.fsum(w * x for w, x in zip(wr, row))
            yp = math.fsum(w * x for w, x in zip(pr, row))
            err += (y - yp) ** 2
            total += y * y
    return {"squared_error": err, "relative_error": err / total if total > 0 else 0.0}


@dataclass(frozen=True)
class ThiNetResult:
    """Outcome of :func:`thinet_prune_layer`."""

    weights: List[float]
    keep: List[int]
    removed: List[int]
    error_before_reconstruction: float
    error_after_reconstruction: float
    relative_error_before: float
    relative_error_after: float
    reconstructed: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "keep": list(self.keep),
            "removed": list(self.removed),
            "error_before_reconstruction": self.error_before_reconstruction,
            "error_after_reconstruction": self.error_after_reconstruction,
            "relative_error_before": self.relative_error_before,
            "relative_error_after": self.relative_error_after,
            "reconstructed": self.reconstructed,
        }


def _zero_channels(weights, shape, removed):
    shape = tuple(int(d) for d in shape)
    inner = _product(shape[2:]) if len(shape) > 2 else 1
    width = shape[1] * inner
    out = [float(w) for w in weights]
    for c in removed:
        for o in range(shape[0]):
            start = o * width + c * inner
            for k in range(inner):
                out[start + k] = 0.0
    return out


def thinet_prune_layer(
    weights: Sequence[float],
    shape: Sequence[int],
    inputs: Sequence[Sequence[float]],
    density: float,
    *,
    selection: str = "thinet",
    reconstruct: bool = True,
    ridge: float = 1e-8,
    min_channels: int = 1,
) -> ThiNetResult:
    """Prune input channels of one layer, optionally refitting the kept weights.

    ``selection="thinet"`` uses the greedy output-aware criterion;
    ``"l1"`` / ``"l2"`` keep the input channels with the largest weight norms
    (the data-free baseline), which is handy for comparing both criteria with
    the same reconstruction step.
    """

    density = _validate_density(density)
    shape_t, n_in, _, _ = _validate(weights, shape, inputs)
    key = str(selection).lower()
    if key == "thinet":
        chosen = thinet_select_channels(weights, shape_t, inputs, density, min_channels=min_channels)
        keep, removed = chosen["keep"], chosen["removed"]
    elif key in {"l1", "l2"}:
        norms = channel_norms(list(weights), shape_t, structure="channel", norm=key)
        n_keep = _keep_count(n_in, density, min_channels)
        order = sorted(range(n_in), key=lambda c: (-norms[c], c))
        keep = sorted(order[:n_keep])
        removed = sorted(order[n_keep:])
    else:
        raise ValueError("selection must be 'thinet', 'l1' or 'l2'")
    zeroed = _zero_channels(weights, shape_t, removed)
    before = reconstruction_error(weights, zeroed, shape_t, inputs)
    if reconstruct and removed:
        new = least_squares_reconstruct(weights, shape_t, inputs, keep, ridge=ridge)
        after = reconstruction_error(weights, new, shape_t, inputs)
    else:
        new, after = zeroed, before
    return ThiNetResult(
        weights=new,
        keep=list(keep),
        removed=list(removed),
        error_before_reconstruction=before["squared_error"],
        error_after_reconstruction=after["squared_error"],
        relative_error_before=before["relative_error"],
        relative_error_after=after["relative_error"],
        reconstructed=bool(reconstruct and removed),
    )


def thinet_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    inputs: Dict[str, Sequence[Sequence[float]]],
    density: float,
    *,
    selection: str = "thinet",
    reconstruct: bool = True,
    ridge: float = 1e-8,
    min_channels: int = 1,
) -> Dict[str, ThiNetResult]:
    """Apply :func:`thinet_prune_layer` to every layer that has calibration inputs.

    Layers absent from ``inputs`` are left untouched (and omitted from the
    result), so the first layer -- whose inputs are raw features -- can be
    skipped by simply not passing calibration rows for it.
    """

    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(inputs) - set(by_name))
    if unknown:
        raise ValueError(f"inputs given for unknown layers: {unknown}")
    results: Dict[str, ThiNetResult] = {}
    for spec in specs:
        if spec.name not in inputs:
            continue
        if spec.name not in model:
            raise KeyError(f"model is missing weights for layer {spec.name!r}")
        results[spec.name] = thinet_prune_layer(
            model[spec.name], spec.shape, inputs[spec.name], density,
            selection=selection, reconstruct=reconstruct, ridge=ridge,
            min_channels=min_channels,
        )
    return results


def thinet_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    inputs: Dict[str, Sequence[Sequence[float]]],
    density: float,
    *,
    ridge: float = 1e-8,
    min_channels: int = 1,
) -> Dict[str, object]:
    """Compare L1, ThiNet and ThiNet + least squares on every calibrated layer."""

    layers = []
    for spec in specs:
        if spec.name not in inputs:
            continue
        common = dict(ridge=ridge, min_channels=min_channels)
        l1 = thinet_prune_layer(model[spec.name], spec.shape, inputs[spec.name], density,
                                selection="l1", reconstruct=False, **common)
        thin = thinet_prune_layer(model[spec.name], spec.shape, inputs[spec.name], density,
                                  selection="thinet", reconstruct=True, **common)
        layers.append({
            "layer": spec.name,
            "channels": int(spec.shape[1]),
            "kept_channels": len(thin.keep),
            "keep": thin.keep,
            "l1_keep": l1.keep,
            "relative_error_l1": l1.relative_error_before,
            "relative_error_thinet": thin.relative_error_before,
            "relative_error_thinet_ls": thin.relative_error_after,
        })
    return {"density": float(density), "layers": layers}
