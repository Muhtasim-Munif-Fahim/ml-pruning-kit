"""Layer-Adaptive Magnitude Pruning (LAMP; Lee et al., ICML 2021).

LAMP scores every scalar weight with a **layer-local** normalization so
that a single global magnitude-style ranking automatically allocates
different sparsities to different layers. The classic score (paper
Eq. 5) sorts the layer by ascending squared magnitude and sets

    score(W_{[i]}) = W_{[i]}^2 / sum_{j >= i} W_{[j]}^2

where ``[i]`` is the ascending-magnitude order inside the layer. The
smallest weight therefore divides by the full Frobenius energy of the
layer; the largest divides by only itself. Ties break toward the
lower flat index.

A practical alternative (``mode="frobenius"``) simply reweights each
connection by the layer Frobenius norm:

    score_i = |w_i| / ||W||_F

(with ``||W||_F = 0`` yielding zeros). Ranking ``|w| / ||W||_F``
globally is a common cheap layer-adaptive heuristic; the classic LAMP
score is strictly more principled and is the default.

``density`` is the fraction of connections to **keep**. The number kept
is ``round(density * N)``, matching the other pruners in this kit.
Default ``scope="global"`` matches the LAMP paper (rank every LAMP
score together). ``scope="layer"`` keeps the same fraction inside each
layer. Dense ``(out, in)`` and conv ``(out, in, kh, kw)`` flat buffers
are both supported; LAMP scores are always computed per layer (the
shape only validates the buffer). A weight that is already zero stays
zero: kept positions copy the original value.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density, sparse_mask_to_dense


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate_density(density: float) -> float:
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    return density


def _validate_scope(scope: str) -> str:
    key = str(scope).lower()
    if key not in {"global", "layer"}:
        raise ValueError("scope must be 'global' or 'layer'")
    return key


def _validate_mode(mode: str) -> str:
    key = str(mode).lower()
    if key not in {"lamp", "frobenius"}:
        raise ValueError("mode must be 'lamp' or 'frobenius'")
    return key


def _validate_shape(weights: Sequence[float], shape: Sequence[int] | None) -> tuple[int, ...]:
    if shape is None:
        return (len(weights),)
    if isinstance(shape, (str, bytes)) or not shape:
        raise ValueError("shape must be non-empty")
    dims = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in dims):
        raise ValueError("shape dimensions must be positive")
    expected = _product(dims)
    if len(weights) != expected:
        raise ValueError(
            f"weights has {len(weights)} values, shape {dims} expects {expected}"
        )
    return dims


def frobenius_norm(weights: Sequence[float]) -> float:
    """Frobenius (L2) norm of a flat weight buffer."""
    if not weights:
        raise ValueError("weights must not be empty")
    total = 0.0
    for value in weights:
        v = float(value)
        total += v * v
    return math.sqrt(total)


def _classic_lamp_scores(weights: Sequence[float]) -> List[float]:
    """Classic LAMP scores with ascending-magnitude cumulative denominators."""
    n = len(weights)
    squares = [float(value) * float(value) for value in weights]
    # Ascending |w| (via square), ties toward lower index.
    order = sorted(range(n), key=lambda index: (squares[index], index))
    scores = [0.0] * n
    denom = 0.0
    # Walk from largest to smallest so the running sum is the tail sum.
    for position in range(n - 1, -1, -1):
        index = order[position]
        denom += squares[index]
        if denom == 0.0:
            scores[index] = 0.0
        else:
            scores[index] = squares[index] / denom
    return scores


def _frobenius_scores(weights: Sequence[float]) -> List[float]:
    """Practical layer-adaptive scores ``|w| / ||W||_F``."""
    norm = frobenius_norm(weights)
    if norm == 0.0:
        return [0.0 for _ in weights]
    return [abs(float(value)) / norm for value in weights]


def lamp_scores(
    weights: Sequence[float],
    *,
    mode: str = "lamp",
    shape: Sequence[int] | None = None,
) -> List[float]:
    """LAMP (or Frobenius-normalized) score of each weight.

    ``mode="lamp"`` (default) is the classic cumulative score of Lee et
    al. ``mode="frobenius"`` returns ``|w| / ||W||_F``. ``shape``, when
    given, must match the flat buffer (dense ``(out, in)`` or conv
    ``(out, in, kh, kw)``) and is used only for validation.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_shape(weights, shape)
    mode = _validate_mode(mode)
    if mode == "lamp":
        return _classic_lamp_scores(weights)
    return _frobenius_scores(weights)


def _keep_count(total: int, density: float) -> int:
    return int(round(density * total))


def _keep_highest(scores: Sequence[float], density: float) -> List[int]:
    """Indices of the highest scores, in ascending index order."""
    total = len(scores)
    n_keep = _keep_count(total, density)
    if n_keep <= 0:
        return []
    if n_keep >= total:
        return list(range(total))
    ranked = sorted(range(total), key=lambda index: (-scores[index], index))
    keep_set = set(ranked[:n_keep])
    return [index for index in range(total) if index in keep_set]


def _apply_mask(
    weights: Sequence[float],
    keep_indices: Sequence[int],
    shape: tuple[int, ...],
) -> Tuple[List[float], List[int]]:
    mask = sparse_mask_to_dense(keep_indices, shape)
    pruned = [
        float(value) if bit else 0.0
        for value, bit in zip(weights, mask)
    ]
    return pruned, mask


def lamp_prune_layer(
    weights: Sequence[float],
    *,
    density: float = 0.5,
    mode: str = "lamp",
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Zero the lowest-LAMP-score connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequence is not modified. Kept positions copy the original weight;
    every dropped position is ``0.0``.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = lamp_scores(weights, mode=mode, shape=resolved)
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _layer_scores(
    spec: LayerSpec,
    weights: Sequence[float],
    mode: str,
) -> List[float]:
    expected = layer_weight_count(spec)
    if len(weights) != expected:
        raise ValueError(
            f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
        )
    return lamp_scores(weights, mode=mode, shape=spec.shape)


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
    mode: str,
) -> Tuple[float, str, str]:
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    spec_names = [spec.name for spec in specs]
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("specs contain duplicate layer names")
    if set(spec_names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    density = _validate_density(density)
    scope = _validate_scope(scope)
    mode = _validate_mode(mode)
    if per_layer:
        if scope != "layer":
            raise ValueError("per_layer requires scope='layer'")
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            _validate_density(layer_density)
    return density, scope, mode


def _lamp_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
    mode: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str, str]:
    density, scope, mode = _validate_model_args(
        specs, model, density, scope, per_layer, mode
    )
    scores = {
        spec.name: _layer_scores(spec, model[spec.name], mode)
        for spec in specs
    }
    keep_by_layer: Dict[str, List[int]] = {}
    if scope == "layer":
        for spec in specs:
            layer_density = (
                density if not per_layer else per_layer.get(spec.name, density)
            )
            keep_by_layer[spec.name] = _keep_highest(
                scores[spec.name], layer_density
            )
    else:
        ranked: List[Tuple[float, int, int, str]] = []
        for order, spec in enumerate(specs):
            for index, score in enumerate(scores[spec.name]):
                ranked.append((score, order, index, spec.name))
        total = len(ranked)
        n_keep = _keep_count(total, density)
        if n_keep <= 0:
            chosen: List[Tuple[float, int, int, str]] = []
        elif n_keep >= total:
            chosen = ranked
        else:
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            chosen = ranked[:n_keep]
        keep_by_layer = {spec.name: [] for spec in specs}
        for _, _, index, name in chosen:
            keep_by_layer[name].append(index)
        for name in keep_by_layer:
            keep_by_layer[name].sort()

    pruned: Dict[str, List[float]] = {}
    masks: Dict[str, List[int]] = {}
    for spec in specs:
        layer_pruned, mask = _apply_mask(
            model[spec.name], keep_by_layer[spec.name], spec.shape
        )
        pruned[spec.name] = layer_pruned
        masks[spec.name] = mask
    return pruned, masks, scores, density, scope, mode


def lamp_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
    mode: str = "lamp",
) -> Dict[str, List[float]]:
    """One-shot LAMP prune of every dense and conv layer.

    Default ``scope="global"`` ranks every LAMP score together (the
    paper's layer-adaptive allocation). ``scope="layer"`` ranks inside
    each layer; ``per_layer`` then overrides ``density`` by layer name.
    ``mode`` selects classic LAMP or Frobenius-normalized magnitude.
    """
    pruned, _, _, _, _, _ = _lamp_prune_detailed(
        specs, model, density, scope, per_layer, mode
    )
    return pruned


def lamp_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
    mode: str = "lamp",
) -> Dict[str, object]:
    """Run LAMP and return per-layer keep counts, masks, and score totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the chosen LAMP / Frobenius scores.
    """
    pruned, masks, scores, density, scope, mode = _lamp_prune_detailed(
        specs, model, density, scope, per_layer, mode
    )
    rows: List[Dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item.name):
        layer_scores = scores[spec.name]
        mask = masks[spec.name]
        total = len(mask)
        kept = sum(mask)
        score_sum = sum(layer_scores)
        max_score = max(layer_scores) if layer_scores else 0.0
        fro_norm = frobenius_norm(model[spec.name]) if model[spec.name] else 0.0
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "total_weights": total,
            "kept_weights": kept,
            "survival_fraction": round(mask_density(mask), 4) if total else 0.0,
            "score_sum": score_sum,
            "max_score": max_score,
            "frobenius_norm": fro_norm,
        })
    total = sum(int(row["total_weights"]) for row in rows)
    total_kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "density": density,
        "sparsity": 1.0 - density,
        "scope": scope,
        "mode": mode,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = [
    "frobenius_norm",
    "lamp_scores",
    "lamp_prune_layer",
    "lamp_prune_model",
    "lamp_prune_summary",
]
