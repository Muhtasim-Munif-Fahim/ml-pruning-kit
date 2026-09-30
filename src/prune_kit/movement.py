"""Movement pruning (Sanh et al., EMNLP 2020).

Movement pruning scores every scalar weight by how far it has moved
from its initialization during fine-tuning:

    score_i = |W_t[i] - W_0[i]|

Connections with the **lowest** scores (smallest absolute movement) are
pruned first; the densest keepers are those that moved the most. This
matches the paper's convention: importance is absolute weight movement
from init, and lowest-importance weights are removed.

Pass either ``initial_weights`` (``W_0``) so scores are computed as
``|W_t - W_0|``, or precomputed ``movements`` (already absolute
movement, e.g. a cumulative sum of per-step ``|ΔW|``). Pass one, not
both.

``density`` is the fraction of connections to **keep**. The number kept
is ``round(density * N)``, matching the other pruners in this kit.
Default ``scope="global"`` ranks every movement score together;
``scope="layer"`` keeps the same fraction inside each layer. Dense
``(out, in)`` and conv ``(out, in, kh, kw)`` flat buffers are both
supported. A weight that is already zero stays zero: kept positions
copy the original (current) value.
"""

from __future__ import annotations

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


def _movement_terms(
    weights: Sequence[float],
    initial_weights: Sequence[float] | None,
    movements: Sequence[float] | None,
) -> List[float]:
    """Per-weight absolute movement scores."""
    if movements is not None and initial_weights is not None:
        raise ValueError("pass initial_weights or movements, not both")
    if movements is None and initial_weights is None:
        raise ValueError("initial_weights or movements is required")
    if movements is not None:
        if len(movements) != len(weights):
            raise ValueError(
                f"movements has {len(movements)} values, "
                f"weights has {len(weights)}"
            )
        return [abs(float(value)) for value in movements]
    assert initial_weights is not None
    if len(initial_weights) != len(weights):
        raise ValueError(
            f"initial_weights has {len(initial_weights)} values, "
            f"weights has {len(weights)}"
        )
    return [
        abs(float(current) - float(initial))
        for current, initial in zip(weights, initial_weights)
    ]


def movement_scores(
    weights: Sequence[float],
    initial_weights: Sequence[float] | None = None,
    *,
    movements: Sequence[float] | None = None,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Absolute movement score of each weight, same length as ``weights``.

    Default scores are ``|W_t - W_0|``. ``movements``, when given,
    replaces that difference with a caller-supplied per-connection term
    (absolute value is taken) — useful for cumulative movement
    ``sum_t |ΔW_t|``. ``shape``, when given, must match the flat buffer
    (dense ``(out, in)`` or conv ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_shape(weights, shape)
    return _movement_terms(weights, initial_weights, movements)


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


def movement_prune_layer(
    weights: Sequence[float],
    initial_weights: Sequence[float] | None = None,
    *,
    movements: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Zero the lowest-movement connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequences are not modified. Kept positions copy the original
    (current) weight; every dropped position is ``0.0``.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = movement_scores(
        weights,
        initial_weights,
        movements=movements,
        shape=resolved,
    )
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    initial_weights: Dict[str, Sequence[float]] | None,
    movements: Dict[str, Sequence[float]] | None,
) -> Tuple[str, Dict[str, Sequence[float]]]:
    if movements is not None and initial_weights is not None:
        raise ValueError("pass initial_weights or movements, not both")
    if movements is None and initial_weights is None:
        raise ValueError("initial_weights or movements is required")
    if movements is not None:
        return "movements", movements
    assert initial_weights is not None
    return "initial_weights", initial_weights


def _layer_scores(
    spec: LayerSpec,
    weights: Sequence[float],
    source: str,
    saliency: Dict[str, Sequence[float]],
) -> List[float]:
    expected = layer_weight_count(spec)
    if len(weights) != expected:
        raise ValueError(
            f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
        )
    if spec.name not in saliency:
        raise ValueError(f"{source} missing layer {spec.name!r}")
    values = saliency[spec.name]
    initial = values if source == "initial_weights" else None
    moves = values if source == "movements" else None
    return movement_scores(
        weights,
        initial,
        movements=moves,
        shape=spec.shape,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    initial_weights: Dict[str, Sequence[float]] | None,
    movements: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[float, str, str, Dict[str, Sequence[float]]]:
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
    if per_layer:
        if scope != "layer":
            raise ValueError("per_layer requires scope='layer'")
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            _validate_density(layer_density)
    source, saliency = _resolve_saliency(initial_weights, movements)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency


def _movement_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    initial_weights: Dict[str, Sequence[float]] | None,
    movements: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency = _validate_model_args(
        specs,
        model,
        initial_weights,
        movements,
        density,
        scope,
        per_layer,
    )
    scores = {
        spec.name: _layer_scores(spec, model[spec.name], source, saliency)
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
    return pruned, masks, scores, density, scope


def movement_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    initial_weights: Dict[str, Sequence[float]] | None = None,
    *,
    movements: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """One-shot movement prune of every dense and conv layer.

    Default ``scope="global"`` ranks every movement score together.
    ``scope="layer"`` ranks inside each layer; ``per_layer`` then
    overrides ``density`` by layer name. Pass ``initial_weights``
    (``W_0``) or precomputed ``movements``, not both.
    """
    pruned, _, _, _, _ = _movement_prune_detailed(
        specs,
        model,
        initial_weights,
        movements,
        density,
        scope,
        per_layer,
    )
    return pruned


def movement_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    initial_weights: Dict[str, Sequence[float]] | None = None,
    *,
    movements: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run movement pruning and return per-layer keep counts and scores.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the absolute movement scores ``|W_t - W_0|`` (or supplied movements).
    """
    pruned, masks, scores, density, scope = _movement_prune_detailed(
        specs,
        model,
        initial_weights,
        movements,
        density,
        scope,
        per_layer,
    )
    rows: List[Dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item.name):
        layer_scores = scores[spec.name]
        mask = masks[spec.name]
        total = len(mask)
        kept = sum(mask)
        score_sum = sum(layer_scores)
        max_score = max(layer_scores) if layer_scores else 0.0
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "total_weights": total,
            "kept_weights": kept,
            "survival_fraction": round(mask_density(mask), 4) if total else 0.0,
            "score_sum": score_sum,
            "max_score": max_score,
        })
    total = sum(int(row["total_weights"]) for row in rows)
    total_kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "density": density,
        "sparsity": 1.0 - density,
        "scope": scope,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = [
    "movement_scores",
    "movement_prune_layer",
    "movement_prune_model",
    "movement_prune_summary",
]
