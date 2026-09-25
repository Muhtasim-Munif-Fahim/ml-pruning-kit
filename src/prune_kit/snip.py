"""Single-shot Network Pruning (SNIP) by connection sensitivity.

SNIP (Lee, Ajanthan, and Torr, ICLR 2019) scores every scalar connection
by the effect of removing it, using one gradient evaluation (typically
a single minibatch at initialization):

    s_j = |dL/dc_j| = |weight_j * grad_j|

evaluated at the auxiliary connection indicator c = 1. The lowest
scores are zeroed in one shot. ``scope="global"`` (the paper) ranks
every dense and conv connection in the model together. ``scope="layer"``
keeps the same fraction inside each layer.

``density`` is the fraction of connections to **keep**. The number kept
is ``round(density * N)``, the same rounding the magnitude pruners use.
Ties break toward the earlier layer in ``specs`` order, then the lower
index. A weight that is already zero stays zero: the kept value is the
original weight, and pruned positions are written as ``0.0``.

Conv kernels are the flattened C-contiguous buffer matching
:func:`prune_kit.layers.conv_layer`. Pass ``grads`` or precomputed
``contributions`` (the raw ``dL/dc`` or ``weight * grad`` products),
not both. Sequences may be lists or any array-like buffer; values are
read with ``float`` so a NumPy array works without taking a NumPy
dependency.

Normalization by the sum of sensitivities (the paper's ``s / sum(s)``)
does not change the ranking. :func:`snip_scores` can still return that
normalized form with ``normalize=True``.
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


def _connection_terms(
    weights: Sequence[float],
    grads: Sequence[float] | None,
    contributions: Sequence[float] | None,
) -> List[float]:
    """Absolute connection sensitivity ``|weight * grad|`` or ``|contribution|``."""
    if contributions is not None and grads is not None:
        raise ValueError("pass grads or contributions, not both")
    if contributions is None and grads is None:
        raise ValueError("grads or contributions is required")
    if contributions is not None:
        if len(contributions) != len(weights):
            raise ValueError(
                f"contributions has {len(contributions)} values, "
                f"weights has {len(weights)}"
            )
        return [abs(float(value)) for value in contributions]
    assert grads is not None
    if len(grads) != len(weights):
        raise ValueError(
            f"grads has {len(grads)} values, weights has {len(weights)}"
        )
    return [
        abs(float(weight) * float(grad))
        for weight, grad in zip(weights, grads)
    ]


def _normalize(scores: Sequence[float]) -> List[float]:
    total = sum(scores)
    if total == 0.0:
        return [0.0 for _ in scores]
    return [score / total for score in scores]


def snip_scores(
    weights: Sequence[float],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    normalize: bool = False,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Connection sensitivity of each weight, same length as ``weights``.

    Each score is ``|weight * grad|``. ``contributions``, when given,
    replaces that product with a caller-supplied per-connection term
    (already ``dL/dc`` or ``weight * grad``); the absolute value is
    taken. ``normalize=True`` divides by the sum of scores in this
    buffer so they sum to 1. A zero sum yields zeros.

    ``shape``, when given, must match the flat buffer (dense
    ``(out, in)`` or conv ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_shape(weights, shape)
    scores = _connection_terms(weights, grads, contributions)
    if normalize:
        return _normalize(scores)
    return scores


def _keep_count(total: int, density: float) -> int:
    return int(round(density * total))


def _keep_highest(scores: Sequence[float], density: float) -> List[int]:
    """Indices of the highest scores, in ascending index order.

    Ties break toward the lower index. ``density`` is the fraction to
    keep; the count is ``round(density * N)``.
    """
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
    """Zero connections outside ``keep_indices``. Already-zero weights stay zero."""
    mask = sparse_mask_to_dense(keep_indices, shape)
    pruned = [
        float(value) if bit else 0.0
        for value, bit in zip(weights, mask)
    ]
    return pruned, mask


def snip_prune_layer(
    weights: Sequence[float],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Zero the lowest-sensitivity connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequences are not modified. Kept positions copy the original weight,
    so a zero stays zero; every dropped position is ``0.0``.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = snip_scores(
        weights,
        grads,
        contributions=contributions,
        shape=resolved,
    )
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
) -> Tuple[str, Dict[str, Sequence[float]]]:
    if contributions is not None and grads is not None:
        raise ValueError("pass grads or contributions, not both")
    if contributions is None and grads is None:
        raise ValueError("grads or contributions is required")
    if contributions is not None:
        return "contributions", contributions
    assert grads is not None
    return "grads", grads


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
    if len(values) != expected:
        raise ValueError(
            f"layer {spec.name!r} {source} has {len(values)} values, expected {expected}"
        )
    grads = values if source == "grads" else None
    contributions = values if source == "contributions" else None
    return snip_scores(
        weights,
        grads,
        contributions=contributions,
        shape=spec.shape,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
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
    source, saliency = _resolve_saliency(grads, contributions)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency


def _snip_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency = _validate_model_args(
        specs, model, grads, contributions, density, scope, per_layer
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
            keep_by_layer[spec.name] = _keep_highest(scores[spec.name], layer_density)
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


def snip_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """One-shot SNIP prune of every dense and conv layer.

    ``scope="global"`` keeps the top ``density`` fraction of connections
    across the whole model. ``scope="layer"`` ranks inside each layer;
    ``per_layer`` then overrides ``density`` by layer name. The input
    mappings are not modified. Already-zero weights stay zero.
    """
    pruned, _, _, _, _ = _snip_prune_detailed(
        specs,
        model,
        grads,
        contributions,
        density,
        scope,
        per_layer,
    )
    return pruned


def snip_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run SNIP and return per-layer keep counts, masks, and score totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. ``masks`` are
    flat 0/1 buffers from :func:`prune_kit.masks.sparse_mask_to_dense`.
    ``pruned_weights`` never writes a non-zero into a previously zero
    position. Score totals use the unnormalized ``|weight * grad|``
    values; ranking uses those same full-precision scores.
    """
    pruned, masks, scores, density, scope = _snip_prune_detailed(
        specs,
        model,
        grads,
        contributions,
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
    "snip_scores",
    "snip_prune_layer",
    "snip_prune_model",
    "snip_prune_summary",
]
