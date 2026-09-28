"""Wanda activation-aware pruning (Sun et al., 2023).

Wanda scores every scalar weight by its magnitude times the L2 norm of
the corresponding **input** activation column collected on a calibration
set:

    S_ij = |W_ij| * ||X_j||_2

The lowest scores are then zeroed with the same magnitude-style keep
count used elsewhere in the kit (``round(density * N)``). Ties break
toward the earlier layer in ``specs`` order, then the lower index. A
weight that is already zero stays zero.

Dense layers use shape ``(out, in)``: ``activation_norms`` has length
``in`` (one norm per input feature / column). Conv kernels use the flat
C-contiguous layout ``(out, in, kh, kw)``: ``activation_norms`` has
length ``in`` (one norm per input channel) and is broadcast across the
spatial dims. You may also pass a full per-weight buffer of the same
length as ``weights``, or precomputed ``contributions`` (already
``|W| * ||X||``). Pass ``activation_norms`` or ``contributions``, not
both.

``structure=None`` (default) prunes individual connections.
``structure="filter"`` / ``"channel"`` aggregates Wanda scores per
output filter or input channel (via
:func:`prune_kit.structured.channel_groups`) and zeros whole groups —
optional structured Wanda on top of the unstructured score.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density, sparse_mask_to_dense
from .structured import channel_groups


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


def _validate_structure(structure: str | None) -> str | None:
    if structure is None:
        return None
    key = str(structure).lower()
    if key not in {"filter", "channel"}:
        raise ValueError("structure must be 'filter', 'channel', or None")
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


def activation_column_norms(
    activations: Sequence[float],
    n_features: int,
) -> List[float]:
    """L2 norm of each column of a row-major ``(n_samples, n_features)`` matrix.

    ``activations`` is the flat calibration buffer of length
    ``n_samples * n_features``. Returns a list of length ``n_features``
    whose ``j``-th entry is ``||X_j||_2``.
    """
    n_features = int(n_features)
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if not activations:
        raise ValueError("activations must not be empty")
    if len(activations) % n_features != 0:
        raise ValueError(
            f"activations length {len(activations)} is not divisible by "
            f"n_features={n_features}"
        )
    n_samples = len(activations) // n_features
    norms: List[float] = []
    for feature in range(n_features):
        total = 0.0
        for sample in range(n_samples):
            value = float(activations[sample * n_features + feature])
            total += value * value
        norms.append(math.sqrt(total))
    return norms


def _expand_activation_norms(
    activation_norms: Sequence[float],
    shape: tuple[int, ...],
    n_weights: int,
) -> List[float]:
    """Broadcast column/channel norms to a flat per-weight buffer."""
    if len(activation_norms) == n_weights:
        return [float(value) for value in activation_norms]
    if len(shape) == 1:
        raise ValueError(
            f"activation_norms has {len(activation_norms)} values, "
            f"expected {n_weights} (flat buffer without column layout)"
        )
    n_in = int(shape[1]) if len(shape) >= 2 else int(shape[0])
    if len(activation_norms) != n_in:
        raise ValueError(
            f"activation_norms has {len(activation_norms)} values, "
            f"expected {n_in} (one per input feature/channel) or "
            f"{n_weights} (per-weight)"
        )
    n_out = int(shape[0])
    inner = _product(shape[2:]) if len(shape) > 2 else 1
    expanded: List[float] = []
    for _ in range(n_out):
        for channel in range(n_in):
            norm = float(activation_norms[channel])
            expanded.extend([norm] * inner)
    return expanded


def _wanda_terms(
    weights: Sequence[float],
    activation_norms: Sequence[float] | None,
    contributions: Sequence[float] | None,
    shape: tuple[int, ...],
) -> List[float]:
    """Per-weight Wanda scores ``|W| * ||X||`` (or absolute contributions)."""
    if contributions is not None and activation_norms is not None:
        raise ValueError("pass activation_norms or contributions, not both")
    if contributions is None and activation_norms is None:
        raise ValueError("activation_norms or contributions is required")
    if contributions is not None:
        if len(contributions) != len(weights):
            raise ValueError(
                f"contributions has {len(contributions)} values, "
                f"weights has {len(weights)}"
            )
        return [abs(float(value)) for value in contributions]
    assert activation_norms is not None
    norms = _expand_activation_norms(activation_norms, shape, len(weights))
    return [
        abs(float(weight)) * float(norm)
        for weight, norm in zip(weights, norms)
    ]


def wanda_scores(
    weights: Sequence[float],
    activation_norms: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Wanda score of each weight, same length as ``weights``.

    Each score is ``|weight| * activation_norm`` for the matching input
    column/channel. ``contributions``, when given, replaces that product
    with a caller-supplied per-connection term; the absolute value is
    taken. ``shape``, when given, must match the flat buffer (dense
    ``(out, in)`` or conv ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    resolved = _validate_shape(weights, shape)
    return _wanda_terms(weights, activation_norms, contributions, resolved)


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


def _structured_keep_weight_indices(
    scores: Sequence[float],
    shape: tuple[int, ...],
    density: float,
    structure: str,
) -> List[int]:
    """Keep whole filters/channels ranked by aggregated Wanda score."""
    groups = channel_groups(shape, structure=structure)
    group_scores = [sum(scores[index] for index in group) for group in groups]
    n_groups = len(group_scores)
    n_keep = _keep_count(n_groups, density)
    if n_keep <= 0:
        kept_groups: List[int] = []
    elif n_keep >= n_groups:
        kept_groups = list(range(n_groups))
    else:
        ranked = sorted(
            range(n_groups), key=lambda index: (-group_scores[index], index)
        )
        keep_set = set(ranked[:n_keep])
        kept_groups = [index for index in range(n_groups) if index in keep_set]
    keep_indices: List[int] = []
    for group_index in kept_groups:
        keep_indices.extend(groups[group_index])
    keep_indices.sort()
    return keep_indices


def wanda_prune_layer(
    weights: Sequence[float],
    activation_norms: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
    structure: str | None = None,
) -> List[float]:
    """Zero the lowest-Wanda-score connections (or groups) in one layer.

    ``density`` is the fraction of connections (or filters/channels when
    ``structure`` is set) to keep. The original sequences are not
    modified. Kept positions copy the original weight; every dropped
    position is ``0.0``.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    structure = _validate_structure(structure)
    resolved = _validate_shape(weights, shape)
    scores = wanda_scores(
        weights,
        activation_norms,
        contributions=contributions,
        shape=resolved,
    )
    if structure is None:
        if len(resolved) < 2 and shape is not None:
            # flat 1-d is fine for unstructured
            pass
        kept = _keep_highest(scores, density)
    else:
        if len(resolved) < 2:
            raise ValueError(
                "structured Wanda requires a shape with at least 2 dimensions"
            )
        kept = _structured_keep_weight_indices(
            scores, resolved, density, structure
        )
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    activation_norms: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
) -> Tuple[str, Dict[str, Sequence[float]]]:
    if contributions is not None and activation_norms is not None:
        raise ValueError("pass activation_norms or contributions, not both")
    if contributions is None and activation_norms is None:
        raise ValueError("activation_norms or contributions is required")
    if contributions is not None:
        return "contributions", contributions
    assert activation_norms is not None
    return "activation_norms", activation_norms


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
    activation_norms = values if source == "activation_norms" else None
    contributions = values if source == "contributions" else None
    return wanda_scores(
        weights,
        activation_norms,
        contributions=contributions,
        shape=spec.shape,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
    structure: str | None,
) -> Tuple[float, str, str, Dict[str, Sequence[float]], str | None]:
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
    structure = _validate_structure(structure)
    if structure is not None and scope == "global":
        # Structured prune is per-layer by nature (group counts differ).
        # Allow global only for unstructured; structured forces layer scope.
        raise ValueError("structured Wanda requires scope='layer'")
    if per_layer:
        if scope != "layer":
            raise ValueError("per_layer requires scope='layer'")
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            _validate_density(layer_density)
    source, saliency = _resolve_saliency(activation_norms, contributions)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency, structure


def _wanda_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
    structure: str | None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency, structure = _validate_model_args(
        specs,
        model,
        activation_norms,
        contributions,
        density,
        scope,
        per_layer,
        structure,
    )
    scores = {
        spec.name: _layer_scores(spec, model[spec.name], source, saliency)
        for spec in specs
    }
    keep_by_layer: Dict[str, List[int]] = {}
    if structure is not None:
        for spec in specs:
            layer_density = (
                density if not per_layer else per_layer.get(spec.name, density)
            )
            keep_by_layer[spec.name] = _structured_keep_weight_indices(
                scores[spec.name], spec.shape, layer_density, structure
            )
    elif scope == "layer":
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


def wanda_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "layer",
    per_layer: Dict[str, float] | None = None,
    structure: str | None = None,
) -> Dict[str, List[float]]:
    """One-shot Wanda prune of every dense and conv layer.

    Default ``scope="layer"`` matches the Wanda paper's per-layer
    sparsity target. ``scope="global"`` ranks every connection together
    (unstructured only). ``structure`` optionally zeros whole filters or
    channels by aggregated Wanda score and requires ``scope="layer"``.
    """
    pruned, _, _, _, _ = _wanda_prune_detailed(
        specs,
        model,
        activation_norms,
        contributions,
        density,
        scope,
        per_layer,
        structure,
    )
    return pruned


def wanda_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    activation_norms: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "layer",
    per_layer: Dict[str, float] | None = None,
    structure: str | None = None,
) -> Dict[str, object]:
    """Run Wanda and return per-layer keep counts, masks, and score totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the Wanda scores ``|W| * ||X||``.
    """
    pruned, masks, scores, density, scope = _wanda_prune_detailed(
        specs,
        model,
        activation_norms,
        contributions,
        density,
        scope,
        per_layer,
        structure,
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
        "structure": structure,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = [
    "activation_column_norms",
    "wanda_scores",
    "wanda_prune_layer",
    "wanda_prune_model",
    "wanda_prune_summary",
]
