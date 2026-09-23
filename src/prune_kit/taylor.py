"""First-order Taylor (soft-filter) pruning for conv-style layers.

Structured L1/L2 pruning ranks a filter by the norm of its weights.
This module ranks each output filter or input channel by a first-order
Taylor saliency and zeros the lowest-scoring groups. The kernel shape
is unchanged, so the result is still a valid conv tensor with some
filters or channels removed — a soft, shape-preserving prune rather
than a hard reshape.

The default per-weight term is ``|grad * weight|``, the absolute
first-order estimate of how much the loss moves if that weight is
removed (Molchanov et al., *Pruning Convolutional Neural Networks for
Resource Efficient Inference*). ``criterion='sq'`` sums the squared
products ``(grad * weight) ** 2`` instead (Molchanov et al.,
*Importance Estimation for Neural Network Pruning*). A filter score is
the sum of its terms, or their mean when ``reduction='mean'``. Within
one layer every group has the same size, so sum and mean rank alike;
mean is the scale-free form of the same proxy.

If the saliency was already computed per weight — for example the
activation-map products ``(dL/dz) * z`` aligned to the kernel layout —
pass those raw products as ``contributions``. The same absolute or
squared reduction is applied. Pass ``grads`` or ``contributions``, not
both.

Layout is C-contiguous ``(out_channels, in_channels, kernel_h, kernel_w)``,
matching :func:`prune_kit.layers.conv_layer`.

``structure='filter'`` ranks and zeros **output filters** (dimension 0).
``structure='channel'`` ranks and zeros **input channels** (dimension 1).
Ranking keeps the highest scores. Ties break toward the lower index,
matching :func:`prune_kit.structured.structured_prune_layer`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .structured import channel_groups


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate_criterion(criterion: str) -> str:
    key = str(criterion).lower()
    if key not in {"abs", "sq"}:
        raise ValueError("criterion must be 'abs' or 'sq'")
    return key


def _validate_reduction(reduction: str) -> str:
    key = str(reduction).lower()
    if key not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")
    return key


def _validate_density(density: float) -> None:
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) < 2:
        raise ValueError(
            "Taylor pruning requires a shape with at least 2 dimensions"
        )
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive")


def _validate_weights_shape(
    weights: Sequence[float],
    shape: tuple[int, ...],
) -> None:
    _validate_shape(shape)
    expected = _product(shape)
    if len(weights) != expected:
        raise ValueError(
            f"weights has {len(weights)} values, shape {shape} expects {expected}"
        )


def _per_weight_terms(
    weights: Sequence[float],
    grads: Sequence[float] | None,
    contributions: Sequence[float] | None,
    criterion: str,
) -> List[float]:
    """Return the reduced per-weight saliency ``|g*w|`` or ``(g*w)**2``."""
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
        raw = [float(value) for value in contributions]
    else:
        assert grads is not None
        if len(grads) != len(weights):
            raise ValueError(
                f"grads has {len(grads)} values, weights has {len(weights)}"
            )
        raw = [
            float(weight) * float(grad)
            for weight, grad in zip(weights, grads)
        ]
    if criterion == "abs":
        return [abs(value) for value in raw]
    return [value * value for value in raw]


def taylor_channel_scores(
    weights: Sequence[float],
    shape: tuple[int, ...],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    structure: str = "filter",
    criterion: str = "abs",
    reduction: str = "sum",
) -> List[float]:
    """First-order Taylor score of every output filter or input channel.

    Each score is the sum (or mean) of ``|grad * weight|`` over the
    group. ``criterion='sq'`` uses ``(grad * weight) ** 2`` before the
    reduction. ``contributions``, when given, replaces ``grad * weight``
    with a caller-supplied per-weight FO term of the same length.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_weights_shape(weights, shape)
    criterion = _validate_criterion(criterion)
    reduction = _validate_reduction(reduction)
    terms = _per_weight_terms(weights, grads, contributions, criterion)
    groups = channel_groups(shape, structure=structure)
    scores: List[float] = []
    for group in groups:
        total = sum(terms[index] for index in group)
        if reduction == "mean":
            total = total / len(group)
        scores.append(total)
    return scores


def _select_keep_indices(
    weights: Sequence[float],
    shape: tuple[int, ...],
    grads: Sequence[float] | None,
    *,
    contributions: Sequence[float] | None,
    density: float,
    structure: str,
    criterion: str,
    reduction: str,
) -> Tuple[List[int], List[int], List[float]]:
    """Return ``(kept, pruned, scores)`` after ranking highest score first."""
    _validate_density(density)
    scores = taylor_channel_scores(
        weights,
        shape,
        grads,
        contributions=contributions,
        structure=structure,
        criterion=criterion,
        reduction=reduction,
    )
    n_groups = len(scores)
    ranked = sorted(range(n_groups), key=lambda index: (-scores[index], index))
    n_keep = int(round(density * n_groups))
    if n_keep <= 0:
        kept: List[int] = []
        pruned = list(range(n_groups))
    elif n_keep >= n_groups:
        kept = list(range(n_groups))
        pruned = []
    else:
        keep_set = set(ranked[:n_keep])
        kept = [index for index in range(n_groups) if index in keep_set]
        pruned = [index for index in range(n_groups) if index not in keep_set]
    return kept, pruned, scores


def taylor_keep_indices(
    weights: Sequence[float],
    shape: tuple[int, ...],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    density: float = 0.5,
    structure: str = "filter",
    criterion: str = "abs",
    reduction: str = "sum",
) -> List[int]:
    """Return the kept filter/channel indices after a Taylor prune."""
    kept, _, _ = _select_keep_indices(
        weights,
        shape,
        grads,
        contributions=contributions,
        density=density,
        structure=structure,
        criterion=criterion,
        reduction=reduction,
    )
    return kept


def taylor_prune_layer(
    weights: Sequence[float],
    shape: tuple[int, ...],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    density: float = 0.5,
    structure: str = "filter",
    criterion: str = "abs",
    reduction: str = "sum",
) -> List[float]:
    """Zero whole filters or channels with the lowest Taylor scores.

    ``density`` is the fraction of filters/channels to **keep**. The
    original ``weights`` and ``grads`` sequences are not modified. The
    returned buffer has the same length (and implied shape) as the input.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    kept, _, _ = _select_keep_indices(
        weights,
        shape,
        grads,
        contributions=contributions,
        density=density,
        structure=structure,
        criterion=criterion,
        reduction=reduction,
    )
    groups = channel_groups(shape, structure=structure)
    keep_weight_indices = {
        index for channel in kept for index in groups[channel]
    }
    return [
        float(value) if index in keep_weight_indices else 0.0
        for index, value in enumerate(weights)
    ]


def _copy_layer(weights: Sequence[float]) -> List[float]:
    return [float(value) for value in weights]


def _resolve_saliency(
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
) -> tuple[str, Dict[str, Sequence[float]]]:
    """Return ``('grads'|'contributions', mapping)`` after validation."""
    if contributions is not None and grads is not None:
        raise ValueError("pass grads or contributions, not both")
    if contributions is None and grads is None:
        raise ValueError("grads or contributions is required")
    if contributions is not None:
        return "contributions", contributions
    assert grads is not None
    return "grads", grads


def taylor_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    structure: str = "filter",
    criterion: str = "abs",
    reduction: str = "sum",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """Taylor-prune every conv layer in ``model``.

    Non-conv layers are copied unchanged. ``per_layer`` overrides
    ``density`` for a subset of layers. ``grads`` (or ``contributions``)
    must include every conv layer. The input mappings are not modified.
    """
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    spec_names = [spec.name for spec in specs]
    if set(spec_names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    if not any(spec.kind == "conv" for spec in specs):
        raise ValueError("Taylor pruning requires at least one conv layer")
    _validate_criterion(criterion)
    _validate_reduction(reduction)
    _validate_density(density)
    source, saliency = _resolve_saliency(model, grads, contributions)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    if per_layer is not None:
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            _validate_density(layer_density)

    pruned: Dict[str, List[float]] = {}
    for spec in specs:
        weights = model[spec.name]
        expected = layer_weight_count(spec)
        if len(weights) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
            )
        if spec.kind != "conv":
            if spec.name in saliency and len(saliency[spec.name]) != expected:
                raise ValueError(
                    f"layer {spec.name!r} {source} has {len(saliency[spec.name])} "
                    f"values, expected {expected}"
                )
            pruned[spec.name] = _copy_layer(weights)
            continue
        if spec.name not in saliency:
            raise ValueError(f"{source} missing conv layer {spec.name!r}")
        layer_saliency = saliency[spec.name]
        if len(layer_saliency) != expected:
            raise ValueError(
                f"layer {spec.name!r} {source} has {len(layer_saliency)} values, "
                f"expected {expected}"
            )
        layer_density = (
            density if per_layer is None else per_layer.get(spec.name, density)
        )
        layer_grads = layer_saliency if source == "grads" else None
        layer_contrib = layer_saliency if source == "contributions" else None
        pruned[spec.name] = taylor_prune_layer(
            weights,
            spec.shape,
            layer_grads,
            contributions=layer_contrib,
            density=layer_density,
            structure=structure,
            criterion=criterion,
            reduction=reduction,
        )
    return pruned


def taylor_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    structure: str = "filter",
    criterion: str = "abs",
    reduction: str = "sum",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run Taylor pruning and return per-channel scores plus aggregate stats.

    Non-conv layers are copied through and listed under ``skipped``.
    ``kept`` follows the FO ranking, including a filter that was already
    zero and was still selected. Scores in the report are rounded for
    display; ranking uses the full-precision scores.
    """
    pruned = taylor_prune_model(
        specs,
        model,
        grads,
        contributions=contributions,
        density=density,
        structure=structure,
        criterion=criterion,
        reduction=reduction,
        per_layer=per_layer,
    )
    structure_key = str(structure).lower()
    criterion_key = str(criterion).lower()
    reduction_key = str(reduction).lower()
    rows: List[Dict[str, object]] = []
    conv_specs = sorted(
        (spec for spec in specs if spec.kind == "conv"),
        key=lambda spec: spec.name,
    )
    for spec in conv_specs:
        layer_density = (
            density if per_layer is None else per_layer.get(spec.name, density)
        )
        layer_grads = None if grads is None else grads.get(spec.name)
        layer_contrib = (
            None if contributions is None else contributions.get(spec.name)
        )
        scores = taylor_channel_scores(
            model[spec.name],
            spec.shape,
            layer_grads,
            contributions=layer_contrib,
            structure=structure_key,
            criterion=criterion_key,
            reduction=reduction_key,
        )
        kept_indices = taylor_keep_indices(
            model[spec.name],
            spec.shape,
            layer_grads,
            contributions=layer_contrib,
            density=layer_density,
            structure=structure_key,
            criterion=criterion_key,
            reduction=reduction_key,
        )
        kept_set = set(kept_indices)
        channels: List[Dict[str, object]] = []
        pruned_indices: List[int] = []
        for index, score in enumerate(scores):
            kept = index in kept_set
            channels.append({
                "index": index,
                "score": round(score, 6),
                "kept": kept,
            })
            if not kept:
                pruned_indices.append(index)
        total_channels = len(scores)
        kept_channels = len(kept_indices)
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "structure": structure_key,
            "criterion": criterion_key,
            "reduction": reduction_key,
            "total_channels": total_channels,
            "kept_channels": kept_channels,
            "survival_fraction": (
                round(kept_channels / total_channels, 4) if total_channels else 0.0
            ),
            "kept_indices": kept_indices,
            "pruned_indices": pruned_indices,
            "channels": channels,
        })
    total = sum(int(row["total_channels"]) for row in rows)
    total_kept = sum(int(row["kept_channels"]) for row in rows)
    skipped = [
        spec.name for spec in sorted(specs, key=lambda item: item.name)
        if spec.kind != "conv"
    ]
    return {
        "density": density,
        "structure": structure_key,
        "criterion": criterion_key,
        "reduction": reduction_key,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "skipped": skipped,
        "total_channels": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "pruned_weights": pruned,
    }


__all__ = [
    "taylor_channel_scores",
    "taylor_keep_indices",
    "taylor_prune_layer",
    "taylor_prune_model",
    "taylor_prune_summary",
]
