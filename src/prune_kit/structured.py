"""Structured (channel / filter) pruning for conv-style layers.

Unlike unstructured magnitude pruning, which zeros individual weights,
structured pruning zeros *entire* output filters or input channels.
Filters and channels are ranked by their L1 or L2 norms. The tensor
shape is unchanged, so the result is still a valid conv kernel — just
with some filters or channels gone.

Layout is C-contiguous ``(out_channels, in_channels, kernel_h, kernel_w)``,
matching :func:`prune_kit.layers.conv_layer`.

``structure='filter'`` ranks and zeros **output filters** (dimension 0).
``structure='channel'`` ranks and zeros **input channels** (dimension 1).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate_norm(norm: str) -> str:
    key = str(norm).lower()
    if key not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")
    return key


def _validate_structure(structure: str) -> str:
    key = str(structure).lower()
    if key not in {"filter", "channel"}:
        raise ValueError("structure must be 'filter' or 'channel'")
    return key


def _validate_density(density: float) -> None:
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) < 2:
        raise ValueError(
            "structured pruning requires a shape with at least 2 dimensions"
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


def channel_groups(
    shape: tuple[int, ...],
    *,
    structure: str = "filter",
) -> List[List[int]]:
    """Return the flat-index group for every filter or input channel.

    Layout is C-contiguous. Each output filter occupies a contiguous
    block of size ``prod(shape[1:])``. Each input channel occupies, for
    every filter, a contiguous block of size ``prod(shape[2:])``.
    """
    _validate_shape(shape)
    structure = _validate_structure(structure)

    if structure == "filter":
        n_groups = int(shape[0])
        group_size = _product(shape[1:])
        return [
            list(range(i * group_size, (i + 1) * group_size))
            for i in range(n_groups)
        ]

    n_out = int(shape[0])
    n_in = int(shape[1])
    inner = _product(shape[2:]) if len(shape) > 2 else 1
    block = n_in * inner
    groups: List[List[int]] = []
    for channel in range(n_in):
        indices: List[int] = []
        for filt in range(n_out):
            start = filt * block + channel * inner
            indices.extend(range(start, start + inner))
        groups.append(indices)
    return groups


def vector_norm(values: Sequence[float], *, norm: str = "l1") -> float:
    """Return the L1 or L2 norm of ``values``."""
    norm = _validate_norm(norm)
    if not values:
        return 0.0
    if norm == "l1":
        return sum(abs(float(value)) for value in values)
    return math.sqrt(sum(float(value) * float(value) for value in values))


def channel_norms(
    weights: Sequence[float],
    shape: tuple[int, ...],
    *,
    norm: str = "l1",
    structure: str = "filter",
) -> List[float]:
    """L1 or L2 norm of every output filter or input channel."""
    _validate_weights_shape(weights, shape)
    groups = channel_groups(shape, structure=structure)
    return [
        vector_norm([weights[index] for index in group], norm=norm)
        for group in groups
    ]


def _select_keep_indices(
    weights: Sequence[float],
    shape: tuple[int, ...],
    *,
    density: float,
    norm: str,
    structure: str,
) -> Tuple[List[int], List[int], List[float]]:
    """Return ``(kept, pruned, norms)`` channel indices and ranking norms.

    Ranking is largest-norm first; ties break toward the lower index,
    matching :func:`prune_kit.prune.magnitude_prune_layer`.
    """
    _validate_density(density)
    norms = channel_norms(weights, shape, norm=norm, structure=structure)
    n_groups = len(norms)
    ranked = sorted(range(n_groups), key=lambda index: (-norms[index], index))
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
    return kept, pruned, norms


def structured_keep_indices(
    weights: Sequence[float],
    shape: tuple[int, ...],
    *,
    density: float = 0.5,
    norm: str = "l1",
    structure: str = "filter",
) -> List[int]:
    """Return the kept filter/channel indices after a structured prune."""
    kept, _, _ = _select_keep_indices(
        weights, shape, density=density, norm=norm, structure=structure
    )
    return kept


def structured_prune_layer(
    weights: Sequence[float],
    shape: tuple[int, ...],
    *,
    density: float = 0.5,
    norm: str = "l1",
    structure: str = "filter",
) -> List[float]:
    """Zero whole filters or channels with the smallest L1/L2 norms.

    ``density`` is the fraction of filters/channels to **keep**. The
    original ``weights`` sequence is not modified. The returned buffer
    has the same length (and implied shape) as the input.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    kept, _, _ = _select_keep_indices(
        weights, shape, density=density, norm=norm, structure=structure
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


def structured_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    norm: str = "l1",
    structure: str = "filter",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """Structured-prune every conv layer in ``model``.

    Non-conv layers are copied unchanged. ``per_layer`` overrides
    ``density`` for a subset of layers. The input mapping is not
    modified.
    """
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    spec_names = [spec.name for spec in specs]
    if set(spec_names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    if not any(spec.kind == "conv" for spec in specs):
        raise ValueError("structured pruning requires at least one conv layer")
    _validate_norm(norm)
    _validate_structure(structure)
    _validate_density(density)
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
            pruned[spec.name] = _copy_layer(weights)
            continue
        layer_density = (
            density if per_layer is None else per_layer.get(spec.name, density)
        )
        pruned[spec.name] = structured_prune_layer(
            weights,
            spec.shape,
            density=layer_density,
            norm=norm,
            structure=structure,
        )
    return pruned


__all__ = [
    "channel_groups",
    "channel_norms",
    "structured_keep_indices",
    "structured_prune_layer",
    "structured_prune_model",
    "vector_norm",
]
