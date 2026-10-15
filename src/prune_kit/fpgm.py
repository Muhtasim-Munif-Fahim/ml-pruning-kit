"""Filter Pruning via Geometric Median (FPGM; He et al., ICCV 2019).

Instead of ranking filters by their own L1/L2 norms, FPGM ranks each
output filter by its *distance to the geometric median of all filters*
in the layer. Filters near the geometric median are redundant and can be
pruned; those far from it carry more unique information.

The geometric median is estimated with Weiszfeld's algorithm (iterative
reweighted means). When a filter lands exactly on the current estimate
the classical update is undefined; we fall back to the arithmetic mean
of the remaining points for that iteration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .structured import (
    _copy_layer,
    _product,
    _validate_density,
    _validate_shape,
    _validate_weights_shape,
    channel_groups,
)

__all__ = [
    "geometric_median",
    "fpgm_distances",
    "fpgm_prune_layer",
    "fpgm_prune_model",
]


def _as_filter_matrix(
    weights: Sequence[float],
    shape: tuple[int, ...],
) -> List[List[float]]:
    """Return one row per output filter (flattened)."""
    _validate_weights_shape(weights, shape)
    groups = channel_groups(shape, structure="filter")
    return [[float(weights[i]) for i in group] for group in groups]


def geometric_median(
    points: Sequence[Sequence[float]],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[float]:
    """Weiszfeld estimate of the geometric median of ``points``."""
    if not points:
        raise ValueError("points must be non-empty")
    dim = len(points[0])
    if dim == 0:
        raise ValueError("points must have positive dimension")
    for p in points:
        if len(p) != dim:
            raise ValueError("all points must share the same dimension")
    # Initialise at the coordinate-wise mean
    estimate = [sum(p[d] for p in points) / len(points) for d in range(dim)]
    for _ in range(max_iter):
        numer = [0.0] * dim
        denom = 0.0
        coincident = False
        for p in points:
            dist = math.sqrt(sum((p[d] - estimate[d]) ** 2 for d in range(dim)))
            if dist < 1e-12:
                coincident = True
                break
            w = 1.0 / dist
            denom += w
            for d in range(dim):
                numer[d] += w * p[d]
        if coincident or denom == 0.0:
            # Fall back to arithmetic mean of non-coincident points
            others = [
                p
                for p in points
                if math.sqrt(sum((p[d] - estimate[d]) ** 2 for d in range(dim))) >= 1e-12
            ]
            if not others:
                return estimate
            new_est = [sum(p[d] for p in others) / len(others) for d in range(dim)]
        else:
            new_est = [numer[d] / denom for d in range(dim)]
        shift = math.sqrt(sum((new_est[d] - estimate[d]) ** 2 for d in range(dim)))
        estimate = new_est
        if shift < tol:
            break
    return estimate


def fpgm_distances(
    weights: Sequence[float],
    shape: tuple[int, ...],
) -> List[float]:
    """Euclidean distance of each output filter to the geometric median."""
    filters = _as_filter_matrix(weights, shape)
    median = geometric_median(filters)
    distances: List[float] = []
    for filt in filters:
        dist = math.sqrt(sum((filt[d] - median[d]) ** 2 for d in range(len(median))))
        distances.append(dist)
    return distances


def fpgm_prune_layer(
    weights: Sequence[float],
    shape: tuple[int, ...],
    *,
    density: float = 0.5,
) -> List[float]:
    """Zero the output filters closest to the geometric median.

    ``density`` is the fraction of filters to **keep** (those farthest
    from the median). The returned buffer has the same length as the
    input; pruned filters are zeroed in place.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_density(density)
    distances = fpgm_distances(weights, shape)
    n_groups = len(distances)
    # Keep filters with *largest* distance to the median
    ranked = sorted(range(n_groups), key=lambda i: (-distances[i], i))
    n_keep = int(round(density * n_groups))
    if n_keep <= 0:
        keep_set: set[int] = set()
    elif n_keep >= n_groups:
        keep_set = set(range(n_groups))
    else:
        keep_set = set(ranked[:n_keep])
    groups = channel_groups(shape, structure="filter")
    keep_weight_indices = {
        index for channel in keep_set for index in groups[channel]
    }
    return [
        float(value) if index in keep_weight_indices else 0.0
        for index, value in enumerate(weights)
    ]


def fpgm_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """Apply :func:`fpgm_prune_layer` to every conv layer in ``model``.

    Non-conv layers are copied unchanged. ``per_layer`` may override
    ``density`` for named layers.
    """
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
        pruned[spec.name] = fpgm_prune_layer(
            weights, spec.shape, density=layer_density
        )
    return pruned
