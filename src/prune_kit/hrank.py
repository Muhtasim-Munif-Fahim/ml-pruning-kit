"""HRank: filter pruning by the average rank of feature maps (Lin et al., CVPR 2020).

HRank observes that the *rank* of the feature map a filter produces is
remarkably stable across input images, even from a small batch, and that
filters whose maps have low rank carry less information. Filters are ranked
by

    R(k) = (1 / G) * sum_g rank(O_k(I_g))

where ``O_k(I_g)`` is the ``h x w`` output map of filter ``k`` for image
``g``. The lowest-``R`` filters are pruned.

Everything here is framework-agnostic plain Python: feature maps are nested
lists ``maps[g][k][row][col]`` (images x filters x height x width), which is
what ``activation.tolist()`` gives for an ``(N, C, H, W)`` tensor. Matrix rank
is computed with Gaussian elimination and partial pivoting, using a tolerance
relative to the largest entry of the map (comparable to
``numpy.linalg.matrix_rank``'s default on well-conditioned maps).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .layers import LayerSpec, layer_weight_count
from .structured import _copy_layer, _validate_density, _validate_weights_shape, channel_groups

__all__ = [
    "matrix_rank",
    "hrank_scores",
    "hrank_keep_filters",
    "hrank_prune_layer",
    "hrank_prune_model",
    "hrank_prune_summary",
]

_DEFAULT_RTOL = 1e-9


def matrix_rank(matrix: Sequence[Sequence[float]], *, rtol: float = _DEFAULT_RTOL) -> int:
    """Numerical rank of a 2-D matrix via Gaussian elimination with partial pivoting.

    A pivot counts when its magnitude exceeds ``rtol * max(rows, cols) *
    max|a_ij|``. An all-zero (or empty) matrix has rank 0.
    """
    rows = [list(map(float, row)) for row in matrix]
    if not rows:
        return 0
    n_cols = len(rows[0])
    for row in rows:
        if len(row) != n_cols:
            raise ValueError("matrix rows must all have the same length")
    if n_cols == 0:
        return 0
    if rtol < 0:
        raise ValueError("rtol must be non-negative")
    scale = max(abs(v) for row in rows for v in row)
    if scale == 0.0:
        return 0
    tol = rtol * max(len(rows), n_cols) * scale
    rank = 0
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= len(rows):
            break
        best = max(range(pivot_row, len(rows)), key=lambda r: abs(rows[r][col]))
        if abs(rows[best][col]) <= tol:
            continue
        rows[pivot_row], rows[best] = rows[best], rows[pivot_row]
        pivot = rows[pivot_row][col]
        for r in range(pivot_row + 1, len(rows)):
            factor = rows[r][col] / pivot
            if factor != 0.0:
                target = rows[r]
                source = rows[pivot_row]
                for c in range(col, n_cols):
                    target[c] -= factor * source[c]
        pivot_row += 1
        rank += 1
    return rank


def hrank_scores(
    feature_maps: Sequence[Sequence[Sequence[Sequence[float]]]],
    *,
    rtol: float = _DEFAULT_RTOL,
) -> List[float]:
    """Average feature-map rank per filter over a batch of images.

    ``feature_maps[g][k]`` is the ``h x w`` output of filter ``k`` on image
    ``g``. Every image must have the same number of filters.
    """
    if not feature_maps:
        raise ValueError("feature_maps must contain at least one image")
    n_filters = len(feature_maps[0])
    if n_filters == 0:
        raise ValueError("feature_maps must contain at least one filter")
    totals = [0.0] * n_filters
    for g, image in enumerate(feature_maps):
        if len(image) != n_filters:
            raise ValueError(
                f"image {g} has {len(image)} filter maps, expected {n_filters}"
            )
        for k, fmap in enumerate(image):
            totals[k] += matrix_rank(fmap, rtol=rtol)
    return [total / len(feature_maps) for total in totals]


def hrank_keep_filters(scores: Sequence[float], density: float) -> List[int]:
    """Indices of the ``round(density * K)`` highest-rank filters (ties -> lower index)."""
    _validate_density(density)
    n = len(scores)
    n_keep = max(0, min(n, int(round(density * n))))
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))
    return sorted(ranked[:n_keep])


def hrank_prune_layer(
    weights: Sequence[float],
    shape: tuple[int, ...],
    feature_maps: Sequence[Sequence[Sequence[Sequence[float]]]],
    *,
    density: float = 0.5,
    rtol: float = _DEFAULT_RTOL,
) -> List[float]:
    """Zero the output filters whose feature maps have the lowest average rank.

    ``density`` is the fraction of filters to keep. ``feature_maps`` must
    have one map per output filter (``shape[0]``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_weights_shape(weights, shape)
    scores = hrank_scores(feature_maps, rtol=rtol)
    if len(scores) != int(shape[0]):
        raise ValueError(
            f"feature maps describe {len(scores)} filters, layer has {shape[0]}"
        )
    keep = set(hrank_keep_filters(scores, density))
    groups = channel_groups(shape, structure="filter")
    kept_indices = {i for k in keep for i in groups[k]}
    return [float(v) if i in kept_indices else 0.0 for i, v in enumerate(weights)]


def _validate_densities(model, density, per_layer):
    _validate_density(density)
    if per_layer is not None:
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(f"per_layer override references unknown layer {name!r}")
            _validate_density(layer_density)


def hrank_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    feature_maps: Dict[str, Sequence],
    *,
    density: float = 0.5,
    per_layer: Dict[str, float] | None = None,
    rtol: float = _DEFAULT_RTOL,
) -> Dict[str, List[float]]:
    """Apply :func:`hrank_prune_layer` to every conv layer with feature maps.

    Layers without an entry in ``feature_maps`` (and all dense layers) are
    copied unchanged.
    """
    _validate_densities(model, density, per_layer)
    for name in feature_maps:
        if name not in model:
            raise ValueError(f"feature_maps references unknown layer {name!r}")
    pruned: Dict[str, List[float]] = {}
    for spec in specs:
        weights = model[spec.name]
        expected = layer_weight_count(spec)
        if len(weights) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
            )
        if spec.kind != "conv" or spec.name not in feature_maps:
            pruned[spec.name] = _copy_layer(weights)
            continue
        layer_density = density if per_layer is None else per_layer.get(spec.name, density)
        pruned[spec.name] = hrank_prune_layer(
            weights, spec.shape, feature_maps[spec.name], density=layer_density, rtol=rtol
        )
    return pruned


def hrank_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    feature_maps: Dict[str, Sequence],
    *,
    density: float = 0.5,
    per_layer: Dict[str, float] | None = None,
    rtol: float = _DEFAULT_RTOL,
) -> dict:
    """Prune with HRank and report per-layer average ranks and kept filters."""
    pruned = hrank_prune_model(
        specs, model, feature_maps, density=density, per_layer=per_layer, rtol=rtol
    )
    layers = []
    total = nonzero = 0
    for spec in specs:
        weights = pruned[spec.name]
        nz = sum(1 for v in weights if v != 0.0)
        total += len(weights)
        nonzero += nz
        row = {
            "layer": spec.name,
            "kind": spec.kind,
            "total_weights": len(weights),
            "nonzero_weights": nz,
            "survival_fraction": nz / len(weights) if weights else 0.0,
            "ranks": None,
            "kept_filters": None,
        }
        if spec.kind == "conv" and spec.name in feature_maps:
            scores = hrank_scores(feature_maps[spec.name], rtol=rtol)
            layer_density = density if per_layer is None else per_layer.get(spec.name, density)
            row["ranks"] = scores
            row["kept_filters"] = hrank_keep_filters(scores, layer_density)
        layers.append(row)
    return {
        "density": density,
        "layers": layers,
        "total_weights": total,
        "nonzero_weights": nonzero,
        "overall_survival": nonzero / total if total else 0.0,
        "pruned_weights": pruned,
    }
