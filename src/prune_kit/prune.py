"""Magnitude-pruning primitives operating on flat weight buffers."""

from __future__ import annotations

from typing import Dict, List, Sequence


def magnitude_prune_layer(
    weights: Sequence[float],
    *,
    density: float = 0.5,
) -> List[float]:
    """Return a new weight vector with the smallest |w| values zeroed.

    ``density`` is the fraction of weights to keep; the rest are set to
    zero. The result is a Python list with the same length as the
    input. The original ``weights`` sequence is not modified.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    keep = int(round(density * len(weights)))
    if keep <= 0:
        return [0.0] * len(weights)
    if keep >= len(weights):
        return [float(value) for value in weights]

    indexed = [(abs(float(value)), index) for index, value in enumerate(weights)]
    indexed.sort(key=lambda pair: (-pair[0], pair[1]))
    keep_indices = {index for _, index in indexed[:keep]}

    return [
        float(value) if index in keep_indices else 0.0
        for index, value in enumerate(weights)
    ]


def magnitude_prune_model(
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """Magnitude-prune every layer in ``model`` and return a new mapping.

    ``per_layer`` can override the density for a subset of layers; any
    layer missing from the override uses the global ``density``. The
    input dictionary is not modified.
    """
    if not model:
        raise ValueError("model must not be empty")
    if per_layer is not None:
        for name in per_layer:
            if name not in model:
                raise ValueError(f"per_layer override references unknown layer {name!r}")
    pruned: Dict[str, List[float]] = {}
    for name, weights in model.items():
        layer_density = density if per_layer is None else per_layer.get(name, density)
        pruned[name] = magnitude_prune_layer(weights, density=layer_density)
    return pruned


def total_pruned(
    original: Dict[str, Sequence[float]],
    pruned: Dict[str, Sequence[float]],
) -> Dict[str, int]:
    """Return the number of weights zeroed in each layer of ``pruned``."""
    if set(original.keys()) != set(pruned.keys()):
        raise ValueError("original and pruned must have the same layer names")
    counts: Dict[str, int] = {}
    for name, original_weights in original.items():
        pruned_weights = pruned[name]
        if len(pruned_weights) != len(original_weights):
            raise ValueError(
                f"layer {name!r} has {len(pruned_weights)} weights after pruning, "
                f"expected {len(original_weights)}"
            )
        counts[name] = sum(
            1 for original_value, pruned_value in zip(original_weights, pruned_weights)
            if pruned_value == 0.0 and original_value != 0.0
        )
    return counts


__all__ = [
    "magnitude_prune_layer",
    "magnitude_prune_model",
    "total_pruned",
]