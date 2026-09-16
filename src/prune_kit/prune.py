"""Magnitude-pruning primitives operating on flat weight buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def weight_density(weights: Sequence[float]) -> float:
    """Return the fraction of non-zero entries in a flat weight buffer."""
    if not weights:
        raise ValueError("weights must not be empty")
    kept = sum(1 for value in weights if float(value) != 0.0)
    return kept / len(weights)


def model_density(model: Dict[str, Sequence[float]]) -> float:
    """Return the overall fraction of non-zero weights across ``model``."""
    if not model:
        raise ValueError("model must not be empty")
    total = 0
    kept = 0
    for weights in model.values():
        total += len(weights)
        kept += sum(1 for value in weights if float(value) != 0.0)
    if total == 0:
        raise ValueError("model has no weights")
    return kept / total


def rewind_layer(
    weights: Sequence[float],
    initial: Sequence[float],
) -> List[float]:
    """Lottery-ticket rewind: reset surviving weights to ``initial``.

    Positions that are already zero in ``weights`` stay zero. Every
    non-zero position is replaced with the corresponding value from
    ``initial``. Neither input sequence is modified.
    """
    if len(weights) != len(initial):
        raise ValueError(
            f"weights has {len(weights)} values, initial has {len(initial)}"
        )
    return [
        float(init) if float(current) != 0.0 else 0.0
        for current, init in zip(weights, initial)
    ]


def rewind_model(
    model: Dict[str, Sequence[float]],
    initial: Dict[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Apply :func:`rewind_layer` to every layer in ``model``.

    ``initial`` must contain the same layer names as ``model``. The
    input dictionaries are not modified.
    """
    if set(model.keys()) != set(initial.keys()):
        raise ValueError("model and initial must have the same layer names")
    return {
        name: rewind_layer(weights, initial[name])
        for name, weights in model.items()
    }


def _copy_layer(weights: Sequence[float]) -> List[float]:
    return [float(value) for value in weights]


def _copy_model(model: Dict[str, Sequence[float]]) -> Dict[str, List[float]]:
    return {name: _copy_layer(weights) for name, weights in model.items()}


def _prune_remaining_layer(
    weights: Sequence[float],
    prune_fraction: float,
) -> List[float]:
    """Zero ``prune_fraction`` of the currently non-zero weights.

    Ranking uses magnitude (largest kept); already-zero entries stay
    zero and are not counted in the surviving pool. Ties break toward
    the lower index, matching :func:`magnitude_prune_layer`.
    """
    if not 0.0 < prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be in (0, 1]")

    surviving = [
        (abs(float(value)), index)
        for index, value in enumerate(weights)
        if float(value) != 0.0
    ]
    n_surviving = len(surviving)
    n_remove = int(round(prune_fraction * n_surviving))
    if n_remove <= 0:
        return _copy_layer(weights)
    if n_remove >= n_surviving:
        return [0.0] * len(weights)

    surviving.sort(key=lambda pair: (-pair[0], pair[1]))
    n_keep = n_surviving - n_remove
    keep_indices = {index for _, index in surviving[:n_keep]}
    return [
        float(value) if index in keep_indices else 0.0
        for index, value in enumerate(weights)
    ]


def iterative_magnitude_prune_layer(
    weights: Sequence[float],
    *,
    prune_fraction: float = 0.2,
    rounds: int = 1,
    rewind: bool = False,
    initial_weights: Sequence[float] | None = None,
) -> List[float]:
    """Iteratively prune ``prune_fraction`` of remaining weights.

    Each round ranks currently surviving (non-zero) entries by
    magnitude and zeros the smallest ``prune_fraction`` of them.
    Ranking always uses the pre-rewind magnitudes so compounding
    sparsity matches one-shot pruning to the product density.

    When ``rewind`` is True, surviving weights are reset to
    ``initial_weights`` after every prune step (the lottery-ticket
    hypothesis reset). If ``initial_weights`` is omitted, the values
    in ``weights`` at call time are used as the rewind target.

    The original ``weights`` sequence is not modified.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if not 0.0 < prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be in (0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if initial_weights is not None and not rewind:
        raise ValueError("initial_weights requires rewind=True")
    if rewind:
        if initial_weights is None:
            initial = _copy_layer(weights)
        else:
            if len(initial_weights) != len(weights):
                raise ValueError(
                    f"weights has {len(weights)} values, "
                    f"initial_weights has {len(initial_weights)}"
                )
            initial = _copy_layer(initial_weights)
    else:
        initial = None

    ranking = _copy_layer(weights)
    current = _copy_layer(weights)
    for _ in range(rounds):
        ranking = _prune_remaining_layer(ranking, prune_fraction)
        if rewind:
            current = rewind_layer(ranking, initial)
        else:
            current = list(ranking)
    return current


@dataclass
class PruneStep:
    """Snapshot of a model after one iterative prune round."""

    round: int
    weights: Dict[str, List[float]]
    density: float
    kept: int
    total: int


@dataclass
class IterativePruneResult:
    """Aggregate result of iterative magnitude pruning.

    ``weights`` is the mapping after the final round. ``steps`` holds
    a snapshot after every round so callers can inspect the density
    schedule (and, when rewind is enabled, the lottery-ticket reset).
    """

    weights: Dict[str, List[float]]
    steps: List[PruneStep] = field(default_factory=list)
    prune_fraction: float = 0.2
    rounds: int = 1
    rewind: bool = False

    def density_curve(self) -> List[float]:
        return [step.density for step in self.steps]

    def final_density(self) -> float:
        if not self.steps:
            raise ValueError("no prune steps have been run")
        return self.steps[-1].density


def _model_kept_total(model: Dict[str, Sequence[float]]) -> tuple[int, int]:
    total = 0
    kept = 0
    for weights in model.values():
        total += len(weights)
        kept += sum(1 for value in weights if float(value) != 0.0)
    return kept, total


def iterative_magnitude_prune_model(
    model: Dict[str, Sequence[float]],
    *,
    prune_fraction: float = 0.2,
    rounds: int = 1,
    rewind: bool = False,
    initial_weights: Dict[str, Sequence[float]] | None = None,
    per_layer: Dict[str, float] | None = None,
) -> IterativePruneResult:
    """Iteratively magnitude-prune every layer and optionally rewind.

    ``per_layer`` overrides ``prune_fraction`` for a subset of layers.
    Ranking uses the magnitudes in ``model`` (typically trained
    weights). When ``rewind`` is True, surviving weights are reset to
    ``initial_weights`` after each prune round. The input dictionaries
    are not modified.
    """
    if not model:
        raise ValueError("model must not be empty")
    if not 0.0 < prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be in (0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if initial_weights is not None and not rewind:
        raise ValueError("initial_weights requires rewind=True")
    if per_layer is not None:
        for name in per_layer:
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            if not 0.0 < per_layer[name] <= 1.0:
                raise ValueError("prune_fraction must be in (0, 1]")
    if rewind:
        if initial_weights is None:
            initial = _copy_model(model)
        else:
            if set(initial_weights.keys()) != set(model.keys()):
                raise ValueError(
                    "model and initial_weights must have the same layer names"
                )
            initial = _copy_model(initial_weights)
    else:
        initial = None

    ranking = _copy_model(model)
    current = _copy_model(model)
    steps: List[PruneStep] = []
    for round_index in range(1, rounds + 1):
        next_ranking: Dict[str, List[float]] = {}
        for name, weights in ranking.items():
            layer_fraction = (
                prune_fraction
                if per_layer is None
                else per_layer.get(name, prune_fraction)
            )
            next_ranking[name] = _prune_remaining_layer(weights, layer_fraction)
        ranking = next_ranking
        if rewind:
            current = rewind_model(ranking, initial)
        else:
            current = _copy_model(ranking)
        kept, total = _model_kept_total(current)
        density = kept / total if total else 0.0
        steps.append(
            PruneStep(
                round=round_index,
                weights=_copy_model(current),
                density=density,
                kept=kept,
                total=total,
            )
        )

    return IterativePruneResult(
        weights=current,
        steps=steps,
        prune_fraction=prune_fraction,
        rounds=rounds,
        rewind=rewind,
    )


__all__ = [
    "magnitude_prune_layer",
    "magnitude_prune_model",
    "total_pruned",
    "weight_density",
    "model_density",
    "rewind_layer",
    "rewind_model",
    "iterative_magnitude_prune_layer",
    "iterative_magnitude_prune_model",
    "PruneStep",
    "IterativePruneResult",
]