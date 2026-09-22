"""Global unstructured magnitude pruning.

Per-layer magnitude pruning (:func:`prune_kit.prune.magnitude_prune_model`)
keeps the same fraction of weights inside every layer. Global pruning
ranks every scalar weight in the model together and zeros the smallest
magnitudes until a target sparsity is reached. A layer of small weights
can therefore lose more parameters than a layer of large weights.

``sparsity`` is the fraction of weights to remove. The number kept is
``round((1 - sparsity) * N)``, the same rounding
:func:`prune_kit.prune.magnitude_prune_layer` uses for
``density = 1 - sparsity``. Ties break toward the earlier layer (dict
order) and the lower index.

An optional iterative schedule reaches that same final mask over
several rounds. Removals are spread evenly across ``rounds`` (earlier
rounds take any remainder), or an explicit cumulative ``schedule`` can
set the sparsity after each round. When ``rewind`` is true, surviving
weights are reset to their initialization after every round. Ranking
always uses the magnitudes passed in ``model`` (typically trained
weights), never the rewound values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .prune import PruneStep, rewind_model


def _copy_model(model: Dict[str, Sequence[float]]) -> Dict[str, List[float]]:
    return {
        name: [float(value) for value in weights]
        for name, weights in model.items()
    }


def _validate_model(model: Dict[str, Sequence[float]]) -> None:
    if not model:
        raise ValueError("model must not be empty")
    for name, weights in model.items():
        if len(weights) == 0:
            raise ValueError(f"layer {name!r} weights must not be empty")


def _validate_sparsity(sparsity: float) -> float:
    sparsity = float(sparsity)
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    return sparsity


def _keep_count(total: int, sparsity: float) -> int:
    """Weights kept at ``sparsity``, matching per-layer density rounding."""
    return int(round((1.0 - sparsity) * total))


def _ranked_best_first(
    model: Dict[str, Sequence[float]],
) -> List[tuple[float, int, int, str]]:
    """Rank ``(magnitude, layer_order, index, name)``, best to keep first.

    Sort key matches :func:`prune_kit.prune.magnitude_prune_layer`:
    larger magnitude first, then earlier layer, then lower index.
    """
    ranked: List[tuple[float, int, int, str]] = []
    for layer_order, (name, weights) in enumerate(model.items()):
        for index, value in enumerate(weights):
            ranked.append((abs(float(value)), layer_order, index, name))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return ranked


def _prune_keys(
    ranked: Sequence[tuple[float, int, int, str]],
    n_prune: int,
) -> set[tuple[str, int]]:
    """Positions of the ``n_prune`` lowest-ranked weights."""
    if n_prune <= 0:
        return set()
    tail = ranked[len(ranked) - n_prune:]
    return {(name, index) for _, _, index, name in tail}


def _zero_positions(
    model: Dict[str, Sequence[float]],
    prune_keys: set[tuple[str, int]],
) -> Dict[str, List[float]]:
    pruned: Dict[str, List[float]] = {}
    for name, weights in model.items():
        pruned[name] = [
            0.0 if (name, index) in prune_keys else float(value)
            for index, value in enumerate(weights)
        ]
    return pruned


def _resolve_prune_counts(
    total: int,
    sparsity: float | None,
    rounds: int,
    schedule: Sequence[float] | None,
) -> tuple[float, List[int]]:
    """Return ``(requested_final_sparsity, cumulative_prune_counts)``."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if schedule is not None:
        if isinstance(schedule, (str, bytes)):
            raise ValueError("schedule must be a sequence of sparsities")
        values = [_validate_sparsity(value) for value in schedule]
        if not values:
            raise ValueError("schedule must not be empty")
        for previous, current in zip(values, values[1:]):
            if current < previous:
                raise ValueError("schedule sparsities must be non-decreasing")
        if rounds != 1 and rounds != len(values):
            raise ValueError("rounds must match the schedule length")
        if sparsity is not None and abs(_validate_sparsity(sparsity) - values[-1]) > 1e-9:
            raise ValueError("sparsity must match the final schedule entry")
        counts = [total - _keep_count(total, value) for value in values]
        return values[-1], counts

    requested = 0.5 if sparsity is None else _validate_sparsity(sparsity)
    n_prune = total - _keep_count(total, requested)
    base, extra = divmod(n_prune, rounds)
    counts: List[int] = []
    removed = 0
    for index in range(rounds):
        removed += base + (1 if index < extra else 0)
        counts.append(removed)
    return requested, counts


def _model_kept(model: Dict[str, Sequence[float]]) -> int:
    return sum(
        1
        for weights in model.values()
        for value in weights
        if float(value) != 0.0
    )


@dataclass
class GlobalPruneResult:
    """Aggregate result of global unstructured magnitude pruning.

    ``weights`` is the mapping after the final round. ``steps`` holds a
    snapshot after every round. ``schedule`` is the cumulative fraction
    of parameters selected for pruning after each round (the mask
    target, which can differ slightly from the requested ``sparsity``
    because the kept count is rounded). ``density_curve`` reports the
    fraction of weights that are actually non-zero, including any zeros
    that were already present outside the mask.
    """

    weights: Dict[str, List[float]]
    steps: List[PruneStep] = field(default_factory=list)
    sparsity: float = 0.5
    rounds: int = 1
    rewind: bool = False
    schedule: List[float] = field(default_factory=list)

    def density_curve(self) -> List[float]:
        return [step.density for step in self.steps]

    def final_density(self) -> float:
        if not self.steps:
            raise ValueError("no prune steps have been run")
        return self.steps[-1].density


def iterative_global_magnitude_prune_model(
    model: Dict[str, Sequence[float]],
    *,
    sparsity: float | None = None,
    rounds: int = 1,
    schedule: Sequence[float] | None = None,
    rewind: bool = False,
    initial_weights: Dict[str, Sequence[float]] | None = None,
) -> GlobalPruneResult:
    """Globally magnitude-prune to ``sparsity``, optionally over rounds.

    Each round zeros the next tranche of smallest-|w| weights. The
    final mask matches one-shot :func:`global_magnitude_prune_model` at
    the same sparsity (or at the last ``schedule`` entry).

    ``schedule``, when given, is the cumulative sparsity after each
    round and replaces the even split of ``sparsity`` across
    ``rounds``. It must be non-empty and non-decreasing. ``rounds``
    may be omitted or must equal ``len(schedule)``. When both
    ``sparsity`` and ``schedule`` are set, ``sparsity`` must equal the
    final schedule entry.

    When ``rewind`` is True, surviving weights are reset to
    ``initial_weights`` after every round. If ``initial_weights`` is
    omitted, ``model`` is the rewind target. The input dictionaries
    are not modified.
    """
    _validate_model(model)
    if initial_weights is not None and not rewind:
        raise ValueError("initial_weights requires rewind=True")

    total = sum(len(weights) for weights in model.values())
    requested, counts = _resolve_prune_counts(total, sparsity, rounds, schedule)

    if rewind:
        if initial_weights is None:
            initial = _copy_model(model)
        else:
            if set(initial_weights.keys()) != set(model.keys()):
                raise ValueError(
                    "model and initial_weights must have the same layer names"
                )
            for name, weights in model.items():
                if len(initial_weights[name]) != len(weights):
                    raise ValueError(
                        f"layer {name!r} has {len(weights)} weights, "
                        f"initial_weights has {len(initial_weights[name])}"
                    )
            initial = _copy_model(initial_weights)
    else:
        initial = None

    original = _copy_model(model)
    ranked = _ranked_best_first(original)
    steps: List[PruneStep] = []
    current = original
    for round_index, count in enumerate(counts, start=1):
        masked = _zero_positions(original, _prune_keys(ranked, count))
        if rewind:
            current = rewind_model(masked, initial)
        else:
            current = masked
        kept = _model_kept(current)
        steps.append(
            PruneStep(
                round=round_index,
                weights=_copy_model(current),
                density=kept / total,
                kept=kept,
                total=total,
            )
        )

    return GlobalPruneResult(
        weights=current,
        steps=steps,
        sparsity=requested,
        rounds=len(counts),
        rewind=rewind,
        schedule=[count / total for count in counts],
    )


def global_magnitude_prune_model(
    model: Dict[str, Sequence[float]],
    *,
    sparsity: float = 0.5,
) -> Dict[str, List[float]]:
    """Zero the globally smallest-|w| weights to reach ``sparsity``.

    ``sparsity`` is the fraction of all scalar weights to prune. The
    original ``model`` mapping is not modified. See the module
    docstring for ranking, rounding, and tie-breaks.
    """
    result = iterative_global_magnitude_prune_model(
        model,
        sparsity=sparsity,
        rounds=1,
    )
    return result.weights


__all__ = [
    "GlobalPruneResult",
    "global_magnitude_prune_model",
    "iterative_global_magnitude_prune_model",
]
