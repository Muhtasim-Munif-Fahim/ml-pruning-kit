"""Gradual magnitude pruning (Zhu & Gupta, 2017).

*To prune, or not to prune: exploring the efficacy of pruning for model
compression* (Zhu & Gupta, ICLR workshop 2018) grows sparsity smoothly
during training instead of pruning once. Starting at training step
``t0 = begin_step`` the target sparsity follows the cubic (more
generally polynomial) schedule

    s_t = s_f + (s_i - s_f) * (1 - (t - t0) / (n * dt)) ** exponent

for ``t in {t0, t0 + dt, ..., t0 + n * dt}`` where ``dt = frequency``,
``t0 + n * dt = end_step``, ``s_i = initial_sparsity`` and
``s_f = final_sparsity``. The default ``exponent=3`` is the paper's
cubic schedule: it prunes aggressively early, when redundant weights
are plentiful, and slowly near the end. Between pruning events the
target is held at the last value (a staircase), before ``begin_step``
no pruning has happened (sparsity ``0``), and from ``end_step`` on the
target is ``s_f``.

At every pruning event the smallest-magnitude **surviving** weights
are masked until ``round((1 - s_t) * N)`` remain, either inside every
layer (``scope="layer"``, the paper's setting) or across the whole
model (``scope="global"``). Masks are monotone: once a weight is pruned
it stays zero, even if an optional ``update_fn`` (a stand-in for the
optimizer step between pruning events) tries to regrow it. Ties break
toward the lower flat index (and the earlier layer for global scope),
matching :func:`prune_kit.magnitude_prune_layer`. Without an
``update_fn`` the final mask therefore equals one-shot magnitude
pruning at ``density = 1 - s_f``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

UpdateFn = Callable[[int, Dict[str, List[float]]], Dict[str, Sequence[float]]]


def _validate_sparsity(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number in [0, 1]")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _validate_int(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validate_scope(scope: str) -> str:
    key = str(scope).lower()
    if key not in {"layer", "global"}:
        raise ValueError("scope must be 'layer' or 'global'")
    return key


def _validate_schedule_args(
    final_sparsity: float,
    initial_sparsity: float,
    begin_step: int,
    end_step: int,
    frequency: int,
    exponent: float,
) -> Tuple[float, float, int, int, int, float]:
    final = _validate_sparsity(final_sparsity, "final_sparsity")
    initial = _validate_sparsity(initial_sparsity, "initial_sparsity")
    if initial > final:
        raise ValueError("initial_sparsity must not exceed final_sparsity")
    begin = _validate_int(begin_step, "begin_step", 0)
    end = _validate_int(end_step, "end_step", 0)
    if end < begin:
        raise ValueError("end_step must be >= begin_step")
    freq = _validate_int(frequency, "frequency", 1)
    if isinstance(exponent, bool):
        raise ValueError("exponent must be a positive number")
    exp = float(exponent)
    if not math.isfinite(exp) or exp <= 0.0:
        raise ValueError("exponent must be a positive finite number")
    return final, initial, begin, end, freq, exp


def _polynomial_value(
    step: int,
    final: float,
    initial: float,
    begin: int,
    end: int,
    freq: int,
    exp: float,
) -> float:
    if step < begin:
        return 0.0
    if step >= end:
        return final
    effective = begin + ((step - begin) // freq) * freq
    progress = (effective - begin) / (end - begin)
    value = final + (initial - final) * (1.0 - progress) ** exp
    # Guard against floating-point drift outside [initial, final].
    return min(final, max(initial, value))


def polynomial_sparsity(
    step: int,
    *,
    final_sparsity: float,
    end_step: int,
    initial_sparsity: float = 0.0,
    begin_step: int = 0,
    frequency: int = 1,
    exponent: float = 3.0,
) -> float:
    """Target sparsity of the Zhu & Gupta polynomial schedule at ``step``.

    Returns ``0.0`` before ``begin_step`` (pruning has not started),
    ``initial_sparsity`` at ``begin_step``, ``final_sparsity`` from
    ``end_step`` on, and the held polynomial value at the most recent
    pruning event (``begin_step + k * frequency``) in between.
    """
    final, initial, begin, end, freq, exp = _validate_schedule_args(
        final_sparsity, initial_sparsity, begin_step, end_step, frequency, exponent
    )
    step = _validate_int(step, "step", 0)
    return _polynomial_value(step, final, initial, begin, end, freq, exp)


def _prune_event_steps(begin: int, end: int, freq: int) -> List[int]:
    steps = list(range(begin, end, freq))
    steps.append(end)
    return steps


def polynomial_sparsity_schedule(
    *,
    final_sparsity: float,
    end_step: int,
    initial_sparsity: float = 0.0,
    begin_step: int = 0,
    frequency: int = 1,
    exponent: float = 3.0,
) -> List[Tuple[int, float]]:
    """List ``(step, target_sparsity)`` for every pruning event.

    Events are ``begin_step, begin_step + frequency, ...`` strictly
    before ``end_step``, plus ``end_step`` itself (always the final
    event, at ``final_sparsity``). Targets are non-decreasing.
    """
    final, initial, begin, end, freq, exp = _validate_schedule_args(
        final_sparsity, initial_sparsity, begin_step, end_step, frequency, exponent
    )
    return [
        (step, _polynomial_value(step, final, initial, begin, end, freq, exp))
        for step in _prune_event_steps(begin, end, freq)
    ]


@dataclass
class GradualPruneStep:
    """Snapshot after one pruning event of the gradual schedule."""

    step: int
    target_sparsity: float
    kept: int
    total: int
    density: float
    layer_kept: Dict[str, int] = field(default_factory=dict)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density


@dataclass
class GradualPruneResult:
    """Result of :func:`gradual_magnitude_prune_model`.

    ``weights`` are the masked weights after the last simulated step and
    ``masks`` the matching 0/1 keep masks. ``steps`` holds one entry per
    pruning event in schedule order.
    """

    weights: Dict[str, List[float]]
    masks: Dict[str, List[int]]
    steps: List[GradualPruneStep] = field(default_factory=list)
    final_sparsity: float = 0.0
    initial_sparsity: float = 0.0
    begin_step: int = 0
    end_step: int = 0
    frequency: int = 1
    exponent: float = 3.0
    scope: str = "layer"

    def target_curve(self) -> List[float]:
        return [step.target_sparsity for step in self.steps]

    def sparsity_curve(self) -> List[float]:
        return [step.sparsity for step in self.steps]

    def density_curve(self) -> List[float]:
        return [step.density for step in self.steps]

    def final_density(self) -> float:
        if not self.steps:
            raise ValueError("no pruning events have been run")
        return self.steps[-1].density


def _copy_model(model: Dict[str, Sequence[float]]) -> Dict[str, List[float]]:
    return {name: [float(value) for value in weights] for name, weights in model.items()}


def _validate_model(model: Dict[str, Sequence[float]]) -> None:
    if not model:
        raise ValueError("model must not be empty")
    for name, weights in model.items():
        if isinstance(weights, (str, bytes)) or len(weights) == 0:
            raise ValueError(f"layer {name!r} must have at least one weight")
        for value in weights:
            if not math.isfinite(float(value)):
                raise ValueError(f"layer {name!r} contains a non-finite weight")


def _apply_masks(
    weights: Dict[str, List[float]],
    masks: Dict[str, List[int]],
) -> Dict[str, List[float]]:
    return {
        name: [value if keep else 0.0 for value, keep in zip(layer, masks[name])]
        for name, layer in weights.items()
    }


def _keep_count(sparsity: float, total: int) -> int:
    return int(round((1.0 - sparsity) * total))


def _prune_layer_mask(
    weights: List[float],
    mask: List[int],
    keep: int,
) -> List[int]:
    alive = [index for index, flag in enumerate(mask) if flag]
    if keep >= len(alive):
        return list(mask)
    ranked = sorted(alive, key=lambda index: (-abs(weights[index]), index))
    survivors = set(ranked[:max(keep, 0)])
    return [1 if index in survivors else 0 for index in range(len(mask))]


def _prune_global_masks(
    weights: Dict[str, List[float]],
    masks: Dict[str, List[int]],
    keep: int,
) -> Dict[str, List[int]]:
    alive: List[Tuple[float, int, int, str]] = []
    for layer_order, (name, layer) in enumerate(weights.items()):
        for index, flag in enumerate(masks[name]):
            if flag:
                alive.append((-abs(layer[index]), layer_order, index, name))
    if keep >= len(alive):
        return {name: list(mask) for name, mask in masks.items()}
    alive.sort(key=lambda item: (item[0], item[1], item[2]))
    survivors = {(item[3], item[2]) for item in alive[:max(keep, 0)]}
    return {
        name: [1 if (name, index) in survivors else 0 for index in range(len(mask))]
        for name, mask in masks.items()
    }


def _check_update(
    updated: Dict[str, Sequence[float]],
    reference: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    if not isinstance(updated, dict) or set(updated.keys()) != set(reference.keys()):
        raise ValueError("update_fn must return a mapping with the same layer names")
    result: Dict[str, List[float]] = {}
    for name in reference:
        layer = updated[name]
        if len(layer) != len(reference[name]):
            raise ValueError(
                f"update_fn changed the size of layer {name!r}: "
                f"{len(layer)} != {len(reference[name])}"
            )
        values = [float(value) for value in layer]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"update_fn produced a non-finite weight in {name!r}")
        result[name] = values
    return result


def gradual_magnitude_prune_model(
    model: Dict[str, Sequence[float]],
    *,
    final_sparsity: float,
    end_step: int,
    initial_sparsity: float = 0.0,
    begin_step: int = 0,
    frequency: int = 1,
    exponent: float = 3.0,
    scope: str = "layer",
    total_steps: int | None = None,
    update_fn: UpdateFn | None = None,
) -> GradualPruneResult:
    """Simulate Zhu & Gupta gradual magnitude pruning over training steps.

    Steps ``0 .. total_steps - 1`` are simulated (default
    ``total_steps = end_step + 1`` so the final event runs). On each
    step ``update_fn(step, weights)`` (if given) returns the new dense
    weights, which are immediately re-masked so pruned weights stay
    zero; then, if ``step`` is a pruning event, the smallest surviving
    magnitudes are masked to reach :func:`polynomial_sparsity` at that
    step. The input ``model`` is not modified.
    """
    final, initial, begin, end, freq, exp = _validate_schedule_args(
        final_sparsity, initial_sparsity, begin_step, end_step, frequency, exponent
    )
    scope_key = _validate_scope(scope)
    _validate_model(model)
    if total_steps is None:
        n_steps = end + 1
    else:
        n_steps = _validate_int(total_steps, "total_steps", 1)
        if n_steps <= end:
            raise ValueError("total_steps must be greater than end_step")
    if update_fn is not None and not callable(update_fn):
        raise ValueError("update_fn must be callable")

    weights = _copy_model(model)
    masks: Dict[str, List[int]] = {name: [1] * len(layer) for name, layer in weights.items()}
    events = set(_prune_event_steps(begin, end, freq))
    grand_total = sum(len(layer) for layer in weights.values())
    steps: List[GradualPruneStep] = []

    for step in range(n_steps):
        if update_fn is not None:
            snapshot = {name: list(layer) for name, layer in weights.items()}
            weights = _check_update(update_fn(step, snapshot), weights)
            weights = _apply_masks(weights, masks)
        if step not in events:
            continue
        target = _polynomial_value(step, final, initial, begin, end, freq, exp)
        if scope_key == "layer":
            masks = {
                name: _prune_layer_mask(
                    weights[name], masks[name], _keep_count(target, len(weights[name]))
                )
                for name in weights
            }
        else:
            masks = _prune_global_masks(weights, masks, _keep_count(target, grand_total))
        weights = _apply_masks(weights, masks)
        layer_kept = {name: sum(mask) for name, mask in masks.items()}
        kept = sum(layer_kept.values())
        steps.append(
            GradualPruneStep(
                step=step,
                target_sparsity=target,
                kept=kept,
                total=grand_total,
                density=kept / grand_total,
                layer_kept=layer_kept,
            )
        )

    return GradualPruneResult(
        weights=weights,
        masks=masks,
        steps=steps,
        final_sparsity=final,
        initial_sparsity=initial,
        begin_step=begin,
        end_step=end,
        frequency=freq,
        exponent=exp,
        scope=scope_key,
    )


__all__ = [
    "GradualPruneStep",
    "GradualPruneResult",
    "polynomial_sparsity",
    "polynomial_sparsity_schedule",
    "gradual_magnitude_prune_model",
]
