"""Soft Filter Pruning (SFP) and Asymptotic Soft Filter Pruning (ASFP).

Hard filter pruning removes filters once and for all, so a filter that was
pruned because of a noisy early norm can never come back. Soft Filter
Pruning (He, Kang, Dong, Fu & Yang, IJCAI 2018) instead *zeros* the
lowest-norm filters at the end of every training epoch but keeps updating
them: a zeroed filter that the loss still needs regrows during the next
epoch and can win its place back. After the last epoch the zeroed filters
are removed for good.

Asymptotic SFP (He et al., IEEE TNNLS 2019) ramps the per-epoch pruning
rate up instead of applying the final rate from the first epoch, so the
information in to-be-pruned filters is moved to the survivors gradually.
The rate follows an exponential curve with ``P_0 = 0``, ``P_E = P`` and
``P_{E/4} = q * P`` (``q = 3/4`` in the paper)::

    P_e = P * (1 - u^{4 e / E}) / (1 - u^4),   (1 - u) / (1 - u^4) = q

Layout is C-contiguous ``(out_channels, in_channels, kh, kw)`` as in
:func:`prune_kit.layers.conv_layer`. The training step is a caller-supplied
``train_epoch(model, epoch) -> model`` callback, so any framework can be
plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .structured import channel_groups, channel_norms

__all__ = [
    "SoftFilterEpoch",
    "SoftFilterPruningResult",
    "asfp_rate_schedule",
    "constant_rate_schedule",
    "soft_filter_prune_layer",
    "soft_filter_pruning",
]


def _validate_rate(rate: float, name: str = "rate") -> float:
    rate = float(rate)
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return rate


def _validate_epochs(epochs: int) -> int:
    if isinstance(epochs, bool) or int(epochs) != epochs or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    return int(epochs)


def constant_rate_schedule(rate: float, epochs: int) -> List[float]:
    """SFP: the target rate at the end of every epoch."""
    rate = _validate_rate(rate)
    return [rate] * _validate_epochs(epochs)


def _solve_u(q: float) -> float:
    """Solve ``(1 - u) / (1 - u^4) = q`` i.e. ``1 + u + u^2 + u^3 = 1 / q`` on ``(0, 1)``."""
    target = 1.0 / q
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 + mid + mid * mid + mid ** 3 < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def asfp_rate_schedule(rate: float, epochs: int, *, quarter_fraction: float = 0.75) -> List[float]:
    """ASFP pruning rate at the end of epochs ``1 .. epochs``.

    Exponential ramp with ``P_0 = 0``, ``P_E = rate`` and
    ``P_{E/4} = quarter_fraction * rate``. ``quarter_fraction`` must be in
    ``(0.25, 1)`` (0.25 would be a straight line, 1 a step).
    """
    rate = _validate_rate(rate)
    epochs = _validate_epochs(epochs)
    q = float(quarter_fraction)
    if not 0.25 < q < 1.0:
        raise ValueError("quarter_fraction must be in (0.25, 1)")
    u = _solve_u(q)
    denom = 1.0 - u ** 4
    return [rate * (1.0 - u ** (4.0 * e / epochs)) / denom for e in range(1, epochs + 1)]


def soft_filter_prune_layer(
    weights: Sequence[float],
    shape: Tuple[int, ...],
    rate: float,
    *,
    norm: str = "l2",
) -> Tuple[List[float], List[int]]:
    """Zero the ``round(rate * n_filters)`` lowest-norm output filters.

    Returns ``(new_weights, zeroed_filter_indices)``. Ties break toward the
    higher index being zeroed (lower index kept), matching
    :func:`prune_kit.structured.structured_prune_layer`.
    """
    rate = _validate_rate(rate)
    if not weights:
        raise ValueError("weights must not be empty")
    norms = channel_norms(weights, shape, norm=norm, structure="filter")
    n = len(norms)
    n_zero = int(round(rate * n))
    n_zero = min(n_zero, n - 1)  # always keep at least one filter
    ranked = sorted(range(n), key=lambda i: (norms[i], -i))
    zeroed = sorted(ranked[:n_zero])
    groups = channel_groups(shape, structure="filter")
    out = [float(v) for v in weights]
    for f in zeroed:
        for idx in groups[f]:
            out[idx] = 0.0
    return out, zeroed


@dataclass
class SoftFilterEpoch:
    """What happened at the end of one epoch."""

    epoch: int
    rate: float
    zeroed: Dict[str, List[int]]
    revived: Dict[str, List[int]]

    @property
    def n_revived(self) -> int:
        return sum(len(v) for v in self.revived.values())


@dataclass
class SoftFilterPruningResult:
    """Outcome of a (A)SFP run."""

    model: Dict[str, List[float]]
    schedule: List[float]
    history: List[SoftFilterEpoch] = field(default_factory=list)
    kept_filters: Dict[str, List[int]] = field(default_factory=dict)
    pruned_filters: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def total_revived(self) -> int:
        return sum(h.n_revived for h in self.history)

    def filter_density(self) -> Dict[str, float]:
        return {
            name: len(kept) / (len(kept) + len(self.pruned_filters[name]))
            for name, kept in self.kept_filters.items()
        }


def soft_filter_pruning(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    train_epoch: Callable[[Dict[str, List[float]], int], Dict[str, Sequence[float]]],
    *,
    epochs: int,
    rate: float,
    schedule: str = "constant",
    quarter_fraction: float = 0.75,
    norm: str = "l2",
    per_layer: Dict[str, float] | None = None,
) -> SoftFilterPruningResult:
    """Run SFP (``schedule="constant"``) or ASFP (``schedule="asymptotic"``).

    Every epoch: ``model = train_epoch(model, epoch)`` (zeroed filters are
    *not* masked, so training may regrow them), then the lowest-norm
    filters of each conv layer are zeroed at that epoch's rate. Dense
    layers are passed through untouched. ``per_layer`` maps a layer name to
    its own final rate (scaled by the same schedule). The returned model is
    the state after the last epoch's soft prune, i.e. the hard-pruned model;
    ``kept_filters`` lists the surviving filters per conv layer for
    physical removal.
    """
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    if {s.name for s in specs} != set(model):
        raise ValueError("specs and model must reference the same layers")
    conv = [s for s in specs if s.kind == "conv"]
    if not conv:
        raise ValueError("soft filter pruning requires at least one conv layer")
    for s in specs:
        if len(model[s.name]) != layer_weight_count(s):
            raise ValueError(f"layer {s.name!r} has {len(model[s.name])} weights, expected {layer_weight_count(s)}")
    if schedule == "constant":
        base = constant_rate_schedule(rate, epochs)
    elif schedule == "asymptotic":
        base = asfp_rate_schedule(rate, epochs, quarter_fraction=quarter_fraction)
    else:
        raise ValueError("schedule must be 'constant' or 'asymptotic'")
    rates = {s.name: _validate_rate(rate) for s in conv}
    if per_layer:
        for name, r in per_layer.items():
            if name not in rates:
                raise ValueError(f"per_layer references unknown conv layer {name!r}")
            rates[name] = _validate_rate(r, f"per_layer[{name!r}]")
    final_rate = float(rate)

    current: Dict[str, List[float]] = {k: [float(v) for v in w] for k, w in model.items()}
    previous_zeroed: Dict[str, List[int]] = {s.name: [] for s in conv}
    history: List[SoftFilterEpoch] = []
    for e, step_rate in enumerate(base, start=1):
        trained = train_epoch({k: list(v) for k, v in current.items()}, e)
        if set(trained) != set(current):
            raise ValueError("train_epoch must return the same layers")
        current = {k: [float(v) for v in trained[k]] for k in current}
        zeroed_now: Dict[str, List[int]] = {}
        revived: Dict[str, List[int]] = {}
        for s in conv:
            scale = step_rate / final_rate if final_rate > 0 else 0.0
            layer_rate = min(rates[s.name] * scale, rates[s.name])
            new_w, zeroed = soft_filter_prune_layer(current[s.name], s.shape, layer_rate, norm=norm)
            current[s.name] = new_w
            zeroed_now[s.name] = zeroed
            revived[s.name] = sorted(set(previous_zeroed[s.name]) - set(zeroed))
        history.append(SoftFilterEpoch(epoch=e, rate=float(step_rate), zeroed=zeroed_now, revived=revived))
        previous_zeroed = zeroed_now

    kept = {}
    pruned = {}
    for s in conv:
        pruned[s.name] = previous_zeroed[s.name]
        kept[s.name] = [i for i in range(s.shape[0]) if i not in set(pruned[s.name])]
    return SoftFilterPruningResult(
        model=current, schedule=base, history=history, kept_filters=kept, pruned_filters=pruned
    )
