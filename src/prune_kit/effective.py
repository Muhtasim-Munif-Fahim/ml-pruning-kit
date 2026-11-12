"""Effective sparsity: remove weights that no longer lie on any input-output path.

Unstructured pruning reports *direct* sparsity -- the fraction of zeroed
weights. At high sparsity many surviving weights are useless because the
neuron they read from receives no input (it is cut off from the network's
inputs) or the neuron they feed reaches no output. Such weights can be
removed without changing the function of a bias-free network, so the
*effective* sparsity (Vysogorets & Kempe, "Connectivity Matters: Neural
Network Pruning Through the Lens of Effective Sparsity", JMLR 2023) can be
much higher than the nominal one and different pruning methods should be
compared at equal effective, not direct, sparsity.

For a chain of dense layers (masks laid out row-major ``(out, in)`` like
the rest of the kit) a weight ``j -> i`` of layer ``l`` is *effective* iff

* neuron ``j`` (input of layer ``l``) is reachable from some network input
  through active weights, and
* neuron ``i`` (output of layer ``l``) reaches some network output through
  active weights.

Biases can keep a neuron with no incoming weights alive (a constant), so
for networks with biases the result is the sparsity of the
input-dependent part; pass ``inputs_alive``/``outputs_alive`` to restrict
the analysis to a subset of inputs or outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .force import _is_dense_chain
from .layers import LayerSpec

__all__ = [
    "LayerEffectiveSparsity",
    "EffectiveSparsityReport",
    "effective_masks",
    "effective_sparsity_report",
    "prune_disconnected",
]


def _check(specs: Sequence[LayerSpec], masks: Dict[str, Sequence[int]]) -> None:
    if not specs:
        raise ValueError("specs must be non-empty")
    if not _is_dense_chain(specs):
        raise ValueError("specs must be a chain of dense layers (fan-in matches previous fan-out)")
    for spec in specs:
        if spec.name not in masks:
            raise KeyError(f"missing mask for layer {spec.name!r}")
        out_f, in_f = spec.shape
        if len(masks[spec.name]) != out_f * in_f:
            raise ValueError(f"mask for {spec.name!r} has wrong length")


def _reachability(specs, masks, inputs_alive, outputs_alive):
    n_in = specs[0].shape[1]
    n_out = specs[-1].shape[0]
    fwd = [list(inputs_alive) if inputs_alive is not None else [True] * n_in]
    if len(fwd[0]) != n_in:
        raise ValueError("inputs_alive must have the first layer's fan-in")
    for spec in specs:
        out_f, in_f = spec.shape
        m = masks[spec.name]
        prev = fwd[-1]
        fwd.append([any(prev[j] and m[o * in_f + j] for j in range(in_f)) for o in range(out_f)])
    bwd = [None] * (len(specs) + 1)
    bwd[-1] = list(outputs_alive) if outputs_alive is not None else [True] * n_out
    if len(bwd[-1]) != n_out:
        raise ValueError("outputs_alive must have the last layer's fan-out")
    for l in range(len(specs) - 1, -1, -1):
        out_f, in_f = specs[l].shape
        m = masks[specs[l].name]
        nxt = bwd[l + 1]
        bwd[l] = [any(nxt[o] and m[o * in_f + j] for o in range(out_f)) for j in range(in_f)]
    return fwd, bwd


def effective_masks(
    specs: Sequence[LayerSpec],
    masks: Dict[str, Sequence[int]],
    inputs_alive: Optional[Sequence[bool]] = None,
    outputs_alive: Optional[Sequence[bool]] = None,
) -> Dict[str, List[int]]:
    """Return masks keeping only weights on some input-to-output path."""
    _check(specs, masks)
    fwd, bwd = _reachability(specs, masks, inputs_alive, outputs_alive)
    out: Dict[str, List[int]] = {}
    for l, spec in enumerate(specs):
        out_f, in_f = spec.shape
        m = masks[spec.name]
        src_ok = fwd[l]
        dst_ok = bwd[l + 1]
        out[spec.name] = [
            1 if (m[o * in_f + j] and src_ok[j] and dst_ok[o]) else 0
            for o in range(out_f)
            for j in range(in_f)
        ]
    return out


@dataclass(frozen=True)
class LayerEffectiveSparsity:
    name: str
    total: int
    active: int
    effective: int
    dead_inputs: int
    dead_outputs: int

    @property
    def direct_density(self) -> float:
        return self.active / self.total

    @property
    def effective_density(self) -> float:
        return self.effective / self.total

    @property
    def inactive_weights(self) -> int:
        """Weights that are unpruned but disconnected."""
        return self.active - self.effective


@dataclass(frozen=True)
class EffectiveSparsityReport:
    layers: tuple
    total: int
    active: int
    effective: int
    connected: bool
    masks: Dict[str, List[int]]

    @property
    def direct_sparsity(self) -> float:
        return 1.0 - self.active / self.total

    @property
    def effective_sparsity(self) -> float:
        return 1.0 - self.effective / self.total

    def table(self) -> str:
        rows = ["| layer | total | active | effective | dead in | dead out |", "|---|---|---|---|---|---|"]
        for r in self.layers:
            rows.append(
                f"| {r.name} | {r.total} | {r.active} | {r.effective} | {r.dead_inputs} | {r.dead_outputs} |"
            )
        rows.append(
            f"| **all** | {self.total} | {self.active} | {self.effective} | | |"
        )
        return "\n".join(rows)


def effective_sparsity_report(
    specs: Sequence[LayerSpec],
    masks: Dict[str, Sequence[int]],
    inputs_alive: Optional[Sequence[bool]] = None,
    outputs_alive: Optional[Sequence[bool]] = None,
) -> EffectiveSparsityReport:
    """Per-layer direct vs. effective density and dead-neuron counts.

    ``dead_inputs`` counts the layer's input neurons not reachable from the
    network inputs; ``dead_outputs`` counts its output neurons that reach
    no network output. ``connected`` is ``False`` on layer collapse.
    """
    _check(specs, masks)
    fwd, bwd = _reachability(specs, masks, inputs_alive, outputs_alive)
    eff = effective_masks(specs, masks, inputs_alive, outputs_alive)
    layers = []
    total = active = effective = 0
    for l, spec in enumerate(specs):
        n = spec.shape[0] * spec.shape[1]
        a = sum(1 for v in masks[spec.name] if v)
        e = sum(eff[spec.name])
        layers.append(
            LayerEffectiveSparsity(
                name=spec.name,
                total=n,
                active=a,
                effective=e,
                dead_inputs=sum(1 for v in fwd[l] if not v),
                dead_outputs=sum(1 for v in bwd[l + 1] if not v),
            )
        )
        total += n
        active += a
        effective += e
    return EffectiveSparsityReport(
        layers=tuple(layers),
        total=total,
        active=active,
        effective=effective,
        connected=effective > 0,
        masks=eff,
    )


def prune_disconnected(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    masks: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, List[float]]:
    """Zero every weight that is pruned or disconnected; returns a new model.

    When ``masks`` is omitted the active set is the non-zero weights.
    """
    if masks is None:
        masks = {s.name: [1 if w != 0 else 0 for w in model[s.name]] for s in specs}
    eff = effective_masks(specs, masks)
    return {
        s.name: [float(w) if k else 0.0 for w, k in zip(model[s.name], eff[s.name])]
        for s in specs
    }
