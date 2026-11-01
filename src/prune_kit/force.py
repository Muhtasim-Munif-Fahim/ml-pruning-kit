"""FORCE and Iterative SNIP: progressive skeletonization at initialization.

One-shot SNIP (:mod:`prune_kit.snip`) scores every connection once, on
the dense network. At extreme sparsity (99%+) that single estimate
breaks down, because pruning many weights changes the gradients of the
ones that survive. Whole layers can be cut off (layer collapse).

de Jorge et al., *Progressive Skeletonization: Trimming more fat from
a network at initialization* (ICLR 2021), prune gradually instead. The
keep-fraction decays over ``T`` steps along an exponential schedule,
``d_t = density ** (t / T)``. At each step the connection sensitivity is
re-evaluated **on the currently pruned network**:

    w_eff = w0 * c            (c = current 0/1 mask)
    s_j   = |dL/dc_j| = |w0_j * dL/dw_eff_j|

Two variants are supported:

* ``mode="force"`` (FORCE): every connection is re-scored, including
  ones pruned earlier, so a pruned weight can come back if the thinner
  network now depends on it.
* ``mode="iter_snip"`` (Iterative SNIP): pruning is permanent and only
  surviving connections compete.

``grad_fn(model) -> grads`` (or ``-> (loss, grads)``) gives the
gradient of your loss with respect to the effective weights. Its
argument is a ``{name: flat list}`` dict holding the masked weights, in
the same layout as :mod:`prune_kit.snip`. :func:`mlp_mse_grads` builds
such a callback for a bias-free dense MLP with a mean-squared-error
loss. It is pure Python, and the tests and the ``prune-kit force`` demo
use it.

``density`` is the final fraction of connections to keep (rounded the
same way as the other pruners). Ties break toward the earlier layer in
``specs`` order, then the lower index. Kept positions hold the
original weight; pruned positions are ``0.0``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

from .layers import LayerSpec, layer_weight_count

GradFn = Callable[[Dict[str, List[float]]], object]


def _validate(specs, model, density, steps, mode, schedule, scope):
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    if mode not in {"force", "iter_snip"}:
        raise ValueError("mode must be 'force' or 'iter_snip'")
    if schedule not in {"exponential", "linear"}:
        raise ValueError("schedule must be 'exponential' or 'linear'")
    if scope not in {"global", "layer"}:
        raise ValueError("scope must be 'global' or 'layer'")
    if not specs:
        raise ValueError("at least one layer spec is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("layer names must be unique")
    for spec in specs:
        if spec.name not in model:
            raise KeyError(f"missing weights for layer {spec.name!r}")
        if len(model[spec.name]) != layer_weight_count(spec):
            raise ValueError(
                f"layer {spec.name!r} has {len(model[spec.name])} weights, "
                f"spec expects {layer_weight_count(spec)}"
            )
    return density


def density_schedule(density: float, steps: int, schedule: str = "exponential") -> List[float]:
    """Keep-fractions ``d_1 .. d_T`` ending exactly at ``density``."""
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    if schedule == "exponential":
        values = [density ** (t / steps) for t in range(1, steps + 1)]
    elif schedule == "linear":
        values = [1.0 - (1.0 - density) * t / steps for t in range(1, steps + 1)]
    else:
        raise ValueError("schedule must be 'exponential' or 'linear'")
    values[-1] = density
    return values


def _call_grad_fn(grad_fn: GradFn, model: Dict[str, List[float]], specs: Sequence[LayerSpec]):
    out = grad_fn(model)
    loss = None
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("grad_fn must return grads or (loss, grads)")
        loss, out = out
        loss = float(loss)
    for spec in specs:
        if spec.name not in out:
            raise KeyError(f"grad_fn returned no gradient for layer {spec.name!r}")
        if len(out[spec.name]) != layer_weight_count(spec):
            raise ValueError(f"gradient for {spec.name!r} has the wrong length")
    return loss, out


def _keep_top(entries: List[tuple], k: int) -> set:
    # entries: (score, layer_order, index, key); highest score first, earlier layer/index on ties
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    return {e[3] for e in entries[:k]}


def force_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grad_fn: GradFn,
    density: float,
    *,
    steps: int = 10,
    mode: str = "force",
    schedule: str = "exponential",
    scope: str = "global",
) -> dict:
    """Progressively prune ``model`` to ``density`` with FORCE or Iterative SNIP.

    Returns a dict with ``pruned_weights``, ``masks`` (0/1 lists),
    per-step ``history`` (target density, kept count, loss reported by
    ``grad_fn`` on the network being scored, connections revived), per-
    layer ``layers`` rows, ``final_loss`` (``grad_fn`` on the pruned
    network, or ``None``) and ``connected`` (see :func:`dense_chain_connected`).
    """
    density = _validate(specs, model, density, steps, mode, schedule, scope)
    w0 = {spec.name: [float(v) for v in model[spec.name]] for spec in specs}
    masks = {spec.name: [1] * layer_weight_count(spec) for spec in specs}
    history: List[dict] = []
    for step, target in enumerate(density_schedule(density, int(steps), schedule), start=1):
        effective = {n: [w * m for w, m in zip(w0[n], masks[n])] for n in w0}
        loss, grads = _call_grad_fn(grad_fn, effective, specs)
        entries_by_layer: Dict[str, List[tuple]] = {}
        for order, spec in enumerate(specs):
            name = spec.name
            rows = []
            for i, (w, g) in enumerate(zip(w0[name], grads[name])):
                if mode == "iter_snip" and not masks[name][i]:
                    continue
                rows.append((abs(w * float(g)), order, i, (name, i)))
            entries_by_layer[name] = rows
        keep: set = set()
        if scope == "global":
            total = sum(layer_weight_count(spec) for spec in specs)
            pool = [e for rows in entries_by_layer.values() for e in rows]
            keep = _keep_top(pool, min(len(pool), int(round(target * total))))
        else:
            for spec in specs:
                n = layer_weight_count(spec)
                rows = entries_by_layer[spec.name]
                keep |= _keep_top(rows, min(len(rows), int(round(target * n))))
        revived = 0
        for spec in specs:
            name = spec.name
            new = [1 if (name, i) in keep else 0 for i in range(len(w0[name]))]
            revived += sum(1 for a, b in zip(masks[name], new) if b and not a)
            masks[name] = new
        history.append(
            {
                "step": step,
                "target_density": target,
                "kept": len(keep),
                "loss": loss,
                "revived": revived,
            }
        )
    pruned = {n: [w if m else 0.0 for w, m in zip(w0[n], masks[n])] for n in w0}
    final_loss, _ = _call_grad_fn(grad_fn, pruned, specs)
    total = sum(len(m) for m in masks.values())
    kept_total = sum(sum(m) for m in masks.values())
    layers = []
    for spec in specs:
        n = len(masks[spec.name])
        kept = sum(masks[spec.name])
        layers.append(
            {
                "layer": spec.name,
                "kind": spec.kind,
                "total_weights": n,
                "kept_weights": kept,
                "survival_fraction": kept / n,
            }
        )
    connected = dense_chain_connected(specs, masks) if _is_dense_chain(specs) else None
    return {
        "mode": mode,
        "schedule": schedule,
        "scope": scope,
        "steps": int(steps),
        "density": density,
        "total": total,
        "total_kept": kept_total,
        "overall_survival": kept_total / total,
        "pruned_weights": pruned,
        "masks": masks,
        "history": history,
        "final_loss": final_loss,
        "connected": connected,
        "layers": layers,
    }


def _is_dense_chain(specs: Sequence[LayerSpec]) -> bool:
    if any(spec.kind != "dense" for spec in specs):
        return False
    return all(specs[i].shape[0] == specs[i + 1].shape[1] for i in range(len(specs) - 1))


def dense_chain_connected(specs: Sequence[LayerSpec], masks: Dict[str, Sequence[int]]) -> bool:
    """True if at least one input-to-output path survives in a dense chain.

    ``False`` signals layer collapse: the pruned network's output no
    longer depends on its input at all.
    """
    if not _is_dense_chain(specs):
        raise ValueError("specs must be a chain of dense layers (fan-in matches previous fan-out)")
    alive = [True] * specs[0].shape[1]
    for spec in specs:
        out_f, in_f = spec.shape
        mask = masks[spec.name]
        alive = [
            any(alive[j] and mask[o * in_f + j] for j in range(in_f)) for o in range(out_f)
        ]
        if not any(alive):
            return False
    return True


def mlp_mse_grads(
    specs: Sequence[LayerSpec],
    inputs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    activation: str = "relu",
) -> GradFn:
    """``grad_fn`` for a bias-free dense MLP with mean-squared-error loss.

    Layers are applied in ``specs`` order with ``activation``
    (``"relu"``, ``"tanh"`` or ``"linear"``) between them and none
    after the last. The loss is ``mean over samples of
    0.5 * ||f(x) - y||^2``. The returned callable maps a weight dict to
    ``(loss, grads)``.
    """
    if not _is_dense_chain(specs):
        raise ValueError("specs must be a chain of dense layers (fan-in matches previous fan-out)")
    if activation not in {"relu", "tanh", "linear"}:
        raise ValueError("activation must be 'relu', 'tanh' or 'linear'")
    xs = [[float(v) for v in row] for row in inputs]
    ys = [[float(v) for v in row] for row in targets]
    if not xs or len(xs) != len(ys):
        raise ValueError("inputs and targets must be non-empty and the same length")
    if any(len(r) != specs[0].shape[1] for r in xs):
        raise ValueError("every input row must have the first layer's fan-in")
    if any(len(r) != specs[-1].shape[0] for r in ys):
        raise ValueError("every target row must have the last layer's fan-out")

    def act(z):
        if activation == "relu":
            return max(z, 0.0)
        if activation == "tanh":
            return math.tanh(z)
        return z

    def dact(z, a):
        if activation == "relu":
            return 1.0 if z > 0.0 else 0.0
        if activation == "tanh":
            return 1.0 - a * a
        return 1.0

    last = len(specs) - 1

    def grad_fn(model: Dict[str, Sequence[float]]):
        grads = {spec.name: [0.0] * layer_weight_count(spec) for spec in specs}
        loss = 0.0
        n = len(xs)
        for x, y in zip(xs, ys):
            acts = [x]
            pre = []
            for li, spec in enumerate(specs):
                out_f, in_f = spec.shape
                w = model[spec.name]
                h = acts[-1]
                z = [sum(w[o * in_f + j] * h[j] for j in range(in_f)) for o in range(out_f)]
                pre.append(z)
                acts.append(z if li == last else [act(v) for v in z])
            delta = [(a - t) for a, t in zip(acts[-1], y)]
            loss += 0.5 * sum(d * d for d in delta)
            for li in range(last, -1, -1):
                spec = specs[li]
                out_f, in_f = spec.shape
                if li != last:
                    delta = [d * dact(z, a) for d, z, a in zip(delta, pre[li], acts[li + 1])]
                h = acts[li]
                g = grads[spec.name]
                w = model[spec.name]
                for o in range(out_f):
                    d = delta[o] / n
                    if d:
                        base = o * in_f
                        for j in range(in_f):
                            g[base + j] += d * h[j]
                if li:
                    delta = [sum(w[o * in_f + j] * delta[o] for o in range(out_f)) for j in range(in_f)]
        return loss / n, grads

    return grad_fn
