"""Dynamic sparse training: RigL and SET prune-and-regrow updates.

Instead of pruning a dense network, dynamic sparse training keeps a
*fixed* number of active connections from the start and periodically
rewires them:

* **SET** (Mocanu et al., Nature Communications 2018, "Sparse
  Evolutionary Training"): drop the smallest-magnitude active weights,
  regrow the same number of connections uniformly at random.
* **RigL** (Evci et al., ICML 2020, "Rigging the Lottery: Making All
  Tickets Winners"): drop the smallest-magnitude active weights, regrow
  the inactive connections with the largest dense-gradient magnitude
  ``|dL/dW|``.

Both keep the layer's density exactly constant. Regrown connections
start at ``regrow_init`` (``0.0`` in the RigL paper) and dropped ones
are zeroed. Because a regrown weight can be exactly zero, the active
set is carried as an explicit 0/1 ``mask`` (not inferred from non-zero
weights). Candidates for regrowth are all positions inactive *after*
the drop, so a just-dropped connection may be regrown when its gradient
is large enough (as in the reference implementation).

The fraction rewired per update follows the paper's cosine decay
``f(t) = alpha / 2 * (1 + cos(pi * t / T_end))``, with no updates after
``T_end`` (see :func:`cosine_drop_fraction` / :func:`rigl_schedule`).
:func:`random_sparse_masks` builds a starting mask with uniform or
ER/ERK layer-wise densities. Everything stays on flat Python lists, like
the rest of the kit.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .erk import erk_densities
from .layers import LayerSpec, layer_weight_count

_REGROW = ("gradient", "random")


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _validate_mask(mask: Sequence[int], n: int) -> List[int]:
    if len(mask) != n:
        raise ValueError(f"mask has {len(mask)} values, weights has {n}")
    out = []
    for bit in mask:
        b = int(bit)
        if b not in (0, 1):
            raise ValueError("mask entries must be 0 or 1")
        out.append(b)
    return out


def cosine_drop_fraction(
    step: int,
    *,
    end_step: int,
    initial_fraction: float = 0.3,
) -> float:
    """RigL cosine-annealed drop fraction at ``step`` (0 after ``end_step``)."""
    if end_step <= 0:
        raise ValueError("end_step must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    alpha = _validate_fraction(initial_fraction, "initial_fraction")
    if step >= end_step:
        return 0.0
    return alpha / 2.0 * (1.0 + math.cos(math.pi * step / end_step))


def rigl_schedule(
    total_steps: int,
    *,
    update_every: int = 100,
    end_fraction: float = 0.75,
    initial_fraction: float = 0.3,
) -> List[Tuple[int, float]]:
    """``[(step, drop_fraction), ...]`` for every mask-update step.

    Updates happen every ``update_every`` steps (starting at
    ``update_every``) and stop at ``T_end = end_fraction * total_steps``.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if update_every <= 0:
        raise ValueError("update_every must be positive")
    end_fraction = float(end_fraction)
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError("end_fraction must be in (0, 1]")
    end_step = max(1, int(round(end_fraction * total_steps)))
    out = []
    for step in range(update_every, end_step, update_every):
        out.append(
            (step, cosine_drop_fraction(step, end_step=end_step, initial_fraction=initial_fraction))
        )
    return out


def rigl_update_layer(
    weights: Sequence[float],
    mask: Sequence[int],
    gradients: Sequence[float] | None = None,
    *,
    drop_fraction: float = 0.3,
    regrow: str = "gradient",
    regrow_init: float = 0.0,
    seed: int | None = None,
) -> Dict[str, object]:
    """One drop-and-regrow update of a single layer.

    ``drop_fraction`` of the active connections (``round`` of it) with
    the smallest ``|w|`` are removed and the same number of inactive
    connections are activated: those with the largest ``|gradient|``
    (``regrow="gradient"``, RigL) or a uniform random subset
    (``regrow="random"``, SET; ``gradients`` not needed). Ties break by
    lower index (gradient) or the seeded RNG (random).

    Returns ``{"weights", "mask", "dropped", "grown", "n_active"}``;
    inputs are not modified.
    """
    n = len(weights)
    if n == 0:
        raise ValueError("weights must not be empty")
    mask_list = _validate_mask(mask, n)
    drop_fraction = _validate_fraction(drop_fraction, "drop_fraction")
    regrow = str(regrow).lower()
    if regrow not in _REGROW:
        raise ValueError(f"regrow must be one of {_REGROW}")
    if regrow == "gradient":
        if gradients is None:
            raise ValueError("gradients are required for regrow='gradient'")
        if len(gradients) != n:
            raise ValueError(f"gradients has {len(gradients)} values, weights has {n}")

    active = [i for i in range(n) if mask_list[i]]
    n_active = len(active)
    n_inactive = n - n_active
    k = int(round(drop_fraction * n_active))
    # cannot rewire more connections than there are dormant ones (a fully
    # dense layer has nothing to swap in)
    k = min(k, n_inactive)

    new_w = [float(w) * b for w, b in zip(weights, mask_list)]
    new_mask = list(mask_list)
    if k == 0:
        return {"weights": new_w, "mask": new_mask, "dropped": [], "grown": [], "n_active": n_active}

    drop_order = sorted(active, key=lambda i: (abs(float(weights[i])), i))
    dropped = sorted(drop_order[:k])
    for i in dropped:
        new_mask[i] = 0
        new_w[i] = 0.0

    candidates = [i for i in range(n) if not new_mask[i]]
    if regrow == "gradient":
        assert gradients is not None
        ranked = sorted(candidates, key=lambda i: (-abs(float(gradients[i])), i))
        grown = sorted(ranked[:k])
    else:
        rng = random.Random(seed)
        grown = sorted(rng.sample(candidates, k))
    for i in grown:
        new_mask[i] = 1
        new_w[i] = float(regrow_init)
    return {
        "weights": new_w,
        "mask": new_mask,
        "dropped": dropped,
        "grown": grown,
        "n_active": sum(new_mask),
    }


def set_update_layer(
    weights: Sequence[float],
    mask: Sequence[int],
    *,
    drop_fraction: float = 0.3,
    regrow_init: float = 0.0,
    seed: int | None = None,
) -> Dict[str, object]:
    """SET update: magnitude drop + uniform random regrowth."""
    return rigl_update_layer(
        weights,
        mask,
        None,
        drop_fraction=drop_fraction,
        regrow="random",
        regrow_init=regrow_init,
        seed=seed,
    )


def random_sparse_masks(
    specs: Sequence[LayerSpec],
    *,
    density: float = 0.1,
    distribution: str = "uniform",
    dense_layers: Sequence[str] = (),
    seed: int | None = None,
) -> Dict[str, List[int]]:
    """Random starting masks for dynamic sparse training.

    ``distribution`` is ``"uniform"`` (every layer at ``density``) or
    ``"er"`` / ``"erk"`` (layer-wise budgets from :func:`erk_densities`).
    Each layer keeps exactly ``round(density_l * N_l)`` (at least one)
    connections chosen uniformly at random.
    """
    if not specs:
        raise ValueError("specs must not be empty")
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    dist = str(distribution).lower()
    if dist not in ("uniform", "er", "erk"):
        raise ValueError("distribution must be 'uniform', 'er' or 'erk'")
    if dist == "uniform":
        per_layer = {s.name: (1.0 if s.name in set(dense_layers) else density) for s in specs}
    else:
        per_layer = erk_densities(specs, density=density, method=dist, dense_layers=dense_layers)
    rng = random.Random(seed)
    masks: Dict[str, List[int]] = {}
    for spec in specs:
        n = layer_weight_count(spec)
        keep = max(1, min(n, int(round(per_layer[spec.name] * n))))
        chosen = set(rng.sample(range(n), keep))
        masks[spec.name] = [1 if i in chosen else 0 for i in range(n)]
    return masks


def rigl_update_model(
    specs: Sequence[LayerSpec],
    weights: Dict[str, Sequence[float]],
    masks: Dict[str, Sequence[int]],
    gradients: Dict[str, Sequence[float]] | None = None,
    *,
    drop_fraction: float = 0.3,
    regrow: str = "gradient",
    regrow_init: float = 0.0,
    dense_layers: Sequence[str] = (),
    seed: int | None = None,
) -> Dict[str, object]:
    """Apply :func:`rigl_update_layer` to every layer (per-layer budgets).

    Layers named in ``dense_layers`` are passed through untouched. Returns
    ``{"weights", "masks", "layers": [...], "density", "n_rewired"}``
    where each layer row reports ``dropped``/``grown`` counts, the
    layer density and the ``mask_overlap`` (fraction of the old active
    set that survived the update).
    """
    if not specs:
        raise ValueError("specs must not be empty")
    skip = set(dense_layers)
    out_w: Dict[str, List[float]] = {}
    out_m: Dict[str, List[int]] = {}
    rows = []
    total = active = rewired = 0
    for idx, spec in enumerate(specs):
        if spec.name not in weights:
            raise KeyError(f"missing weights for layer {spec.name!r}")
        if spec.name not in masks:
            raise KeyError(f"missing mask for layer {spec.name!r}")
        w = weights[spec.name]
        n = layer_weight_count(spec)
        if len(w) != n:
            raise ValueError(f"layer {spec.name!r} has {len(w)} weights, spec expects {n}")
        m = _validate_mask(masks[spec.name], n)
        if spec.name in skip:
            res = {"weights": [float(v) for v in w], "mask": list(m), "dropped": [], "grown": []}
        else:
            g = None
            if regrow == "gradient":
                if gradients is None or spec.name not in gradients:
                    raise KeyError(f"missing gradients for layer {spec.name!r}")
                g = gradients[spec.name]
            res = rigl_update_layer(
                w,
                m,
                g,
                drop_fraction=drop_fraction,
                regrow=regrow,
                regrow_init=regrow_init,
                seed=None if seed is None else seed + idx,
            )
        new_m = res["mask"]
        old_active = sum(m)
        kept = sum(1 for a, b in zip(m, new_m) if a and b)
        out_w[spec.name] = res["weights"]
        out_m[spec.name] = new_m
        rows.append(
            {
                "name": spec.name,
                "size": n,
                "active": sum(new_m),
                "density": sum(new_m) / n,
                "dropped": len(res["dropped"]),
                "grown": len(res["grown"]),
                "mask_overlap": kept / old_active if old_active else 1.0,
            }
        )
        total += n
        active += sum(new_m)
        rewired += len(res["grown"])
    return {
        "weights": out_w,
        "masks": out_m,
        "layers": rows,
        "density": active / total,
        "n_rewired": rewired,
    }
