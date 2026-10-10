"""Erdős–Rényi (ER) and Erdős–Rényi-Kernel (ERK) layer-wise sparsity.

Sparse-training methods such as SET (Mocanu et al., 2018) and RigL
(Evci et al., ICML 2020) do not give every layer the same density.
Instead each layer's density scales with how "wide" it is relative to
its parameter count:

    ER  : density_l  ∝  (n_out + n_in) / (n_out * n_in)
    ERK : density_l  ∝  (sum of all dims) / (product of all dims)

For a dense ``(out, in)`` layer ER and ERK coincide. For a conv
``(out, in, kh, kw)`` layer ERK also counts the kernel dimensions, so a
conv layer's density reflects its full parameter count; ER only looks at
the channel dims and therefore over-allocates density to wide kernels.
Larger layers end up sparser and small layers (first / last / narrow
ones) denser.

A single scale factor ``epsilon`` is chosen so that the parameter-weighted
average density over all layers equals the requested global ``density``:

    sum_l n_l * density_l = density * sum_l n_l

If ``epsilon * raw_l`` would exceed 1 for some layer, that layer is made
fully dense and ``epsilon`` is recomputed for the remaining layers (the
RigL ``get_sparsities`` procedure). ``erk_power_scale`` (RigL's
``erk_power_scale``) raises the raw ratios to a power: ``1.0`` is
standard ERK, ``0.0`` recovers a uniform allocation. ``dense_layers``
names layers that must stay dense (e.g. the input stem).

``method="uniform"`` gives every non-dense layer the same density, with
any forced-dense layers' budget taken out of the others.

After allocating densities, ``erk_prune_model`` applies per-layer
magnitude pruning with those densities (``round(density_l * n_l)``
weights kept per layer, ties toward the lower index).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density

_METHODS = ("erk", "er", "uniform")


def _validate_density(density: float) -> float:
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    return density


def _validate_method(method: str) -> str:
    key = str(method).lower()
    if key not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    return key


def _raw_ratio(spec: LayerSpec, method: str, power: float) -> float:
    dims = [int(dim) for dim in spec.shape]
    if method == "uniform":
        return 1.0
    if method == "er":
        if len(dims) >= 2:
            n_out, n_in = dims[0], dims[1]
            ratio = (n_out + n_in) / float(n_out * n_in)
        else:
            ratio = 1.0 / float(dims[0])
    else:  # erk
        count = layer_weight_count(spec)
        ratio = float(sum(dims)) / float(count)
    return ratio ** float(power)


def erk_densities(
    specs: Sequence[LayerSpec],
    *,
    density: float = 0.5,
    method: str = "erk",
    erk_power_scale: float = 1.0,
    dense_layers: Iterable[str] = (),
) -> Dict[str, float]:
    """Per-layer keep densities whose parameter-weighted mean is ``density``.

    Returns ``{layer_name: density_l}`` with every value in ``(0, 1]``.
    Layers whose scaled ratio would exceed 1 (and any in ``dense_layers``)
    are set to ``1.0``. Raises ``ValueError`` if the forced-dense layers
    alone already exceed the global parameter budget.
    """
    if not specs:
        raise ValueError("specs must not be empty")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("specs contain duplicate layer names")
    density = _validate_density(density)
    method = _validate_method(method)
    power = float(erk_power_scale)
    if power < 0.0:
        raise ValueError("erk_power_scale must be non-negative")
    dense = set(dense_layers)
    unknown = dense - set(names)
    if unknown:
        raise ValueError(f"dense_layers references unknown layer(s) {sorted(unknown)}")

    counts = {spec.name: layer_weight_count(spec) for spec in specs}
    raw = {spec.name: _raw_ratio(spec, method, power) for spec in specs}
    total = sum(counts.values())
    budget = density * total

    if density == 1.0:
        return {name: 1.0 for name in names}

    while True:
        dense_params = sum(counts[name] for name in dense)
        sparse_names = [name for name in names if name not in dense]
        if not sparse_names:
            if dense_params > budget + 1e-9:
                raise ValueError("dense_layers exceed the global density budget")
            return {name: 1.0 for name in names}
        remaining = budget - dense_params
        if remaining <= 0.0:
            raise ValueError("dense_layers exceed the global density budget")
        divisor = sum(raw[name] * counts[name] for name in sparse_names)
        epsilon = remaining / divisor
        overflow = [name for name in sparse_names if epsilon * raw[name] > 1.0]
        if not overflow:
            break
        # make the layer with the largest scaled ratio dense and retry
        worst = max(overflow, key=lambda name: (raw[name], -names.index(name)))
        dense.add(worst)

    result: Dict[str, float] = {}
    for name in names:
        result[name] = 1.0 if name in dense else epsilon * raw[name]
    return result


def _magnitude_keep(weights: Sequence[float], density: float) -> List[int]:
    total = len(weights)
    n_keep = int(round(density * total))
    if n_keep >= total:
        return [1] * total
    mask = [0] * total
    if n_keep <= 0:
        return mask
    ranked = sorted(range(total), key=lambda i: (-abs(float(weights[i])), i))
    for index in ranked[:n_keep]:
        mask[index] = 1
    return mask


def _erk_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    density: float,
    method: str,
    erk_power_scale: float,
    dense_layers: Iterable[str],
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, float]]:
    if not model:
        raise ValueError("model must not be empty")
    if set(spec.name for spec in specs) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    densities = erk_densities(
        specs,
        density=density,
        method=method,
        erk_power_scale=erk_power_scale,
        dense_layers=dense_layers,
    )
    pruned: Dict[str, List[float]] = {}
    masks: Dict[str, List[int]] = {}
    for spec in specs:
        weights = model[spec.name]
        expected = layer_weight_count(spec)
        if len(weights) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
            )
        mask = _magnitude_keep(weights, densities[spec.name])
        masks[spec.name] = mask
        pruned[spec.name] = [
            float(value) if bit else 0.0 for value, bit in zip(weights, mask)
        ]
    return pruned, masks, densities


def erk_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    method: str = "erk",
    erk_power_scale: float = 1.0,
    dense_layers: Iterable[str] = (),
) -> Dict[str, List[float]]:
    """Magnitude-prune each layer to its ER / ERK / uniform density."""
    pruned, _, _ = _erk_prune_detailed(
        specs, model, density, method, erk_power_scale, dense_layers
    )
    return pruned


def erk_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    method: str = "erk",
    erk_power_scale: float = 1.0,
    dense_layers: Iterable[str] = (),
) -> Dict[str, object]:
    """Allocate ERK densities, prune, and report per-layer budgets."""
    dense_layers = list(dense_layers)
    pruned, masks, densities = _erk_prune_detailed(
        specs, model, density, method, erk_power_scale, dense_layers
    )
    rows: List[Dict[str, object]] = []
    for spec in specs:
        mask = masks[spec.name]
        total = len(mask)
        kept = sum(mask)
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "shape": list(spec.shape),
            "total_weights": total,
            "target_density": densities[spec.name],
            "kept_weights": kept,
            "survival_fraction": round(mask_density(mask), 4) if total else 0.0,
            "dense": densities[spec.name] >= 1.0,
        })
    total = sum(int(row["total_weights"]) for row in rows)
    total_kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "method": _validate_method(method),
        "density": float(density),
        "sparsity": 1.0 - float(density),
        "erk_power_scale": float(erk_power_scale),
        "dense_layers": dense_layers,
        "layer_densities": densities,
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = ["erk_densities", "erk_prune_model", "erk_prune_summary"]
