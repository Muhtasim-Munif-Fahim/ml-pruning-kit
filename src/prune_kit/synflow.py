"""SynFlow synaptic-flow pruning (Tanaka et al., NeurIPS 2020).

SynFlow scores every scalar connection by the product of its magnitude
and the synaptic saliency gradient of a data-free objective ``R``
evaluated on **unit** inputs:

    score = |W| * |∂R/∂W|

In the paper, parameters are first linearized (absolute values; ReLU
replaced by identity) so that ``R = 1^T (∏_l |W^l|) 1`` measures
synaptic flow through the network. The lowest scores are then zeroed.

This module mirrors the SNIP / Movement / Wanda APIs:

* :func:`synflow_scores` — per-weight scores for one flat buffer
* :func:`synflow_prune_layer` — one-shot prune of a single layer
* :func:`synflow_prune_model` / :func:`synflow_prune_summary` — model

Pass ``grads`` (``∂R/∂W`` from your own forward/backward) or
precomputed ``contributions`` (already ``|W| * |∂R/∂W|``), not both.
When neither is given, a practical **exponential synaptic-flow proxy**
is used for dense (and flattened conv) layers:

    score_ij = |W_ij| * exp(∑_k |W_ik|)

i.e. each connection is weighted by the exponential of its output
neuron's absolute incoming mass — the common single-layer exponential
SynFlow form used in the pruning literature when a full multi-layer
autodiff graph is unavailable.

At the model level, when ``grads`` / ``contributions`` are omitted,
:func:`synflow_unit_grads` first tries a **linearized multi-layer**
unit-input SynFlow (chaining dense/conv layers in ``specs`` order
when fan-in matches the previous fan-out). If shapes do not chain,
each layer falls back to the exponential proxy independently.

``density`` is the fraction of connections to **keep**
(``round(density * N)``). Default ``scope="global"`` ranks every score
together; ``scope="layer"`` keeps the same fraction inside each layer.
Dense ``(out, in)`` and conv ``(out, in, kh, kw)`` flat buffers are
both supported. Already-zero weights stay zero when kept.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density, sparse_mask_to_dense


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate_density(density: float) -> float:
    density = float(density)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    return density


def _validate_scope(scope: str) -> str:
    key = str(scope).lower()
    if key not in {"global", "layer"}:
        raise ValueError("scope must be 'global' or 'layer'")
    return key


def _validate_shape(weights: Sequence[float], shape: Sequence[int] | None) -> tuple[int, ...]:
    if shape is None:
        return (len(weights),)
    if isinstance(shape, (str, bytes)) or not shape:
        raise ValueError("shape must be non-empty")
    dims = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in dims):
        raise ValueError("shape dimensions must be positive")
    expected = _product(dims)
    if len(weights) != expected:
        raise ValueError(
            f"weights has {len(weights)} values, shape {dims} expects {expected}"
        )
    return dims


def _fan_out_in(shape: tuple[int, ...]) -> Tuple[int, int]:
    """Return (fan_out, fan_in) for dense/conv/flat layouts."""
    if len(shape) == 1:
        return 1, int(shape[0])
    fan_out = int(shape[0])
    fan_in = _product(shape[1:])
    return fan_out, fan_in


def _exponential_proxy_scores(
    weights: Sequence[float],
    shape: tuple[int, ...],
) -> List[float]:
    """Single-layer exponential SynFlow: |W_ij| * exp(row_sum_i |W|)."""
    fan_out, fan_in = _fan_out_in(shape)
    abs_w = [abs(float(value)) for value in weights]
    scores: List[float] = []
    for row in range(fan_out):
        start = row * fan_in
        row_sum = sum(abs_w[start : start + fan_in])
        # Clamp exponent to avoid overflow on huge rows; ranking is preserved
        # for finite positive scale factors shared within a row.
        scale = math.exp(min(row_sum, 80.0))
        for col in range(fan_in):
            scores.append(abs_w[start + col] * scale)
    return scores


def _synflow_terms(
    weights: Sequence[float],
    grads: Sequence[float] | None,
    contributions: Sequence[float] | None,
    shape: tuple[int, ...],
) -> List[float]:
    """Per-weight SynFlow scores ``|W| * |∂R/∂W|`` or absolute contributions."""
    if contributions is not None and grads is not None:
        raise ValueError("pass grads or contributions, not both")
    if contributions is not None:
        if len(contributions) != len(weights):
            raise ValueError(
                f"contributions has {len(contributions)} values, "
                f"weights has {len(weights)}"
            )
        return [abs(float(value)) for value in contributions]
    if grads is not None:
        if len(grads) != len(weights):
            raise ValueError(
                f"grads has {len(grads)} values, weights has {len(weights)}"
            )
        return [
            abs(float(weight)) * abs(float(grad))
            for weight, grad in zip(weights, grads)
        ]
    return _exponential_proxy_scores(weights, shape)


def synflow_scores(
    weights: Sequence[float],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Synaptic-flow score of each weight, same length as ``weights``.

    Default (when ``grads`` / ``contributions`` are omitted) is the
    exponential single-layer proxy ``|W_ij| * exp(∑_k |W_ik|)``. With
    ``grads``, each score is ``|weight| * |grad|`` (``|W| * |∂R/∂W|``).
    ``contributions``, when given, replaces that product with a
    caller-supplied per-connection term; the absolute value is taken.

    ``shape``, when given, must match the flat buffer (dense
    ``(out, in)`` or conv ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    resolved = _validate_shape(weights, shape)
    return _synflow_terms(weights, grads, contributions, resolved)


def _matvec_abs(weights: Sequence[float], shape: tuple[int, ...], x: Sequence[float]) -> List[float]:
    """y = |W| @ x for dense/conv flattened as (fan_out, fan_in)."""
    fan_out, fan_in = _fan_out_in(shape)
    if len(x) != fan_in:
        raise ValueError(
            f"activation has {len(x)} values, layer fan-in expects {fan_in}"
        )
    y: List[float] = []
    for row in range(fan_out):
        start = row * fan_in
        total = 0.0
        for col in range(fan_in):
            total += abs(float(weights[start + col])) * float(x[col])
        y.append(total)
    return y


def _matvec_abs_t(weights: Sequence[float], shape: tuple[int, ...], g: Sequence[float]) -> List[float]:
    """x = |W|^T @ g for dense/conv flattened as (fan_out, fan_in)."""
    fan_out, fan_in = _fan_out_in(shape)
    if len(g) != fan_out:
        raise ValueError(
            f"gradient has {len(g)} values, layer fan-out expects {fan_out}"
        )
    x = [0.0] * fan_in
    for row in range(fan_out):
        start = row * fan_in
        coeff = float(g[row])
        for col in range(fan_in):
            x[col] += abs(float(weights[start + col])) * coeff
    return x


def _layers_chain(specs: Sequence[LayerSpec]) -> bool:
    """True when each layer's fan-in equals the previous layer's fan-out."""
    if len(specs) <= 1:
        return True
    prev_out, _ = _fan_out_in(specs[0].shape)
    for spec in specs[1:]:
        fan_out, fan_in = _fan_out_in(spec.shape)
        if fan_in != prev_out:
            return False
        prev_out = fan_out
    return True


def synflow_unit_grads(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Linearized multi-layer SynFlow gradients ``∂R/∂W`` on unit inputs.

    Parameters are treated as absolute values (paper linearization).
    ``R = 1^T x_L`` where ``x_0 = 1`` and ``x_{l+1} = |W^{l+1}| x_l``.
    Requires consecutive layers to chain (fan-in of layer ``l+1`` equals
    fan-out of layer ``l``). Raises ``ValueError`` if they do not.
    """
    if not specs:
        raise ValueError("specs must not be empty")
    if not model:
        raise ValueError("model must not be empty")
    spec_names = [spec.name for spec in specs]
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("specs contain duplicate layer names")
    if set(spec_names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    if not _layers_chain(specs):
        raise ValueError(
            "synflow_unit_grads requires chained layer shapes "
            "(each fan-in equals previous fan-out)"
        )
    for spec in specs:
        expected = layer_weight_count(spec)
        weights = model[spec.name]
        if len(weights) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
            )

    # Forward activations x[0] = ones(fan_in of first), x[l+1] = |W| x[l]
    activations: List[List[float]] = []
    _, fan_in0 = _fan_out_in(specs[0].shape)
    activations.append([1.0] * fan_in0)
    for spec in specs:
        activations.append(
            _matvec_abs(model[spec.name], spec.shape, activations[-1])
        )

    # Backward: g[L] = ones(fan_out of last), g[l] = |W^{l+1}|^T g[l+1]
    n_layers = len(specs)
    grads_act: List[List[float] | None] = [None] * (n_layers + 1)
    fan_out_last, _ = _fan_out_in(specs[-1].shape)
    grads_act[n_layers] = [1.0] * fan_out_last
    for index in range(n_layers - 1, -1, -1):
        spec = specs[index]
        assert grads_act[index + 1] is not None
        grads_act[index] = _matvec_abs_t(
            model[spec.name], spec.shape, grads_act[index + 1]
        )

    # ∂R/∂W_ij = g_out_i * x_in_j
    result: Dict[str, List[float]] = {}
    for index, spec in enumerate(specs):
        fan_out, fan_in = _fan_out_in(spec.shape)
        x_in = activations[index]
        g_out = grads_act[index + 1]
        assert g_out is not None
        flat: List[float] = []
        for row in range(fan_out):
            for col in range(fan_in):
                flat.append(float(g_out[row]) * float(x_in[col]))
        result[spec.name] = flat
    return result


def _keep_count(total: int, density: float) -> int:
    return int(round(density * total))


def _keep_highest(scores: Sequence[float], density: float) -> List[int]:
    """Indices of the highest scores, in ascending index order."""
    total = len(scores)
    n_keep = _keep_count(total, density)
    if n_keep <= 0:
        return []
    if n_keep >= total:
        return list(range(total))
    ranked = sorted(range(total), key=lambda index: (-scores[index], index))
    keep_set = set(ranked[:n_keep])
    return [index for index in range(total) if index in keep_set]


def _apply_mask(
    weights: Sequence[float],
    keep_indices: Sequence[int],
    shape: tuple[int, ...],
) -> Tuple[List[float], List[int]]:
    mask = sparse_mask_to_dense(keep_indices, shape)
    pruned = [
        float(value) if bit else 0.0
        for value, bit in zip(weights, mask)
    ]
    return pruned, mask


def synflow_prune_layer(
    weights: Sequence[float],
    grads: Sequence[float] | None = None,
    *,
    contributions: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Zero the lowest SynFlow-score connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequences are not modified. Kept positions copy the original weight;
    every dropped position is ``0.0``. When ``grads`` and
    ``contributions`` are both omitted, the exponential proxy is used.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = synflow_scores(
        weights,
        grads,
        contributions=contributions,
        shape=resolved,
    )
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
) -> Tuple[str | None, Dict[str, Sequence[float]] | None]:
    if contributions is not None and grads is not None:
        raise ValueError("pass grads or contributions, not both")
    if contributions is not None:
        return "contributions", contributions
    if grads is not None:
        return "grads", grads
    return None, None


def _auto_layer_scores(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Data-free SynFlow scores: unit-grads when chained, else exponential."""
    if _layers_chain(specs):
        unit_grads = synflow_unit_grads(specs, model)
        return {
            spec.name: synflow_scores(
                model[spec.name],
                unit_grads[spec.name],
                shape=spec.shape,
            )
            for spec in specs
        }
    return {
        spec.name: synflow_scores(model[spec.name], shape=spec.shape)
        for spec in specs
    }


def _layer_scores(
    spec: LayerSpec,
    weights: Sequence[float],
    source: str | None,
    saliency: Dict[str, Sequence[float]] | None,
) -> List[float]:
    expected = layer_weight_count(spec)
    if len(weights) != expected:
        raise ValueError(
            f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
        )
    if source is None:
        return synflow_scores(weights, shape=spec.shape)
    assert saliency is not None
    if spec.name not in saliency:
        raise ValueError(f"{source} missing layer {spec.name!r}")
    values = saliency[spec.name]
    if len(values) != expected:
        raise ValueError(
            f"layer {spec.name!r} {source} has {len(values)} values, "
            f"expected {expected}"
        )
    grads = values if source == "grads" else None
    contributions = values if source == "contributions" else None
    return synflow_scores(
        weights,
        grads,
        contributions=contributions,
        shape=spec.shape,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[float, str, str | None, Dict[str, Sequence[float]] | None]:
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    spec_names = [spec.name for spec in specs]
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("specs contain duplicate layer names")
    if set(spec_names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    density = _validate_density(density)
    scope = _validate_scope(scope)
    if per_layer:
        if scope != "layer":
            raise ValueError("per_layer requires scope='layer'")
        for name, layer_density in per_layer.items():
            if name not in model:
                raise ValueError(
                    f"per_layer override references unknown layer {name!r}"
                )
            _validate_density(layer_density)
    source, saliency = _resolve_saliency(grads, contributions)
    if saliency is not None:
        for name in saliency:
            if name not in model:
                raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency


def _synflow_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None,
    contributions: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency = _validate_model_args(
        specs, model, grads, contributions, density, scope, per_layer
    )
    if source is None:
        scores = _auto_layer_scores(specs, model)
    else:
        scores = {
            spec.name: _layer_scores(spec, model[spec.name], source, saliency)
            for spec in specs
        }
    keep_by_layer: Dict[str, List[int]] = {}
    if scope == "layer":
        for spec in specs:
            layer_density = (
                density if not per_layer else per_layer.get(spec.name, density)
            )
            keep_by_layer[spec.name] = _keep_highest(
                scores[spec.name], layer_density
            )
    else:
        ranked: List[Tuple[float, int, int, str]] = []
        for order, spec in enumerate(specs):
            for index, score in enumerate(scores[spec.name]):
                ranked.append((score, order, index, spec.name))
        total = len(ranked)
        n_keep = _keep_count(total, density)
        if n_keep <= 0:
            chosen: List[Tuple[float, int, int, str]] = []
        elif n_keep >= total:
            chosen = ranked
        else:
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            chosen = ranked[:n_keep]
        keep_by_layer = {spec.name: [] for spec in specs}
        for _, _, index, name in chosen:
            keep_by_layer[name].append(index)
        for name in keep_by_layer:
            keep_by_layer[name].sort()

    pruned: Dict[str, List[float]] = {}
    masks: Dict[str, List[int]] = {}
    for spec in specs:
        layer_pruned, mask = _apply_mask(
            model[spec.name], keep_by_layer[spec.name], spec.shape
        )
        pruned[spec.name] = layer_pruned
        masks[spec.name] = mask
    return pruned, masks, scores, density, scope


def synflow_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """One-shot SynFlow prune of every dense and conv layer.

    ``scope="global"`` keeps the top ``density`` fraction of connections
    across the whole model. ``scope="layer"`` ranks inside each layer;
    ``per_layer`` then overrides ``density`` by layer name. When
    ``grads`` and ``contributions`` are omitted, scores come from
    linearized unit-input SynFlow (chained layers) or the exponential
    proxy (otherwise).
    """
    pruned, _, _, _, _ = _synflow_prune_detailed(
        specs,
        model,
        grads,
        contributions,
        density,
        scope,
        per_layer,
    )
    return pruned


def synflow_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    grads: Dict[str, Sequence[float]] | None = None,
    *,
    contributions: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run SynFlow and return per-layer keep counts, masks, and score totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the SynFlow scores ``|W| * |∂R/∂W|`` (or the exponential proxy).
    """
    pruned, masks, scores, density, scope = _synflow_prune_detailed(
        specs,
        model,
        grads,
        contributions,
        density,
        scope,
        per_layer,
    )
    rows: List[Dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item.name):
        layer_scores = scores[spec.name]
        mask = masks[spec.name]
        total = len(mask)
        kept = sum(mask)
        score_sum = sum(layer_scores)
        max_score = max(layer_scores) if layer_scores else 0.0
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "total_weights": total,
            "kept_weights": kept,
            "survival_fraction": round(mask_density(mask), 4) if total else 0.0,
            "score_sum": score_sum,
            "max_score": max_score,
        })
    total = sum(int(row["total_weights"]) for row in rows)
    total_kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "density": density,
        "sparsity": 1.0 - density,
        "scope": scope,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = [
    "synflow_scores",
    "synflow_unit_grads",
    "synflow_prune_layer",
    "synflow_prune_model",
    "synflow_prune_summary",
]
