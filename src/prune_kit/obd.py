"""Optimal Brain Damage (OBD) pruning (LeCun, Denker, Solla, 1990).

OBD scores every scalar connection by a second-order saliency that
approximates the change in loss when that weight is set to zero:

    saliency_i ≈ (1/2) * H_ii * w_i^2

where ``H_ii`` is the corresponding diagonal entry of the Hessian of
the loss with respect to the parameters. Connections with the
**lowest** saliency are pruned first (they hurt the loss least).

This module mirrors the SynFlow / Movement / Wanda APIs:

* :func:`obd_scores` — per-weight saliency for one flat buffer
* :func:`obd_prune_layer` — one-shot prune of a single layer
* :func:`obd_prune_model` / :func:`obd_prune_summary` — model

Pass either ``hess_diag`` (precomputed diagonal Hessian ``H_ii``) **or**
``grads`` (first-order gradients). When ``grads`` are given, the kit
uses the common **squared-gradient diagonal proxy**

    H_ii ≈ g_i^2

so that

    saliency_i ≈ (1/2) * g_i^2 * w_i^2

This proxy is well-documented in the pruning literature as a practical
stand-in when a full (or even diagonal) Hessian is unavailable; it is
**not** the exact OBD formula. Prefer ``hess_diag`` whenever you can
compute or estimate true diagonal second derivatives (e.g. via
Hessian-vector products with coordinate basis vectors, or empirical
Fisher / Gauss-Newton diagonals).

Pass one of ``hess_diag`` / ``grads``, not both. ``density`` is the
fraction of connections to **keep** (``round(density * N)``). Default
``scope="global"`` ranks every saliency together; ``scope="layer"``
keeps the same fraction inside each layer. Dense ``(out, in)`` and
conv ``(out, in, kh, kw)`` flat buffers are both supported.
Already-zero weights stay zero when kept.
"""

from __future__ import annotations

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


def _obd_terms(
    weights: Sequence[float],
    hess_diag: Sequence[float] | None,
    grads: Sequence[float] | None,
) -> List[float]:
    """Per-weight OBD saliency ``(1/2) * H_ii * w_i^2``.

    ``hess_diag`` supplies ``H_ii`` directly. ``grads`` approximates
    ``H_ii ≈ g_i^2`` (squared-gradient / empirical-Fisher diagonal
    proxy). Pass exactly one.
    """
    if hess_diag is not None and grads is not None:
        raise ValueError("pass hess_diag or grads, not both")
    if hess_diag is None and grads is None:
        raise ValueError("hess_diag or grads is required")
    if hess_diag is not None:
        if len(hess_diag) != len(weights):
            raise ValueError(
                f"hess_diag has {len(hess_diag)} values, "
                f"weights has {len(weights)}"
            )
        return [
            0.5 * float(h) * (float(w) ** 2)
            for w, h in zip(weights, hess_diag)
        ]
    assert grads is not None
    if len(grads) != len(weights):
        raise ValueError(
            f"grads has {len(grads)} values, weights has {len(weights)}"
        )
    # Squared-gradient diagonal Hessian proxy: H_ii ≈ g_i^2
    return [
        0.5 * (float(g) ** 2) * (float(w) ** 2)
        for w, g in zip(weights, grads)
    ]


def obd_scores(
    weights: Sequence[float],
    hess_diag: Sequence[float] | None = None,
    *,
    grads: Sequence[float] | None = None,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """OBD saliency of each weight, same length as ``weights``.

    With ``hess_diag``, each score is ``(1/2) * H_ii * w_i^2`` (LeCun
    et al.). With ``grads``, each score is
    ``(1/2) * g_i^2 * w_i^2``, using ``g_i^2`` as a diagonal Hessian
    proxy when the true ``H_ii`` is unavailable. Pass exactly one of
    ``hess_diag`` / ``grads``. ``shape``, when given, must match the
    flat buffer (dense ``(out, in)`` or conv ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_shape(weights, shape)
    return _obd_terms(weights, hess_diag, grads)


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


def obd_prune_layer(
    weights: Sequence[float],
    hess_diag: Sequence[float] | None = None,
    *,
    grads: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
) -> List[float]:
    """Zero the lowest-OBD-saliency connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequences are not modified. Kept positions copy the original
    weight; every dropped position is ``0.0``. Pass ``hess_diag`` or
    ``grads`` (squared-gradient proxy), not both.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = obd_scores(
        weights,
        hess_diag,
        grads=grads,
        shape=resolved,
    )
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    hess_diag: Dict[str, Sequence[float]] | None,
    grads: Dict[str, Sequence[float]] | None,
) -> Tuple[str, Dict[str, Sequence[float]]]:
    if hess_diag is not None and grads is not None:
        raise ValueError("pass hess_diag or grads, not both")
    if hess_diag is None and grads is None:
        raise ValueError("hess_diag or grads is required")
    if hess_diag is not None:
        return "hess_diag", hess_diag
    assert grads is not None
    return "grads", grads


def _layer_scores(
    spec: LayerSpec,
    weights: Sequence[float],
    source: str,
    saliency: Dict[str, Sequence[float]],
) -> List[float]:
    expected = layer_weight_count(spec)
    if len(weights) != expected:
        raise ValueError(
            f"layer {spec.name!r} has {len(weights)} weights, expected {expected}"
        )
    if spec.name not in saliency:
        raise ValueError(f"{source} missing layer {spec.name!r}")
    values = saliency[spec.name]
    if len(values) != expected:
        raise ValueError(
            f"layer {spec.name!r} {source} has {len(values)} values, "
            f"expected {expected}"
        )
    hess = values if source == "hess_diag" else None
    layer_grads = values if source == "grads" else None
    return obd_scores(
        weights,
        hess,
        grads=layer_grads,
        shape=spec.shape,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_diag: Dict[str, Sequence[float]] | None,
    grads: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[float, str, str, Dict[str, Sequence[float]]]:
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
    source, saliency = _resolve_saliency(hess_diag, grads)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency


def _obd_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_diag: Dict[str, Sequence[float]] | None,
    grads: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency = _validate_model_args(
        specs,
        model,
        hess_diag,
        grads,
        density,
        scope,
        per_layer,
    )
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


def obd_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_diag: Dict[str, Sequence[float]] | None = None,
    *,
    grads: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, List[float]]:
    """One-shot OBD prune of every dense and conv layer.

    ``scope="global"`` keeps the top ``density`` fraction of connections
    across the whole model. ``scope="layer"`` ranks inside each layer;
    ``per_layer`` then overrides ``density`` by layer name. Pass
    ``hess_diag`` (diagonal Hessian) or ``grads`` (squared-gradient
    proxy), not both.
    """
    pruned, _, _, _, _ = _obd_prune_detailed(
        specs,
        model,
        hess_diag,
        grads,
        density,
        scope,
        per_layer,
    )
    return pruned


def obd_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_diag: Dict[str, Sequence[float]] | None = None,
    *,
    grads: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run OBD and return per-layer keep counts, masks, and saliency totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the OBD saliency ``(1/2) * H_ii * w_i^2`` (or the ``g_i^2`` proxy).
    """
    pruned, masks, scores, density, scope = _obd_prune_detailed(
        specs,
        model,
        hess_diag,
        grads,
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
    "obd_scores",
    "obd_prune_layer",
    "obd_prune_model",
    "obd_prune_summary",
]
