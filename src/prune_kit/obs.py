"""Optimal Brain Surgeon (OBS) pruning (Hassibi & Stork, 1992/1993).

OBS scores every scalar connection by a second-order saliency that uses
the **inverse** Hessian diagonal:

    saliency_i ≈ w_i^2 / (2 * H^{-1}_ii)

Connections with the **lowest** saliency are pruned first (they hurt the
loss least). When the Hessian is diagonal, ``H^{-1}_ii = 1/H_ii`` and
OBS reduces to Optimal Brain Damage (OBD); the classic OBS result also
updates the remaining weights with an optimal perturbation, which this
kit does **not** implement — only the saliency ranking / one-shot mask.

This module mirrors the OBD / SynFlow / Wanda APIs:

* :func:`obs_scores` — per-weight saliency for one flat buffer
* :func:`obs_prune_layer` — one-shot prune of a single layer
* :func:`obs_prune_model` / :func:`obs_prune_summary` — model

Pass **exactly one** of:

* ``hess_inv_diag`` — diagonal of ``H^{-1}`` directly (**exact OBS**
  saliency formula above).
* ``hess_diag`` — diagonal of ``H``. Uses the reciprocal proxy
  ``H^{-1}_ii ≈ 1 / (H_ii + eps)`` (with a small ``eps`` for zeros /
  near-zeros). Equivalent to OBD when ``H`` is diagonal and ``eps`` is
  negligible; documented here as a **proxy**, not full-matrix OBS.
* ``grads`` — first-order gradients. Uses the squared-gradient diagonal
  proxy ``H_ii ≈ g_i^2`` then ``H^{-1}_ii ≈ 1 / (g_i^2 + eps)``, so
  saliency becomes ``w_i^2 / (2 / (g_i^2 + eps))``. This is a
  **practical stand-in** when neither ``H`` nor ``H^{-1}`` is
  available; it is **not** the exact OBS formula.

``density`` is the fraction of connections to **keep**
(``round(density * N)``). Default ``scope="global"`` ranks every
saliency together; ``scope="layer"`` keeps the same fraction inside
each layer. Dense ``(out, in)`` and conv ``(out, in, kh, kw)`` flat
buffers are both supported. Already-zero weights stay zero when kept.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density, sparse_mask_to_dense

_EPS = 1e-12


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


def _obs_terms(
    weights: Sequence[float],
    hess_inv_diag: Sequence[float] | None,
    hess_diag: Sequence[float] | None,
    grads: Sequence[float] | None,
    eps: float = _EPS,
) -> List[float]:
    """Per-weight OBS saliency ``w_i^2 / (2 * H^{-1}_ii)``.

    Pass exactly one of ``hess_inv_diag`` / ``hess_diag`` / ``grads``.
    """
    provided = [
        name
        for name, value in (
            ("hess_inv_diag", hess_inv_diag),
            ("hess_diag", hess_diag),
            ("grads", grads),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError(
            "pass hess_inv_diag, hess_diag, or grads — exactly one, not both"
        )
    if not provided:
        raise ValueError("hess_inv_diag, hess_diag, or grads is required")

    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    if hess_inv_diag is not None:
        if len(hess_inv_diag) != len(weights):
            raise ValueError(
                f"hess_inv_diag has {len(hess_inv_diag)} values, "
                f"weights has {len(weights)}"
            )
        # Exact OBS: saliency = w^2 / (2 * H^{-1}_ii)
        out: List[float] = []
        for w, hinv in zip(weights, hess_inv_diag):
            denom = 2.0 * max(float(hinv), eps)
            out.append((float(w) ** 2) / denom)
        return out

    if hess_diag is not None:
        if len(hess_diag) != len(weights):
            raise ValueError(
                f"hess_diag has {len(hess_diag)} values, "
                f"weights has {len(weights)}"
            )
        # Proxy: H^{-1}_ii ≈ 1/(H_ii + eps)
        out = []
        for w, h in zip(weights, hess_diag):
            hinv = 1.0 / (float(h) + eps)
            denom = 2.0 * max(hinv, eps)
            out.append((float(w) ** 2) / denom)
        return out

    assert grads is not None
    if len(grads) != len(weights):
        raise ValueError(
            f"grads has {len(grads)} values, weights has {len(weights)}"
        )
    # Squared-gradient proxy: H_ii ≈ g^2, H^{-1}_ii ≈ 1/(g^2 + eps)
    out = []
    for w, g in zip(weights, grads):
        hinv = 1.0 / ((float(g) ** 2) + eps)
        denom = 2.0 * max(hinv, eps)
        out.append((float(w) ** 2) / denom)
    return out


def obs_scores(
    weights: Sequence[float],
    hess_inv_diag: Sequence[float] | None = None,
    *,
    hess_diag: Sequence[float] | None = None,
    grads: Sequence[float] | None = None,
    shape: Sequence[int] | None = None,
    eps: float = _EPS,
) -> List[float]:
    """OBS saliency of each weight, same length as ``weights``.

    With ``hess_inv_diag``, each score is ``w_i^2 / (2 * H^{-1}_ii)``
    (Hassibi & Stork). With ``hess_diag``, uses
    ``H^{-1}_ii ≈ 1/(H_ii + eps)``. With ``grads``, uses
    ``H^{-1}_ii ≈ 1/(g_i^2 + eps)``. Pass exactly one. ``shape``, when
    given, must match the flat buffer (dense ``(out, in)`` or conv
    ``(out, in, kh, kw)``).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    _validate_shape(weights, shape)
    return _obs_terms(weights, hess_inv_diag, hess_diag, grads, eps=eps)


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


def obs_prune_layer(
    weights: Sequence[float],
    hess_inv_diag: Sequence[float] | None = None,
    *,
    hess_diag: Sequence[float] | None = None,
    grads: Sequence[float] | None = None,
    density: float = 0.5,
    shape: Sequence[int] | None = None,
    eps: float = _EPS,
) -> List[float]:
    """Zero the lowest-OBS-saliency connections in one layer.

    ``density`` is the fraction of connections to keep. The original
    sequences are not modified. Kept positions copy the original
    weight; every dropped position is ``0.0``. Pass exactly one of
    ``hess_inv_diag`` / ``hess_diag`` / ``grads``.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    density = _validate_density(density)
    resolved = _validate_shape(weights, shape)
    scores = obs_scores(
        weights,
        hess_inv_diag,
        hess_diag=hess_diag,
        grads=grads,
        shape=resolved,
        eps=eps,
    )
    kept = _keep_highest(scores, density)
    pruned, _ = _apply_mask(weights, kept, resolved)
    return pruned


def _resolve_saliency(
    hess_inv_diag: Dict[str, Sequence[float]] | None,
    hess_diag: Dict[str, Sequence[float]] | None,
    grads: Dict[str, Sequence[float]] | None,
) -> Tuple[str, Dict[str, Sequence[float]]]:
    provided = [
        name
        for name, value in (
            ("hess_inv_diag", hess_inv_diag),
            ("hess_diag", hess_diag),
            ("grads", grads),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError(
            "pass hess_inv_diag, hess_diag, or grads — exactly one, not both"
        )
    if not provided:
        raise ValueError("hess_inv_diag, hess_diag, or grads is required")
    source = provided[0]
    if source == "hess_inv_diag":
        assert hess_inv_diag is not None
        return source, hess_inv_diag
    if source == "hess_diag":
        assert hess_diag is not None
        return source, hess_diag
    assert grads is not None
    return source, grads


def _layer_scores(
    spec: LayerSpec,
    weights: Sequence[float],
    source: str,
    saliency: Dict[str, Sequence[float]],
    eps: float,
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
    hinv = values if source == "hess_inv_diag" else None
    hess = values if source == "hess_diag" else None
    layer_grads = values if source == "grads" else None
    return obs_scores(
        weights,
        hinv,
        hess_diag=hess,
        grads=layer_grads,
        shape=spec.shape,
        eps=eps,
    )


def _validate_model_args(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_inv_diag: Dict[str, Sequence[float]] | None,
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
    source, saliency = _resolve_saliency(hess_inv_diag, hess_diag, grads)
    for name in saliency:
        if name not in model:
            raise ValueError(f"{source} references unknown layer {name!r}")
    return density, scope, source, saliency


def _obs_prune_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_inv_diag: Dict[str, Sequence[float]] | None,
    hess_diag: Dict[str, Sequence[float]] | None,
    grads: Dict[str, Sequence[float]] | None,
    density: float,
    scope: str,
    per_layer: Dict[str, float] | None,
    eps: float,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, List[float]], float, str]:
    density, scope, source, saliency = _validate_model_args(
        specs,
        model,
        hess_inv_diag,
        hess_diag,
        grads,
        density,
        scope,
        per_layer,
    )
    scores = {
        spec.name: _layer_scores(spec, model[spec.name], source, saliency, eps)
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


def obs_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_inv_diag: Dict[str, Sequence[float]] | None = None,
    *,
    hess_diag: Dict[str, Sequence[float]] | None = None,
    grads: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
    eps: float = _EPS,
) -> Dict[str, List[float]]:
    """One-shot OBS prune of every dense and conv layer.

    ``scope="global"`` keeps the top ``density`` fraction of connections
    across the whole model. ``scope="layer"`` ranks inside each layer;
    ``per_layer`` then overrides ``density`` by layer name. Pass exactly
    one of ``hess_inv_diag`` / ``hess_diag`` / ``grads``.
    """
    pruned, _, _, _, _ = _obs_prune_detailed(
        specs,
        model,
        hess_inv_diag,
        hess_diag,
        grads,
        density,
        scope,
        per_layer,
        eps,
    )
    return pruned


def obs_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    hess_inv_diag: Dict[str, Sequence[float]] | None = None,
    *,
    hess_diag: Dict[str, Sequence[float]] | None = None,
    grads: Dict[str, Sequence[float]] | None = None,
    density: float = 0.5,
    scope: str = "global",
    per_layer: Dict[str, float] | None = None,
    eps: float = _EPS,
) -> Dict[str, object]:
    """Run OBS and return per-layer keep counts, masks, and saliency totals.

    ``kept_weights`` counts connections retained by the mask, including
    a connection whose stored weight was already zero. Score totals use
    the OBS saliency ``w_i^2 / (2 * H^{-1}_ii)`` (or a documented proxy).
    """
    pruned, masks, scores, density, scope = _obs_prune_detailed(
        specs,
        model,
        hess_inv_diag,
        hess_diag,
        grads,
        density,
        scope,
        per_layer,
        eps,
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
    "obs_scores",
    "obs_prune_layer",
    "obs_prune_model",
    "obs_prune_summary",
]
