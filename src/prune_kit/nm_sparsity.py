"""N:M semi-structured sparsity (e.g. NVIDIA Ampere 2:4).

Fine-grained structured sparsity keeps at most ``n`` non-zero weights in
every group of ``m`` consecutive weights along the **input** dimension
(Mishra et al., *Accelerating Sparse Deep Neural Networks*, 2021; Zhou et
al., *Learning N:M Fine-grained Structured Sparse Neural Networks From
Scratch*, ICLR 2021). Sparse tensor cores execute 2:4 layers at roughly
twice the dense throughput, so the pattern is the standard target for
hardware-friendly pruning.

Groups are formed as follows:

- dense ``(out, in)``: ``m`` consecutive input columns of each row;
- conv ``(out, in, kh, kw)``: ``m`` consecutive input channels at a fixed
  ``(out, kh, kw)`` position (the layout sparse tensor cores consume);
- a 1-D buffer ``(n,)``: ``m`` consecutive entries.

Inside each group the ``n`` highest-scoring weights survive. The score is
``|w|`` by default, but any per-weight importance (Wanda, SNIP, movement,
...) can be passed through ``scores`` to get e.g. "Wanda 2:4". Ties break
toward the lower flat index. The grouped dimension must be divisible by
``m`` unless ``allow_partial=True``, in which case the short trailing group
keeps at most ``n`` of its weights.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .masks import mask_density


def _product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _validate_nm(n: int, m: int) -> Tuple[int, int]:
    if isinstance(n, bool) or isinstance(m, bool):
        raise ValueError("n and m must be integers")
    if int(n) != n or int(m) != m:
        raise ValueError("n and m must be integers")
    n, m = int(n), int(m)
    if m < 1:
        raise ValueError("m must be a positive integer")
    if not 0 <= n <= m:
        raise ValueError("n must satisfy 0 <= n <= m")
    return n, m


def _resolve_shape(weights: Sequence[float], shape: Sequence[int] | None) -> Tuple[int, ...]:
    if shape is None:
        return (len(weights),)
    if isinstance(shape, (str, bytes)) or not shape:
        raise ValueError("shape must be non-empty")
    dims = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in dims):
        raise ValueError("shape dimensions must be positive")
    if len(weights) != _product(dims):
        raise ValueError(
            f"weights has {len(weights)} values, shape {dims} expects {_product(dims)}"
        )
    return dims


def nm_groups(
    shape: Sequence[int],
    m: int = 4,
    *,
    allow_partial: bool = False,
) -> List[List[int]]:
    """Flat-index groups of ``m`` consecutive weights along the input axis.

    The input axis is axis 1 for 2-D and 4-D shapes and axis 0 for 1-D.
    Raises ``ValueError`` when that axis is not divisible by ``m`` and
    ``allow_partial`` is false.
    """
    _, m = _validate_nm(0, m)
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("shape must be non-empty with positive dimensions")
    if len(dims) == 1:
        outer, axis_len, inner = 1, dims[0], 1
    else:
        outer, axis_len, inner = dims[0], dims[1], _product(dims[2:])
    if axis_len % m and not allow_partial:
        raise ValueError(
            f"input dimension {axis_len} is not divisible by m={m}; "
            "pass allow_partial=True to keep a short trailing group"
        )
    groups: List[List[int]] = []
    for o in range(outer):
        base = o * axis_len * inner
        for pos in range(inner):
            for start in range(0, axis_len, m):
                stop = min(start + m, axis_len)
                groups.append([base + i * inner + pos for i in range(start, stop)])
    return groups


def nm_mask(
    weights: Sequence[float],
    n: int = 2,
    m: int = 4,
    *,
    shape: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    allow_partial: bool = False,
) -> List[int]:
    """Flat 0/1 keep-mask with at most ``n`` ones in every ``m``-group."""
    if not weights:
        raise ValueError("weights must not be empty")
    n, m = _validate_nm(n, m)
    dims = _resolve_shape(weights, shape)
    if scores is None:
        values = [abs(float(w)) for w in weights]
    else:
        if len(scores) != len(weights):
            raise ValueError("scores must have one value per weight")
        values = [float(s) for s in scores]
    mask = [0] * len(weights)
    for group in nm_groups(dims, m, allow_partial=allow_partial):
        ranked = sorted(group, key=lambda idx: (-values[idx], idx))
        for idx in ranked[: min(n, len(group))]:
            mask[idx] = 1
    return mask


def nm_prune_layer(
    weights: Sequence[float],
    n: int = 2,
    m: int = 4,
    *,
    shape: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    allow_partial: bool = False,
) -> List[float]:
    """Return a copy of ``weights`` pruned to the N:M pattern."""
    mask = nm_mask(weights, n, m, shape=shape, scores=scores, allow_partial=allow_partial)
    return [float(w) if bit else 0.0 for w, bit in zip(weights, mask)]


def is_nm_sparse(
    weights: Sequence[float],
    n: int = 2,
    m: int = 4,
    *,
    shape: Sequence[int] | None = None,
    allow_partial: bool = False,
) -> bool:
    """True when every ``m``-group holds at most ``n`` non-zero weights."""
    if not weights:
        raise ValueError("weights must not be empty")
    n, m = _validate_nm(n, m)
    dims = _resolve_shape(weights, shape)
    return all(
        sum(1 for idx in group if float(weights[idx]) != 0.0) <= n
        for group in nm_groups(dims, m, allow_partial=allow_partial)
    )


def _validate_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    scores: Dict[str, Sequence[float]] | None,
    skip: Sequence[str],
) -> None:
    if not model:
        raise ValueError("model must not be empty")
    if not specs:
        raise ValueError("specs must not be empty")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("specs contain duplicate layer names")
    if set(names) != set(model.keys()):
        raise ValueError("specs and model must reference the same layers")
    for name in skip:
        if name not in model:
            raise ValueError(f"skip references unknown layer {name!r}")
    if scores:
        for name in scores:
            if name not in model:
                raise ValueError(f"scores references unknown layer {name!r}")
    for spec in specs:
        expected = layer_weight_count(spec)
        if len(model[spec.name]) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(model[spec.name])} weights, expected {expected}"
            )


def _nm_detailed(specs, model, n, m, scores, skip, allow_partial):
    n, m = _validate_nm(n, m)
    skip = tuple(skip or ())
    _validate_model(specs, model, scores, skip)
    pruned: Dict[str, List[float]] = {}
    masks: Dict[str, List[int]] = {}
    for spec in specs:
        weights = model[spec.name]
        if spec.name in skip:
            masks[spec.name] = [1] * len(weights)
            pruned[spec.name] = [float(w) for w in weights]
            continue
        layer_scores = None if not scores else scores.get(spec.name)
        mask = nm_mask(weights, n, m, shape=spec.shape, scores=layer_scores,
                       allow_partial=allow_partial)
        masks[spec.name] = mask
        pruned[spec.name] = [float(w) if bit else 0.0 for w, bit in zip(weights, mask)]
    return pruned, masks, n, m, skip


def nm_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    n: int = 2,
    m: int = 4,
    scores: Dict[str, Sequence[float]] | None = None,
    skip: Sequence[str] = (),
    allow_partial: bool = False,
) -> Dict[str, List[float]]:
    """Prune every dense / conv layer to the N:M pattern.

    ``scores`` optionally maps layer name to a per-weight importance buffer
    (layers without an entry fall back to ``|w|``). Layers listed in
    ``skip`` stay dense, which is common for the first and last layers.
    """
    pruned, _, _, _, _ = _nm_detailed(specs, model, n, m, scores, skip, allow_partial)
    return pruned


def nm_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    *,
    n: int = 2,
    m: int = 4,
    scores: Dict[str, Sequence[float]] | None = None,
    skip: Sequence[str] = (),
    allow_partial: bool = False,
) -> Dict[str, object]:
    """Run N:M pruning and report per-layer keep counts and compliance."""
    pruned, masks, n, m, skip = _nm_detailed(specs, model, n, m, scores, skip, allow_partial)
    rows: List[Dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item.name):
        mask = masks[spec.name]
        dense = spec.name in skip
        groups = 0 if dense else len(nm_groups(spec.shape, m, allow_partial=allow_partial))
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "total_weights": len(mask),
            "kept_weights": sum(mask),
            "survival_fraction": round(mask_density(mask), 4),
            "groups": groups,
            "skipped": dense,
            "nm_compliant": True if dense else is_nm_sparse(
                pruned[spec.name], n, m, shape=spec.shape, allow_partial=allow_partial
            ),
        })
    total = sum(int(row["total_weights"]) for row in rows)
    kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "n": n,
        "m": m,
        "pattern": f"{n}:{m}",
        "target_density": n / m,
        "skip": list(skip),
        "layers": rows,
        "total": total,
        "total_kept": kept,
        "overall_survival": round(kept / total, 4) if total else 0.0,
        "masks": masks,
        "pruned_weights": pruned,
    }


__all__ = [
    "nm_groups",
    "nm_mask",
    "nm_prune_layer",
    "is_nm_sparse",
    "nm_prune_model",
    "nm_prune_summary",
]
