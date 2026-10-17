"""Network Slimming: channel pruning by batch-norm scaling factors.

Liu, Li, Shen, Huang, Yan & Zhang, "Learning Efficient Convolutional
Networks through Network Slimming" (ICCV 2017).

Each convolutional output channel is followed by a batch-norm layer with
a learnable scale ``gamma``. Training with an extra L1 penalty
``lam * sum |gamma|`` drives unimportant channels' scales towards zero;
afterwards **one global threshold** over all scales (the
``prune_ratio`` quantile of ``|gamma|``) decides which channels to drop
in every layer at once, so layers end up with *different* widths.

This module provides the sparsity-training helpers
(:func:`slimming_l1_penalty`, :func:`slimming_subgradient`), the global
channel selection (:func:`slimming_threshold`,
:func:`slimming_channel_masks`) and the model-level prune
(:func:`slimming_prune_model`, :func:`slimming_prune_summary`), which
zeroes the pruned output filters and, optionally, the matching input
channels of the following layer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .structured import _copy_layer, channel_groups

__all__ = [
    "slimming_l1_penalty",
    "slimming_subgradient",
    "slimming_threshold",
    "slimming_channel_masks",
    "slimming_prune_model",
    "slimming_prune_summary",
]


def _validate_scales(bn_scales: Dict[str, Sequence[float]]) -> None:
    if not bn_scales:
        raise ValueError("bn_scales must contain at least one layer")
    for name, scales in bn_scales.items():
        if len(scales) == 0:
            raise ValueError(f"bn_scales[{name!r}] must not be empty")
        for value in scales:
            if not math.isfinite(float(value)):
                raise ValueError(f"bn_scales[{name!r}] contains a non-finite value")


def _validate_ratio(prune_ratio: float) -> float:
    ratio = float(prune_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("prune_ratio must be in [0, 1)")
    return ratio


def _validate_lambda(lam: float) -> float:
    lam = float(lam)
    if lam < 0.0 or not math.isfinite(lam):
        raise ValueError("lam must be a finite non-negative number")
    return lam


def slimming_l1_penalty(bn_scales: Dict[str, Sequence[float]], lam: float = 1e-4) -> float:
    """Sparsity penalty ``lam * sum |gamma|`` added to the training loss."""
    _validate_scales(bn_scales)
    lam = _validate_lambda(lam)
    return lam * sum(abs(float(v)) for scales in bn_scales.values() for v in scales)


def slimming_subgradient(
    bn_scales: Dict[str, Sequence[float]],
    lam: float = 1e-4,
) -> Dict[str, List[float]]:
    """Per-scale subgradient ``lam * sign(gamma)`` of the L1 penalty.

    Add it to each batch-norm scale's loss gradient during sparsity
    training (``sign(0) = 0``, the minimum-norm subgradient).
    """
    _validate_scales(bn_scales)
    lam = _validate_lambda(lam)
    out: Dict[str, List[float]] = {}
    for name, scales in bn_scales.items():
        row = []
        for value in scales:
            v = float(value)
            row.append(lam if v > 0.0 else (-lam if v < 0.0 else 0.0))
        out[name] = row
    return out


def _ranked_channels(bn_scales: Dict[str, Sequence[float]]) -> List[Tuple[float, str, int]]:
    """All channels ordered smallest |gamma| first (ties: layer name, index)."""
    entries = [
        (abs(float(value)), name, idx)
        for name, scales in bn_scales.items()
        for idx, value in enumerate(scales)
    ]
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return entries


def slimming_threshold(
    bn_scales: Dict[str, Sequence[float]],
    prune_ratio: float = 0.5,
) -> Optional[float]:
    """Global ``|gamma|`` cut-off for removing ``prune_ratio`` of all channels.

    Returns the ``|gamma|`` of the ``round(prune_ratio * total)``-th
    smallest channel, i.e. the largest scale that gets pruned (the
    ``prune_ratio`` quantile used by Liu et al.). ``None`` when the ratio
    rounds to zero channels.
    """
    _validate_scales(bn_scales)
    ratio = _validate_ratio(prune_ratio)
    ranked = _ranked_channels(bn_scales)
    n_prune = int(round(ratio * len(ranked)))
    if n_prune <= 0:
        return None
    return ranked[n_prune - 1][0]


def slimming_channel_masks(
    bn_scales: Dict[str, Sequence[float]],
    prune_ratio: float = 0.5,
    *,
    min_channels: int = 1,
) -> Dict[str, List[int]]:
    """Global keep masks (1 = keep) over every layer's channels.

    The ``round(prune_ratio * total)`` channels with the smallest
    ``|gamma|`` across *all* layers are pruned (ties broken by layer name
    then channel index, so results are deterministic). Each layer keeps at
    least ``min_channels`` channels (its largest-``|gamma|`` ones) so no
    layer is cut entirely; protected channels are skipped and the
    next-smallest channels elsewhere are pruned instead, so the requested
    count is met whenever enough unprotected channels exist.
    """
    _validate_scales(bn_scales)
    ratio = _validate_ratio(prune_ratio)
    if isinstance(min_channels, bool) or not isinstance(min_channels, int) or min_channels < 0:
        raise ValueError("min_channels must be a non-negative integer")
    masks = {name: [1] * len(scales) for name, scales in bn_scales.items()}
    ranked = _ranked_channels(bn_scales)
    n_prune = int(round(ratio * len(ranked)))
    protected: Dict[str, set] = {}
    for name, scales in bn_scales.items():
        order = sorted(range(len(scales)), key=lambda i: (-abs(float(scales[i])), i))
        protected[name] = set(order[: min(min_channels, len(scales))])
    pruned = 0
    for _, name, idx in ranked:
        if pruned >= n_prune:
            break
        if idx in protected[name]:
            continue
        masks[name][idx] = 0
        pruned += 1
    return masks


def _next_layer_map(specs: Sequence[LayerSpec]) -> Dict[str, LayerSpec]:
    return {specs[i].name: specs[i + 1] for i in range(len(specs) - 1)}


def _slimming_detailed(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    bn_scales: Dict[str, Sequence[float]],
    prune_ratio: float,
    min_channels: int,
    prune_next_inputs: bool,
):
    spec_by_name = {spec.name: spec for spec in specs}
    for name in bn_scales:
        if name not in spec_by_name:
            raise ValueError(f"bn_scales references unknown layer {name!r}")
        out_channels = int(spec_by_name[name].shape[0])
        if len(bn_scales[name]) != out_channels:
            raise ValueError(
                f"bn_scales[{name!r}] has {len(bn_scales[name])} values, "
                f"layer has {out_channels} output channels"
            )
        if len(spec_by_name[name].shape) < 2:
            raise ValueError(f"layer {name!r} needs at least 2 dimensions")
    for spec in specs:
        if spec.name not in model:
            raise ValueError(f"model is missing layer {spec.name!r}")
        expected = layer_weight_count(spec)
        if len(model[spec.name]) != expected:
            raise ValueError(
                f"layer {spec.name!r} has {len(model[spec.name])} weights, expected {expected}"
            )
    masks = slimming_channel_masks(bn_scales, prune_ratio, min_channels=min_channels)
    threshold = slimming_threshold(bn_scales, prune_ratio)
    pruned = {spec.name: _copy_layer(model[spec.name]) for spec in specs}
    input_masks: Dict[str, List[int]] = {}
    nxt = _next_layer_map(specs)
    for name, mask in masks.items():
        spec = spec_by_name[name]
        groups = channel_groups(spec.shape, structure="filter")
        for ch, keep in enumerate(mask):
            if not keep:
                for index in groups[ch]:
                    pruned[name][index] = 0.0
        if prune_next_inputs and name in nxt:
            follower = nxt[name]
            if len(follower.shape) >= 2 and int(follower.shape[1]) == len(mask):
                in_groups = channel_groups(follower.shape, structure="channel")
                for ch, keep in enumerate(mask):
                    if not keep:
                        for index in in_groups[ch]:
                            pruned[follower.name][index] = 0.0
                input_masks[follower.name] = list(mask)
    return pruned, masks, input_masks, threshold


def slimming_prune_model(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    bn_scales: Dict[str, Sequence[float]],
    *,
    prune_ratio: float = 0.5,
    min_channels: int = 1,
    prune_next_inputs: bool = False,
) -> Dict[str, List[float]]:
    """Zero the output filters whose BN scale falls below the global threshold.

    ``bn_scales`` maps a layer name to one ``gamma`` per output channel
    (``shape[0]``); layers without scales are copied unchanged. With
    ``prune_next_inputs=True`` the matching input channels of the next
    layer in ``specs`` are zeroed too, when its ``shape[1]`` equals the
    pruned layer's channel count (a dense layer fed by a flattened conv
    with spatial size > 1 is left untouched).
    """
    pruned, _, _, _ = _slimming_detailed(
        specs, model, bn_scales, prune_ratio, min_channels, prune_next_inputs
    )
    return pruned


def slimming_prune_summary(
    specs: Sequence[LayerSpec],
    model: Dict[str, Sequence[float]],
    bn_scales: Dict[str, Sequence[float]],
    *,
    prune_ratio: float = 0.5,
    min_channels: int = 1,
    prune_next_inputs: bool = False,
) -> Dict[str, object]:
    """Run Network Slimming and report per-layer channel and weight survival."""
    pruned, masks, input_masks, threshold = _slimming_detailed(
        specs, model, bn_scales, prune_ratio, min_channels, prune_next_inputs
    )
    rows: List[Dict[str, object]] = []
    for spec in specs:
        weights = pruned[spec.name]
        total = len(weights)
        nonzero = sum(1 for v in weights if v != 0.0)
        mask = masks.get(spec.name)
        rows.append({
            "layer": spec.name,
            "kind": spec.kind,
            "channels": len(mask) if mask is not None else None,
            "kept_channels": sum(mask) if mask is not None else None,
            "inputs_pruned": spec.name in input_masks,
            "total_weights": total,
            "nonzero_weights": nonzero,
            "survival_fraction": round(nonzero / total, 4) if total else 0.0,
        })
    total_channels = sum(len(m) for m in masks.values())
    kept_channels = sum(sum(m) for m in masks.values())
    return {
        "prune_ratio": float(prune_ratio),
        "threshold": threshold,
        "min_channels": min_channels,
        "prune_next_inputs": bool(prune_next_inputs),
        "total_channels": total_channels,
        "kept_channels": kept_channels,
        "channel_survival": round(kept_channels / total_channels, 4) if total_channels else 0.0,
        "layers": rows,
        "masks": masks,
        "pruned_weights": pruned,
    }
