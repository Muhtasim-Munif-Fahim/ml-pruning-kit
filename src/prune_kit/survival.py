"""Survival reporting: per-layer and per-channel stats after a prune pass."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .layers import LayerSpec, layer_weight_count
from .prune import magnitude_prune_model
from .structured import channel_groups, channel_norms, structured_prune_model


def per_layer_survival(
    specs: Sequence[LayerSpec],
    original: Dict[str, Sequence[float]],
    pruned: Dict[str, Sequence[float]],
) -> List[Dict[str, object]]:
    """Per-layer survival report after a pruning pass.

    Each row carries the layer name, the original weight count, the
    kept-weight count, the survival fraction (kept / original), and the
    number of weights that were exactly zero in the original model
    (which are unaffected by magnitude pruning). Rows are sorted by
    layer name so the report is reproducible.
    """
    if set(spec.name for spec in specs) != set(original.keys()):
        raise ValueError("specs and original must reference the same layers")
    if set(original.keys()) != set(pruned.keys()):
        raise ValueError("original and pruned must reference the same layers")
    rows: List[Dict[str, object]] = []
    for spec in sorted(specs, key=lambda s: s.name):
        name = spec.name
        total = layer_weight_count(spec)
        original_weights = list(original[name])
        pruned_weights = list(pruned[name])
        if len(original_weights) != total:
            raise ValueError(
                f"layer {name!r} has {len(original_weights)} weights, expected {total}"
            )
        if len(pruned_weights) != total:
            raise ValueError(
                f"layer {name!r} pruned buffer has {len(pruned_weights)} weights, expected {total}"
            )
        kept = sum(1 for value in pruned_weights if value != 0.0)
        nonzero_original = sum(1 for value in original_weights if value != 0.0)
        rows.append({
            "layer": name,
            "kind": spec.kind,
            "total_weights": total,
            "kept_weights": kept,
            "survival_fraction": round(kept / total, 4) if total else 0.0,
            "nonzero_in_original": nonzero_original,
        })
    return rows


def model_survival_summary(
    specs: Sequence[LayerSpec],
    original: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run a magnitude-pruning pass and return both per-layer and aggregate stats.

    Useful for one-shot CLI output: the per-layer rows are returned
    under the ``layers`` key, the aggregate totals under ``total`` /
    ``total_kept`` / ``overall_survival``. The pruned weights are also
    returned so callers can feed them to the next step of a pipeline.
    """
    pruned = magnitude_prune_model(original, density=density, per_layer=per_layer)
    rows = per_layer_survival(specs, original, pruned)
    total = sum(layer_weight_count(spec) for spec in specs)
    total_kept = sum(int(row["kept_weights"]) for row in rows)
    return {
        "density": density,
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "total": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "pruned_weights": pruned,
    }


def per_channel_survival(
    specs: Sequence[LayerSpec],
    original: Dict[str, Sequence[float]],
    pruned: Dict[str, Sequence[float]],
    *,
    structure: str = "filter",
    norm: str = "l1",
) -> List[Dict[str, object]]:
    """Per-channel (or per-filter) survival report after structured pruning.

    Only conv layers are included. Each row lists the layer, how many
    filters/channels survived, the kept/pruned index lists, and a
    per-channel breakdown with L1 and L2 norms computed on the
    **original** weights. A channel is kept if any of its weights are
    non-zero in ``pruned``. Rows are sorted by layer name.
    """
    if set(spec.name for spec in specs) != set(original.keys()):
        raise ValueError("specs and original must reference the same layers")
    if set(original.keys()) != set(pruned.keys()):
        raise ValueError("original and pruned must reference the same layers")
    if str(norm).lower() not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")
    structure = str(structure).lower()
    if structure not in {"filter", "channel"}:
        raise ValueError("structure must be 'filter' or 'channel'")

    rows: List[Dict[str, object]] = []
    conv_specs = sorted(
        (spec for spec in specs if spec.kind == "conv"),
        key=lambda spec: spec.name,
    )
    for spec in conv_specs:
        name = spec.name
        total_weights = layer_weight_count(spec)
        original_weights = list(original[name])
        pruned_weights = list(pruned[name])
        if len(original_weights) != total_weights:
            raise ValueError(
                f"layer {name!r} has {len(original_weights)} weights, expected {total_weights}"
            )
        if len(pruned_weights) != total_weights:
            raise ValueError(
                f"layer {name!r} pruned buffer has {len(pruned_weights)} weights, "
                f"expected {total_weights}"
            )
        groups = channel_groups(spec.shape, structure=structure)
        l1_norms = channel_norms(
            original_weights, spec.shape, norm="l1", structure=structure
        )
        l2_norms = channel_norms(
            original_weights, spec.shape, norm="l2", structure=structure
        )
        ranking_norms = l1_norms if str(norm).lower() == "l1" else l2_norms
        channels: List[Dict[str, object]] = []
        kept_indices: List[int] = []
        pruned_indices: List[int] = []
        for index, group in enumerate(groups):
            kept = any(float(pruned_weights[pos]) != 0.0 for pos in group)
            entry = {
                "index": index,
                "l1": round(l1_norms[index], 6),
                "l2": round(l2_norms[index], 6),
                "norm": round(ranking_norms[index], 6),
                "kept": kept,
            }
            channels.append(entry)
            if kept:
                kept_indices.append(index)
            else:
                pruned_indices.append(index)
        total_channels = len(groups)
        kept_channels = len(kept_indices)
        rows.append({
            "layer": name,
            "kind": spec.kind,
            "structure": structure,
            "norm": str(norm).lower(),
            "total_channels": total_channels,
            "kept_channels": kept_channels,
            "survival_fraction": (
                round(kept_channels / total_channels, 4) if total_channels else 0.0
            ),
            "kept_indices": kept_indices,
            "pruned_indices": pruned_indices,
            "channels": channels,
        })
    return rows


def model_channel_survival_summary(
    specs: Sequence[LayerSpec],
    original: Dict[str, Sequence[float]],
    *,
    density: float = 0.5,
    norm: str = "l1",
    structure: str = "filter",
    per_layer: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run structured pruning and return per-channel plus aggregate stats.

    Non-conv layers are copied through and listed under ``skipped``.
    The pruned weights are returned so callers can feed them to the
    next step of a pipeline.
    """
    pruned = structured_prune_model(
        specs,
        original,
        density=density,
        norm=norm,
        structure=structure,
        per_layer=per_layer,
    )
    rows = per_channel_survival(
        specs, original, pruned, structure=structure, norm=norm
    )
    total = sum(int(row["total_channels"]) for row in rows)
    total_kept = sum(int(row["kept_channels"]) for row in rows)
    skipped = [
        spec.name for spec in sorted(specs, key=lambda item: item.name)
        if spec.kind != "conv"
    ]
    return {
        "density": density,
        "norm": str(norm).lower(),
        "structure": str(structure).lower(),
        "per_layer_density": dict(per_layer) if per_layer else {},
        "layers": rows,
        "skipped": skipped,
        "total_channels": total,
        "total_kept": total_kept,
        "overall_survival": round(total_kept / total, 4) if total else 0.0,
        "pruned_weights": pruned,
    }


__all__ = [
    "per_layer_survival",
    "model_survival_summary",
    "per_channel_survival",
    "model_channel_survival_summary",
]