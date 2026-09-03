"""Survival reporting: per-layer weight survival after a pruning pass."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .layers import LayerSpec, layer_weight_count
from .prune import magnitude_prune_model


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


__all__ = ["per_layer_survival", "model_survival_summary"]