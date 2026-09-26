"""Command-line interface for prune_kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .global_unstructured import iterative_global_magnitude_prune_model
from .layers import LayerSpec, conv_layer, dense_layer
from .masks import dense_mask, mask_density, sparse_mask_to_dense
from .prune import iterative_magnitude_prune_model
from .snip import snip_prune_summary
from .grasp import grasp_prune_summary
from .survival import model_channel_survival_summary, model_survival_summary
from .taylor import taylor_prune_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prune-kit")
    sub = parser.add_subparsers(dest="command", required=True)

    survival = sub.add_parser(
        "survival",
        help="Magnitude-prune a model and report per-layer weight survival",
    )
    survival.add_argument(
        "--density", type=float, default=0.5,
        help="Global density to keep (default: 0.5)",
    )
    survival.add_argument(
        "--per-layer-density", default=None,
        help="Comma-separated name=density overrides, e.g. fc1=0.9,fc2=0.1",
    )
    survival.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. 'fc1=dense:784x256,fc2=dense:256x10'. "
            "The first N weights in --weights are assigned to the first spec in order."
        ),
    )
    survival.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (the flat weight buffer for the model)",
    )
    survival.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    survival.add_argument(
        "--json", action="store_true",
        help="Print the summary as JSON",
    )

    masks = sub.add_parser(
        "mask",
        help="Generate a 0/1 mask of the requested shape and density",
    )
    masks.add_argument("--shape", required=True, help="Comma-separated dimensions, e.g. 3,4,5")
    masks.add_argument("--density", type=float, required=True, help="Fraction of 1s")
    masks.add_argument("--seed", type=int, default=None)
    masks.add_argument("--json", action="store_true", help="Print mask stats as JSON")

    imp = sub.add_parser(
        "imp",
        help="Iterative magnitude pruning with optional lottery-ticket rewind",
    )
    imp.add_argument(
        "--prune-fraction", type=float, default=0.2,
        help="Fraction of currently surviving weights to zero each round (default: 0.2)",
    )
    imp.add_argument(
        "--rounds", type=int, default=1,
        help="Number of prune rounds (default: 1)",
    )
    imp.add_argument(
        "--rewind", action="store_true",
        help="Reset surviving weights to --initial-weights after each prune round",
    )
    imp.add_argument(
        "--per-layer-fraction", default=None,
        help="Comma-separated name=fraction overrides, e.g. fc1=0.1,fc2=0.3",
    )
    imp.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. 'fc1=dense:784x256,fc2=dense:256x10'. "
            "The first N weights in --weights are assigned to the first spec in order."
        ),
    )
    imp.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (typically trained weights)",
    )
    imp.add_argument(
        "--initial-weights", default=None,
        help=(
            "Comma-separated floats used as the lottery-ticket rewind target. "
            "Defaults to --weights when --rewind is set."
        ),
    )
    imp.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    imp.add_argument(
        "--json", action="store_true",
        help="Print the result as JSON",
    )

    structured = sub.add_parser(
        "structured",
        help="Prune whole conv filters or channels by L1/L2 norm",
    )
    structured.add_argument(
        "--density", type=float, default=0.5,
        help="Fraction of filters/channels to keep (default: 0.5)",
    )
    structured.add_argument(
        "--norm", choices=("l1", "l2"), default="l1",
        help="Ranking norm over each filter or channel (default: l1)",
    )
    structured.add_argument(
        "--structure", choices=("filter", "channel"), default="filter",
        help=(
            "filter: prune output filters (dim 0); "
            "channel: prune input channels (dim 1). Default: filter"
        ),
    )
    structured.add_argument(
        "--per-layer-density", default=None,
        help="Comma-separated name=density overrides, e.g. conv1=0.75,conv2=0.25",
    )
    structured.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. "
            "'conv1=conv:8x3x3x3,fc1=dense:256x10'. "
            "Non-conv layers are copied unchanged."
        ),
    )
    structured.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (the flat weight buffer for the model)",
    )
    structured.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    structured.add_argument(
        "--json", action="store_true",
        help="Print the summary as JSON",
    )

    taylor = sub.add_parser(
        "taylor",
        help="Prune conv filters or channels by first-order Taylor saliency",
    )
    taylor.add_argument(
        "--density", type=float, default=0.5,
        help="Fraction of filters/channels to keep (default: 0.5)",
    )
    taylor.add_argument(
        "--structure", choices=("filter", "channel"), default="filter",
        help=(
            "filter: prune output filters (dim 0); "
            "channel: prune input channels (dim 1). Default: filter"
        ),
    )
    taylor.add_argument(
        "--criterion", choices=("abs", "sq"), default="abs",
        help=(
            "abs: sum |grad * weight|; sq: sum (grad * weight)^2. Default: abs"
        ),
    )
    taylor.add_argument(
        "--reduction", choices=("sum", "mean"), default="sum",
        help="Reduce per-weight terms by sum or mean (default: sum)",
    )
    taylor.add_argument(
        "--per-layer-density", default=None,
        help="Comma-separated name=density overrides, e.g. conv1=0.75,conv2=0.25",
    )
    taylor.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. "
            "'conv1=conv:8x3x3x3,fc1=dense:256x10'. "
            "Non-conv layers are copied unchanged."
        ),
    )
    taylor.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (the flat weight buffer for the model)",
    )
    taylor.add_argument(
        "--grads", required=True,
        help=(
            "Comma-separated floats aligned with --weights. "
            "Scores use |grad * weight| (or the squared criterion)."
        ),
    )
    taylor.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    taylor.add_argument(
        "--json", action="store_true",
        help="Print the summary as JSON",
    )

    snip = sub.add_parser(
        "snip",
        help="One-shot SNIP prune by connection sensitivity |weight * grad|",
    )
    snip.add_argument(
        "--density", type=float, default=0.5,
        help="Fraction of connections to keep (default: 0.5)",
    )
    snip.add_argument(
        "--scope", choices=("global", "layer"), default="global",
        help=(
            "global: rank every connection together (the SNIP paper); "
            "layer: keep --density inside each layer. Default: global"
        ),
    )
    snip.add_argument(
        "--per-layer-density", default=None,
        help=(
            "Comma-separated name=density overrides, e.g. fc1=0.9,conv1=0.25. "
            "Requires --scope layer."
        ),
    )
    snip.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. "
            "'fc1=dense:8x4,conv1=conv:4x1x3x3'. Dense and conv layers are both pruned."
        ),
    )
    snip.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (the flat weight buffer for the model)",
    )
    snip.add_argument(
        "--grads", required=True,
        help=(
            "Comma-separated floats aligned with --weights. "
            "Scores are |grad * weight|."
        ),
    )
    snip.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    snip.add_argument(
        "--json", action="store_true",
        help="Print the summary as JSON",
    )

    grasp = sub.add_parser(
        "grasp",
        help="One-shot GraSP prune by gradient-signal scores -w*(Hg)",
    )
    grasp.add_argument(
        "--density", type=float, default=0.5,
        help="Fraction of connections to keep (default: 0.5)",
    )
    grasp.add_argument(
        "--scope", choices=("global", "layer"), default="global",
        help=(
            "global: rank every connection together (the GraSP paper); "
            "layer: keep --density inside each layer. Default: global"
        ),
    )
    grasp.add_argument(
        "--per-layer-density", default=None,
        help=(
            "Comma-separated name=density overrides, e.g. fc1=0.9,conv1=0.25. "
            "Requires --scope layer."
        ),
    )
    grasp.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. "
            "'fc1=dense:8x4,conv1=conv:4x1x3x3'. Dense and conv layers are both pruned."
        ),
    )
    grasp.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (the flat weight buffer for the model)",
    )
    grasp.add_argument(
        "--hg", required=True,
        help=(
            "Comma-separated Hessian-vector products Hg aligned with --weights. "
            "Scores are -weight * hg."
        ),
    )
    grasp.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    grasp.add_argument(
        "--json", action="store_true",
        help="Print the summary as JSON",
    )

    global_cmd = sub.add_parser(
        "global",
        help="Global unstructured magnitude pruning to a target sparsity",
    )
    global_cmd.add_argument(
        "--sparsity", type=float, default=None,
        help=(
            "Fraction of all weights to prune (default: 0.5). "
            "With --schedule, must match the final entry."
        ),
    )
    global_cmd.add_argument(
        "--rounds", type=int, default=1,
        help="Split the prune evenly across this many rounds (default: 1)",
    )
    global_cmd.add_argument(
        "--schedule", default=None,
        help=(
            "Comma-separated cumulative sparsities, one per round, "
            "e.g. 0.25,0.5,0.9. Replaces the even split of --sparsity."
        ),
    )
    global_cmd.add_argument(
        "--rewind", action="store_true",
        help="Reset surviving weights to --initial-weights after each round",
    )
    global_cmd.add_argument(
        "--specs", required=True,
        help=(
            "Comma-separated layer specs, e.g. 'fc1=dense:784x256,fc2=dense:256x10'. "
            "The first N weights in --weights are assigned to the first spec in order."
        ),
    )
    global_cmd.add_argument(
        "--weights", required=True,
        help="Comma-separated floats (typically trained weights)",
    )
    global_cmd.add_argument(
        "--initial-weights", default=None,
        help=(
            "Comma-separated floats used as the rewind target. "
            "Defaults to --weights when --rewind is set."
        ),
    )
    global_cmd.add_argument(
        "--output", "-o", default=None,
        help="Write the Markdown report to a file instead of stdout",
    )
    global_cmd.add_argument(
        "--json", action="store_true",
        help="Print the result as JSON",
    )

    return parser


def _parse_specs(spec: str) -> list:
    """Parse 'name=dense:8x4' style spec strings into LayerSpec objects."""
    parsed: list[LayerSpec] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"spec entry must be 'name=kind:...': {piece!r}")
        name, kind_shape = piece.split("=", 1)
        if ":" not in kind_shape:
            raise ValueError(f"spec entry must be 'name=kind:...': {piece!r}")
        kind, shape = kind_shape.split(":", 1)
        dims = [int(dim) for dim in shape.split("x")]
        if kind.lower() == "dense":
            if len(dims) != 2:
                raise ValueError(f"dense spec expects 2 dims, got {dims}")
            parsed.append(dense_layer(name.strip(), dims[0], dims[1]))
        elif kind.lower() == "conv":
            if len(dims) != 4:
                raise ValueError(f"conv spec expects 4 dims, got {dims}")
            parsed.append(conv_layer(name.strip(), dims[0], dims[1], dims[2], dims[3]))
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")
    if not parsed:
        raise ValueError("at least one spec is required")
    return parsed


def _parse_per_layer(raw: str | None) -> dict:
    if not raw:
        return {}
    result: dict = {}
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"per-layer override must be 'name=density': {piece!r}")
        name, value = piece.split("=", 1)
        result[name.strip()] = float(value)
    return result


def _parse_weights(raw: str) -> list:
    return [float(item) for item in raw.split(",") if item.strip()]


def _parse_schedule(raw: str | None) -> list | None:
    if not raw:
        return None
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("schedule must not be empty")
    return values


def _weights_per_layer(specs, flat_weights: list) -> dict:
    result: dict = {}
    cursor = 0
    for spec in specs:
        size = 1
        for dim in spec.shape:
            size *= int(dim)
        result[spec.name] = flat_weights[cursor:cursor + size]
        cursor += size
    if cursor != len(flat_weights):
        raise ValueError(
            f"got {len(flat_weights)} weights but spec wants {cursor} (per layer: {[(s.name, 1) for s in specs]})"
        )
    return result


def _render_survival_markdown(summary: dict) -> str:
    lines: list[str] = [
        "# Per-layer pruning survival",
        "",
        f"- Density: {summary['density']:.4f}",
        f"- Total weights: {summary['total']}",
        f"- Total kept: {summary['total_kept']}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
        "",
        "| Layer | Kind | Total | Kept | Survival |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["layers"]:
        lines.append(
            f"| {row['layer']} | {row['kind']} | {row['total_weights']} | "
            f"{row['kept_weights']} | {row['survival_fraction']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _render_channel_markdown(summary: dict) -> str:
    lines: list[str] = [
        "# Channel / filter pruning survival",
        "",
        f"- Structure: {summary['structure']}",
        f"- Norm: {summary['norm']}",
        f"- Density: {summary['density']:.4f}",
        f"- Total channels: {summary['total_channels']}",
        f"- Total kept: {summary['total_kept']}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
    ]
    if summary["skipped"]:
        skipped = ", ".join(summary["skipped"])
        lines.append(f"- Skipped (non-conv): {skipped}")
    lines.extend([
        "",
        "| Layer | Kind | Structure | Total | Kept | Survival | Kept indices |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for row in summary["layers"]:
        kept = ",".join(str(index) for index in row["kept_indices"]) or "—"
        lines.append(
            f"| {row['layer']} | {row['kind']} | {row['structure']} | "
            f"{row['total_channels']} | {row['kept_channels']} | "
            f"{row['survival_fraction']:.4f} | {kept} |"
        )
    for row in summary["layers"]:
        unit = "filters" if row["structure"] == "filter" else "channels"
        lines.extend([
            "",
            f"## {row['layer']} ({row['kind']}, {unit})",
            "",
            "| Index | L1 | L2 | Kept |",
            "| ---: | ---: | ---: | --- |",
        ])
        for channel in row["channels"]:
            kept_label = "yes" if channel["kept"] else "no"
            lines.append(
                f"| {channel['index']} | {channel['l1']:.6f} | "
                f"{channel['l2']:.6f} | {kept_label} |"
            )
    return "\n".join(lines) + "\n"


def _render_taylor_markdown(summary: dict) -> str:
    lines: list[str] = [
        "# First-order Taylor channel pruning",
        "",
        f"- Structure: {summary['structure']}",
        f"- Criterion: {summary['criterion']}",
        f"- Reduction: {summary['reduction']}",
        f"- Density: {summary['density']:.4f}",
        f"- Total channels: {summary['total_channels']}",
        f"- Total kept: {summary['total_kept']}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
    ]
    if summary["skipped"]:
        skipped = ", ".join(summary["skipped"])
        lines.append(f"- Skipped (non-conv): {skipped}")
    lines.extend([
        "",
        "| Layer | Kind | Structure | Total | Kept | Survival | Kept indices |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for row in summary["layers"]:
        kept = ",".join(str(index) for index in row["kept_indices"]) or "—"
        lines.append(
            f"| {row['layer']} | {row['kind']} | {row['structure']} | "
            f"{row['total_channels']} | {row['kept_channels']} | "
            f"{row['survival_fraction']:.4f} | {kept} |"
        )
    for row in summary["layers"]:
        unit = "filters" if row["structure"] == "filter" else "channels"
        lines.extend([
            "",
            f"## {row['layer']} ({row['kind']}, {unit})",
            "",
            "| Index | Score | Kept |",
            "| ---: | ---: | --- |",
        ])
        for channel in row["channels"]:
            kept_label = "yes" if channel["kept"] else "no"
            lines.append(
                f"| {channel['index']} | {channel['score']:.6f} | {kept_label} |"
            )
    return "\n".join(lines) + "\n"


def _render_imp_markdown(result) -> str:
    rewind_label = "yes" if result.rewind else "no"
    lines: list[str] = [
        "# Iterative magnitude pruning",
        "",
        f"- Prune fraction: {result.prune_fraction:.4f}",
        f"- Rounds: {result.rounds}",
        f"- Rewind: {rewind_label}",
        f"- Final density: {result.final_density():.4f}",
        f"- Final kept: {result.steps[-1].kept}/{result.steps[-1].total}",
        "",
        "| Round | Kept | Total | Density |",
        "| --- | ---: | ---: | ---: |",
    ]
    for step in result.steps:
        lines.append(
            f"| {step.round} | {step.kept} | {step.total} | {step.density:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _render_grasp_markdown(summary: dict) -> str:
    lines: list[str] = [
        "# GraSP gradient-signal pruning",
        "",
        f"- Scope: {summary['scope']}",
        f"- Density: {summary['density']:.4f}",
        f"- Sparsity: {summary['sparsity']:.4f}",
        f"- Total weights: {summary['total']}",
        f"- Total kept: {summary['total_kept']}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
    ]
    if summary["per_layer_density"]:
        overrides = ", ".join(
            f"{name}={float(value):.4f}"
            for name, value in summary["per_layer_density"].items()
        )
        lines.append(f"- Per-layer density: {overrides}")
    lines.extend([
        "",
        "| Layer | Kind | Total | Kept | Survival | Score sum |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["layers"]:
        lines.append(
            f"| {row['layer']} | {row['kind']} | {row['total_weights']} | "
            f"{row['kept_weights']} | {row['survival_fraction']:.4f} | "
            f"{row['score_sum']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _render_snip_markdown(summary: dict) -> str:

    lines: list[str] = [
        "# SNIP connection-sensitivity pruning",
        "",
        f"- Scope: {summary['scope']}",
        f"- Density: {summary['density']:.4f}",
        f"- Sparsity: {summary['sparsity']:.4f}",
        f"- Total weights: {summary['total']}",
        f"- Total kept: {summary['total_kept']}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
    ]
    if summary["per_layer_density"]:
        overrides = ", ".join(
            f"{name}={float(value):.4f}"
            for name, value in summary["per_layer_density"].items()
        )
        lines.append(f"- Per-layer density: {overrides}")
    lines.extend([
        "",
        "| Layer | Kind | Total | Kept | Survival | Score sum |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["layers"]:
        lines.append(
            f"| {row['layer']} | {row['kind']} | {row['total_weights']} | "
            f"{row['kept_weights']} | {row['survival_fraction']:.4f} | "
            f"{row['score_sum']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _render_global_markdown(result) -> str:
    rewind_label = "yes" if result.rewind else "no"
    schedule = ", ".join(f"{value:.4f}" for value in result.schedule)
    lines: list[str] = [
        "# Global unstructured magnitude pruning",
        "",
        f"- Sparsity: {result.sparsity:.4f}",
        f"- Rounds: {result.rounds}",
        f"- Rewind: {rewind_label}",
        f"- Schedule: {schedule}",
        f"- Final density: {result.final_density():.4f}",
        f"- Final kept: {result.steps[-1].kept}/{result.steps[-1].total}",
        "",
        "| Round | Target sparsity | Kept | Total | Density |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for step, target in zip(result.steps, result.schedule):
        lines.append(
            f"| {step.round} | {target:.4f} | {step.kept} | "
            f"{step.total} | {step.density:.4f} |"
        )
    lines.extend([
        "",
        "| Layer | Kept | Total | Density |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, weights in result.weights.items():
        total = len(weights)
        kept = sum(1 for value in weights if float(value) != 0.0)
        density = kept / total if total else 0.0
        lines.append(f"| {name} | {kept} | {total} | {density:.4f} |")
    return "\n".join(lines) + "\n"


def cmd_survival(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        per_layer = _parse_per_layer(args.per_layer_density)
    except ValueError as exc:
        print(f"survival: {exc}", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        summary = model_survival_summary(specs, model, density=args.density, per_layer=per_layer)
    except (KeyError, ValueError) as exc:
        print(f"survival: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = dict(summary)
        printable.pop("pruned_weights", None)
        print(json.dumps(printable, indent=2))
        return 0
    text = _render_survival_markdown(summary)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def cmd_mask(args: argparse.Namespace) -> int:
    shape = tuple(int(dim) for dim in args.shape.split(","))
    mask = dense_mask(shape, args.density, seed=args.seed)
    if args.json:
        print(json.dumps({
            "shape": list(shape),
            "density": args.density,
            "seed": args.seed,
            "mask_density": mask_density(mask),
            "total": len(mask),
            "kept": sum(mask),
        }, indent=2))
        return 0
    print(f"shape: {shape}, density: {args.density}, kept: {sum(mask)}/{len(mask)}")
    return 0


def cmd_imp(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        per_layer = _parse_per_layer(args.per_layer_fraction)
        initial_flat = (
            _parse_weights(args.initial_weights)
            if args.initial_weights is not None
            else None
        )
    except ValueError as exc:
        print(f"imp: {exc}", file=sys.stderr)
        return 2
    if initial_flat is not None and not args.rewind:
        print("imp: --initial-weights requires --rewind", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        initial = (
            _weights_per_layer(specs, initial_flat)
            if initial_flat is not None
            else None
        )
        result = iterative_magnitude_prune_model(
            model,
            prune_fraction=args.prune_fraction,
            rounds=args.rounds,
            rewind=args.rewind,
            initial_weights=initial,
            per_layer=per_layer or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"imp: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = {
            "prune_fraction": result.prune_fraction,
            "rounds": result.rounds,
            "rewind": result.rewind,
            "final_density": result.final_density(),
            "steps": [
                {
                    "round": step.round,
                    "kept": step.kept,
                    "total": step.total,
                    "density": step.density,
                }
                for step in result.steps
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    text = _render_imp_markdown(result)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def cmd_structured(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        per_layer = _parse_per_layer(args.per_layer_density)
    except ValueError as exc:
        print(f"structured: {exc}", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        summary = model_channel_survival_summary(
            specs,
            model,
            density=args.density,
            norm=args.norm,
            structure=args.structure,
            per_layer=per_layer or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"structured: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = dict(summary)
        printable.pop("pruned_weights", None)
        print(json.dumps(printable, indent=2))
        return 0
    text = _render_channel_markdown(summary)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def cmd_taylor(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        grad_flat = _parse_weights(args.grads)
        per_layer = _parse_per_layer(args.per_layer_density)
        if len(grad_flat) != len(flat):
            raise ValueError(
                f"grads has {len(grad_flat)} values, weights has {len(flat)}"
            )
    except ValueError as exc:
        print(f"taylor: {exc}", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        grads = _weights_per_layer(specs, grad_flat)
        summary = taylor_prune_summary(
            specs,
            model,
            grads,
            density=args.density,
            structure=args.structure,
            criterion=args.criterion,
            reduction=args.reduction,
            per_layer=per_layer or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"taylor: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = dict(summary)
        printable.pop("pruned_weights", None)
        print(json.dumps(printable, indent=2))
        return 0
    text = _render_taylor_markdown(summary)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def cmd_snip(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        grad_flat = _parse_weights(args.grads)
        per_layer = _parse_per_layer(args.per_layer_density)
        if len(grad_flat) != len(flat):
            raise ValueError(
                f"grads has {len(grad_flat)} values, weights has {len(flat)}"
            )
    except ValueError as exc:
        print(f"snip: {exc}", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        grads = _weights_per_layer(specs, grad_flat)
        summary = snip_prune_summary(
            specs,
            model,
            grads,
            density=args.density,
            scope=args.scope,
            per_layer=per_layer or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"snip: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = dict(summary)
        printable.pop("pruned_weights", None)
        printable.pop("masks", None)
        print(json.dumps(printable, indent=2))
        return 0
    text = _render_snip_markdown(summary)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0



def cmd_grasp(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        hg_flat = _parse_weights(args.hg)
        per_layer = _parse_per_layer(args.per_layer_density)
        if len(hg_flat) != len(flat):
            raise ValueError(
                f"hg has {len(hg_flat)} values, weights has {len(flat)}"
            )
    except ValueError as exc:
        print(f"grasp: {exc}", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        hg = _weights_per_layer(specs, hg_flat)
        summary = grasp_prune_summary(
            specs,
            model,
            hg,
            density=args.density,
            scope=args.scope,
            per_layer=per_layer or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"grasp: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = dict(summary)
        printable.pop("pruned_weights", None)
        printable.pop("masks", None)
        print(json.dumps(printable, indent=2))
        return 0
    text = _render_grasp_markdown(summary)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def cmd_global(args: argparse.Namespace) -> int:
    try:
        specs = _parse_specs(args.specs)
        flat = _parse_weights(args.weights)
        schedule = _parse_schedule(args.schedule)
        initial_flat = (
            _parse_weights(args.initial_weights)
            if args.initial_weights is not None
            else None
        )
    except ValueError as exc:
        print(f"global: {exc}", file=sys.stderr)
        return 2
    if initial_flat is not None and not args.rewind:
        print("global: --initial-weights requires --rewind", file=sys.stderr)
        return 2
    try:
        model = _weights_per_layer(specs, flat)
        initial = (
            _weights_per_layer(specs, initial_flat)
            if initial_flat is not None
            else None
        )
        result = iterative_global_magnitude_prune_model(
            model,
            sparsity=args.sparsity,
            rounds=args.rounds,
            schedule=schedule,
            rewind=args.rewind,
            initial_weights=initial,
        )
    except (KeyError, ValueError) as exc:
        print(f"global: {exc}", file=sys.stderr)
        return 2
    if args.json:
        layers = []
        for name, weights in result.weights.items():
            total = len(weights)
            kept = sum(1 for value in weights if float(value) != 0.0)
            layers.append({
                "layer": name,
                "kept": kept,
                "total": total,
                "density": kept / total if total else 0.0,
            })
        payload = {
            "sparsity": result.sparsity,
            "rounds": result.rounds,
            "rewind": result.rewind,
            "schedule": result.schedule,
            "final_density": result.final_density(),
            "final_kept": result.steps[-1].kept,
            "final_total": result.steps[-1].total,
            "steps": [
                {
                    "round": step.round,
                    "kept": step.kept,
                    "total": step.total,
                    "density": step.density,
                }
                for step in result.steps
            ],
            "layers": layers,
        }
        print(json.dumps(payload, indent=2))
        return 0
    text = _render_global_markdown(result)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "survival":
        return cmd_survival(args)
    if args.command == "mask":
        return cmd_mask(args)
    if args.command == "imp":
        return cmd_imp(args)
    if args.command == "structured":
        return cmd_structured(args)
    if args.command == "taylor":
        return cmd_taylor(args)
    if args.command == "snip":
        return cmd_snip(args)
    if args.command == "grasp":
        return cmd_grasp(args)
    if args.command == "global":
        return cmd_global(args)
    parser.error(f"unknown command: {args.command}")
    return 2