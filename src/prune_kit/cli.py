"""Command-line interface for prune_kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .layers import LayerSpec, conv_layer, dense_layer
from .masks import dense_mask, mask_density, sparse_mask_to_dense
from .prune import magnitude_prune_model, total_pruned
from .survival import model_survival_summary, per_layer_survival


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "survival":
        return cmd_survival(args)
    if args.command == "mask":
        return cmd_mask(args)
    parser.error(f"unknown command: {args.command}")
    return 2