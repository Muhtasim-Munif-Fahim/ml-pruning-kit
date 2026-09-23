"""Demo: magnitude, structured, Taylor, and global unstructured pruning."""

from __future__ import annotations

from pathlib import Path

from prune_kit import (
    conv_layer,
    dense_layer,
    global_magnitude_prune_model,
    iterative_global_magnitude_prune_model,
    model_channel_survival_summary,
    model_survival_summary,
    taylor_prune_summary,
)


def main() -> None:
    specs = [
        dense_layer("fc1", in_features=784, out_features=128),
        dense_layer("fc2", in_features=128, out_features=10),
        conv_layer("conv1", out_channels=8, in_channels=1, kernel_h=3, kernel_w=3),
    ]
    weights = {}
    cursor = 1
    for spec in specs:
        size = 1
        for dim in spec.shape:
            size *= int(dim)
        weights[spec.name] = [0.01 * cursor + 0.001 * i for i in range(size)]
        cursor += 1

    summary = model_survival_summary(specs, weights, density=0.5)
    out = Path("examples/output/survival.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo: survival report",
        "",
        f"- Density: {summary['density']:.2f}",
        f"- Overall survival: {summary['overall_survival']:.4f}",
        "",
        "| Layer | Total | Kept |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["layers"]:
        lines.append(f"| {row['layer']} | {row['total_weights']} | {row['kept_weights']} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for row in summary["layers"]:
        print(
            f"  {row['layer']:<8} total={row['total_weights']} kept={row['kept_weights']} "
            f"survival={row['survival_fraction']:.4f}"
        )

    channel = model_channel_survival_summary(
        specs, weights, density=0.5, norm="l1", structure="filter"
    )
    channel_out = Path("examples/output/channel_survival.md")
    clines = [
        "# Demo: channel / filter survival",
        "",
        f"- Structure: {channel['structure']}",
        f"- Norm: {channel['norm']}",
        f"- Overall survival: {channel['overall_survival']:.4f}",
        "",
        "| Layer | Total | Kept | Indices |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in channel["layers"]:
        kept = ",".join(str(index) for index in row["kept_indices"])
        clines.append(
            f"| {row['layer']} | {row['total_channels']} | {row['kept_channels']} | {kept} |"
        )
    channel_out.write_text("\n".join(clines) + "\n", encoding="utf-8")
    print(f"Wrote {channel_out}")
    for row in channel["layers"]:
        print(
            f"  {row['layer']:<8} channels={row['total_channels']} "
            f"kept={row['kept_channels']} indices={row['kept_indices']}"
        )

    taylor_grads = {}
    for spec in specs:
        size = 1
        for dim in spec.shape:
            size *= int(dim)
        if spec.kind == "conv":
            taylor_grads[spec.name] = [0.001 * (i + 1) for i in range(size)]
        else:
            taylor_grads[spec.name] = [0.0] * size
    taylor = taylor_prune_summary(
        specs, weights, taylor_grads, density=0.5, structure="filter"
    )
    taylor_out = Path("examples/output/taylor.md")
    tlines = [
        "# Demo: first-order Taylor channel pruning",
        "",
        f"- Criterion: {taylor['criterion']}",
        f"- Overall survival: {taylor['overall_survival']:.4f}",
        "",
        "| Layer | Total | Kept | Indices |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in taylor["layers"]:
        kept = ",".join(str(index) for index in row["kept_indices"])
        tlines.append(
            f"| {row['layer']} | {row['total_channels']} | {row['kept_channels']} | {kept} |"
        )
    taylor_out.write_text("\n".join(tlines) + "\n", encoding="utf-8")
    print(f"Wrote {taylor_out}")
    for row in taylor["layers"]:
        print(
            f"  {row['layer']:<8} channels={row['total_channels']} "
            f"kept={row['kept_channels']} indices={row['kept_indices']}"
        )

    one_shot = global_magnitude_prune_model(weights, sparsity=0.5)
    iterative = iterative_global_magnitude_prune_model(
        weights, sparsity=0.5, rounds=2
    )
    global_out = Path("examples/output/global.md")
    glines = [
        "# Demo: global unstructured magnitude pruning",
        "",
        f"- Sparsity: {iterative.sparsity:.2f}",
        f"- Rounds: {iterative.rounds}",
        f"- Final density: {iterative.final_density():.4f}",
        "",
        "| Layer | Kept | Total |",
        "| --- | ---: | ---: |",
    ]
    for spec in specs:
        layer = one_shot[spec.name]
        kept = sum(1 for value in layer if value != 0.0)
        glines.append(f"| {spec.name} | {kept} | {len(layer)} |")
    global_out.write_text("\n".join(glines) + "\n", encoding="utf-8")
    print(f"Wrote {global_out}")
    print(f"  schedule={iterative.schedule} density={iterative.density_curve()}")


if __name__ == "__main__":
    main()