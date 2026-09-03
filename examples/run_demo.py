"""Demo: magnitude-prune a tiny two-layer model and report survival."""

from __future__ import annotations

from pathlib import Path

from prune_kit import (
    conv_layer,
    dense_layer,
    model_survival_summary,
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


if __name__ == "__main__":
    main()