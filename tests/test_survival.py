"""Tests for the survival module and the CLI."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    magnitude_prune_model,
    model_survival_summary,
    per_layer_survival,
)
from prune_kit.cli import main


def _toy_model():
    specs = [
        dense_layer("fc1", in_features=10, out_features=4),
        dense_layer("fc2", in_features=4, out_features=2),
    ]
    weights = {
        "fc1": [0.01 * i for i in range(40)],
        "fc2": [0.01 * i for i in range(8)],
    }
    return specs, weights


def test_per_layer_survival_basic() -> None:
    specs, weights = _toy_model()
    pruned = magnitude_prune_model(weights, density=0.5)
    rows = per_layer_survival(specs, weights, pruned)
    assert [row["layer"] for row in rows] == ["fc1", "fc2"]
    assert rows[0]["kind"] == "dense"
    assert rows[0]["total_weights"] == 40
    assert 0 <= rows[0]["kept_weights"] <= 40
    assert rows[0]["nonzero_in_original"] == 39  # 0.01 * 0 == 0


def test_per_layer_survival_rejects_mismatched_layers() -> None:
    specs, weights = _toy_model()
    pruned = magnitude_prune_model(weights, density=0.5)
    with pytest.raises(ValueError, match="same layers"):
        per_layer_survival(specs[:1], weights, pruned)


def test_per_layer_survival_rejects_length_mismatch() -> None:
    from prune_kit.layers import LayerSpec
    specs, weights = _toy_model()
    pruned = magnitude_prune_model(weights, density=0.5)
    bad_specs = [
        dense_layer("fc1", in_features=10, out_features=4),
        LayerSpec(name="fc2", shape=(3,)),
    ]
    with pytest.raises(ValueError, match="expected"):
        per_layer_survival(bad_specs, weights, pruned)


def test_model_survival_summary_returns_expected_keys() -> None:
    specs, weights = _toy_model()
    summary = model_survival_summary(specs, weights, density=0.5)
    assert set(summary) == {
        "density", "per_layer_density", "layers", "total", "total_kept",
        "overall_survival", "pruned_weights",
    }
    assert summary["total"] == 48
    assert summary["overall_survival"] >= 0.4
    assert summary["overall_survival"] <= 0.6


def test_model_survival_summary_respects_per_layer_density() -> None:
    specs, weights = _toy_model()
    summary = model_survival_summary(specs, weights, density=0.5, per_layer={"fc1": 0.9})
    # Per-layer overrides are reported in the summary.
    assert summary["per_layer_density"] == {"fc1": 0.9}


def test_model_survival_summary_handles_gaussian_like_zero_weights() -> None:
    from prune_kit.layers import LayerSpec
    specs = [dense_layer("fc1", in_features=4, out_features=2)]
    # Half of the original weights are already zero.
    weights = {"fc1": [0.0, 0.1, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7]}
    pruned = magnitude_prune_model(weights, density=0.5)
    summary = model_survival_summary(specs, weights, density=0.5)
    row = summary["layers"][0]
    assert row["nonzero_in_original"] == 4
    # The kept set is the top-4 of {0.1, 0.3, 0.5, 0.7} -> all 4.
    assert row["kept_weights"] == 4


def test_cli_survival_writes_markdown(tmp_path: Path) -> None:
    specs_str = "fc1=dense:10x4,fc2=dense:4x2"
    flat = ",".join(f"{0.01 * i:.3f}" for i in range(48))
    out = tmp_path / "report.md"
    rc = main([
        "survival",
        "--specs", specs_str,
        "--weights", flat,
        "--density", "0.5",
        "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Per-layer pruning survival" in text
    assert "fc1" in text
    assert "fc2" in text


def test_cli_survival_json_output() -> None:
    specs_str = "fc1=dense:10x4,fc2=dense:4x2"
    flat = ",".join(f"{0.01 * i:.3f}" for i in range(48))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "survival",
            "--specs", specs_str,
            "--weights", flat,
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "layers" in payload
    assert payload["total"] == 48


def test_cli_survival_rejects_bad_spec() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "survival",
                "--specs", "fc1=dense:10x4",
                "--weights", "0.1,0.2",
                "--density", "0.5",
            ])
    assert rc == 2


def test_cli_mask_prints_stats() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["mask", "--shape", "3,4", "--density", "0.5", "--seed", "7"])
    assert rc == 0
    assert "shape: (3, 4)" in buf.getvalue()
    assert "kept: 6/12" in buf.getvalue()