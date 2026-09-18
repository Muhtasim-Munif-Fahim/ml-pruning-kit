"""Tests for structured channel/filter pruning and channel survival."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from prune_kit import (
    channel_groups,
    channel_norms,
    conv_layer,
    dense_layer,
    model_channel_survival_summary,
    per_channel_survival,
    structured_keep_indices,
    structured_prune_layer,
    structured_prune_model,
    vector_norm,
)
from prune_kit.cli import main


def test_vector_norm_l1_and_l2() -> None:
    values = [3.0, -4.0]
    assert vector_norm(values, norm="l1") == pytest.approx(7.0)
    assert vector_norm(values, norm="l2") == pytest.approx(5.0)


def test_channel_groups_filters_are_contiguous_blocks() -> None:
    # 2 filters, 3 in-channels, 1x1 kernel: each filter is 3 weights.
    groups = channel_groups((2, 3, 1, 1), structure="filter")
    assert groups == [[0, 1, 2], [3, 4, 5]]


def test_channel_groups_input_channels_stride_across_filters() -> None:
    groups = channel_groups((2, 3, 1, 1), structure="channel")
    assert groups == [[0, 3], [1, 4], [2, 5]]


def test_channel_groups_dense_style_2d_shape() -> None:
    groups = channel_groups((2, 4), structure="filter")
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    channel = channel_groups((2, 4), structure="channel")
    assert channel == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_channel_groups_3x3_kernel_strided_input_channels() -> None:
    groups = channel_groups((2, 2, 3, 3), structure="filter")
    assert groups[0] == list(range(18))
    assert groups[1] == list(range(18, 36))
    channels = channel_groups((2, 2, 3, 3), structure="channel")
    assert channels[0] == list(range(0, 9)) + list(range(18, 27))
    assert channels[1] == list(range(9, 18)) + list(range(27, 36))


def test_structured_prune_layer_zeros_whole_filters_by_l1() -> None:
    # 4 filters, 2 in-channels, 1x1. Filter L1 norms: 0.2, 1.0, 0.4, 1.6.
    shape = (4, 2, 1, 1)
    weights = [
        0.1, 0.1,
        0.5, 0.5,
        0.2, 0.2,
        0.8, 0.8,
    ]
    pruned = structured_prune_layer(weights, shape, density=0.5, norm="l1")
    assert pruned == [
        0.0, 0.0,
        0.5, 0.5,
        0.0, 0.0,
        0.8, 0.8,
    ]
    assert structured_keep_indices(weights, shape, density=0.5, norm="l1") == [1, 3]


def test_structured_prune_layer_l1_and_l2_rank_differently() -> None:
    # Filter 0 is sparse/large; filter 1 is dense/small.
    # L1(filter0)=1.0, L2=1.0; L1(filter1)=1.5, L2=sqrt(0.75)≈0.866.
    shape = (2, 1, 1, 3)
    weights = [1.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    l1 = structured_prune_layer(weights, shape, density=0.5, norm="l1")
    l2 = structured_prune_layer(weights, shape, density=0.5, norm="l2")
    assert l1 == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    assert l2 == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_structured_prune_layer_zeros_whole_input_channels() -> None:
    # 2 filters, 3 in-channels, 1x1.
    # Channel L1: [2.0, 0.2, 1.0] -> keep 2 of 3: channels 0 and 2.
    shape = (2, 3, 1, 1)
    weights = [
        1.0, 0.1, 0.5,
        1.0, 0.1, 0.5,
    ]
    pruned = structured_prune_layer(
        weights, shape, density=2 / 3, norm="l1", structure="channel"
    )
    assert pruned == [1.0, 0.0, 0.5, 1.0, 0.0, 0.5]


def test_structured_prune_layer_keeps_all_at_density_one() -> None:
    shape = (2, 2, 1, 1)
    weights = [0.1, 0.2, 0.3, 0.4]
    assert structured_prune_layer(weights, shape, density=1.0) == weights


def test_structured_prune_layer_does_not_modify_input() -> None:
    shape = (2, 2, 1, 1)
    weights = [0.1, 0.2, 0.3, 0.4]
    original = list(weights)
    structured_prune_layer(weights, shape, density=0.5)
    assert weights == original


def test_structured_prune_layer_tie_breaks_toward_lower_index() -> None:
    # Two equal-norm filters; keep one. Lower index (0) wins the tie.
    shape = (2, 1, 1, 1)
    weights = [0.5, 0.5]
    pruned = structured_prune_layer(weights, shape, density=0.5, norm="l1")
    assert pruned == [0.5, 0.0]


def test_structured_prune_layer_already_zero_filter_is_dropped_first() -> None:
    shape = (3, 1, 1, 1)
    weights = [0.0, 0.4, 0.8]
    pruned = structured_prune_layer(weights, shape, density=2 / 3, norm="l1")
    assert pruned == [0.0, 0.4, 0.8]


def test_structured_prune_layer_rejects_empty() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        structured_prune_layer([], (1, 1, 1, 1))


def test_structured_prune_layer_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="density"):
        structured_prune_layer([0.1, 0.2], (2, 1, 1, 1), density=0.0)
    with pytest.raises(ValueError, match="density"):
        structured_prune_layer([0.1, 0.2], (2, 1, 1, 1), density=1.5)


def test_structured_prune_layer_rejects_invalid_norm_and_structure() -> None:
    weights = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="norm"):
        structured_prune_layer(weights, (2, 2, 1, 1), norm="linf")
    with pytest.raises(ValueError, match="structure"):
        structured_prune_layer(weights, (2, 2, 1, 1), structure="neuron")


def test_structured_prune_layer_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expects"):
        structured_prune_layer([0.1, 0.2], (2, 2, 1, 1))


def test_channel_norms_match_manual_l1() -> None:
    shape = (2, 2, 1, 1)
    weights = [0.1, -0.2, 0.3, 0.4]
    assert channel_norms(weights, shape, norm="l1") == pytest.approx([0.3, 0.7])


def test_structured_prune_model_prunes_conv_and_copies_dense() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        conv_layer("conv1", out_channels=4, in_channels=2, kernel_h=1, kernel_w=1),
    ]
    model = {
        "fc1": [1.0, 2.0, 3.0, 4.0],
        "conv1": [
            0.1, 0.1,
            0.5, 0.5,
            0.2, 0.2,
            0.8, 0.8,
        ],
    }
    original_fc1 = list(model["fc1"])
    pruned = structured_prune_model(specs, model, density=0.5, norm="l1")
    assert pruned["fc1"] == original_fc1
    assert pruned["conv1"] == [
        0.0, 0.0,
        0.5, 0.5,
        0.0, 0.0,
        0.8, 0.8,
    ]
    assert model["fc1"] == original_fc1


def test_structured_prune_model_respects_per_layer_density() -> None:
    specs = [
        conv_layer("conv1", out_channels=4, in_channels=1, kernel_h=1, kernel_w=1),
        conv_layer("conv2", out_channels=4, in_channels=1, kernel_h=1, kernel_w=1),
    ]
    model = {
        "conv1": [0.1, 0.2, 0.3, 0.4],
        "conv2": [0.1, 0.2, 0.3, 0.4],
    }
    pruned = structured_prune_model(
        specs, model, density=0.5, per_layer={"conv1": 0.25}
    )
    assert sum(1 for value in pruned["conv1"] if value != 0.0) == 1
    assert sum(1 for value in pruned["conv2"] if value != 0.0) == 2


def test_structured_prune_model_rejects_no_conv_layers() -> None:
    specs = [dense_layer("fc1", in_features=2, out_features=2)]
    with pytest.raises(ValueError, match="at least one conv"):
        structured_prune_model(specs, {"fc1": [0.1, 0.2, 0.3, 0.4]})


def test_structured_prune_model_rejects_unknown_per_layer() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="unknown layer"):
        structured_prune_model(
            specs, {"conv1": [0.1, 0.2]}, per_layer={"ghost": 0.5}
        )


def test_structured_prune_model_rejects_empty_model() -> None:
    specs = [conv_layer("conv1", out_channels=1, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="not be empty"):
        structured_prune_model(specs, {})


def test_per_channel_survival_reports_kept_filters() -> None:
    specs = [conv_layer("conv1", out_channels=4, in_channels=2, kernel_h=1, kernel_w=1)]
    weights = {
        "conv1": [
            0.1, 0.1,
            0.5, 0.5,
            0.2, 0.2,
            0.8, 0.8,
        ],
    }
    pruned = structured_prune_model(specs, weights, density=0.5, norm="l1")
    rows = per_channel_survival(specs, weights, pruned, structure="filter", norm="l1")
    assert len(rows) == 1
    row = rows[0]
    assert row["layer"] == "conv1"
    assert row["total_channels"] == 4
    assert row["kept_channels"] == 2
    assert row["kept_indices"] == [1, 3]
    assert row["pruned_indices"] == [0, 2]
    assert row["survival_fraction"] == pytest.approx(0.5)
    assert [channel["kept"] for channel in row["channels"]] == [False, True, False, True]


def test_per_channel_survival_skips_dense_layers() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1),
    ]
    weights = {"fc1": [1.0, 2.0, 3.0, 4.0], "conv1": [0.1, 0.9]}
    pruned = structured_prune_model(specs, weights, density=0.5)
    rows = per_channel_survival(specs, weights, pruned)
    assert [row["layer"] for row in rows] == ["conv1"]


def test_model_channel_survival_summary_keys_and_skipped() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        conv_layer("conv1", out_channels=4, in_channels=1, kernel_h=1, kernel_w=1),
    ]
    weights = {
        "fc1": [1.0, 2.0, 3.0, 4.0],
        "conv1": [0.1, 0.2, 0.3, 0.4],
    }
    summary = model_channel_survival_summary(
        specs, weights, density=0.5, norm="l2", structure="filter"
    )
    assert summary["norm"] == "l2"
    assert summary["structure"] == "filter"
    assert summary["skipped"] == ["fc1"]
    assert summary["total_channels"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["pruned_weights"]["fc1"] == [1.0, 2.0, 3.0, 4.0]
    assert "layers" in summary


def test_model_channel_survival_summary_channel_structure() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=4, kernel_h=1, kernel_w=1)]
    weights = {
        "conv1": [
            1.0, 0.1, 0.5, 0.2,
            1.0, 0.1, 0.5, 0.2,
        ],
    }
    summary = model_channel_survival_summary(
        specs, weights, density=0.5, structure="channel", norm="l1"
    )
    row = summary["layers"][0]
    assert row["total_channels"] == 4
    assert row["kept_channels"] == 2
    assert row["kept_indices"] == [0, 2]


def test_per_channel_survival_rejects_mismatched_layers() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="same layers"):
        per_channel_survival(specs, {"conv1": [0.1, 0.2]}, {"other": [0.1, 0.2]})


def _flat_conv_weights(n: int) -> str:
    return ",".join(f"{0.1 * (i + 1):.1f}" for i in range(n))


def test_cli_structured_json_reports_channel_survival() -> None:
    # 4 filters x 2 in x 1 x 1 = 8 weights.
    specs_str = "conv1=conv:4x2x1x1"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "structured",
            "--specs", specs_str,
            "--weights", _flat_conv_weights(8),
            "--density", "0.5",
            "--norm", "l1",
            "--structure", "filter",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["structure"] == "filter"
    assert payload["norm"] == "l1"
    assert payload["total_channels"] == 4
    assert payload["total_kept"] == 2
    assert "pruned_weights" not in payload
    assert payload["layers"][0]["kept_channels"] == 2


def test_cli_structured_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "structured.md"
    rc = main([
        "structured",
        "--specs", "fc1=dense:2x2,conv1=conv:4x1x1x1",
        "--weights", "1,2,3,4,0.1,0.2,0.3,0.4",
        "--density", "0.5",
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Channel / filter pruning survival" in text
    assert "conv1" in text
    assert "Skipped (non-conv): fc1" in text
    assert "Kept indices" in text


def test_cli_structured_channel_mode_json() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "structured",
            "--specs", "conv1=conv:2x4x1x1",
            "--weights", "1.0,0.1,0.5,0.2,1.0,0.1,0.5,0.2",
            "--density", "0.5",
            "--structure", "channel",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["structure"] == "channel"
    assert payload["layers"][0]["kept_indices"] == [0, 2]


def test_cli_structured_rejects_no_conv() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "structured",
                "--specs", "fc1=dense:2x2",
                "--weights", "0.1,0.2,0.3,0.4",
            ])
    assert rc == 2
    assert "conv" in buf_err.getvalue()


def test_cli_structured_rejects_weight_count_mismatch() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "structured",
                "--specs", "conv1=conv:2x2x1x1",
                "--weights", "0.1,0.2",
            ])
    assert rc == 2
