"""Tests for first-order Taylor (soft-filter) channel pruning."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    structured_prune_layer,
    taylor_channel_scores,
    taylor_keep_indices,
    taylor_prune_layer,
    taylor_prune_model,
    taylor_prune_summary,
)
from prune_kit.cli import main


def test_scores_are_sum_of_absolute_grad_weight_products() -> None:
    # 2 filters, 1 in-channel, 1x2 kernel.
    shape = (2, 1, 1, 2)
    weights = [2.0, -2.0, 0.5, 0.5]
    grads = [0.5, 0.5, 4.0, 4.0]
    scores = taylor_channel_scores(weights, shape, grads)
    # |2*0.5| + |-2*0.5| = 2; |0.5*4| + |0.5*4| = 4
    assert scores == pytest.approx([2.0, 4.0])


def test_squared_criterion_uses_squared_products() -> None:
    shape = (2, 1, 1, 2)
    weights = [3.0, 0.0, 2.0, 2.0]
    grads = [1.0, 1.0, 1.0, 1.0]
    scores = taylor_channel_scores(weights, shape, grads, criterion="sq")
    # 3^2 + 0 = 9; 2^2 + 2^2 = 8
    assert scores == pytest.approx([9.0, 8.0])


def test_abs_and_sq_rank_filters_differently() -> None:
    shape = (2, 1, 1, 2)
    weights = [3.0, 0.0, 2.0, 2.0]
    grads = [1.0, 1.0, 1.0, 1.0]
    by_abs = taylor_prune_layer(weights, shape, grads, density=0.5, criterion="abs")
    by_sq = taylor_prune_layer(weights, shape, grads, density=0.5, criterion="sq")
    # abs scores: 3 and 4 -> keep filter 1. sq scores: 9 and 8 -> keep filter 0.
    assert by_abs == [0.0, 0.0, 2.0, 2.0]
    assert by_sq == [3.0, 0.0, 0.0, 0.0]


def test_mean_reduction_scales_scores_without_changing_rank() -> None:
    shape = (2, 1, 1, 2)
    weights = [2.0, -2.0, 0.5, 0.5]
    grads = [0.5, 0.5, 4.0, 4.0]
    scores = taylor_channel_scores(weights, shape, grads, reduction="mean")
    assert scores == pytest.approx([1.0, 2.0])
    summed = taylor_keep_indices(weights, shape, grads, density=0.5, reduction="sum")
    averaged = taylor_keep_indices(weights, shape, grads, density=0.5, reduction="mean")
    assert summed == averaged == [1]


def test_negative_gradients_use_absolute_product() -> None:
    shape = (1, 1, 1, 2)
    weights = [2.0, -3.0]
    grads = [-4.0, 5.0]
    scores = taylor_channel_scores(weights, shape, grads)
    assert scores == pytest.approx([8.0 + 15.0])


def test_prune_keeps_highest_scores_not_largest_weights() -> None:
    # Filter 0 is large with zero gradient; filter 1 is small but salient.
    # L1 keeps filter 0. FO keeps filter 1.
    shape = (2, 1, 1, 2)
    weights = [10.0, 10.0, 0.1, 0.1]
    grads = [0.0, 0.0, 5.0, 5.0]
    pruned = taylor_prune_layer(weights, shape, grads, density=0.5)
    assert pruned == [0.0, 0.0, 0.1, 0.1]
    assert taylor_keep_indices(weights, shape, grads, density=0.5) == [1]
    l1 = structured_prune_layer(weights, shape, density=0.5, norm="l1")
    assert l1 == [10.0, 10.0, 0.0, 0.0]
    assert pruned != l1


def test_prune_zeros_whole_input_channels() -> None:
    # 2 filters, 2 in-channels, 1x1.
    # Channel 0: |1*1| + |1*1| = 2. Channel 1: |10*0| + |10*0| = 0.
    shape = (2, 2, 1, 1)
    weights = [1.0, 10.0, 1.0, 10.0]
    grads = [1.0, 0.0, 1.0, 0.0]
    pruned = taylor_prune_layer(
        weights, shape, grads, density=0.5, structure="channel"
    )
    assert pruned == [1.0, 0.0, 1.0, 0.0]
    assert taylor_keep_indices(
        weights, shape, grads, density=0.5, structure="channel"
    ) == [0]


def test_contributions_replace_grad_weight_product() -> None:
    shape = (2, 1, 1, 2)
    weights = [4.0, 4.0, 0.2, 0.2]
    contributions = [0.0, 0.0, 3.0, -1.0]
    scores = taylor_channel_scores(
        weights, shape, contributions=contributions
    )
    assert scores == pytest.approx([0.0, 4.0])
    pruned = taylor_prune_layer(
        weights, shape, contributions=contributions, density=0.5
    )
    assert pruned == [0.0, 0.0, 0.2, 0.2]


def test_kernel_groups_follow_conv_layout() -> None:
    # One 1x2 filter, two spatial taps. Only the second tap is salient.
    shape = (1, 1, 1, 2)
    weights = [0.0, 2.0]
    grads = [9.0, 0.5]
    assert taylor_channel_scores(weights, shape, grads) == pytest.approx([1.0])


def test_keeps_all_at_density_one() -> None:
    shape = (2, 1, 1, 1)
    weights = [0.1, 0.2]
    grads = [1.0, 0.0]
    assert taylor_prune_layer(weights, shape, grads, density=1.0) == [0.1, 0.2]


def test_does_not_modify_inputs() -> None:
    shape = (2, 1, 1, 1)
    weights = [0.1, 0.9]
    grads = [5.0, 0.1]
    original_w = list(weights)
    original_g = list(grads)
    taylor_prune_layer(weights, shape, grads, density=0.5)
    assert weights == original_w
    assert grads == original_g


def test_tie_breaks_toward_lower_index() -> None:
    shape = (2, 1, 1, 1)
    weights = [0.5, 0.5]
    grads = [1.0, 1.0]
    pruned = taylor_prune_layer(weights, shape, grads, density=0.5)
    assert pruned == [0.5, 0.0]


def test_zero_score_filter_is_dropped_first() -> None:
    shape = (3, 1, 1, 1)
    weights = [0.0, 0.4, 0.8]
    grads = [1.0, 1.0, 1.0]
    pruned = taylor_prune_layer(weights, shape, grads, density=2 / 3)
    assert pruned == [0.0, 0.4, 0.8]


def test_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        taylor_prune_layer([], (1, 1, 1, 1), [])


def test_rejects_missing_saliency() -> None:
    with pytest.raises(ValueError, match="grads or contributions"):
        taylor_channel_scores([0.1, 0.2], (2, 1, 1, 1))


def test_rejects_both_grads_and_contributions() -> None:
    weights = [0.1, 0.2]
    with pytest.raises(ValueError, match="not both"):
        taylor_channel_scores(
            weights, (2, 1, 1, 1), [1.0, 1.0], contributions=[1.0, 1.0]
        )


def test_rejects_grad_length_mismatch() -> None:
    with pytest.raises(ValueError, match="grads has"):
        taylor_prune_layer([0.1, 0.2], (2, 1, 1, 1), [1.0])


def test_rejects_contribution_length_mismatch() -> None:
    with pytest.raises(ValueError, match="contributions has"):
        taylor_channel_scores(
            [0.1, 0.2], (2, 1, 1, 1), contributions=[1.0]
        )


def test_rejects_invalid_density() -> None:
    weights = [0.1, 0.2]
    grads = [1.0, 1.0]
    shape = (2, 1, 1, 1)
    with pytest.raises(ValueError, match="density"):
        taylor_prune_layer(weights, shape, grads, density=0.0)
    with pytest.raises(ValueError, match="density"):
        taylor_prune_layer(weights, shape, grads, density=1.5)


def test_rejects_invalid_criterion_reduction_and_structure() -> None:
    weights = [0.1, 0.2, 0.3, 0.4]
    grads = [1.0, 1.0, 1.0, 1.0]
    shape = (2, 2, 1, 1)
    with pytest.raises(ValueError, match="criterion"):
        taylor_prune_layer(weights, shape, grads, criterion="l1")
    with pytest.raises(ValueError, match="reduction"):
        taylor_prune_layer(weights, shape, grads, reduction="max")
    with pytest.raises(ValueError, match="structure"):
        taylor_prune_layer(weights, shape, grads, structure="neuron")


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expects"):
        taylor_prune_layer([0.1, 0.2], (2, 2, 1, 1), [1.0, 1.0, 1.0, 1.0])


def test_model_prunes_conv_and_copies_dense() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2),
    ]
    model = {
        "fc1": [1.0, 2.0, 3.0, 4.0],
        "conv1": [10.0, 10.0, 0.1, 0.1],
    }
    grads = {"conv1": [0.0, 0.0, 5.0, 5.0]}
    original_fc1 = list(model["fc1"])
    pruned = taylor_prune_model(specs, model, grads, density=0.5)
    assert pruned["fc1"] == original_fc1
    assert pruned["conv1"] == [0.0, 0.0, 0.1, 0.1]
    assert model["fc1"] == original_fc1
    assert model["conv1"] == [10.0, 10.0, 0.1, 0.1]


def test_model_respects_per_layer_density() -> None:
    specs = [
        conv_layer("conv1", out_channels=4, in_channels=1, kernel_h=1, kernel_w=1),
        conv_layer("conv2", out_channels=4, in_channels=1, kernel_h=1, kernel_w=1),
    ]
    model = {
        "conv1": [0.1, 0.2, 0.3, 0.4],
        "conv2": [0.1, 0.2, 0.3, 0.4],
    }
    grads = {
        "conv1": [1.0, 1.0, 1.0, 1.0],
        "conv2": [1.0, 1.0, 1.0, 1.0],
    }
    pruned = taylor_prune_model(
        specs, model, grads, density=0.5, per_layer={"conv1": 0.25}
    )
    assert sum(1 for value in pruned["conv1"] if value != 0.0) == 1
    assert sum(1 for value in pruned["conv2"] if value != 0.0) == 2


def test_model_accepts_contributions_mapping() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    model = {"conv1": [5.0, 0.2]}
    contributions = {"conv1": [0.0, 3.0]}
    pruned = taylor_prune_model(
        specs, model, contributions=contributions, density=0.5
    )
    assert pruned["conv1"] == [0.0, 0.2]


def test_model_rejects_no_conv_layers() -> None:
    specs = [dense_layer("fc1", in_features=2, out_features=2)]
    with pytest.raises(ValueError, match="at least one conv"):
        taylor_prune_model(specs, {"fc1": [0.1, 0.2, 0.3, 0.4]}, {"fc1": [1, 1, 1, 1]})


def test_model_rejects_missing_conv_grads() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="missing conv layer"):
        taylor_prune_model(specs, {"conv1": [0.1, 0.2]}, {})


def test_model_rejects_neither_saliency() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="grads or contributions"):
        taylor_prune_model(specs, {"conv1": [0.1, 0.2]})


def test_model_rejects_unknown_grad_layer() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="unknown layer"):
        taylor_prune_model(
            specs,
            {"conv1": [0.1, 0.2]},
            {"conv1": [1.0, 1.0], "ghost": [1.0]},
        )


def test_model_rejects_unknown_per_layer() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="unknown layer"):
        taylor_prune_model(
            specs,
            {"conv1": [0.1, 0.2]},
            {"conv1": [1.0, 1.0]},
            per_layer={"ghost": 0.5},
        )


def test_model_rejects_empty_model() -> None:
    specs = [conv_layer("conv1", out_channels=1, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="not be empty"):
        taylor_prune_model(specs, {}, {})


def test_model_rejects_grad_length_mismatch() -> None:
    specs = [conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="grads has"):
        taylor_prune_model(specs, {"conv1": [0.1, 0.2]}, {"conv1": [1.0]})


def test_summary_reports_scores_and_skipped_dense() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2),
    ]
    model = {
        "fc1": [1.0, 2.0, 3.0, 4.0],
        "conv1": [10.0, 10.0, 0.1, 0.1],
    }
    grads = {"conv1": [0.0, 0.0, 5.0, 5.0]}
    summary = taylor_prune_summary(specs, model, grads, density=0.5)
    assert summary["criterion"] == "abs"
    assert summary["reduction"] == "sum"
    assert summary["structure"] == "filter"
    assert summary["skipped"] == ["fc1"]
    assert summary["total_channels"] == 2
    assert summary["total_kept"] == 1
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["pruned_weights"]["fc1"] == [1.0, 2.0, 3.0, 4.0]
    row = summary["layers"][0]
    assert row["kept_indices"] == [1]
    assert row["pruned_indices"] == [0]
    assert [channel["score"] for channel in row["channels"]] == pytest.approx([0.0, 1.0])
    assert [channel["kept"] for channel in row["channels"]] == [False, True]


def test_public_api_exports_taylor_helpers() -> None:
    import prune_kit

    for name in (
        "taylor_channel_scores",
        "taylor_keep_indices",
        "taylor_prune_layer",
        "taylor_prune_model",
        "taylor_prune_summary",
    ):
        assert name in prune_kit.__all__
        assert callable(getattr(prune_kit, name))


def test_cli_taylor_json_keeps_salient_filter() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "taylor",
            "--specs", "conv1=conv:2x1x1x2",
            "--weights", "10,10,0.1,0.1",
            "--grads", "0,0,5,5",
            "--density", "0.5",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["criterion"] == "abs"
    assert payload["structure"] == "filter"
    assert payload["total_channels"] == 2
    assert payload["total_kept"] == 1
    assert "pruned_weights" not in payload
    assert payload["layers"][0]["kept_indices"] == [1]
    assert payload["layers"][0]["channels"][1]["score"] == pytest.approx(1.0)


def test_cli_taylor_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "taylor.md"
    rc = main([
        "taylor",
        "--specs", "fc1=dense:2x2,conv1=conv:2x1x1x2",
        "--weights", "1,2,3,4,10,10,0.1,0.1",
        "--grads", "0,0,0,0,0,0,5,5",
        "--density", "0.5",
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "First-order Taylor channel pruning" in text
    assert "conv1" in text
    assert "Skipped (non-conv): fc1" in text
    assert "Score" in text


def test_cli_taylor_channel_and_squared_criterion() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "taylor",
            "--specs", "conv1=conv:2x2x1x1",
            "--weights", "1,10,1,10",
            "--grads", "1,0,1,0",
            "--density", "0.5",
            "--structure", "channel",
            "--criterion", "sq",
            "--reduction", "mean",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["structure"] == "channel"
    assert payload["criterion"] == "sq"
    assert payload["reduction"] == "mean"
    assert payload["layers"][0]["kept_indices"] == [0]


def test_cli_taylor_rejects_no_conv() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "taylor",
                "--specs", "fc1=dense:2x2",
                "--weights", "0.1,0.2,0.3,0.4",
                "--grads", "1,1,1,1",
            ])
    assert rc == 2
    assert "conv" in buf_err.getvalue()


def test_cli_taylor_rejects_grad_count_mismatch() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "taylor",
                "--specs", "conv1=conv:2x1x1x1",
                "--weights", "0.1,0.2",
                "--grads", "1",
            ])
    assert rc == 2
    assert "grads" in buf_err.getvalue()
