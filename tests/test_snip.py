"""Tests for SNIP single-shot connection-sensitivity pruning."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    layer_weight_count,
    magnitude_prune_layer,
    mask_density,
    snip_prune_layer,
    snip_prune_model,
    snip_prune_summary,
    snip_scores,
)
from prune_kit.cli import main


def test_score_shape_matches_dense_and_conv_layers() -> None:
    dense = dense_layer("fc", in_features=5, out_features=3)
    dense_n = layer_weight_count(dense)
    dense_scores = snip_scores(
        [0.1] * dense_n, [0.2] * dense_n, shape=dense.shape
    )
    assert len(dense_scores) == dense_n == 15
    assert dense_scores == pytest.approx([0.02] * dense_n)

    conv = conv_layer("conv", out_channels=2, in_channels=3, kernel_h=2, kernel_w=2)
    conv_n = layer_weight_count(conv)
    conv_scores = snip_scores(
        [0.5] * conv_n, [-0.4] * conv_n, shape=conv.shape
    )
    assert len(conv_scores) == conv_n == 24
    assert conv_scores == pytest.approx([0.2] * conv_n)


def test_scores_are_absolute_weight_grad_products() -> None:
    weights = [2.0, -3.0, 0.0, 4.0]
    grads = [0.5, -1.0, 9.0, 0.25]
    assert snip_scores(weights, grads) == pytest.approx([1.0, 3.0, 0.0, 1.0])


def test_normalize_divides_by_score_sum() -> None:
    scores = snip_scores([1.0, -2.0, 3.0], [1.0, 1.0, 1.0], normalize=True)
    assert scores == pytest.approx([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    assert sum(scores) == pytest.approx(1.0)


def test_normalize_zero_sum_stays_zero() -> None:
    assert snip_scores([0.0, 0.0], [4.0, -8.0], normalize=True) == [0.0, 0.0]


def test_contributions_replace_weight_grad_product() -> None:
    weights = [4.0, 0.2]
    contributions = [0.0, -3.0]
    assert snip_scores(weights, contributions=contributions) == pytest.approx([0.0, 3.0])


def test_target_sparsity_matches_density_rounding() -> None:
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    grads = [1.0] * len(weights)
    density = 0.25
    pruned = snip_prune_layer(weights, grads, density=density)
    n_zero = sum(value == 0.0 for value in pruned)
    assert n_zero == 6
    assert n_zero / len(pruned) == pytest.approx(1.0 - density)
    assert pruned == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8]


def test_zero_weights_stay_zero_even_with_large_gradients() -> None:
    weights = [0.0, 3.0, 0.0, 1.0]
    grads = [1.0e6, 0.01, 1.0e6, 0.01]
    pruned = snip_prune_layer(weights, grads, density=0.5)
    assert pruned == [0.0, 3.0, 0.0, 1.0]
    assert pruned[0] == 0.0
    assert pruned[2] == 0.0


def test_zero_weights_stay_zero_when_contribution_ranks_them_high() -> None:
    weights = [0.0, 2.0]
    contributions = [5.0, 0.1]
    pruned = snip_prune_layer(weights, contributions=contributions, density=0.5)
    assert pruned == [0.0, 0.0]
    summary = snip_prune_summary(
        [dense_layer("fc", in_features=2, out_features=1)],
        {"fc": weights},
        contributions={"fc": contributions},
        density=0.5,
        scope="layer",
    )
    assert summary["masks"]["fc"] == [1, 0]
    assert summary["pruned_weights"]["fc"] == [0.0, 0.0]
    assert summary["layers"][0]["kept_weights"] == 1


def test_second_pass_does_not_revive_pruned_weights() -> None:
    weights = [0.2, 0.9, 0.1, 0.8]
    grads = [1.0, 1.0, 1.0, 1.0]
    once = snip_prune_layer(weights, grads, density=0.5)
    assert once == [0.0, 0.9, 0.0, 0.8]
    twice = snip_prune_layer(once, grads, density=0.5)
    assert twice == once
    assert snip_prune_layer(once, grads, density=1.0) == [0.0, 0.9, 0.0, 0.8]


def test_keeps_salient_small_weights_not_largest_magnitude() -> None:
    weights = [10.0, 10.0, 0.1, 0.1]
    grads = [0.0, 0.0, 5.0, 5.0]
    pruned = snip_prune_layer(weights, grads, density=0.5)
    assert pruned == [0.0, 0.0, 0.1, 0.1]
    assert magnitude_prune_layer(weights, density=0.5) == [10.0, 10.0, 0.0, 0.0]


def test_conv_layer_uses_flat_kernel_layout() -> None:
    spec = conv_layer("conv1", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2)
    weights = [10.0, 10.0, 0.1, 0.1]
    grads = [0.0, 0.0, 5.0, 5.0]
    pruned = snip_prune_layer(weights, grads, density=0.5, shape=spec.shape)
    assert len(pruned) == layer_weight_count(spec)
    assert pruned == [0.0, 0.0, 0.1, 0.1]


def test_tie_breaks_toward_lower_index() -> None:
    weights = [0.4, 0.4, 0.4, 0.4]
    grads = [1.0, 1.0, 1.0, 1.0]
    assert snip_prune_layer(weights, grads, density=0.5) == [0.4, 0.4, 0.0, 0.0]


def test_density_one_keeps_values_including_zeros() -> None:
    weights = [0.0, -1.5, 2.0]
    assert snip_prune_layer(weights, [1.0, 1.0, 1.0], density=1.0) == [0.0, -1.5, 2.0]


def test_rounding_can_drop_every_connection() -> None:
    pruned = snip_prune_layer([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], density=0.1)
    assert pruned == [0.0, 0.0, 0.0, 0.0]


def test_does_not_modify_inputs() -> None:
    weights = [0.1, 0.9, 0.2, 0.8]
    grads = [1.0, 0.0, 3.0, 0.5]
    original_w = list(weights)
    original_g = list(grads)
    snip_prune_layer(weights, grads, density=0.5)
    assert weights == original_w
    assert grads == original_g


def test_mask_density_matches_target_on_summary() -> None:
    spec = dense_layer("fc", in_features=4, out_features=2)
    weights = [0.1 * (i + 1) for i in range(8)]
    grads = [1.0] * 8
    summary = snip_prune_summary(
        [spec], {"fc": weights}, {"fc": grads}, density=0.5, scope="layer"
    )
    mask = summary["masks"]["fc"]
    assert len(mask) == 8
    assert mask_density(mask) == pytest.approx(0.5)
    assert summary["sparsity"] == pytest.approx(0.5)
    assert summary["total_kept"] == 4
    assert summary["overall_survival"] == pytest.approx(0.5)
    pruned = summary["pruned_weights"]["fc"]
    for bit, original, value in zip(mask, weights, pruned):
        if bit:
            assert value == pytest.approx(original)
        else:
            assert value == 0.0


def test_global_scope_keeps_salient_layer() -> None:
    specs = [
        dense_layer("big_dead", in_features=4, out_features=1),
        conv_layer("small_live", out_channels=1, in_channels=1, kernel_h=1, kernel_w=4),
    ]
    model = {
        "big_dead": [10.0, 10.0, 10.0, 10.0],
        "small_live": [0.1, 0.1, 0.1, 0.1],
    }
    grads = {
        "big_dead": [0.0, 0.0, 0.0, 0.0],
        "small_live": [1.0, 1.0, 1.0, 1.0],
    }
    pruned = snip_prune_model(specs, model, grads, density=0.5, scope="global")
    assert pruned["big_dead"] == [0.0, 0.0, 0.0, 0.0]
    assert pruned["small_live"] == [0.1, 0.1, 0.1, 0.1]
    layered = snip_prune_model(specs, model, grads, density=0.5, scope="layer")
    assert sum(value != 0.0 for value in layered["big_dead"]) == 2
    assert sum(value != 0.0 for value in layered["small_live"]) == 2
    assert pruned != layered


def test_global_tie_breaks_toward_earlier_spec() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [0.2, 0.2], "b": [0.2, 0.9]}
    grads = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    # scores: a=0.2, 0.2; b=0.2, 0.9. density 0.5 keeps 2: b[1] and a[0].
    pruned = snip_prune_model(specs, model, grads, density=0.5)
    assert pruned == {"a": [0.2, 0.0], "b": [0.0, 0.9]}


def test_layer_scope_respects_per_layer_density() -> None:
    specs = [
        dense_layer("fc1", in_features=4, out_features=1),
        dense_layer("fc2", in_features=4, out_features=1),
    ]
    model = {
        "fc1": [0.1, 0.2, 0.3, 0.4],
        "fc2": [0.1, 0.2, 0.3, 0.4],
    }
    grads = {"fc1": [1.0, 1.0, 1.0, 1.0], "fc2": [1.0, 1.0, 1.0, 1.0]}
    summary = snip_prune_summary(
        specs,
        model,
        grads,
        density=0.5,
        scope="layer",
        per_layer={"fc1": 0.25},
    )
    by_name = {row["layer"]: row for row in summary["layers"]}
    assert by_name["fc1"]["kept_weights"] == 1
    assert by_name["fc2"]["kept_weights"] == 2
    assert summary["per_layer_density"] == {"fc1": 0.25}


def test_model_accepts_contributions_for_dense_and_conv() -> None:
    specs = [
        dense_layer("fc", in_features=2, out_features=1),
        conv_layer("conv", out_channels=1, in_channels=1, kernel_h=1, kernel_w=2),
    ]
    model = {"fc": [5.0, 5.0], "conv": [0.2, 0.2]}
    contributions = {"fc": [0.0, 0.0], "conv": [1.0, -2.0]}
    pruned = snip_prune_model(
        specs, model, contributions=contributions, density=0.5, scope="global"
    )
    assert pruned["fc"] == [0.0, 0.0]
    assert pruned["conv"] == [0.2, 0.2]


def test_model_does_not_modify_inputs() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=2)]
    model = {"fc": [0.1, 0.9, 0.2, 0.8]}
    grads = {"fc": [1.0, 0.0, 3.0, 0.5]}
    original = {name: list(values) for name, values in model.items()}
    original_grads = {name: list(values) for name, values in grads.items()}
    snip_prune_model(specs, model, grads, density=0.5)
    assert model == original
    assert grads == original_grads


def test_summary_reports_score_sums() -> None:
    specs = [conv_layer("conv", out_channels=1, in_channels=1, kernel_h=1, kernel_w=2)]
    model = {"conv": [2.0, -3.0]}
    grads = {"conv": [0.5, 1.0]}
    summary = snip_prune_summary(specs, model, grads, density=1.0)
    assert summary["scope"] == "global"
    assert summary["layers"][0]["kind"] == "conv"
    assert summary["layers"][0]["score_sum"] == pytest.approx(4.0)
    assert summary["layers"][0]["max_score"] == pytest.approx(3.0)
    assert summary["masks"]["conv"] == [1, 1]


def test_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        snip_prune_layer([], [])


def test_rejects_missing_saliency() -> None:
    with pytest.raises(ValueError, match="grads or contributions"):
        snip_scores([0.1, 0.2])


def test_rejects_both_grads_and_contributions() -> None:
    with pytest.raises(ValueError, match="not both"):
        snip_scores([0.1, 0.2], [1.0, 1.0], contributions=[1.0, 1.0])


def test_rejects_grad_length_mismatch() -> None:
    with pytest.raises(ValueError, match="grads has"):
        snip_prune_layer([0.1, 0.2], [1.0])


def test_rejects_contribution_length_mismatch() -> None:
    with pytest.raises(ValueError, match="contributions has"):
        snip_scores([0.1, 0.2], contributions=[1.0])


def test_rejects_invalid_density_and_scope() -> None:
    weights = [0.1, 0.2]
    grads = [1.0, 1.0]
    with pytest.raises(ValueError, match="density"):
        snip_prune_layer(weights, grads, density=0.0)
    with pytest.raises(ValueError, match="density"):
        snip_prune_layer(weights, grads, density=1.5)
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    with pytest.raises(ValueError, match="scope"):
        snip_prune_model(specs, {"fc": weights}, {"fc": grads}, scope="filter")


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expects"):
        snip_scores([0.1, 0.2], [1.0, 1.0], shape=(2, 2))


def test_model_rejects_per_layer_on_global_scope() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    with pytest.raises(ValueError, match="scope='layer'"):
        snip_prune_model(
            specs,
            {"fc": [0.1, 0.2]},
            {"fc": [1.0, 1.0]},
            scope="global",
            per_layer={"fc": 0.5},
        )


def test_model_rejects_unknown_grad_layer() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    with pytest.raises(ValueError, match="unknown layer"):
        snip_prune_model(
            specs,
            {"fc": [0.1, 0.2]},
            {"fc": [1.0, 1.0], "ghost": [1.0]},
        )


def test_model_rejects_missing_grads() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    with pytest.raises(ValueError, match="missing layer"):
        snip_prune_model(specs, {"fc": [0.1, 0.2]}, {})


def test_model_rejects_neither_saliency() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    with pytest.raises(ValueError, match="grads or contributions"):
        snip_prune_model(specs, {"fc": [0.1, 0.2]})


def test_model_rejects_empty_model() -> None:
    specs = [dense_layer("fc", in_features=1, out_features=1)]
    with pytest.raises(ValueError, match="not be empty"):
        snip_prune_model(specs, {}, {})


def test_model_rejects_weight_length_mismatch() -> None:
    specs = [conv_layer("conv", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    with pytest.raises(ValueError, match="expected"):
        snip_prune_model(specs, {"conv": [0.1]}, {"conv": [1.0]})


def test_public_api_exports_snip_helpers() -> None:
    import prune_kit

    for name in (
        "snip_scores",
        "snip_prune_layer",
        "snip_prune_model",
        "snip_prune_summary",
    ):
        assert name in prune_kit.__all__
        assert callable(getattr(prune_kit, name))


def test_cli_snip_json_keeps_salient_connections() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "snip",
            "--specs", "fc1=dense:2x2,conv1=conv:2x1x1x1",
            "--weights", "10,10,0.1,0.1,0.2,0.2",
            "--grads", "0,0,5,5,4,4",
            "--density", "0.5",
            "--scope", "global",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["scope"] == "global"
    assert payload["sparsity"] == pytest.approx(0.5)
    assert payload["total"] == 6
    assert payload["total_kept"] == 3
    assert "pruned_weights" not in payload
    assert "masks" not in payload
    by_name = {row["layer"]: row for row in payload["layers"]}
    assert by_name["conv1"]["kept_weights"] == 2
    assert by_name["fc1"]["kept_weights"] == 1
    assert by_name["conv1"]["kind"] == "conv"
    assert by_name["fc1"]["kind"] == "dense"


def test_cli_snip_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "snip.md"
    rc = main([
        "snip",
        "--specs", "fc1=dense:2x2",
        "--weights", "0.1,0.4,0.2,0.3",
        "--grads", "1,1,1,1",
        "--density", "0.5",
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "SNIP connection-sensitivity pruning" in text
    assert "fc1" in text
    assert "Score sum" in text


def test_cli_snip_layer_scope_per_layer_density() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "snip",
            "--specs", "fc1=dense:4x1,fc2=dense:4x1",
            "--weights", "0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4",
            "--grads", "1,1,1,1,1,1,1,1",
            "--density", "0.5",
            "--scope", "layer",
            "--per-layer-density", "fc1=0.25",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["scope"] == "layer"
    by_name = {row["layer"]: row for row in payload["layers"]}
    assert by_name["fc1"]["kept_weights"] == 1
    assert by_name["fc2"]["kept_weights"] == 2


def test_cli_snip_rejects_grad_count_mismatch() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "snip",
                "--specs", "fc1=dense:2x1",
                "--weights", "0.1,0.2",
                "--grads", "1",
            ])
    assert rc == 2
    assert "grads" in buf_err.getvalue()


def test_cli_snip_rejects_per_layer_on_global_scope() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "snip",
                "--specs", "fc1=dense:2x1",
                "--weights", "0.1,0.2",
                "--grads", "1,1",
                "--per-layer-density", "fc1=0.5",
            ])
    assert rc == 2
    assert "layer" in buf_err.getvalue()
