"""Tests for Wanda activation-aware pruning."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    activation_column_norms,
    conv_layer,
    dense_layer,
    layer_weight_count,
    wanda_prune_layer,
    wanda_prune_model,
    wanda_prune_summary,
    wanda_scores,
)
from prune_kit.cli import main


def test_scores_are_abs_weight_times_column_norm() -> None:
    # dense (2 out, 2 in): weights row-major [w00, w01, w10, w11]
    weights = [2.0, -3.0, 0.5, 4.0]
    norms = [0.5, 2.0]  # columns
    assert wanda_scores(weights, norms, shape=(2, 2)) == pytest.approx(
        [1.0, 6.0, 0.25, 8.0]
    )


def test_activation_column_norms_l2() -> None:
    # 2 samples x 2 features: [[3, 4], [0, 0]] => norms [3, 4]
    activations = [3.0, 4.0, 0.0, 0.0]
    assert activation_column_norms(activations, 2) == pytest.approx([3.0, 4.0])
    # [[1, 0], [0, 1], [0, 0]] => [1, 1]
    assert activation_column_norms([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2) == pytest.approx(
        [1.0, 1.0]
    )


def test_keeps_highest_wanda_scores() -> None:
    # scores = |w| * norms with norms=[1,1,1,1] => magnitude prune
    weights = [0.1, 0.2, 0.3, 0.4]
    norms = [1.0, 1.0, 1.0, 1.0]
    pruned = wanda_prune_layer(weights, norms, density=0.5)
    assert pruned == [0.0, 0.0, 0.3, 0.4]


def test_large_activation_protects_small_weight() -> None:
    # column 0 has tiny weight but huge activation; column 1 opposite
    weights = [0.1, 10.0]  # shape (1, 2)
    norms = [100.0, 0.1]  # scores = [10.0, 1.0]
    pruned = wanda_prune_layer(weights, norms, density=0.5, shape=(1, 2))
    assert pruned == [0.1, 0.0]


def test_contributions_replace_product() -> None:
    assert wanda_scores([4.0, 0.2], contributions=[0.0, -3.0]) == pytest.approx(
        [0.0, 3.0]
    )


def test_conv_broadcasts_per_input_channel() -> None:
    # (out=1, in=2, kh=1, kw=2) => 4 weights; norms length 2
    weights = [1.0, 1.0, 2.0, 2.0]
    norms = [3.0, 0.5]
    # expanded norms: [3, 3, 0.5, 0.5]; scores [3, 3, 1, 1]
    scores = wanda_scores(weights, norms, shape=(1, 2, 1, 2))
    assert scores == pytest.approx([3.0, 3.0, 1.0, 1.0])


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    norms = [1.0] * 8
    pruned = wanda_prune_layer(weights, norms, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    norms = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = wanda_prune_model(
        specs, model, norms, density=0.5, scope="layer"
    )
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    norms = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = wanda_prune_model(
        specs, model, norms, density=0.5, scope="global"
    )
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_structured_filter_keeps_highest_group() -> None:
    # 2 filters, 1 in, 1x2 kernel
    weights = [1.0, 1.0, 10.0, 10.0]  # filter0 small, filter1 large
    norms = [1.0]  # one input channel
    pruned = wanda_prune_layer(
        weights,
        norms,
        density=0.5,
        shape=(2, 1, 1, 2),
        structure="filter",
    )
    assert pruned == [0.0, 0.0, 10.0, 10.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    norms = {"fc": [1.0, 1.0, 1.0, 1.0]}
    summary = wanda_prune_summary(specs, model, norms, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["structure"] is None
    assert summary["layers"][0]["kept_weights"] == 2


def test_rejects_both_norms_and_contributions() -> None:
    with pytest.raises(ValueError):
        wanda_scores([1.0], [1.0], contributions=[1.0])


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        wanda_prune_layer([], activation_norms=[])


def test_structured_requires_layer_scope() -> None:
    specs = [conv_layer("c", out_channels=2, in_channels=1, kernel_h=1, kernel_w=1)]
    model = {"c": [1.0, 2.0]}
    norms = {"c": [1.0]}
    with pytest.raises(ValueError, match="structured"):
        wanda_prune_model(
            specs, model, norms, density=0.5, scope="global", structure="filter"
        )


def test_zero_weights_stay_zero() -> None:
    weights = [0.0, 3.0]
    norms = [1e6, 0.01]  # scores [0, 0.03]
    pruned = wanda_prune_layer(weights, norms, density=0.5, shape=(1, 2))
    assert pruned[0] == 0.0
    assert pruned[1] == 3.0


def test_cli_wanda_json_column_norms() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([
            "wanda",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--activation-norms", "1,1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["total_kept"] == 2
    assert payload["scope"] == "layer"


def test_cli_wanda_markdown_mentions_wanda() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "wanda",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--activation-norms", "1,1,1,1",
            "--density", "0.5",
        ])
    assert code == 0
    assert "Wanda" in stdout.getvalue()


def test_score_shape_matches_dense_and_conv() -> None:
    dense = dense_layer("fc", in_features=5, out_features=3)
    n = layer_weight_count(dense)
    scores = wanda_scores([0.1] * n, [2.0] * 5, shape=dense.shape)
    assert len(scores) == n == 15
    assert scores == pytest.approx([0.2] * n)

    conv = conv_layer("conv", out_channels=2, in_channels=3, kernel_h=2, kernel_w=2)
    cn = layer_weight_count(conv)
    scores = wanda_scores([0.5] * cn, [4.0] * 3, shape=conv.shape)
    assert len(scores) == cn == 24
    assert scores == pytest.approx([2.0] * cn)
