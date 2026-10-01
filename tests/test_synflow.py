"""Tests for SynFlow synaptic-flow pruning (Tanaka et al.)."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    layer_weight_count,
    synflow_prune_layer,
    synflow_prune_model,
    synflow_prune_summary,
    synflow_scores,
    synflow_unit_grads,
)
from prune_kit.cli import main


def test_scores_with_grads_are_abs_w_times_abs_grad() -> None:
    weights = [2.0, -3.0, 0.5, -0.25]
    grads = [0.5, -1.0, 2.0, 4.0]
    # |2|*|0.5|, |3|*|1|, |0.5|*|2|, |0.25|*|4|
    assert synflow_scores(weights, grads) == pytest.approx([1.0, 3.0, 1.0, 1.0])


def test_exponential_proxy_row_scale() -> None:
    # shape (2, 2): row0 sum=0.3, row1 sum=1.2
    weights = [0.1, 0.2, 0.4, 0.8]
    scores = synflow_scores(weights, shape=(2, 2))
    s0 = math.exp(0.3)
    s1 = math.exp(1.2)
    assert scores == pytest.approx([0.1 * s0, 0.2 * s0, 0.4 * s1, 0.8 * s1])


def test_keeps_highest_synflow() -> None:
    weights = [1.0, 2.0, 3.0, 4.0]
    grads = [1.0, 1.0, 1.0, 1.0]  # scores == |W|
    pruned = synflow_prune_layer(weights, grads, density=0.5)
    assert pruned == [0.0, 0.0, 3.0, 4.0]


def test_contributions_replace_product() -> None:
    assert synflow_scores(
        [4.0, 0.2], contributions=[0.0, -3.0]
    ) == pytest.approx([0.0, 3.0])


def test_rejects_both_grads_and_contributions() -> None:
    with pytest.raises(ValueError, match="not both"):
        synflow_scores([1.0], [1.0], contributions=[1.0])


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    grads = [1.0] * 8
    pruned = synflow_prune_layer(weights, grads, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    grads = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = synflow_prune_model(
        specs, model, grads, density=0.5, scope="layer"
    )
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    grads = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = synflow_prune_model(
        specs, model, grads, density=0.5, scope="global"
    )
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    grads = {"fc": [1.0, 1.0, 1.0, 1.0]}
    summary = synflow_prune_summary(specs, model, grads, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["scope"] == "global"
    assert summary["layers"][0]["kept_weights"] == 2
    assert summary["layers"][0]["score_sum"] == pytest.approx(10.0)


def test_conv_shape_validated() -> None:
    conv = conv_layer("c", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2)
    n = layer_weight_count(conv)
    weights = [float(i + 1) for i in range(n)]
    grads = [1.0] * n
    scores = synflow_scores(weights, grads, shape=conv.shape)
    assert len(scores) == n
    pruned = synflow_prune_layer(
        weights, grads, density=0.5, shape=conv.shape
    )
    assert sum(v == 0.0 for v in pruned) == n // 2


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        synflow_prune_layer([])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="grads"):
        synflow_scores([1.0, 2.0], [1.0])


def test_zero_weights_stay_zero() -> None:
    pruned = synflow_prune_layer(
        [0.0, 3.0], [5.0, 0.1], density=0.5, shape=(1, 2)
    )
    # scores 0, 0.3 — keep index 1
    assert pruned == [0.0, 3.0]
    pruned_keep_zero = synflow_prune_layer(
        [0.0, 0.1], [10.0, 0.01], density=0.5, shape=(1, 2)
    )
    # scores 0, 0.001 — keep index 0 (higher? wait 0 vs 0.001 — keep index 1)
    # Actually 0 * 10 = 0, 0.1 * 0.01 = 0.001 — keep index 1
    assert pruned_keep_zero == [0.0, 0.1]


def test_per_layer_override() -> None:
    specs = [
        dense_layer("a", in_features=4, out_features=1),
        dense_layer("b", in_features=4, out_features=1),
    ]
    model = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
    }
    grads = {
        "a": [1.0, 1.0, 1.0, 1.0],
        "b": [1.0, 1.0, 1.0, 1.0],
    }
    pruned = synflow_prune_model(
        specs,
        model,
        grads,
        density=0.5,
        scope="layer",
        per_layer={"a": 0.25, "b": 0.75},
    )
    assert sum(v != 0.0 for v in pruned["a"]) == 1
    assert sum(v != 0.0 for v in pruned["b"]) == 3


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        synflow_scores([1.0, 2.0], [1.0, 1.0], shape=(2, 2))


def test_unit_grads_two_layer_chain() -> None:
    # W1 (2, 2), W2 (1, 2) — chains
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        dense_layer("fc2", in_features=2, out_features=1),
    ]
    model = {
        "fc1": [1.0, 0.0, 0.0, 1.0],  # identity abs
        "fc2": [1.0, 1.0],
    }
    grads = synflow_unit_grads(specs, model)
    # x0 = [1, 1]; x1 = [1, 1]; x2 = [2]
    # g2 = [1]; g1 = |W2|^T g2 = [1, 1]; g0 = |W1|^T g1 = [1, 1]
    # ∂R/∂W2 = g2_i * x1_j => [1*1, 1*1] = [1, 1]
    # ∂R/∂W1 row0: g1_0 * x0 = [1, 1]; row1: [1, 1]
    assert grads["fc2"] == pytest.approx([1.0, 1.0])
    assert grads["fc1"] == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_model_auto_unit_grads_when_chained() -> None:
    specs = [
        dense_layer("fc1", in_features=2, out_features=2),
        dense_layer("fc2", in_features=2, out_features=1),
    ]
    model = {
        "fc1": [2.0, 0.1, 0.1, 2.0],
        "fc2": [3.0, 0.05],
    }
    pruned = synflow_prune_model(specs, model, density=0.5, scope="global")
    assert sum(1 for layer in pruned.values() for v in layer if v != 0.0) == 3
    # fc2's large weight and fc1's large diagonal should survive preferentially
    assert pruned["fc2"][0] == 3.0


def test_model_exponential_fallback_when_unchained() -> None:
    specs = [
        dense_layer("a", in_features=3, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 2.0, 3.0], "b": [4.0, 0.1]}
    # No chain (3 != previous out 1); still works via exponential proxy
    pruned = synflow_prune_model(specs, model, density=0.5, scope="layer")
    assert sum(v != 0.0 for v in pruned["a"]) == 2  # round(0.5*3)=2
    assert sum(v != 0.0 for v in pruned["b"]) == 1


def test_proxy_differs_from_pure_magnitude_ranking() -> None:
    # Two rows: small weights in a huge row beat a lone large weight in a tiny row?
    # Row0: [0.5, 0.5] sum=1 -> scale e^1; scores 0.5e, 0.5e
    # Row1: [0.9, 0.0] sum=0.9 -> scale e^0.9; scores 0.9*e^0.9, 0
    # 0.5*e^1 ≈ 1.359 vs 0.9*e^0.9 ≈ 2.213 — keep 0.9 first
    # Make row0 massive so small weights outrank a larger weight in a weak row:
    # Row0: [0.4, 0.4, 0.4, 0.4] sum=1.6
    # Row1: [1.0, 0, 0, 0] sum=1.0
    # score 0.4*exp(1.6)≈1.98 > 1.0*exp(1.0)≈2.718? No still less.
    # Row0 sum even larger: [0.5]*4 sum=2.0 -> 0.5*e^2≈3.69 > 1.0*e^1≈2.72
    weights = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]
    scores = synflow_scores(weights, shape=(2, 4))
    # Highest should be the 0.5s (proxy), not the 1.0 (magnitude winner)
    assert scores[0] > scores[4]
    pruned = synflow_prune_layer(weights, density=0.5, shape=(2, 4))
    # Keep 4 of 8 — all four 0.5s (higher proxy scores)
    assert pruned[:4] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert pruned[4:] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_cli_synflow_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([
            "synflow",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--grads", "1,1,1,1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["total_kept"] == 2
    assert payload["scope"] == "global"


def test_cli_synflow_proxy_without_grads() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "synflow",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--density", "0.5",
            "--scope", "layer",
        ])
    assert code == 0
    text = stdout.getvalue()
    assert "SynFlow" in text


def test_cli_synflow_markdown_mentions_synflow() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "synflow",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--grads", "1,1,1,1",
            "--density", "0.5",
        ])
    assert code == 0
    assert "SynFlow" in stdout.getvalue()


def test_unit_grads_unchained_raises() -> None:
    specs = [
        dense_layer("a", in_features=3, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0]}
    with pytest.raises(ValueError, match="chained"):
        synflow_unit_grads(specs, model)
