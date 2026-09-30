"""Tests for Movement pruning (Sanh et al.)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    layer_weight_count,
    movement_prune_layer,
    movement_prune_model,
    movement_prune_summary,
    movement_scores,
)
from prune_kit.cli import main


def test_scores_are_abs_delta_from_init() -> None:
    weights = [2.0, -3.0, 0.5, 4.0]
    initial = [1.0, -1.0, 0.5, 0.0]
    # |1|, |2|, |0|, |4|
    assert movement_scores(weights, initial) == pytest.approx([1.0, 2.0, 0.0, 4.0])


def test_keeps_highest_movement() -> None:
    # movements 0.1, 0.2, 0.3, 0.4 — keep top half
    weights = [1.1, 1.2, 1.3, 1.4]
    initial = [1.0, 1.0, 1.0, 1.0]
    pruned = movement_prune_layer(weights, initial, density=0.5)
    assert pruned == [0.0, 0.0, 1.3, 1.4]


def test_small_movement_pruned_first() -> None:
    # weight that barely moved is pruned despite large magnitude
    weights = [10.0, 0.5]
    initial = [9.9, 0.0]  # movements 0.1, 0.5
    pruned = movement_prune_layer(weights, initial, density=0.5, shape=(1, 2))
    assert pruned == [0.0, 0.5]


def test_movements_replace_delta() -> None:
    assert movement_scores(
        [4.0, 0.2], movements=[0.0, -3.0]
    ) == pytest.approx([0.0, 3.0])


def test_rejects_both_initial_and_movements() -> None:
    with pytest.raises(ValueError, match="not both"):
        movement_scores([1.0], [1.0], movements=[1.0])


def test_requires_saliency() -> None:
    with pytest.raises(ValueError, match="required"):
        movement_scores([1.0])


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    initial = [0.0] * 8  # movement == |w|
    pruned = movement_prune_layer(weights, initial, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    initial = {"a": [0.0, 0.0], "b": [0.0, 0.0]}
    pruned = movement_prune_model(
        specs, model, initial, density=0.5, scope="layer"
    )
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    initial = {"a": [0.0, 0.0], "b": [0.0, 0.0]}
    pruned = movement_prune_model(
        specs, model, initial, density=0.5, scope="global"
    )
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    initial = {"fc": [0.0, 0.0, 0.0, 0.0]}
    summary = movement_prune_summary(specs, model, initial, density=0.5)
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
    initial = [0.0] * n
    scores = movement_scores(weights, initial, shape=conv.shape)
    assert len(scores) == n
    pruned = movement_prune_layer(
        weights, initial, density=0.5, shape=conv.shape
    )
    assert sum(v == 0.0 for v in pruned) == n // 2


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        movement_prune_layer([], [])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="initial_weights"):
        movement_scores([1.0, 2.0], [1.0])


def test_zero_weights_stay_zero() -> None:
    # Large movement on an already-zero weight: kept slot stays 0.0
    pruned = movement_prune_layer(
        [0.0, 3.0], [-5.0, 2.9], density=0.5, shape=(1, 2)
    )
    # movements 5.0, 0.1 — keep index 0; stored value was already zero
    assert pruned == [0.0, 0.0]


def test_sign_of_delta_ignored() -> None:
    pos = movement_scores([1.0, 2.0], [0.0, 0.0])
    neg = movement_scores([-1.0, -2.0], [0.0, 0.0])
    assert pos == pytest.approx(neg)


def test_per_layer_override() -> None:
    specs = [
        dense_layer("a", in_features=4, out_features=1),
        dense_layer("b", in_features=4, out_features=1),
    ]
    model = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
    }
    initial = {
        "a": [0.0, 0.0, 0.0, 0.0],
        "b": [0.0, 0.0, 0.0, 0.0],
    }
    pruned = movement_prune_model(
        specs,
        model,
        initial,
        density=0.5,
        scope="layer",
        per_layer={"a": 0.25, "b": 0.75},
    )
    assert sum(v != 0.0 for v in pruned["a"]) == 1
    assert sum(v != 0.0 for v in pruned["b"]) == 3


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        movement_scores([1.0, 2.0], [0.0, 0.0], shape=(2, 2))


def test_cli_movement_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([
            "movement",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--initial-weights", "0,0,0,0",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["total_kept"] == 2
    assert payload["scope"] == "global"


def test_cli_movement_markdown_mentions_movement() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "movement",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--initial-weights", "0,0,0,0",
            "--density", "0.5",
            "--scope", "layer",
        ])
    assert code == 0
    text = stdout.getvalue()
    assert "Movement" in text


def test_cumulative_movements_path() -> None:
    # Precomputed cumulative |ΔW| prefers the heavily moved weight
    weights = [1.0, 1.0]
    pruned = movement_prune_layer(
        weights, movements=[0.01, 5.0], density=0.5, shape=(1, 2)
    )
    assert pruned == [0.0, 1.0]
