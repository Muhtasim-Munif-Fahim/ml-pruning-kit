"""Tests for Layer-Adaptive Magnitude Pruning (LAMP)."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    frobenius_norm,
    lamp_prune_layer,
    lamp_prune_model,
    lamp_prune_summary,
    lamp_scores,
    layer_weight_count,
)
from prune_kit.cli import main


def test_classic_lamp_scores_ascending_cumulative() -> None:
    # weights [1, 2, 3]: squares 1, 4, 9
    # ascending order indices 0,1,2
    # score[0] = 1/(1+4+9) = 1/14
    # score[1] = 4/(4+9) = 4/13
    # score[2] = 9/9 = 1
    scores = lamp_scores([1.0, 2.0, 3.0], mode="lamp")
    assert scores == pytest.approx([1.0 / 14.0, 4.0 / 13.0, 1.0])


def test_frobenius_scores_divide_by_norm() -> None:
    weights = [3.0, 4.0]  # ||W||_F = 5
    scores = lamp_scores(weights, mode="frobenius")
    assert scores == pytest.approx([0.6, 0.8])
    assert frobenius_norm(weights) == pytest.approx(5.0)


def test_zero_layer_frobenius_yields_zeros() -> None:
    assert lamp_scores([0.0, 0.0], mode="frobenius") == [0.0, 0.0]


def test_keeps_highest_lamp_scores() -> None:
    # With equal-ish magnitudes the largest |w| has the highest LAMP score.
    weights = [0.1, 0.2, 0.3, 0.4]
    pruned = lamp_prune_layer(weights, density=0.5, mode="lamp")
    assert pruned == [0.0, 0.0, 0.3, 0.4]


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    pruned = lamp_prune_layer(weights, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_global_scope_adapts_across_layers() -> None:
    # Small-magnitude layer vs large-magnitude layer: classic LAMP global
    # ranking should prune more from the small layer (layer-adaptive).
    specs = [
        dense_layer("small", in_features=4, out_features=1),
        dense_layer("large", in_features=4, out_features=1),
    ]
    model = {
        "small": [0.1, 0.2, 0.3, 0.4],
        "large": [1.0, 2.0, 3.0, 4.0],
    }
    pruned = lamp_prune_model(specs, model, density=0.5, scope="global")
    small_kept = sum(1 for v in pruned["small"] if v != 0.0)
    large_kept = sum(1 for v in pruned["large"] if v != 0.0)
    assert small_kept + large_kept == 4
    # Large layer should keep at least as many as the small layer.
    assert large_kept >= small_kept


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    pruned = lamp_prune_model(specs, model, density=0.5, scope="layer")
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_frobenius_global_prefers_relative_large() -> None:
    # Within each layer |w|/||W||_F; globally rank those.
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 4.0], "b": [0.3, 0.4]}  # same relative scores
    scores_a = lamp_scores(model["a"], mode="frobenius")
    scores_b = lamp_scores(model["b"], mode="frobenius")
    assert scores_a == pytest.approx(scores_b)


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    summary = lamp_prune_summary(specs, model, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["mode"] == "lamp"
    assert summary["scope"] == "global"
    assert summary["layers"][0]["kept_weights"] == 2
    assert summary["layers"][0]["frobenius_norm"] == pytest.approx(math.sqrt(30.0))


def test_conv_shape_validated() -> None:
    conv = conv_layer("c", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2)
    n = layer_weight_count(conv)
    weights = [float(i + 1) for i in range(n)]
    scores = lamp_scores(weights, shape=conv.shape)
    assert len(scores) == n
    pruned = lamp_prune_layer(weights, density=0.5, shape=conv.shape)
    assert sum(v == 0.0 for v in pruned) == n // 2


def test_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        lamp_scores([1.0], mode="nope")


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        lamp_prune_layer([])


def test_zero_weights_stay_zero() -> None:
    weights = [0.0, 3.0, 0.0, 4.0]
    pruned = lamp_prune_layer(weights, density=0.5)
    # Keep the two non-zeros (highest LAMP); zeros stay zero even if kept.
    assert pruned[0] == 0.0
    assert pruned[2] == 0.0
    assert pruned[1] == 3.0
    assert pruned[3] == 4.0


def test_sign_ignored_in_scores() -> None:
    pos = lamp_scores([1.0, -2.0, 3.0], mode="lamp")
    neg = lamp_scores([-1.0, 2.0, -3.0], mode="lamp")
    assert pos == pytest.approx(neg)


def test_cli_lamp_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([
            "lamp",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["total_kept"] == 2
    assert payload["scope"] == "global"
    assert payload["mode"] == "lamp"


def test_cli_lamp_markdown_mentions_lamp() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "lamp",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--density", "0.5",
            "--mode", "frobenius",
            "--scope", "layer",
        ])
    assert code == 0
    text = stdout.getvalue()
    assert "LAMP" in text
    assert "frobenius" in text


def test_per_layer_override() -> None:
    specs = [
        dense_layer("a", in_features=4, out_features=1),
        dense_layer("b", in_features=4, out_features=1),
    ]
    model = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
    }
    pruned = lamp_prune_model(
        specs,
        model,
        density=0.5,
        scope="layer",
        per_layer={"a": 0.25, "b": 0.75},
    )
    assert sum(v != 0.0 for v in pruned["a"]) == 1
    assert sum(v != 0.0 for v in pruned["b"]) == 3


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        lamp_scores([1.0, 2.0], shape=(2, 2))
