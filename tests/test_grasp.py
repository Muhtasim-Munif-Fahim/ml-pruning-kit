"""Tests for GraSP gradient-signal-preservation pruning."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    grasp_prune_layer,
    grasp_prune_model,
    grasp_prune_summary,
    grasp_scores,
    layer_weight_count,
    mask_density,
)
from prune_kit.cli import main


def test_scores_are_negated_weight_hg_products() -> None:
    weights = [2.0, -3.0, 0.5, 4.0]
    hg = [0.5, -1.0, 2.0, 0.25]
    assert grasp_scores(weights, hg) == pytest.approx([-1.0, -3.0, -1.0, -1.0])


def test_contributions_are_negated() -> None:
    assert grasp_scores([1.0, 2.0], contributions=[0.5, -0.25]) == pytest.approx([-0.5, 0.25])


def test_keeps_highest_scores() -> None:
    # scores = -w*hg => [ -1, -2, -3, -4 ] when w=hg=1,2,3,4
    weights = [1.0, 2.0, 3.0, 4.0]
    hg = [1.0, 1.0, 1.0, 1.0]
    pruned = grasp_prune_layer(weights, hg, density=0.5)
    # highest scores are -1 and -2 (indices 0,1)
    assert pruned == [1.0, 2.0, 0.0, 0.0]


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    hg = [-1.0] * 8  # scores = -w*(-1) = w, so keep largest weights
    pruned = grasp_prune_layer(weights, hg, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    hg = {"a": [-1.0, -1.0], "b": [-1.0, -1.0]}  # scores = w
    pruned = grasp_prune_model(specs, model, hg, density=0.5, scope="global")
    # top half of {1.0, 0.1, 0.5, 2.0} => keep 2.0 and 1.0
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    hg = {"a": [-1.0, -1.0], "b": [-1.0, -1.0]}
    pruned = grasp_prune_model(specs, model, hg, density=0.5, scope="layer")
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    hg = {"fc": [-1.0, -1.0, -1.0, -1.0]}
    summary = grasp_prune_summary(specs, model, hg, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["layers"][0]["kept_weights"] == 2


def test_rejects_both_hg_and_contributions() -> None:
    with pytest.raises(ValueError):
        grasp_scores([1.0], hg=[1.0], contributions=[1.0])


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        grasp_prune_layer([], hg=[])


def test_cli_grasp_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([
            "grasp",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--hg=-1,-1,-1,-1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["total_kept"] == 2
    assert payload["scope"] == "global"


def test_cli_grasp_markdown_mentions_grasp() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = main([
            "grasp",
            "--specs", "fc1=dense:2x2",
            "--weights", "1,2,3,4",
            "--hg=-1,-1,-1,-1",
            "--density", "0.5",
        ])
    assert code == 0
    assert "GraSP" in stdout.getvalue()
