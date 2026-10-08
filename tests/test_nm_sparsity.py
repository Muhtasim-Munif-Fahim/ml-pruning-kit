"""Tests for N:M semi-structured sparsity."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    is_nm_sparse,
    nm_groups,
    nm_mask,
    nm_prune_layer,
    nm_prune_model,
    nm_prune_summary,
)
from prune_kit.cli import main


def test_two_four_keeps_two_largest_per_group() -> None:
    weights = [0.1, -0.9, 0.3, 0.2, 5.0, 0.0, -4.0, 1.0]
    assert nm_prune_layer(weights, 2, 4) == [0.0, -0.9, 0.3, 0.0, 5.0, 0.0, -4.0, 0.0]


def test_dense_groups_run_along_input_columns() -> None:
    # shape (out=2, in=4): one group per row
    assert nm_groups((2, 4), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert nm_groups((1, 8), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_conv_groups_run_along_input_channels() -> None:
    # shape (out=1, in=4, kh=1, kw=2): stride 2 between consecutive channels
    assert nm_groups((1, 4, 1, 2), 4) == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_conv_prune_is_compliant_and_half_dense() -> None:
    spec = conv_layer("c", 2, 4, 3, 3)
    weights = [((i * 37) % 11) - 5.0 for i in range(72)]
    pruned = nm_prune_layer(weights, 2, 4, shape=spec.shape)
    assert is_nm_sparse(pruned, 2, 4, shape=spec.shape)
    assert sum(1 for w in pruned if w != 0.0) <= 36
    assert not is_nm_sparse([1.0] * 72, 2, 4, shape=spec.shape)


def test_ties_break_toward_lower_index() -> None:
    assert nm_mask([1.0, 1.0, 1.0, 1.0], 2, 4) == [1, 1, 0, 0]


def test_custom_scores_override_magnitude() -> None:
    weights = [10.0, 9.0, 0.1, 0.2]
    scores = [0.0, 0.0, 1.0, 2.0]  # e.g. Wanda-style importance
    assert nm_prune_layer(weights, 2, 4, scores=scores) == [0.0, 0.0, 0.1, 0.2]


def test_other_patterns() -> None:
    weights = [float(i) for i in range(8)]
    assert nm_prune_layer(weights, 1, 2) == [0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0]
    assert nm_prune_layer(weights, 4, 8) == [0.0] * 4 + [4.0, 5.0, 6.0, 7.0]
    assert nm_prune_layer(weights, 0, 4) == [0.0] * 8
    assert nm_prune_layer(weights, 4, 4) == weights


def test_partial_group_handling() -> None:
    with pytest.raises(ValueError, match="divisible"):
        nm_groups((2, 6), 4)
    groups = nm_groups((1, 6), 4, allow_partial=True)
    assert groups == [[0, 1, 2, 3], [4, 5]]
    pruned = nm_prune_layer([1, 2, 3, 4, 5, 6], 2, 4, shape=(1, 6), allow_partial=True)
    assert pruned == [0.0, 0.0, 3.0, 4.0, 5.0, 6.0]


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        nm_prune_layer([], 2, 4)
    with pytest.raises(ValueError, match="n must"):
        nm_prune_layer([1.0] * 4, 5, 4)
    with pytest.raises(ValueError, match="m must"):
        nm_prune_layer([1.0] * 4, 0, 0)
    with pytest.raises(ValueError, match="integers"):
        nm_prune_layer([1.0] * 4, 1.5, 4)
    with pytest.raises(ValueError, match="shape"):
        nm_prune_layer([1.0] * 4, 2, 4, shape=(2, 4))
    with pytest.raises(ValueError, match="scores"):
        nm_prune_layer([1.0] * 4, 2, 4, scores=[1.0])


def test_model_prune_with_skip_and_scores() -> None:
    specs = [dense_layer("fc1", 4, 2), dense_layer("fc2", 4, 1)]
    model = {"fc1": [1.0, 2.0, 3.0, 4.0, -8.0, 7.0, 0.5, 0.1], "fc2": [1.0, 2.0, 3.0, 4.0]}
    pruned = nm_prune_model(specs, model, n=2, m=4, skip=["fc2"],
                            scores={"fc1": [9, 8, 0, 0, 0, 0, 1, 2]})
    assert pruned["fc1"] == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1]
    assert pruned["fc2"] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="unknown"):
        nm_prune_model(specs, model, skip=["nope"])
    with pytest.raises(ValueError, match="same layers"):
        nm_prune_model(specs, {"fc1": model["fc1"]})


def test_summary_reports_compliance() -> None:
    specs = [dense_layer("a", 8, 2), conv_layer("b", 1, 4, 1, 1)]
    model = {"a": [float(i + 1) for i in range(16)], "b": [4.0, 3.0, 2.0, 1.0]}
    summary = nm_prune_summary(specs, model, skip=["b"])
    assert summary["pattern"] == "2:4"
    assert summary["target_density"] == 0.5
    rows = {row["layer"]: row for row in summary["layers"]}
    assert rows["a"]["kept_weights"] == 8 and rows["a"]["groups"] == 4
    assert rows["a"]["nm_compliant"] is True
    assert rows["b"]["skipped"] is True and rows["b"]["kept_weights"] == 4
    assert summary["total_kept"] == 12
    assert summary["overall_survival"] == pytest.approx(0.6)


def test_cli_nm_json_and_markdown() -> None:
    weights = ",".join(str(float(i)) for i in range(16))
    out = io.StringIO()
    with redirect_stdout(out):
        code = main(["nm", "--specs", "fc1=dense:8x2", "--weights", weights, "--json"])
    assert code == 0
    data = json.loads(out.getvalue())
    assert data["total_kept"] == 8
    assert "masks" not in data

    out = io.StringIO()
    with redirect_stdout(out):
        code = main(["nm", "--n", "1", "--m", "4", "--specs", "fc1=dense:8x2",
                     "--weights", weights])
    assert code == 0
    assert "1:4" in out.getvalue()

    err = io.StringIO()
    with redirect_stderr(err):
        code = main(["nm", "--specs", "fc1=dense:6x1", "--weights", "1,2,3,4,5,6"])
    assert code == 2
    assert "divisible" in err.getvalue()
