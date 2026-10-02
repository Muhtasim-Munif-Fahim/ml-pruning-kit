"""Tests for Optimal Brain Damage (OBD) pruning (LeCun et al.)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    layer_weight_count,
    obd_prune_layer,
    obd_prune_model,
    obd_prune_summary,
    obd_scores,
)
from prune_kit.cli import main


def test_scores_with_hess_diag_are_half_h_w_squared() -> None:
    weights = [2.0, -3.0, 0.5, -0.25]
    hess = [1.0, 2.0, 4.0, 8.0]
    # 0.5 * H * w^2
    expected = [
        0.5 * 1.0 * 4.0,
        0.5 * 2.0 * 9.0,
        0.5 * 4.0 * 0.25,
        0.5 * 8.0 * 0.0625,
    ]
    assert obd_scores(weights, hess) == pytest.approx(expected)


def test_scores_with_grads_use_squared_gradient_proxy() -> None:
    weights = [2.0, -1.0, 0.5]
    grads = [0.5, 2.0, -1.0]
    # H_ii ≈ g^2 => 0.5 * g^2 * w^2
    expected = [
        0.5 * 0.25 * 4.0,
        0.5 * 4.0 * 1.0,
        0.5 * 1.0 * 0.25,
    ]
    assert obd_scores(weights, grads=grads) == pytest.approx(expected)


def test_keeps_highest_saliency() -> None:
    # Uniform Hessian => scores proportional to w^2
    weights = [1.0, 2.0, 3.0, 4.0]
    hess = [1.0, 1.0, 1.0, 1.0]
    pruned = obd_prune_layer(weights, hess, density=0.5)
    assert pruned == [0.0, 0.0, 3.0, 4.0]


def test_rejects_both_hess_and_grads() -> None:
    with pytest.raises(ValueError, match="not both"):
        obd_scores([1.0], [1.0], grads=[1.0])


def test_requires_hess_or_grads() -> None:
    with pytest.raises(ValueError, match="required"):
        obd_scores([1.0, 2.0])


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    hess = [1.0] * 8
    pruned = obd_prune_layer(weights, hess, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    hess = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = obd_prune_model(
        specs, model, hess, density=0.5, scope="layer"
    )
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    # Uniform H => rank by |w|
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    hess = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = obd_prune_model(
        specs, model, hess, density=0.5, scope="global"
    )
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    hess = {"fc": [1.0, 1.0, 1.0, 1.0]}
    summary = obd_prune_summary(specs, model, hess, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["scope"] == "global"
    assert summary["layers"][0]["kept_weights"] == 2
    # score_sum = 0.5*(1+4+9+16) = 15
    assert summary["layers"][0]["score_sum"] == pytest.approx(15.0)


def test_conv_shape_validated() -> None:
    conv = conv_layer("c", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2)
    n = layer_weight_count(conv)
    weights = [float(i + 1) for i in range(n)]
    hess = [1.0] * n
    scores = obd_scores(weights, hess, shape=conv.shape)
    assert len(scores) == n
    pruned = obd_prune_layer(
        weights, hess, density=0.5, shape=conv.shape
    )
    assert sum(v == 0.0 for v in pruned) == n // 2


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        obd_prune_layer([], [1.0])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="hess_diag"):
        obd_scores([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="grads"):
        obd_scores([1.0, 2.0], grads=[1.0])


def test_zero_weights_stay_zero() -> None:
    # score(0)=0, score(3)=0.5*1*9=4.5 — keep index 1
    pruned = obd_prune_layer(
        [0.0, 3.0], [1.0, 1.0], density=0.5, shape=(1, 2)
    )
    assert pruned == [0.0, 3.0]


def test_per_layer_override() -> None:
    specs = [
        dense_layer("a", in_features=4, out_features=1),
        dense_layer("b", in_features=4, out_features=1),
    ]
    model = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
    }
    hess = {
        "a": [1.0, 1.0, 1.0, 1.0],
        "b": [1.0, 1.0, 1.0, 1.0],
    }
    pruned = obd_prune_model(
        specs,
        model,
        hess,
        density=0.5,
        scope="layer",
        per_layer={"a": 0.25, "b": 0.75},
    )
    assert sum(v != 0.0 for v in pruned["a"]) == 1
    assert sum(v != 0.0 for v in pruned["b"]) == 3


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        obd_scores([1.0, 2.0], [1.0, 1.0], shape=(2, 2))


def test_invalid_density_raises() -> None:
    with pytest.raises(ValueError, match="density"):
        obd_prune_layer([1.0, 2.0], [1.0, 1.0], density=0.0)
    with pytest.raises(ValueError, match="density"):
        obd_prune_layer([1.0, 2.0], [1.0, 1.0], density=1.5)


def test_invalid_scope_raises() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    hess = {"fc": [1.0, 1.0]}
    with pytest.raises(ValueError, match="scope"):
        obd_prune_model(specs, model, hess, scope="weird")


def test_per_layer_requires_layer_scope() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    hess = {"fc": [1.0, 1.0]}
    with pytest.raises(ValueError, match="per_layer"):
        obd_prune_model(
            specs, model, hess, scope="global", per_layer={"fc": 0.5}
        )


def test_model_rejects_both_hess_and_grads() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    with pytest.raises(ValueError, match="not both"):
        obd_prune_model(
            specs, model, {"fc": [1.0, 1.0]}, grads={"fc": [1.0, 1.0]}
        )


def test_model_requires_hess_or_grads() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    with pytest.raises(ValueError, match="required"):
        obd_prune_model(specs, model)


def test_grads_proxy_model_prune() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    # g=1 => same ranking as uniform H=1
    grads = {"fc": [1.0, 1.0, 1.0, 1.0]}
    pruned = obd_prune_model(specs, model, grads=grads, density=0.5)
    assert pruned["fc"] == [0.0, 0.0, 3.0, 4.0]


def test_cli_obd_json_with_hess() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obd",
            "--specs", "fc=dense:1x4",
            "--weights", "1,2,3,4",
            "--hess-diag", "1,1,1,1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["total"] == 4
    assert payload["total_kept"] == 2
    assert payload["overall_survival"] == pytest.approx(0.5)


def test_cli_obd_markdown_with_grads() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obd",
            "--specs", "fc=dense:1x4",
            "--weights", "1,2,3,4",
            "--grads", "1,1,1,1",
            "--density", "0.5",
        ])
    assert code == 0
    text = buf.getvalue()
    assert "OBD pruning" in text
    assert "fc" in text


def test_cli_obd_requires_hess_or_grads() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obd",
            "--specs", "fc=dense:1x2",
            "--weights", "1,2",
            "--density", "0.5",
        ])
    assert code == 2
    assert "required" in err.getvalue().lower() or "required" in err.getvalue()


def test_cli_obd_rejects_both() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obd",
            "--specs", "fc=dense:1x2",
            "--weights", "1,2",
            "--hess-diag", "1,1",
            "--grads", "1,1",
        ])
    assert code == 2
    assert "not both" in err.getvalue()


def test_exports_in_package() -> None:
    import prune_kit as pk

    assert hasattr(pk, "obd_scores")
    assert hasattr(pk, "obd_prune_layer")
    assert hasattr(pk, "obd_prune_model")
    assert hasattr(pk, "obd_prune_summary")
