"""Tests for Optimal Brain Surgeon (OBS) pruning (Hassibi & Stork)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    layer_weight_count,
    obs_prune_layer,
    obs_prune_model,
    obs_prune_summary,
    obs_scores,
)
from prune_kit.cli import main


def test_scores_with_hess_inv_diag_are_w_squared_over_two_hinv() -> None:
    weights = [2.0, -3.0, 0.5, -0.25]
    hinv = [1.0, 2.0, 4.0, 8.0]
    # w^2 / (2 * H^{-1})
    expected = [
        4.0 / 2.0,
        9.0 / 4.0,
        0.25 / 8.0,
        0.0625 / 16.0,
    ]
    assert obs_scores(weights, hinv) == pytest.approx(expected)


def test_scores_with_hess_diag_use_reciprocal_proxy() -> None:
    weights = [2.0, -1.0]
    hess = [1.0, 4.0]
    # Hinv ≈ 1/H (eps negligible), saliency = w^2 / (2/H) = w^2 * H / 2
    expected = [
        4.0 * 1.0 / 2.0,
        1.0 * 4.0 / 2.0,
    ]
    assert obs_scores(weights, hess_diag=hess) == pytest.approx(expected, rel=1e-9)


def test_scores_with_grads_use_squared_gradient_proxy() -> None:
    weights = [2.0, -1.0, 0.5]
    grads = [0.5, 2.0, -1.0]
    # H ≈ g^2, Hinv ≈ 1/g^2, saliency = w^2 * g^2 / 2
    expected = [
        4.0 * 0.25 / 2.0,
        1.0 * 4.0 / 2.0,
        0.25 * 1.0 / 2.0,
    ]
    assert obs_scores(weights, grads=grads) == pytest.approx(expected, rel=1e-9)


def test_keeps_highest_saliency() -> None:
    # Uniform H^{-1} => scores proportional to w^2
    weights = [1.0, 2.0, 3.0, 4.0]
    hinv = [1.0, 1.0, 1.0, 1.0]
    pruned = obs_prune_layer(weights, hinv, density=0.5)
    assert pruned == [0.0, 0.0, 3.0, 4.0]


def test_rejects_both_hess_inv_and_grads() -> None:
    with pytest.raises(ValueError, match="not both"):
        obs_scores([1.0], [1.0], grads=[1.0])


def test_rejects_hess_inv_and_hess_diag() -> None:
    with pytest.raises(ValueError, match="not both"):
        obs_scores([1.0], [1.0], hess_diag=[1.0])


def test_rejects_hess_diag_and_grads() -> None:
    with pytest.raises(ValueError, match="not both"):
        obs_scores([1.0], hess_diag=[1.0], grads=[1.0])


def test_requires_one_saliency_source() -> None:
    with pytest.raises(ValueError, match="required"):
        obs_scores([1.0, 2.0])


def test_density_rounding() -> None:
    weights = [0.1 * (i + 1) for i in range(8)]
    hinv = [1.0] * 8
    pruned = obs_prune_layer(weights, hinv, density=0.25)
    assert sum(v == 0.0 for v in pruned) == 6
    assert pruned[-2:] == pytest.approx([0.7, 0.8])


def test_layer_scope_independent() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [3.0, 1.0], "b": [0.4, 0.2]}
    hinv = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = obs_prune_model(
        specs, model, hinv, density=0.5, scope="layer"
    )
    assert pruned["a"] == [3.0, 0.0]
    assert pruned["b"] == [0.4, 0.0]


def test_global_scope_ranks_across_layers() -> None:
    specs = [
        dense_layer("a", in_features=2, out_features=1),
        dense_layer("b", in_features=2, out_features=1),
    ]
    model = {"a": [1.0, 0.1], "b": [0.5, 2.0]}
    hinv = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    pruned = obs_prune_model(
        specs, model, hinv, density=0.5, scope="global"
    )
    assert pruned["a"] == [1.0, 0.0]
    assert pruned["b"] == [0.0, 2.0]


def test_summary_counts() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    hinv = {"fc": [1.0, 1.0, 1.0, 1.0]}
    summary = obs_prune_summary(specs, model, hinv, density=0.5)
    assert summary["total"] == 4
    assert summary["total_kept"] == 2
    assert summary["overall_survival"] == pytest.approx(0.5)
    assert summary["scope"] == "global"
    assert summary["layers"][0]["kept_weights"] == 2
    # score_sum = (1+4+9+16)/2 = 15
    assert summary["layers"][0]["score_sum"] == pytest.approx(15.0)


def test_conv_shape_validated() -> None:
    conv = conv_layer("c", out_channels=2, in_channels=1, kernel_h=1, kernel_w=2)
    n = layer_weight_count(conv)
    weights = [float(i + 1) for i in range(n)]
    hinv = [1.0] * n
    scores = obs_scores(weights, hinv, shape=conv.shape)
    assert len(scores) == n
    pruned = obs_prune_layer(
        weights, hinv, density=0.5, shape=conv.shape
    )
    assert sum(v == 0.0 for v in pruned) == n // 2


def test_empty_weights_raise() -> None:
    with pytest.raises(ValueError):
        obs_prune_layer([], [1.0])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="hess_inv_diag"):
        obs_scores([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="hess_diag"):
        obs_scores([1.0, 2.0], hess_diag=[1.0])
    with pytest.raises(ValueError, match="grads"):
        obs_scores([1.0, 2.0], grads=[1.0])


def test_zero_weights_stay_zero() -> None:
    # score(0)=0, score(3)=9/2 — keep index 1
    pruned = obs_prune_layer(
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
    hinv = {
        "a": [1.0, 1.0, 1.0, 1.0],
        "b": [1.0, 1.0, 1.0, 1.0],
    }
    pruned = obs_prune_model(
        specs,
        model,
        hinv,
        density=0.5,
        scope="layer",
        per_layer={"a": 0.25, "b": 0.75},
    )
    assert sum(v != 0.0 for v in pruned["a"]) == 1
    assert sum(v != 0.0 for v in pruned["b"]) == 3


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        obs_scores([1.0, 2.0], [1.0, 1.0], shape=(2, 2))


def test_invalid_density_raises() -> None:
    with pytest.raises(ValueError, match="density"):
        obs_prune_layer([1.0, 2.0], [1.0, 1.0], density=0.0)
    with pytest.raises(ValueError, match="density"):
        obs_prune_layer([1.0, 2.0], [1.0, 1.0], density=1.5)


def test_invalid_scope_raises() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    hinv = {"fc": [1.0, 1.0]}
    with pytest.raises(ValueError, match="scope"):
        obs_prune_model(specs, model, hinv, scope="weird")


def test_per_layer_requires_layer_scope() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    hinv = {"fc": [1.0, 1.0]}
    with pytest.raises(ValueError, match="per_layer"):
        obs_prune_model(
            specs, model, hinv, scope="global", per_layer={"fc": 0.5}
        )


def test_model_rejects_multiple_sources() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    with pytest.raises(ValueError, match="not both"):
        obs_prune_model(
            specs, model, {"fc": [1.0, 1.0]}, grads={"fc": [1.0, 1.0]}
        )
    with pytest.raises(ValueError, match="not both"):
        obs_prune_model(
            specs,
            model,
            {"fc": [1.0, 1.0]},
            hess_diag={"fc": [1.0, 1.0]},
        )


def test_model_requires_saliency_source() -> None:
    specs = [dense_layer("fc", in_features=2, out_features=1)]
    model = {"fc": [1.0, 2.0]}
    with pytest.raises(ValueError, match="required"):
        obs_prune_model(specs, model)


def test_grads_proxy_model_prune() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    grads = {"fc": [1.0, 1.0, 1.0, 1.0]}
    pruned = obs_prune_model(specs, model, grads=grads, density=0.5)
    assert pruned["fc"] == [0.0, 0.0, 3.0, 4.0]


def test_hess_diag_proxy_model_prune() -> None:
    specs = [dense_layer("fc", in_features=4, out_features=1)]
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}
    hess = {"fc": [1.0, 1.0, 1.0, 1.0]}
    pruned = obs_prune_model(specs, model, hess_diag=hess, density=0.5)
    assert pruned["fc"] == [0.0, 0.0, 3.0, 4.0]


def test_cli_obs_json_with_hess_inv() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obs",
            "--specs", "fc=dense:1x4",
            "--weights", "1,2,3,4",
            "--hess-inv-diag", "1,1,1,1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["total"] == 4
    assert payload["total_kept"] == 2
    assert payload["overall_survival"] == pytest.approx(0.5)


def test_cli_obs_markdown_with_grads() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obs",
            "--specs", "fc=dense:1x4",
            "--weights", "1,2,3,4",
            "--grads", "1,1,1,1",
            "--density", "0.5",
        ])
    assert code == 0
    text = buf.getvalue()
    assert "OBS pruning" in text
    assert "fc" in text


def test_cli_obs_with_hess_diag() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obs",
            "--specs", "fc=dense:1x4",
            "--weights", "1,2,3,4",
            "--hess-diag", "1,1,1,1",
            "--density", "0.5",
            "--json",
        ])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["total_kept"] == 2


def test_cli_obs_requires_source() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obs",
            "--specs", "fc=dense:1x2",
            "--weights", "1,2",
            "--density", "0.5",
        ])
    assert code == 2
    assert "required" in err.getvalue().lower() or "required" in err.getvalue()


def test_cli_obs_rejects_both() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "obs",
            "--specs", "fc=dense:1x2",
            "--weights", "1,2",
            "--hess-inv-diag", "1,1",
            "--grads", "1,1",
        ])
    assert code == 2
    assert "not both" in err.getvalue()


def test_exports_in_package() -> None:
    import prune_kit as pk

    assert hasattr(pk, "obs_scores")
    assert hasattr(pk, "obs_prune_layer")
    assert hasattr(pk, "obs_prune_model")
    assert hasattr(pk, "obs_prune_summary")
