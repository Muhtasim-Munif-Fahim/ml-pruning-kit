"""Tests for OWL outlier-weighed layer-wise sparsity."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    dense_layer,
    layer_outlier_ratio,
    owl_layer_densities,
    owl_outlier_ratios,
    owl_prune_model,
    owl_prune_summary,
    wanda_prune_model,
)
from prune_kit.cli import main


def _model():
    specs = [dense_layer("fc1", 8, 4), dense_layer("fc2", 8, 4), dense_layer("fc3", 4, 2)]
    # fc1: four strong outliers among 32 weights; fc2: smooth; fc3: one outlier in 8.
    fc1 = [1.0 + 0.01 * i for i in range(32)]
    for i in (0, 9, 18, 27):
        fc1[i] = 40.0
    fc2 = [1.0 + 0.02 * i for i in range(32)]
    fc3 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 30.0]
    model = {"fc1": fc1, "fc2": fc2, "fc3": fc3}
    norms = {"fc1": [1.0] * 8, "fc2": [1.0] * 8, "fc3": [1.0] * 4}
    return specs, model, norms


def test_layer_outlier_ratio():
    assert layer_outlier_ratio([1, 1, 1, 1, 1, 1, 1, 1, 1, 100], m=5) == pytest.approx(0.1)
    assert layer_outlier_ratio([0.0, 0.0], m=5) == 0.0
    assert layer_outlier_ratio([1.0, 1.0], m=0.5) == 1.0
    with pytest.raises(ValueError):
        layer_outlier_ratio([], m=5)
    with pytest.raises(ValueError):
        layer_outlier_ratio([1.0], m=0)


def test_outlier_ratios_use_wanda_scores():
    specs, model, norms = _model()
    ratios = owl_outlier_ratios(specs, model, norms, m=3.0)
    assert ratios["fc1"] == pytest.approx(4 / 32)
    assert ratios["fc2"] == 0.0
    assert ratios["fc3"] == pytest.approx(1 / 8)
    # Activation norms change the outlier picture: boosting one fc2 column.
    norms2 = dict(norms, fc2=[1.0] * 7 + [50.0])
    assert owl_outlier_ratios(specs, model, norms2, m=3.0)["fc2"] > 0


def test_layer_densities_formula():
    specs, _, _ = _model()
    ratios = {"fc1": 0.2, "fc2": 0.0, "fc3": 0.1}
    dens = owl_layer_densities(specs, ratios, density=0.5, lam=0.1)
    # scaled = [0.2, 0.0, 0.1], mean 0.1 -> densities 0.6, 0.4, 0.5
    assert dens == pytest.approx({"fc1": 0.6, "fc2": 0.4, "fc3": 0.5})
    assert sum(dens.values()) / 3 == pytest.approx(0.5)


def test_weighted_densities_hit_overall_target():
    specs, _, _ = _model()
    ratios = {"fc1": 0.2, "fc2": 0.0, "fc3": 0.1}
    dens = owl_layer_densities(specs, ratios, density=0.5, lam=0.1, weighted=True)
    sizes = {"fc1": 32, "fc2": 32, "fc3": 8}
    overall = sum(dens[n] * sizes[n] for n in sizes) / sum(sizes.values())
    assert overall == pytest.approx(0.5)
    assert dens["fc1"] - dens["fc2"] == pytest.approx(0.2)


def test_uniform_ratios_reduce_to_wanda():
    specs, model, norms = _model()
    dens = owl_layer_densities(specs, {"fc1": 0.1, "fc2": 0.1, "fc3": 0.1}, density=0.3)
    assert list(dens.values()) == pytest.approx([0.3, 0.3, 0.3])
    zero_lam = owl_prune_model(specs, model, norms, density=0.5, lam=0.0)
    assert zero_lam == wanda_prune_model(specs, model, norms, density=0.5)


def test_densities_clip_and_validate():
    specs, _, _ = _model()
    dens = owl_layer_densities(specs, {"fc1": 1.0, "fc2": 0.0, "fc3": 0.0}, density=0.05, lam=0.5)
    assert all(0.0 < d <= 1.0 for d in dens.values())
    assert dens["fc2"] < 1e-9
    with pytest.raises(ValueError):
        owl_layer_densities(specs, {"fc1": 0.1}, density=0.5)
    with pytest.raises(ValueError):
        owl_layer_densities(specs, {"fc1": 0, "fc2": 0, "fc3": 0}, lam=0.6)
    with pytest.raises(ValueError):
        owl_layer_densities(specs, {"fc1": 0, "fc2": 0, "fc3": 0}, density=0.0)


def test_owl_keeps_more_in_outlier_heavy_layer():
    specs, model, norms = _model()
    summary = owl_prune_summary(specs, model, norms, density=0.5, m=3.0, lam=0.1)
    rows = {row["layer"]: row for row in summary["layers"]}
    assert rows["fc1"]["owl_density"] > rows["fc2"]["owl_density"]
    assert rows["fc1"]["kept_weights"] > rows["fc2"]["kept_weights"]
    assert summary["method"] == "owl"
    assert summary["outlier_ratios"]["fc1"] == pytest.approx(rows["fc1"]["outlier_ratio"])
    # Outliers always survive and pruned weights are a masked copy.
    pruned = summary["pruned_weights"]
    for i in (0, 9, 18, 27):
        assert pruned["fc1"][i] == 40.0
    for name in model:
        for orig, new, bit in zip(model[name], pruned[name], summary["masks"][name]):
            assert new == (orig if bit else 0.0)


def test_owl_with_contributions():
    specs, model, norms = _model()
    contributions = {n: [abs(w) for w in model[n]] for n in model}
    a = owl_prune_model(specs, model, contributions=contributions, density=0.5, m=3.0)
    b = owl_prune_model(specs, model, {"fc1": [1.0] * 8, "fc2": [1.0] * 8, "fc3": [1.0] * 4}, density=0.5, m=3.0)
    assert a == b
    with pytest.raises(ValueError):
        owl_prune_model(specs, model, norms, contributions=contributions)


def _cli_args():
    specs, model, _ = _model()
    flat = model["fc1"] + model["fc2"] + model["fc3"]
    return [
        "--specs", "fc1=dense:4x8,fc2=dense:4x8,fc3=dense:2x4",
        "--weights", ",".join(str(v) for v in flat),
        "--activation-norms", ",".join("1" for _ in flat),
        "--density", "0.5", "-m", "3", "--lam", "0.1",
    ]


def test_cli_owl_json_and_markdown():
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["owl", *_cli_args(), "--json"]) == 0
    data = json.loads(out.getvalue())
    assert data["method"] == "owl" and "pruned_weights" not in data
    layers = {row["layer"]: row for row in data["layers"]}
    assert layers["fc1"]["owl_density"] > layers["fc2"]["owl_density"]
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["owl", *_cli_args()]) == 0
    assert "OWL outlier-weighed" in out.getvalue() and "| fc1 |" in out.getvalue()


def test_cli_owl_bad_input():
    err = io.StringIO()
    with redirect_stderr(err):
        assert main(["owl", *_cli_args(), "--lam", "0.9"]) == 2
    assert "owl:" in err.getvalue()
