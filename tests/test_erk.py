"""Tests for ER / ERK layer-wise sparsity allocation."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    erk_densities,
    erk_prune_model,
    erk_prune_summary,
    layer_weight_count,
)
from prune_kit.cli import main


def _weighted_density(specs, densities):
    total = sum(layer_weight_count(s) for s in specs)
    return sum(layer_weight_count(s) * densities[s.name] for s in specs) / total


def _specs():
    return [
        conv_layer("conv1", 16, 3, 3, 3),
        conv_layer("conv2", 32, 16, 3, 3),
        dense_layer("fc1", in_features=512, out_features=128),
        dense_layer("fc2", in_features=128, out_features=10),
    ]


@pytest.mark.parametrize("method", ["erk", "er", "uniform"])
@pytest.mark.parametrize("density", [0.05, 0.2, 0.5, 0.9])
def test_budget_is_matched(method, density):
    specs = _specs()
    dens = erk_densities(specs, density=density, method=method)
    assert set(dens) == {s.name for s in specs}
    assert all(0.0 < d <= 1.0 for d in dens.values())
    assert _weighted_density(specs, dens) == pytest.approx(density, rel=1e-9)


def test_erk_formula_for_two_dense_layers():
    # a: 10x10 (raw 20/100 = 0.2), b: 100x100 (raw 200/10000 = 0.02)
    specs = [dense_layer("a", 10, 10), dense_layer("b", 100, 100)]
    dens = erk_densities(specs, density=0.1)
    eps = 0.1 * 10100 / (0.2 * 100 + 0.02 * 10000)
    assert dens["a"] == pytest.approx(eps * 0.2)
    assert dens["b"] == pytest.approx(eps * 0.02)
    assert dens["a"] / dens["b"] == pytest.approx(10.0)


def test_er_equals_erk_on_dense_but_not_conv():
    dense_specs = [dense_layer("a", 8, 4), dense_layer("b", 64, 32)]
    assert erk_densities(dense_specs, density=0.3, method="er") == pytest.approx(
        erk_densities(dense_specs, density=0.3, method="erk")
    )
    conv_specs = [conv_layer("c1", 8, 4, 3, 3), conv_layer("c2", 64, 32, 1, 1)]
    er = erk_densities(conv_specs, density=0.3, method="er")
    erk = erk_densities(conv_specs, density=0.3, method="erk")
    # ERK counts the 3x3 kernel parameters, so the 3x3 conv no longer gets the
    # outsized share ER gives it from its channel dims alone
    assert erk["c1"] / erk["c2"] < er["c1"] / er["c2"]
    # ER's channel-only ratio overflows and forces the 3x3 conv fully dense
    assert er["c1"] == 1.0 and erk["c1"] < 1.0


def test_small_layers_become_dense_and_budget_redistributed():
    specs = [dense_layer("tiny", 2, 2), dense_layer("big", 200, 200)]
    dens = erk_densities(specs, density=0.5)
    assert dens["tiny"] == 1.0
    expected_big = (0.5 * (4 + 40000) - 4) / 40000
    assert dens["big"] == pytest.approx(expected_big)


def test_larger_layers_are_sparser():
    specs = _specs()
    dens = erk_densities(specs, density=0.1)
    assert dens["fc1"] < dens["fc2"]
    assert dens["conv2"] < dens["conv1"]


def test_power_scale_zero_is_uniform():
    specs = _specs()
    dens = erk_densities(specs, density=0.3, erk_power_scale=0.0)
    assert all(d == pytest.approx(0.3) for d in dens.values())


def test_forced_dense_layers():
    specs = _specs()
    dens = erk_densities(specs, density=0.3, method="uniform", dense_layers=["conv1"])
    assert dens["conv1"] == 1.0
    others = {dens[n] for n in ("conv2", "fc1", "fc2")}
    assert len({round(v, 12) for v in others}) == 1
    assert _weighted_density(specs, dens) == pytest.approx(0.3)


def test_full_density_and_errors():
    specs = _specs()
    assert set(erk_densities(specs, density=1.0).values()) == {1.0}
    with pytest.raises(ValueError):
        erk_densities(specs, density=0.0)
    with pytest.raises(ValueError):
        erk_densities(specs, density=0.5, method="random")
    with pytest.raises(ValueError):
        erk_densities(specs, density=0.5, erk_power_scale=-1)
    with pytest.raises(ValueError):
        erk_densities(specs, density=0.5, dense_layers=["nope"])
    with pytest.raises(ValueError):
        erk_densities([], density=0.5)
    with pytest.raises(ValueError):
        erk_densities([dense_layer("a", 2, 2), dense_layer("a", 3, 3)])
    with pytest.raises(ValueError):
        erk_densities(specs, density=0.01, dense_layers=["fc1"])


def test_prune_model_keeps_largest_magnitudes():
    specs = [dense_layer("a", 4, 2), dense_layer("b", 16, 16)]
    model = {
        "a": [0.1, -0.8, 0.3, 0.2, 0.5, -0.05, 0.9, 0.0],
        "b": [((-1) ** i) * (i % 17) / 17.0 for i in range(256)],
    }
    dens = erk_densities(specs, density=0.25)
    pruned = erk_prune_model(specs, model, density=0.25)
    for spec in specs:
        kept = [abs(v) for v in pruned[spec.name] if v != 0.0]
        dropped = [abs(o) for o, v in zip(model[spec.name], pruned[spec.name]) if v == 0.0]
        assert len([v for v in pruned[spec.name] if v != 0.0]) <= round(dens[spec.name] * len(model[spec.name]))
        if kept and dropped:
            assert min(kept) >= max(dropped)
    # input not mutated
    assert model["a"][1] == -0.8


def test_summary_rows_and_survival():
    specs = _specs()
    model = {s.name: [float((i * 7919) % 101) - 50 for i in range(layer_weight_count(s))] for s in specs}
    summary = erk_prune_summary(specs, model, density=0.2)
    assert summary["method"] == "erk"
    assert [row["layer"] for row in summary["layers"]] == [s.name for s in specs]
    assert summary["total"] == sum(layer_weight_count(s) for s in specs)
    assert summary["overall_survival"] == pytest.approx(0.2, abs=0.01)
    for row in summary["layers"]:
        assert row["kept_weights"] == sum(summary["masks"][row["layer"]])


def test_model_mismatch_errors():
    specs = [dense_layer("a", 2, 2)]
    with pytest.raises(ValueError):
        erk_prune_model(specs, {"b": [1.0] * 4})
    with pytest.raises(ValueError):
        erk_prune_model(specs, {"a": [1.0] * 3})
    with pytest.raises(ValueError):
        erk_prune_model(specs, {})


def test_cli_json_and_markdown(tmp_path):
    weights = ",".join(str(float(i % 9) - 4) for i in range(4 + 40000))
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["erk", "--specs", "tiny=dense:2x2,big=dense:200x200",
                     f"--weights={weights}", "--density", "0.5", "--json"])
    assert code == 0
    data = json.loads(buf.getvalue())
    assert data["layer_densities"]["tiny"] == 1.0
    assert "masks" not in data
    out = tmp_path / "erk.md"
    with redirect_stdout(io.StringIO()):
        code = main(["erk", "--specs", "tiny=dense:2x2,big=dense:200x200",
                     f"--weights={weights}", "--method", "uniform", "--dense", "tiny",
                     "-o", str(out)])
    assert code == 0
    text = out.read_text()
    assert "Forced dense: tiny" in text and "| big | dense | 200x200 |" in text


def test_cli_errors():
    err = io.StringIO()
    with redirect_stderr(err):
        assert main(["erk", "--specs", "a=dense:2x2", "--weights", "1,2,3"]) == 2
        assert main(["erk", "--specs", "a=dense:2x2", "--weights", "1,2,3,4", "--dense", "zz"]) == 2
    assert "erk:" in err.getvalue()
