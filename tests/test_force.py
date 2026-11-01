"""Tests for FORCE / Iterative SNIP progressive pruning."""

from __future__ import annotations

import io
import json
import random
from contextlib import redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    dense_chain_connected,
    dense_layer,
    density_schedule,
    force_prune_model,
    mlp_mse_grads,
    snip_prune_summary,
)
from prune_kit.cli import main


def _mlp(seed=0, sizes=(6, 10, 10, 2), samples=16):
    rng = random.Random(seed)
    specs = [dense_layer(f"fc{i + 1}", a, b) for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]
    model = {s.name: [rng.gauss(0.0, 1.0) for _ in range(s.shape[0] * s.shape[1])] for s in specs}
    xs = [[rng.gauss(0.0, 1.0) for _ in range(sizes[0])] for _ in range(samples)]
    ys = [[x[0] - x[1], 0.5 * x[2]] for x in xs]
    return specs, model, xs, ys


def test_density_schedule():
    assert density_schedule(0.01, 2) == pytest.approx([0.1, 0.01])
    assert density_schedule(0.5, 4, "linear") == pytest.approx([0.875, 0.75, 0.625, 0.5])
    assert density_schedule(0.3, 7)[-1] == 0.3
    with pytest.raises(ValueError):
        density_schedule(0.0, 3)
    with pytest.raises(ValueError):
        density_schedule(0.5, 0)
    with pytest.raises(ValueError):
        density_schedule(0.5, 3, "cosine")


@pytest.mark.parametrize("activation", ["relu", "tanh", "linear"])
def test_mlp_grads_match_finite_differences(activation):
    specs, model, xs, ys = _mlp(sizes=(3, 4, 4, 2), samples=5)
    grad_fn = mlp_mse_grads(specs, xs, ys, activation)
    _, grads = grad_fn(model)
    eps = 1e-6
    for name, weights in model.items():
        for i in range(len(weights)):
            up = {k: list(v) for k, v in model.items()}
            dn = {k: list(v) for k, v in model.items()}
            up[name][i] += eps
            dn[name][i] -= eps
            numeric = (grad_fn(up)[0] - grad_fn(dn)[0]) / (2 * eps)
            assert grads[name][i] == pytest.approx(numeric, abs=1e-6)


@pytest.mark.parametrize("scope", ["global", "layer"])
def test_single_step_iter_snip_equals_snip(scope):
    specs, model, xs, ys = _mlp()
    grad_fn = mlp_mse_grads(specs, xs, ys)
    _, grads = grad_fn(model)
    ours = force_prune_model(specs, model, grad_fn, 0.2, steps=1, mode="iter_snip", scope=scope)
    ref = snip_prune_summary(specs, model, grads, density=0.2, scope=scope)
    assert ours["masks"] == {k: [int(v) for v in m] for k, m in ref["masks"].items()}


def test_kept_counts_follow_schedule_and_pruned_weights():
    specs, model, xs, ys = _mlp()
    grad_fn = mlp_mse_grads(specs, xs, ys)
    total = sum(len(w) for w in model.values())
    for mode in ("force", "iter_snip"):
        res = force_prune_model(specs, model, grad_fn, 0.05, steps=5, mode=mode)
        targets = density_schedule(0.05, 5)
        assert [h["kept"] for h in res["history"]] == [round(t * total) for t in targets]
        assert res["total_kept"] == round(0.05 * total)
        for name, mask in res["masks"].items():
            for w, w0, m in zip(res["pruned_weights"][name], model[name], mask):
                assert w == (w0 if m else 0.0)
        assert all(h["loss"] is not None for h in res["history"])
        assert res["final_loss"] == pytest.approx(grad_fn(res["pruned_weights"])[0])
    iter_res = force_prune_model(specs, model, grad_fn, 0.05, steps=5, mode="iter_snip")
    assert all(h["revived"] == 0 for h in iter_res["history"])


def test_force_revives_connection_that_becomes_important():
    spec = dense_layer("fc", 4, 1)
    model = {"fc": [1.0, 1.0, 1.0, 0.5]}

    def grad_fn(m):
        w = m["fc"]
        # the loss only becomes sensitive to weight 3 once it has been pruned
        return {"fc": [3.0, 2.0, 1.0, 1.0 if w[3] else 10.0]}

    force = force_prune_model([spec], model, grad_fn, 0.5, steps=2, mode="force", schedule="linear")
    snip_it = force_prune_model([spec], model, grad_fn, 0.5, steps=2, mode="iter_snip", schedule="linear")
    # step 1 keeps 3 of 4 (drops weight 3); step 2 keeps 2
    assert force["masks"]["fc"] == [1, 0, 0, 1]
    assert force["history"][1]["revived"] == 1
    assert snip_it["masks"]["fc"] == [1, 1, 0, 0]


def test_layer_scope_keeps_fraction_per_layer():
    specs, model, xs, ys = _mlp()
    res = force_prune_model(specs, model, mlp_mse_grads(specs, xs, ys), 0.3, steps=3, scope="layer")
    for row in res["layers"]:
        assert row["kept_weights"] == round(0.3 * row["total_weights"])


def test_dense_chain_connected_detects_collapse():
    specs = [dense_layer("a", 2, 2), dense_layer("b", 2, 1)]
    assert dense_chain_connected(specs, {"a": [1, 0, 0, 0], "b": [1, 0]})
    assert not dense_chain_connected(specs, {"a": [1, 0, 0, 0], "b": [0, 1]})
    assert not dense_chain_connected(specs, {"a": [0, 0, 0, 0], "b": [1, 1]})
    with pytest.raises(ValueError):
        dense_chain_connected([dense_layer("a", 2, 3), dense_layer("b", 2, 1)], {})


def test_conv_models_are_supported_without_connectivity():
    specs = [conv_layer("c", 2, 1, 2, 2), dense_layer("d", 4, 1)]
    model = {"c": [float(i + 1) for i in range(8)], "d": [1.0, -2.0, 3.0, -4.0]}
    res = force_prune_model(specs, model, lambda m: {k: [1.0] * len(v) for k, v in m.items()}, 0.5, steps=3)
    assert res["connected"] is None
    assert res["total_kept"] == 6
    assert res["final_loss"] is None


def test_validation_errors():
    specs, model, xs, ys = _mlp()
    grad_fn = mlp_mse_grads(specs, xs, ys)
    with pytest.raises(ValueError):
        force_prune_model(specs, model, grad_fn, 1.5)
    with pytest.raises(ValueError):
        force_prune_model(specs, model, grad_fn, 0.5, steps=0)
    with pytest.raises(ValueError):
        force_prune_model(specs, model, grad_fn, 0.5, mode="snap")
    with pytest.raises(KeyError):
        force_prune_model(specs, {"fc1": model["fc1"]}, grad_fn, 0.5)
    with pytest.raises(KeyError):
        force_prune_model(specs, model, lambda m: {}, 0.5)
    with pytest.raises(ValueError):
        mlp_mse_grads(specs, xs, ys[:-1])
    with pytest.raises(ValueError):
        mlp_mse_grads(specs, xs, ys, activation="gelu")


def test_cli_force_markdown_and_json():
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["force", "--specs", "fc1=dense:4x8,fc2=dense:8x2", "--density", "0.2", "--steps", "4"]) == 0
    out = buf.getvalue()
    assert "| force |" in out and "| iter_snip |" in out and "| snip |" in out
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["force", "--specs", "fc1=dense:4x8,fc2=dense:8x2", "--density", "0.2", "--json"]) == 0
    data = json.loads(buf.getvalue())
    assert set(data) == {"snip", "iter_snip", "force", "dense_loss"}
    assert data["force"]["total_kept"] == round(0.2 * 48)
    assert main(["force", "--specs", "fc1=dense:4x8,fc2=dense:3x2"]) == 2
