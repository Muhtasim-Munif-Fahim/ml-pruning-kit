"""Tests for RigL / SET dynamic sparse training updates."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    conv_layer,
    cosine_drop_fraction,
    dense_layer,
    erk_densities,
    layer_weight_count,
    random_sparse_masks,
    rigl_schedule,
    rigl_update_layer,
    rigl_update_model,
    set_update_layer,
)
from prune_kit.cli import main


def test_rigl_layer_drops_smallest_and_grows_largest_gradient():
    w = [0.9, 0.0, -0.1, 0.5, 0.0, 0.0]
    m = [1, 0, 1, 1, 0, 0]
    g = [0.0, 2.0, 0.0, 0.0, -3.0, 0.1]
    out = rigl_update_layer(w, m, g, drop_fraction=1 / 3)
    assert out["dropped"] == [2]
    assert out["grown"] == [4]
    assert out["mask"] == [1, 0, 0, 1, 1, 0]
    assert out["weights"] == [0.9, 0.0, 0.0, 0.5, 0.0, 0.0]
    assert out["n_active"] == 3
    # inputs untouched
    assert m == [1, 0, 1, 1, 0, 0]


def test_density_is_preserved_and_regrow_init():
    w = [0.1 * (i + 1) for i in range(20)]
    m = [1 if i % 2 == 0 else 0 for i in range(20)]
    g = [float(i) for i in range(20)]
    out = rigl_update_layer(w, m, g, drop_fraction=0.4, regrow_init=0.01)
    assert sum(out["mask"]) == sum(m)
    assert len(out["dropped"]) == len(out["grown"]) == 4
    for i in out["grown"]:
        assert out["weights"][i] == 0.01
    for i, bit in enumerate(out["mask"]):
        if not bit:
            assert out["weights"][i] == 0.0


def test_just_dropped_connection_can_regrow_with_large_gradient():
    w = [0.01, 0.5, 0.0]
    m = [1, 1, 0]
    g = [10.0, 0.0, 1.0]
    out = rigl_update_layer(w, m, g, drop_fraction=0.5)
    assert out["dropped"] == [0]
    assert out["grown"] == [0]
    assert out["weights"][0] == 0.0  # reset to regrow_init


def test_zero_drop_fraction_and_fully_dense_are_noops():
    w = [0.3, -0.2, 0.1]
    out = rigl_update_layer(w, [1, 1, 1], [1.0, 1.0, 1.0], drop_fraction=0.5)
    assert out["dropped"] == [] and out["grown"] == []
    assert out["mask"] == [1, 1, 1]
    out = rigl_update_layer(w, [1, 0, 1], [1.0, 1.0, 1.0], drop_fraction=0.0)
    assert out["mask"] == [1, 0, 1]
    assert out["weights"] == [0.3, 0.0, 0.1]


def test_set_random_regrowth_is_seeded():
    w = [float(i) for i in range(1, 31)]
    m = [1] * 10 + [0] * 20
    a = set_update_layer(w, m, drop_fraction=0.3, seed=7)
    b = set_update_layer(w, m, drop_fraction=0.3, seed=7)
    assert a == b
    assert a["dropped"] == [0, 1, 2]
    assert len(a["grown"]) == 3 and sum(a["mask"]) == 10


def test_validation_errors():
    with pytest.raises(ValueError):
        rigl_update_layer([], [], [])
    with pytest.raises(ValueError):
        rigl_update_layer([1.0, 2.0], [1, 2], [0.0, 0.0])
    with pytest.raises(ValueError):
        rigl_update_layer([1.0, 2.0], [1, 0], None)
    with pytest.raises(ValueError):
        rigl_update_layer([1.0, 2.0], [1, 0], [0.0], drop_fraction=0.5)
    with pytest.raises(ValueError):
        rigl_update_layer([1.0, 2.0], [1, 0], [0.0, 1.0], drop_fraction=1.5)
    with pytest.raises(ValueError):
        rigl_update_layer([1.0, 2.0], [1, 0], [0.0, 1.0], regrow="magic")


def test_cosine_drop_fraction_and_schedule():
    assert cosine_drop_fraction(0, end_step=100, initial_fraction=0.3) == pytest.approx(0.3)
    assert cosine_drop_fraction(50, end_step=100, initial_fraction=0.3) == pytest.approx(0.15)
    assert cosine_drop_fraction(100, end_step=100) == 0.0
    assert cosine_drop_fraction(25, end_step=100, initial_fraction=0.2) == pytest.approx(
        0.1 * (1 + math.cos(math.pi / 4))
    )
    sched = rigl_schedule(1000, update_every=100, end_fraction=0.75)
    assert [s for s, _ in sched] == [100, 200, 300, 400, 500, 600, 700]
    fracs = [f for _, f in sched]
    assert all(a > b for a, b in zip(fracs, fracs[1:]))
    with pytest.raises(ValueError):
        rigl_schedule(0)
    with pytest.raises(ValueError):
        cosine_drop_fraction(1, end_step=0)


def test_random_sparse_masks_uniform_and_erk():
    specs = [conv_layer("conv1", 8, 3, 3, 3), dense_layer("fc1", 128, 32), dense_layer("fc2", 32, 10)]
    masks = random_sparse_masks(specs, density=0.2, seed=1)
    for s in specs:
        n = layer_weight_count(s)
        assert len(masks[s.name]) == n
        assert sum(masks[s.name]) == max(1, round(0.2 * n))
    erk = random_sparse_masks(specs, density=0.2, distribution="erk", seed=1)
    dens = erk_densities(specs, density=0.2, method="erk")
    for s in specs:
        n = layer_weight_count(s)
        assert sum(erk[s.name]) == max(1, min(n, round(dens[s.name] * n)))
    assert random_sparse_masks(specs, density=0.2, seed=1) == masks
    dense = random_sparse_masks(specs, density=0.2, dense_layers=["fc2"], seed=1)
    assert all(dense["fc2"])


def test_rigl_update_model_reports_and_skips_dense_layers():
    specs = [dense_layer("fc1", 4, 5), dense_layer("fc2", 5, 2)]
    weights = {
        "fc1": [0.05 * (i + 1) * (-1) ** i for i in range(20)],
        "fc2": [0.1 * (i + 1) for i in range(10)],
    }
    masks = {"fc1": [1 if i < 10 else 0 for i in range(20)], "fc2": [1] * 5 + [0] * 5}
    grads = {"fc1": [float(i) for i in range(20)], "fc2": [1.0] * 10}
    out = rigl_update_model(specs, weights, masks, grads, drop_fraction=0.2, dense_layers=["fc2"])
    rows = {r["name"]: r for r in out["layers"]}
    assert rows["fc1"]["dropped"] == rows["fc1"]["grown"] == 2
    assert rows["fc1"]["active"] == 10
    assert rows["fc1"]["mask_overlap"] == pytest.approx(0.8)
    assert rows["fc2"]["dropped"] == 0 and out["masks"]["fc2"] == masks["fc2"]
    assert out["density"] == pytest.approx(15 / 30)
    assert out["n_rewired"] == 2
    # grown fc1 indices are the dormant ones with the largest gradients: 18, 19
    assert out["masks"]["fc1"][18] == 1 and out["masks"]["fc1"][19] == 1
    assert out["masks"]["fc1"][0] == 0 and out["masks"]["fc1"][1] == 0
    with pytest.raises(KeyError):
        rigl_update_model(specs, weights, masks, None, drop_fraction=0.2)
    random_out = rigl_update_model(specs, weights, masks, None, drop_fraction=0.2, regrow="random", seed=3)
    assert random_out["density"] == pytest.approx(0.5)


def test_cli_rigl_json_and_markdown():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main([
            "rigl", "--specs", "fc=dense:2x3", "--weights", "0.9,0,-0.1,0.5,0,0",
            "--gradients", "0,2,0,0,-3,0.1", "--drop-fraction", "0.34", "--json",
        ])
    assert code == 0
    data = json.loads(buf.getvalue())
    assert data["masks"]["fc"] == [1, 0, 0, 1, 1, 0]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main([
            "rigl", "--specs", "fc=dense:2x3", "--weights", "0.9,0,-0.1,0.5,0,0",
            "--regrow", "random", "--seed", "0", "--drop-fraction", "0.34",
        ])
    assert code == 0
    assert "SET (random regrowth)" in buf.getvalue()
    err = io.StringIO()
    with redirect_stderr(err):
        code = main(["rigl", "--specs", "fc=dense:2x3", "--weights", "0.9,0,-0.1,0.5,0,0"])
    assert code == 2
    assert "--gradients" in err.getvalue()
