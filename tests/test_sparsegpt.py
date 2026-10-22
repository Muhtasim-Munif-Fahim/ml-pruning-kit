"""Tests for SparseGPT one-shot pruning."""

from __future__ import annotations

import json
import math
import random

import pytest

from prune_kit import (
    dense_layer,
    layer_hessian,
    output_reconstruction_error,
    sparsegpt_compare,
    sparsegpt_prune_layer,
    sparsegpt_prune_model,
)
from prune_kit.cli import main


def _problem(d=16, out=6, n=120, seed=0):
    rng = random.Random(seed)
    mix = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(d)]
    rows = []
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(d)]
        rows.append([sum(z[k] * mix[k][j] for k in range(d)) for j in range(d)])
    weights = [rng.gauss(0, 1) for _ in range(out * d)]
    return weights, (out, d), rows


def _reference(weights, shape, rows, density, block_size, percdamp=0.01):
    """Independent NumPy port of the reference SparseGPT loop (exact-k masks)."""
    np = pytest.importorskip("numpy")
    out, d = shape
    W = np.array(weights, dtype=float).reshape(out, d)
    X = np.array(rows, dtype=float)
    H = X.T @ X / len(X)
    H[np.diag_indices(d)] += percdamp * np.mean(np.diag(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    for i1 in range(0, d, block_size):
        i2 = min(i1 + block_size, d)
        W1 = W[:, i1:i2].copy()
        Hinv1 = U[i1:i2, i1:i2]
        Err1 = np.zeros_like(W1)
        tmp = W1 ** 2 / np.diag(Hinv1)[None, :] ** 2
        k = int(round(tmp.size * (1 - density)))
        mask = np.zeros(tmp.size, bool)
        mask[np.argsort(tmp.ravel(), kind="stable")[:k]] = True
        mask = mask.reshape(tmp.shape)
        for i in range(i2 - i1):
            w = W1[:, i]
            q = w.copy()
            q[mask[:, i]] = 0
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= np.outer(err, Hinv1[i, i:])
            W1[:, i] = q
            Err1[:, i] = err
        W[:, i1:i2] = W1
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]
    return W.ravel().tolist()


@pytest.mark.parametrize("block_size", [128, 5, 4])
def test_matches_reference_port(block_size):
    weights, shape, rows = _problem(d=12, out=4, n=80, seed=block_size)
    ours = sparsegpt_prune_layer(weights, shape, rows, 0.5, block_size=block_size).weights
    ref = _reference(weights, shape, rows, 0.5, block_size)
    assert max(abs(a - b) for a, b in zip(ours, ref)) < 1e-10


def test_density_and_mask_consistency():
    weights, shape, rows = _problem()
    res = sparsegpt_prune_layer(weights, shape, rows, 0.25)
    assert res.density == pytest.approx(0.25, abs=1 / len(weights))
    assert all((w == 0.0) == (m == 0) for w, m in zip(res.weights, res.mask) if m == 0)
    assert res.pattern == "unstructured"
    assert res.relative_error == pytest.approx(
        output_reconstruction_error(weights, res.weights, shape, rows)["relative_error"])


def test_full_density_is_identity():
    weights, shape, rows = _problem()
    res = sparsegpt_prune_layer(weights, shape, rows, 1.0)
    assert res.weights == pytest.approx(weights)
    assert res.squared_error == pytest.approx(0.0, abs=1e-12)


def test_beats_magnitude_and_wanda_and_update_helps():
    weights, shape, rows = _problem(d=20, out=8, n=200, seed=3)
    cmp = sparsegpt_compare(weights, shape, rows, 0.5)
    err = {k: v["relative_error"] for k, v in cmp.items()}
    assert err["sparsegpt"] < err["magnitude"]
    assert err["sparsegpt"] < err["wanda"]
    assert err["sparsegpt"] < err["sparsegpt_mask_only"]


def test_two_four_pattern():
    weights, shape, rows = _problem(d=16, out=5, seed=4)
    res = sparsegpt_prune_layer(weights, shape, rows, nm=(2, 4), block_size=6)  # rounded to a multiple of 4
    d = shape[1]
    for o in range(shape[0]):
        for g in range(0, d, 4):
            assert sum(res.mask[o * d + g:o * d + g + 4]) == 2
    assert res.pattern == "2:4" and res.density == 0.5
    cmp = sparsegpt_compare(weights, shape, rows, nm=(2, 4))
    assert cmp["sparsegpt"]["relative_error"] < cmp["magnitude"]["relative_error"]
    with pytest.raises(ValueError, match="divisible"):
        sparsegpt_prune_layer(weights[: 5 * 15], (5, 15), [r[:15] for r in rows], nm=(2, 4))
    with pytest.raises(ValueError):
        sparsegpt_prune_layer(weights, shape, rows, nm=(4, 4))


def test_exact_recovery_when_pruned_inputs_are_redundant():
    # Column 0 duplicates column 3: pruning w_0 and folding it into the later w_3 is lossless
    # (SparseGPT only updates columns to the right of the pruned one).
    rng = random.Random(9)
    rows = []
    for _ in range(60):
        a, b, c = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        rows.append([a, b, c, a])
    weights = [0.01, 2.0, 3.0, 1.0]
    res = sparsegpt_prune_layer(weights, (1, 4), rows, 0.75, percdamp=1e-9)
    assert res.mask == [0, 1, 1, 1]
    assert res.weights[3] == pytest.approx(1.01, abs=1e-4)
    assert res.relative_error < 1e-8


def test_conv_layout_and_dead_inputs():
    weights, _, rows = _problem(d=8, out=3, seed=5)
    rows = [r[:4] + [0.0] + r[5:] for r in rows]  # input 4 is never active
    res = sparsegpt_prune_layer(weights[: 3 * 8], (3, 2, 2, 2), rows, 0.5)
    assert len(res.weights) == 24
    H = layer_hessian(rows)
    assert H[4][4] > 0  # dead input regularised to keep H positive definite


def test_model_wrapper_and_validation():
    weights, shape, rows = _problem(d=6, out=3)
    specs = [dense_layer("fc1", 6, 3), dense_layer("fc2", 3, 2)]
    model = {"fc1": weights, "fc2": [0.5] * 6}
    out = sparsegpt_prune_model(specs, model, {"fc1": rows}, density=0.5)
    assert list(out) == ["fc1"]
    with pytest.raises(ValueError, match="unknown"):
        sparsegpt_prune_model(specs, model, {"fc9": rows})
    with pytest.raises(ValueError):
        sparsegpt_prune_layer(weights, shape, rows, 0.0)
    with pytest.raises(ValueError):
        sparsegpt_prune_layer(weights, shape, [r[:3] for r in rows])
    with pytest.raises(ValueError):
        sparsegpt_prune_layer(weights[:-1], shape, rows)
    with pytest.raises(ValueError):
        sparsegpt_prune_layer(weights, shape, [])


def test_cli_sparsegpt(capsys):
    assert main(["sparsegpt", "--in-features", "12", "--out-features", "4", "--samples", "60", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    errs = payload["relative_error"]
    assert errs["sparsegpt"] < errs["magnitude"]
    assert main(["sparsegpt", "--in-features", "12", "--out-features", "4", "--nm", "2:4"]) == 0
    assert "| sparsegpt |" in capsys.readouterr().out
    assert main(["sparsegpt", "--nm", "bad"]) == 2
