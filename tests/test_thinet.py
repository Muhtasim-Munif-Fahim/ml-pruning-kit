"""Tests for ThiNet channel selection and least-squares reconstruction."""

from __future__ import annotations

import io
import itertools
import json
import random
from contextlib import redirect_stdout

import pytest

from prune_kit import (
    channel_contribution_gram,
    dense_layer,
    least_squares_reconstruct,
    reconstruction_error,
    thinet_prune_layer,
    thinet_prune_model,
    thinet_prune_summary,
    thinet_select_channels,
)
from prune_kit.cli import main


def _random_layer(n_in=6, out=4, n=80, seed=0):
    rng = random.Random(seed)
    weights = [rng.gauss(0, 1) for _ in range(out * n_in)]
    rows = [[rng.gauss(0, 1) for _ in range(n_in)] for _ in range(n)]
    return weights, (out, n_in), rows


def _brute_error(weights, shape, rows, removed):
    out, n_in = shape
    err = 0.0
    for x in rows:
        for o in range(out):
            err += sum(weights[o * n_in + c] * x[c] for c in removed) ** 2
    return err


def test_gram_matches_definition_and_error_identity():
    weights, shape, rows = _random_layer()
    G = channel_contribution_gram(weights, shape, rows)
    out, n_in = shape
    for c, d in [(0, 0), (1, 3), (5, 2)]:
        direct = sum(
            sum(weights[o * n_in + c] * x[c] * weights[o * n_in + d] * x[d] for o in range(out))
            for x in rows
        )
        assert G[c][d] == pytest.approx(direct)
        assert G[c][d] == pytest.approx(G[d][c])
    T = [1, 4]
    assert sum(G[a][b] for a in T for b in T) == pytest.approx(_brute_error(weights, shape, rows, T))


def test_greedy_errors_are_cumulative_and_first_step_is_optimal():
    weights, shape, rows = _random_layer(seed=1)
    chosen = thinet_select_channels(weights, shape, rows, density=0.5)
    assert len(chosen["keep"]) == 3 and len(chosen["removed"]) == 3
    assert sorted(chosen["keep"] + chosen["removed"]) == list(range(6))
    for i in range(len(chosen["removed"])):
        assert chosen["errors"][i] == pytest.approx(
            _brute_error(weights, shape, rows, chosen["removed"][: i + 1])
        )
    singles = {c: _brute_error(weights, shape, rows, [c]) for c in range(6)}
    assert chosen["removed"][0] == min(singles, key=singles.get)
    # greedy is near the exhaustive optimum on this small problem
    best = min(_brute_error(weights, shape, rows, list(T)) for T in itertools.combinations(range(6), 3))
    assert chosen["errors"][-1] <= 1.5 * best


def test_thinet_drops_silent_high_norm_channel_that_l1_keeps():
    rng = random.Random(2)
    n_in, out = 5, 3
    weights = [rng.gauss(0, 1) for _ in range(out * n_in)]
    for o in range(out):
        weights[o * n_in + 0] = 20.0  # huge weights ...
    rows = [[rng.gauss(0, 1) * (1e-3 if c == 0 else 1.0) for c in range(n_in)] for _ in range(100)]  # ... silent input
    l1 = thinet_prune_layer(weights, (out, n_in), rows, 0.6, selection="l1", reconstruct=False)
    thin = thinet_prune_layer(weights, (out, n_in), rows, 0.6, reconstruct=False)
    assert 0 in l1.keep
    assert 0 not in thin.keep
    assert thin.relative_error_before < l1.relative_error_before


def test_greedy_uses_interactions_with_already_removed_channels():
    rng = random.Random(3)
    n_in, out = 4, 2
    weights = [rng.gauss(0, 1.0) for _ in range(out * n_in)]
    rows = []
    for _ in range(60):
        x = [rng.gauss(0, 1) for _ in range(n_in)]
        x[3] = x[2]  # channel 3 duplicates channel 2 ...
        rows.append(x)
    for o in range(out):
        weights[o * n_in + 0] = 1.5
        weights[o * n_in + 1] = -1.5
        weights[o * n_in + 2] = 0.2
        weights[o * n_in + 3] = -0.25  # ... with nearly opposite weights
    G = channel_contribution_gram(weights, (out, n_in), rows)
    pair = G[2][2] + G[3][3] + 2 * G[2][3]
    assert pair < 0.05 * G[3][3]  # the Gram form sees the cancellation
    chosen = thinet_select_channels(weights, (out, n_in), rows, density=0.5)
    # channel 2 is cheapest alone; once it is gone, channel 3 almost cancels it
    assert chosen["removed"] == [2, 3]
    assert chosen["errors"][1] < chosen["errors"][0]


def test_least_squares_recovers_redundant_channel_exactly():
    rng = random.Random(4)
    n_in, out = 5, 3
    weights = [rng.gauss(0, 1) for _ in range(out * n_in)]
    rows = []
    for _ in range(50):
        x = [rng.gauss(0, 1) for _ in range(4)]
        rows.append(x + [2.0 * x[0] - x[1]])
    res = thinet_prune_layer(weights, (out, n_in), rows, density=0.8, ridge=0.0)
    assert len(res.keep) == 4
    if 4 not in res.keep:
        assert res.relative_error_after == pytest.approx(0.0, abs=1e-12)
    # explicit: drop the redundant channel and refit
    new = least_squares_reconstruct(weights, (out, n_in), rows, keep=[0, 1, 2, 3], ridge=0.0)
    assert reconstruction_error(weights, new, (out, n_in), rows)["relative_error"] < 1e-20
    for o in range(out):
        assert new[o * n_in + 4] == 0.0
        assert new[o * n_in + 0] == pytest.approx(weights[o * n_in + 0] + 2 * weights[o * n_in + 4])
        assert new[o * n_in + 1] == pytest.approx(weights[o * n_in + 1] - weights[o * n_in + 4])


def test_least_squares_never_worse_than_zeroing_and_ridge_handles_collinearity():
    for seed in range(5):
        weights, shape, rows = _random_layer(n_in=7, out=3, n=40, seed=seed)
        for sel in ("thinet", "l1", "l2"):
            res = thinet_prune_layer(weights, shape, rows, 0.5, selection=sel)
            assert res.error_after_reconstruction <= res.error_before_reconstruction + 1e-9
            assert res.reconstructed
    weights, shape, rows = _random_layer(n_in=3, out=2, n=20, seed=9)
    dup = [[x[0], x[0], x[2]] for x in rows]  # kept channels 0 and 1 are identical
    new = least_squares_reconstruct(weights, shape, dup, keep=[0, 1])
    assert all(abs(v) < 1e6 for v in new)
    with pytest.raises(ValueError, match="singular"):
        least_squares_reconstruct(weights, shape, dup, keep=[0, 1], ridge=0.0)


def test_conv_layer_with_im2col_rows():
    rng = random.Random(5)
    out, n_in, kh, kw = 3, 4, 1, 2
    weights = [rng.gauss(0, 1) for _ in range(out * n_in * kh * kw)]
    rows = [[rng.gauss(0, 1) for _ in range(n_in * kh * kw)] for _ in range(60)]
    res = thinet_prune_layer(weights, (out, n_in, kh, kw), rows, 0.5)
    assert len(res.keep) == 2
    inner = kh * kw
    for c in res.removed:
        for o in range(out):
            block = res.weights[o * n_in * inner + c * inner : o * n_in * inner + (c + 1) * inner]
            assert block == [0.0] * inner
    G = channel_contribution_gram(weights, (out, n_in, kh, kw), rows)
    assert len(G) == n_in


def test_model_and_summary():
    weights, shape, rows = _random_layer(n_in=6, out=4, seed=6)
    specs = [dense_layer("fc1", 3, 6), dense_layer("fc2", 6, 4)]
    model = {"fc1": [0.1] * 18, "fc2": weights}
    result = thinet_prune_model(specs, model, {"fc2": rows}, density=0.5)
    assert set(result) == {"fc2"}
    assert len(result["fc2"].keep) == 3
    summary = thinet_prune_summary(specs, model, {"fc2": rows}, density=0.5)
    row = summary["layers"][0]
    assert row["relative_error_thinet"] <= row["relative_error_l1"] + 1e-12
    assert row["relative_error_thinet_ls"] <= row["relative_error_thinet"] + 1e-12
    with pytest.raises(ValueError, match="unknown layers"):
        thinet_prune_model(specs, model, {"nope": rows}, density=0.5)
    with pytest.raises(KeyError):
        thinet_prune_model(specs, {"fc1": [0.1] * 18}, {"fc2": rows}, density=0.5)
    assert result["fc2"].as_dict()["reconstructed"] is True


def test_density_one_and_min_channels():
    weights, shape, rows = _random_layer(seed=7)
    full = thinet_prune_layer(weights, shape, rows, 1.0)
    assert full.removed == [] and not full.reconstructed
    assert full.weights == pytest.approx(weights)
    tiny = thinet_select_channels(weights, shape, rows, density=0.01, min_channels=2)
    assert len(tiny["keep"]) == 2


def test_validation():
    weights, shape, rows = _random_layer()
    with pytest.raises(ValueError, match="density"):
        thinet_select_channels(weights, shape, rows, density=0)
    with pytest.raises(ValueError, match="min_channels"):
        thinet_select_channels(weights, shape, rows, 0.5, min_channels=0)
    with pytest.raises(ValueError, match="values"):
        thinet_prune_layer(weights[:-1], shape, rows, 0.5)
    with pytest.raises(ValueError, match="input row"):
        thinet_prune_layer(weights, shape, [r[:-1] for r in rows], 0.5)
    with pytest.raises(ValueError, match="at least one"):
        thinet_prune_layer(weights, shape, [], 0.5)
    with pytest.raises(ValueError, match="selection"):
        thinet_prune_layer(weights, shape, rows, 0.5, selection="random")
    with pytest.raises(ValueError, match="keep"):
        least_squares_reconstruct(weights, shape, rows, keep=[9])
    with pytest.raises(ValueError, match="ridge"):
        least_squares_reconstruct(weights, shape, rows, keep=[0], ridge=-1)
    with pytest.raises(ValueError, match="finite"):
        thinet_prune_layer(weights, shape, [[float("nan")] * 6], 0.5)
    with pytest.raises(ValueError, match="differ"):
        reconstruction_error(weights, weights[:-1], shape, rows)


def test_cli_thinet():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["thinet", "--in-channels", "8", "--density", "0.5", "--seed", "1"])
    assert rc == 0
    assert "ThiNet channel pruning" in buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["thinet", "--json", "--seed", "1"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    row = payload["layers"][0]
    assert 0 in row["l1_keep"] and 0 not in row["keep"]
    assert row["relative_error_thinet_ls"] < row["relative_error_l1"]
    assert main(["thinet", "--redundant", "9"]) == 2
