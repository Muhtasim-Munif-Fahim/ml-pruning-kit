"""Tests for HRank feature-map-rank filter pruning."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from prune_kit import (
    conv_layer,
    dense_layer,
    hrank_keep_filters,
    hrank_prune_layer,
    hrank_prune_model,
    hrank_prune_summary,
    hrank_scores,
    matrix_rank,
)
from prune_kit.cli import main


def _rank_r_map(rng, h, w, r):
    """Random h x w map with exact rank r (sum of r outer products)."""
    out = [[0.0] * w for _ in range(h)]
    for _ in range(r):
        u = [rng.uniform(-1, 1) for _ in range(h)]
        v = [rng.uniform(-1, 1) for _ in range(w)]
        for i in range(h):
            for j in range(w):
                out[i][j] += u[i] * v[j]
    return out


def test_matrix_rank_matches_numpy_on_random_low_rank_maps():
    rng = random.Random(0)
    for h, w in [(5, 5), (6, 4), (3, 7)]:
        for r in range(0, min(h, w) + 1):
            m = _rank_r_map(rng, h, w, r)
            assert matrix_rank(m) == r == np.linalg.matrix_rank(np.array(m))


def test_matrix_rank_edge_cases():
    assert matrix_rank([]) == 0
    assert matrix_rank([[0.0, 0.0], [0.0, 0.0]]) == 0
    assert matrix_rank([[1.0, 2.0], [2.0, 4.0]]) == 1
    assert matrix_rank([[1e-20, 0.0], [0.0, 1.0]]) == 1  # tiny relative pivot ignored
    assert matrix_rank([[1.0, 0.0], [0.0, 1.0]]) == 2
    with pytest.raises(ValueError):
        matrix_rank([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        matrix_rank([[1.0]], rtol=-1.0)


def _maps(rng, ranks, n_images=3, size=6):
    return [[_rank_r_map(rng, size, size, r) for r in ranks] for _ in range(n_images)]


def test_scores_average_over_images():
    rng = random.Random(1)
    maps = _maps(rng, [1, 4, 2, 6])
    assert hrank_scores(maps) == [1.0, 4.0, 2.0, 6.0]
    maps[0][0] = _rank_r_map(rng, 6, 6, 4)  # one image disagrees
    assert hrank_scores(maps)[0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        hrank_scores([])
    with pytest.raises(ValueError):
        hrank_scores([[[[1.0]]], [[[1.0]], [[1.0]]]])


def test_keep_filters_ranks_highest_and_breaks_ties_by_index():
    assert hrank_keep_filters([1.0, 4.0, 2.0, 6.0], 0.5) == [1, 3]
    assert hrank_keep_filters([3.0, 3.0, 3.0, 3.0], 0.5) == [0, 1]
    assert hrank_keep_filters([1.0, 2.0], 1.0) == [0, 1]
    with pytest.raises(ValueError):
        hrank_keep_filters([1.0], 0.0)


def test_prune_layer_zeroes_low_rank_filters():
    rng = random.Random(2)
    shape = (4, 2, 3, 3)
    weights = [float(i + 1) for i in range(4 * 18)]
    maps = _maps(rng, [1, 4, 2, 6])
    pruned = hrank_prune_layer(weights, shape, maps, density=0.5)
    for k in range(4):
        block = pruned[k * 18:(k + 1) * 18]
        if k in (1, 3):
            assert block == weights[k * 18:(k + 1) * 18]
        else:
            assert all(v == 0.0 for v in block)
    with pytest.raises(ValueError, match="filters"):
        hrank_prune_layer(weights, shape, _maps(rng, [1, 2, 3]), density=0.5)


def test_prune_model_leaves_dense_and_unmapped_layers():
    rng = random.Random(3)
    specs = [conv_layer("c1", 2, 1, 1, 2), conv_layer("c2", 2, 2, 1, 1), dense_layer("fc", 2, 2)]
    model = {"c1": [1.0, 2.0, 3.0, 4.0], "c2": [5.0, 6.0, 7.0, 8.0], "fc": [1.0] * 4}
    maps = {"c1": _maps(rng, [3, 1], size=4)}
    pruned = hrank_prune_model(specs, model, maps, density=0.5)
    assert pruned["c1"] == [1.0, 2.0, 0.0, 0.0]
    assert pruned["c2"] == model["c2"] and pruned["fc"] == model["fc"]
    assert hrank_prune_model(specs, model, maps, density=0.5, per_layer={"c1": 1.0})["c1"] == model["c1"]
    with pytest.raises(ValueError, match="unknown"):
        hrank_prune_model(specs, model, {"nope": maps["c1"]})
    with pytest.raises(ValueError, match="unknown"):
        hrank_prune_model(specs, model, maps, per_layer={"nope": 0.5})


def test_summary_reports_ranks_and_survival():
    rng = random.Random(4)
    specs = [conv_layer("c1", 4, 1, 1, 1), dense_layer("fc", 2, 4)]
    model = {"c1": [1.0, 2.0, 3.0, 4.0], "fc": [1.0] * 8}
    summary = hrank_prune_summary(specs, model, {"c1": _maps(rng, [2, 5, 1, 3])}, density=0.5)
    c1 = summary["layers"][0]
    assert c1["ranks"] == [2.0, 5.0, 1.0, 3.0]
    assert c1["kept_filters"] == [1, 3]
    assert c1["nonzero_weights"] == 2
    assert summary["layers"][1]["ranks"] is None
    assert summary["nonzero_weights"] == 10


def test_cli_hrank(tmp_path, capsys):
    rng = random.Random(5)
    path = tmp_path / "maps.json"
    path.write_text(json.dumps({"c1": _maps(rng, [2, 5, 1, 3])}), encoding="utf-8")
    args = ["hrank", "--specs", "c1=conv:4x1x1x1", "--weights", "1,2,3,4",
            "--feature-maps", str(path), "--density", "0.5"]
    assert main(args + ["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["layers"][0]["kept_filters"] == [1, 3]
    assert "pruned_weights" not in data
    assert main(args) == 0
    assert "HRank" in capsys.readouterr().out
    out = tmp_path / "r.md"
    assert main(args + ["-o", str(out)]) == 0
    assert "| c1 | conv |" in out.read_text(encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert main(["hrank", "--specs", "c1=conv:4x1x1x1", "--weights", "1,2,3,4",
                 "--feature-maps", str(bad)]) == 2
    assert main(["hrank", "--specs", "c1=conv:4x1x1x1", "--weights", "1,2,3,4",
                 "--feature-maps", str(tmp_path / "missing.json")]) == 2
