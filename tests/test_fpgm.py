"""Tests for Filter Pruning via Geometric Median (FPGM)."""

from __future__ import annotations

import math

import pytest

from prune_kit.fpgm import (
    fpgm_distances,
    fpgm_prune_layer,
    fpgm_prune_model,
    geometric_median,
)
from prune_kit.layers import LayerSpec, conv_layer


def test_geometric_median_of_identical_points():
    pts = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    med = geometric_median(pts)
    assert med == pytest.approx([1.0, 2.0])


def test_geometric_median_of_collinear_points():
    # Median of 0, 1, 100 on a line should be near 1
    pts = [[0.0], [1.0], [100.0]]
    med = geometric_median(pts)
    assert med[0] == pytest.approx(1.0, abs=0.05)


def test_fpgm_prunes_near_median_filter():
    # 4 filters of shape (4, 1, 1, 1): three clustered near 1.0, one outlier at 10
    shape = (4, 1, 1, 1)
    weights = [1.0, 1.05, 0.95, 10.0]
    pruned = fpgm_prune_layer(weights, shape, density=0.25)  # keep 1
    # The outlier (index 3) should be kept; the near-median ones zeroed
    assert pruned[3] == pytest.approx(10.0)
    assert sum(1 for v in pruned if v != 0.0) == 1


def test_fpgm_density_one_keeps_all():
    shape = (3, 2, 1, 1)
    weights = [float(i) for i in range(6)]
    pruned = fpgm_prune_layer(weights, shape, density=1.0)
    assert pruned == pytest.approx(weights)


def test_fpgm_distances_outlier_largest():
    shape = (3, 1, 1, 1)
    weights = [0.0, 0.1, 5.0]
    dists = fpgm_distances(weights, shape)
    assert dists[2] == max(dists)


def test_fpgm_prune_model_skips_dense():
    specs = [
        LayerSpec(name="c1", kind="conv", shape=(2, 1, 1, 1)),
        LayerSpec(name="d1", kind="dense", shape=(4, 3)),
    ]
    model = {
        "c1": [1.0, 9.0],
        "d1": [0.1] * 12,
    }
    out = fpgm_prune_model(specs, model, density=0.5)
    assert out["d1"] == pytest.approx(model["d1"])
    assert len(out["c1"]) == 2
