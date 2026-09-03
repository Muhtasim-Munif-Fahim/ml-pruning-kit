"""Tests for the magnitude-pruning module."""

from __future__ import annotations

import pytest

from prune_kit import (
    magnitude_prune_layer,
    magnitude_prune_model,
    total_pruned,
)


def test_magnitude_prune_layer_keeps_top_magnitudes() -> None:
    weights = [0.1, 0.5, 0.2, 0.9, 0.3]
    pruned = magnitude_prune_layer(weights, density=0.4)
    assert pruned == [0.0, 0.5, 0.0, 0.9, 0.0]


def test_magnitude_prune_layer_keeps_all_at_density_one() -> None:
    weights = [0.1, 0.2, 0.3]
    pruned = magnitude_prune_layer(weights, density=1.0)
    assert pruned == [0.1, 0.2, 0.3]


def test_magnitude_prune_layer_zeros_all_below_threshold() -> None:
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    pruned = magnitude_prune_layer(weights, density=0.2)
    assert pruned == [0.0, 0.0, 0.0, 0.0, 0.5]


def test_magnitude_prune_layer_does_not_modify_input() -> None:
    weights = [0.1, 0.2, 0.3]
    original = list(weights)
    magnitude_prune_layer(weights, density=0.5)
    assert weights == original


def test_magnitude_prune_layer_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        magnitude_prune_layer([])


def test_magnitude_prune_layer_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="density"):
        magnitude_prune_layer([0.1, 0.2], density=0.0)
    with pytest.raises(ValueError, match="density"):
        magnitude_prune_layer([0.1, 0.2], density=1.5)


def test_magnitude_prune_model_returns_one_pruned_dict() -> None:
    model = {"fc1": [0.1, 0.2, 0.3], "fc2": [0.4, 0.5, 0.6]}
    pruned = magnitude_prune_model(model, density=0.5)
    assert set(pruned.keys()) == {"fc1", "fc2"}
    assert len(pruned["fc1"]) == 3


def test_magnitude_prune_model_respects_per_layer_density() -> None:
    model = {"fc1": [0.1, 0.2, 0.3, 0.4], "fc2": [0.5, 0.6, 0.7, 0.8]}
    pruned = magnitude_prune_model(model, density=0.5, per_layer={"fc1": 0.25})
    # fc1 keeps one weight; fc2 keeps two.
    assert sum(1 for value in pruned["fc1"] if value != 0.0) == 1
    assert sum(1 for value in pruned["fc2"] if value != 0.0) == 2


def test_magnitude_prune_model_rejects_unknown_per_layer() -> None:
    with pytest.raises(ValueError, match="unknown layer"):
        magnitude_prune_model({"fc1": [0.1]}, per_layer={"ghost": 0.5})


def test_magnitude_prune_model_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        magnitude_prune_model({})


def test_total_pruned_counts_zeroed_weights() -> None:
    original = {"fc1": [0.1, 0.2, 0.3, 0.4], "fc2": [0.5, 0.6, 0.7, 0.8]}
    pruned = magnitude_prune_model(original, density=0.5)
    counts = total_pruned(original, pruned)
    assert sum(counts.values()) == 4  # 2 pruned in each of 2 layers


def test_total_pruned_ignores_weights_already_zero() -> None:
    original = {"fc1": [0.0, 0.0, 0.1, 0.2]}
    pruned = {"fc1": [0.0, 0.0, 0.0, 0.0]}
    counts = total_pruned(original, pruned)
    assert counts == {"fc1": 2}


def test_total_pruned_rejects_layer_count_mismatch() -> None:
    original = {"fc1": [0.1, 0.2]}
    pruned = {"fc1": [0.1]}
    with pytest.raises(ValueError, match="pruning, expected"):
        total_pruned(original, pruned)


def test_total_pruned_rejects_layer_name_mismatch() -> None:
    with pytest.raises(ValueError, match="same layer names"):
        total_pruned({"a": [0.1]}, {"b": [0.0]})