"""Unit tests for prune_kit.train (simulated IMP-style training)."""

from __future__ import annotations

import pytest

from prune_kit import (
    TrainingConfig,
    TrainingHistory,
    EpochResult,
    train_with_pruning,
    dense_layer,
    layer_weight_count,
)
from prune_kit.layers import LayerSpec


def make_specs():
    return [
        dense_layer("fc1", in_features=4, out_features=3),
        dense_layer("fc2", in_features=3, out_features=2),
    ]


def test_training_config_rejects_invalid_epochs():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)


def test_training_config_rejects_invalid_steps():
    with pytest.raises(ValueError):
        TrainingConfig(steps_per_epoch=0)


def test_training_config_rejects_negative_lr():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=-0.1)


def test_training_config_rejects_invalid_density():
    with pytest.raises(ValueError):
        TrainingConfig(prune_density=0.0)
    with pytest.raises(ValueError):
        TrainingConfig(prune_density=1.5)


def test_train_returns_history_with_epochs():
    specs = make_specs()
    config = TrainingConfig(epochs=3, steps_per_epoch=5, prune_density=0.5, seed=42)
    history = train_with_pruning(specs, config=config)
    assert isinstance(history, TrainingHistory)
    assert len(history.epochs) == 3
    for i, epoch_result in enumerate(history.epochs):
        assert epoch_result.epoch == i
        assert isinstance(epoch_result.weights, dict)
        assert epoch_result.loss >= 0.0
        assert 0.0 <= epoch_result.density <= 1.0


def test_train_is_deterministic_for_same_seed():
    specs = make_specs()
    config = TrainingConfig(epochs=3, steps_per_epoch=5, seed=99)
    history_a = train_with_pruning(specs, config=config)
    history_b = train_with_pruning(specs, config=config)
    assert history_a.loss_curve() == history_b.loss_curve()
    assert history_a.density_curve() == history_b.density_curve()


def test_train_pruning_reduces_density_monotonically():
    specs = make_specs()
    config = TrainingConfig(epochs=4, steps_per_epoch=5, prune_density=0.5, seed=42)
    history = train_with_pruning(specs, config=config)
    densities = history.density_curve()
    for d in densities:
        assert d <= 0.5 + 1e-9  # density never exceeds prune_density (after first prune)


def test_train_accepts_initial_weights():
    specs = make_specs()
    total = sum(layer_weight_count(s) for s in specs)
    init = {}
    cursor = 0
    for spec in specs:
        count = layer_weight_count(spec)
        init[spec.name] = [0.5 + 0.01 * i for i in range(count)]
    config = TrainingConfig(epochs=2, steps_per_epoch=3, prune_density=0.8, seed=42)
    history = train_with_pruning(specs, config=config, initial_weights=init)
    assert len(history.epochs) == 2
    assert history.epochs[0].weights["fc1"] != init["fc1"]  # pruned after training


def test_train_default_config_works():
    specs = make_specs()
    history = train_with_pruning(specs)
    assert len(history.epochs) == TrainingConfig().epochs


def test_final_weights_matches_last_epoch():
    specs = make_specs()
    config = TrainingConfig(epochs=2, steps_per_epoch=5, prune_density=0.6, seed=7)
    history = train_with_pruning(specs, config=config)
    final = history.final_weights()
    assert final == history.epochs[-1].weights


def test_loss_curve_matches_epochs():
    specs = make_specs()
    config = TrainingConfig(epochs=3, steps_per_epoch=4, prune_density=0.7, seed=3)
    history = train_with_pruning(specs, config=config)
    assert history.loss_curve() == [e.loss for e in history.epochs]
