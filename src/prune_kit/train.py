"""Training integration: simulate iterative magnitude pruning (IMP).

This module provides a lightweight, framework-free training loop that
applies magnitude pruning after each epoch and tracks survival across
pruning rounds.  It mirrors the Iterative Magnitude Pruning (IMP) workflow
without requiring a real neural-network framework — weights are simple
flat lists of floats and the loss is a toy quadratic proxy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .layers import LayerSpec, layer_weight_count
from .prune import magnitude_prune_model


@dataclass
class TrainingConfig:
    """Hyper-parameters for the simulated training loop."""

    epochs: int = 5
    steps_per_epoch: int = 20
    learning_rate: float = 0.01
    noise_scale: float = 0.05
    prune_density: float = 0.7
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if not 0.0 < self.prune_density <= 1.0:
            raise ValueError("prune_density must be in (0, 1]")


@dataclass
class EpochResult:
    """Snapshot of weights and loss after a single training epoch."""

    epoch: int
    weights: Dict[str, List[float]]
    loss: float
    density: float


@dataclass
class TrainingHistory:
    """Aggregate result of an IMP-style training+pruning run."""

    config: TrainingConfig
    epochs: List[EpochResult] = field(default_factory=list)

    def final_weights(self) -> Dict[str, List[float]]:
        """Return the weights after the final epoch (post-pruning)."""
        if not self.epochs:
            raise ValueError("no epochs have been run")
        return self.epochs[-1].weights

    def loss_curve(self) -> List[float]:
        return [e.loss for e in self.epochs]

    def density_curve(self) -> List[float]:
        return [e.density for e in self.epochs]


def _initial_weights(
    specs: Sequence[LayerSpec],
    *,
    seed: int,
) -> Dict[str, List[float]]:
    """Generate deterministic pseudo-random initial weights for each layer."""
    rng = random.Random(seed)
    weights: Dict[str, List[float]] = {}
    for spec in specs:
        count = layer_weight_count(spec)
        weights[spec.name] = [rng.gauss(0.0, 0.1) for _ in range(count)]
    return weights


def _loss(
    weights: Dict[str, List[float]],
    target: float = 1.0,
) -> float:
    """Toy loss: mean squared distance of all weight magnitudes from *target*."""
    if not weights:
        raise ValueError("weights must not be empty")
    total_sq = 0.0
    total = 0
    for layer_weights in weights.values():
        for value in layer_weights:
            total_sq += (abs(value) - target) ** 2
            total += 1
    return total_sq / total if total else 0.0


def _step(
    weights: Dict[str, List[float]],
    learning_rate: float,
    noise_scale: float,
    *,
    rng: random.Random,
) -> None:
    """Apply one gradient-descent-like step to *weights* in place.

    The 'gradient' is a synthetic signal that pushes weight magnitudes
    toward unity; Gaussian noise simulates stochasticity.
    """
    for layer_weights in weights.values():
        for i, value in enumerate(layer_weights):
            if value == 0.0:
                continue
            grad = value - math.copysign(1.0, value)
            value -= learning_rate * grad
            if noise_scale > 0.0:
                value += noise_scale * rng.gauss(0.0, 1.0)
            layer_weights[i] = value


def train_with_pruning(
    specs: Sequence[LayerSpec],
    *,
    config: TrainingConfig | None = None,
    initial_weights: Dict[str, List[float]] | None = None,
) -> TrainingHistory:
    """Run a simulated training loop with iterative magnitude pruning.

    Parameters
    ----------
    specs:
        Layer specifications defining the model topology.
    config:
        Training hyper-parameters (defaults to sensible values).
    initial_weights:
        Optional pre-computed initial weights.  When ``None`` a fresh
        deterministic set is generated from ``config.seed``.
    """
    config = config or TrainingConfig()
    weights = (
        initial_weights
        if initial_weights is not None
        else _initial_weights(specs, seed=config.seed)
    )

    rng = random.Random(config.seed)
    history = TrainingHistory(config=config)

    for epoch in range(config.epochs):
        for _ in range(config.steps_per_epoch):
            _step(weights, config.learning_rate, config.noise_scale, rng=rng)

        pruned = magnitude_prune_model(
            weights, density=config.prune_density
        )
        weights = pruned
        loss = _loss(weights)
        total = sum(layer_weight_count(spec) for spec in specs)
        kept = sum(
            1 for layer_weights in weights.values()
            for value in layer_weights
            if value != 0.0
        )
        density = kept / total if total else 0.0
        history.epochs.append(EpochResult(
            epoch=epoch,
            weights={k: list(v) for k, v in weights.items()},
            loss=loss,
            density=density,
        ))

    return history


__all__ = [
    "TrainingConfig",
    "EpochResult",
    "TrainingHistory",
    "train_with_pruning",
]
