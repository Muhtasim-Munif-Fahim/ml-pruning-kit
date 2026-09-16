"""Pruning toolkit (full public surface)."""

from .layers import LayerSpec, dense_layer, conv_layer, layer_weight_count
from .masks import dense_mask, mask_density, sparse_mask_to_dense
from .prune import (
    IterativePruneResult,
    PruneStep,
    iterative_magnitude_prune_layer,
    iterative_magnitude_prune_model,
    magnitude_prune_layer,
    magnitude_prune_model,
    model_density,
    rewind_layer,
    rewind_model,
    total_pruned,
    weight_density,
)
from .survival import model_survival_summary, per_layer_survival
from .train import (
    EpochResult,
    TrainingConfig,
    TrainingHistory,
    train_with_pruning,
)

__all__ = [
    "LayerSpec",
    "dense_layer",
    "conv_layer",
    "layer_weight_count",
    "dense_mask",
    "sparse_mask_to_dense",
    "mask_density",
    "magnitude_prune_layer",
    "magnitude_prune_model",
    "total_pruned",
    "weight_density",
    "model_density",
    "rewind_layer",
    "rewind_model",
    "iterative_magnitude_prune_layer",
    "iterative_magnitude_prune_model",
    "PruneStep",
    "IterativePruneResult",
    "per_layer_survival",
    "model_survival_summary",
    "TrainingConfig",
    "EpochResult",
    "TrainingHistory",
    "train_with_pruning",
]