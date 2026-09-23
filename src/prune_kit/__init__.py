"""Pruning toolkit (full public surface)."""

from .global_unstructured import (
    GlobalPruneResult,
    global_magnitude_prune_model,
    iterative_global_magnitude_prune_model,
)
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
from .structured import (
    channel_groups,
    channel_norms,
    structured_keep_indices,
    structured_prune_layer,
    structured_prune_model,
    vector_norm,
)
from .survival import (
    model_channel_survival_summary,
    model_survival_summary,
    per_channel_survival,
    per_layer_survival,
)
from .taylor import (
    taylor_channel_scores,
    taylor_keep_indices,
    taylor_prune_layer,
    taylor_prune_model,
    taylor_prune_summary,
)
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
    "channel_groups",
    "channel_norms",
    "vector_norm",
    "structured_keep_indices",
    "structured_prune_layer",
    "structured_prune_model",
    "per_layer_survival",
    "model_survival_summary",
    "per_channel_survival",
    "model_channel_survival_summary",
    "taylor_channel_scores",
    "taylor_keep_indices",
    "taylor_prune_layer",
    "taylor_prune_model",
    "taylor_prune_summary",
    "TrainingConfig",
    "EpochResult",
    "TrainingHistory",
    "train_with_pruning",
    "GlobalPruneResult",
    "global_magnitude_prune_model",
    "iterative_global_magnitude_prune_model",
]