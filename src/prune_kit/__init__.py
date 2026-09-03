"""Pruning toolkit: magnitude pruning, sparse masks, lottery-ticket reproduction, and per-layer survival reporting."""

from .layers import LayerSpec, dense_layer, conv_layer, layer_weight_count
from .masks import dense_mask, sparse_mask_to_dense, mask_density
from .prune import magnitude_prune_layer, magnitude_prune_model, total_pruned
from .survival import per_layer_survival, model_survival_summary

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
    "per_layer_survival",
    "model_survival_summary",
]