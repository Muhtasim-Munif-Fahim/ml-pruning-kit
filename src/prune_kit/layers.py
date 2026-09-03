"""Layer specifications used by the pruning routines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    """A tensor-shaped weight variable.

    Only the shape (and therefore the parameter count) is needed for
    the magnitude-pruning primitives; the actual values live on the
    caller's tensors. ``fan_in`` is the per-neuron input size used to
    compute fan-in-aware density targets.
    """

    name: str
    shape: tuple[int, ...]
    kind: str = "dense"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if not self.shape:
            raise ValueError("layer shape must be non-empty")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("layer shape dimensions must be positive")
        if self.kind not in {"dense", "conv"}:
            raise ValueError("kind must be 'dense' or 'conv'")


def dense_layer(name: str, in_features: int, out_features: int) -> LayerSpec:
    """Construct a fully-connected layer spec of shape (out, in)."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    return LayerSpec(name=name, shape=(out_features, in_features), kind="dense")


def conv_layer(
    name: str,
    out_channels: int,
    in_channels: int,
    kernel_h: int,
    kernel_w: int,
) -> LayerSpec:
    """Construct a 2D conv layer spec of shape (out, in, kh, kw)."""
    if out_channels <= 0 or in_channels <= 0 or kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("conv dimensions must all be positive")
    return LayerSpec(
        name=name,
        shape=(out_channels, in_channels, kernel_h, kernel_w),
        kind="conv",
    )


def layer_weight_count(spec: LayerSpec) -> int:
    """Number of scalar weights in the layer."""
    count = 1
    for dim in spec.shape:
        count *= int(dim)
    return count


__all__ = ["LayerSpec", "dense_layer", "conv_layer", "layer_weight_count"]