"""Tests for the layers module."""

from __future__ import annotations

import pytest

from prune_kit import LayerSpec, conv_layer, dense_layer, layer_weight_count


def test_dense_layer_shape() -> None:
    spec = dense_layer("fc1", in_features=784, out_features=256)
    assert spec.name == "fc1"
    assert spec.shape == (256, 784)
    assert spec.kind == "dense"


def test_dense_layer_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        dense_layer("fc1", in_features=0, out_features=10)
    with pytest.raises(ValueError):
        dense_layer("fc1", in_features=10, out_features=0)


def test_conv_layer_shape() -> None:
    spec = conv_layer("conv1", out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
    assert spec.shape == (16, 3, 3, 3)
    assert spec.kind == "conv"


def test_conv_layer_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        conv_layer("conv1", out_channels=0, in_channels=3, kernel_h=3, kernel_w=3)
    with pytest.raises(ValueError):
        conv_layer("conv1", out_channels=16, in_channels=3, kernel_h=0, kernel_w=3)


def test_layer_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LayerSpec(name="", shape=(3, 4))


def test_layer_spec_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LayerSpec(name="x", shape=())


def test_layer_spec_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="positive"):
        LayerSpec(name="x", shape=(0, 4))


def test_layer_spec_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        LayerSpec(name="x", shape=(3, 4), kind="recurrent")


def test_layer_weight_count_dense() -> None:
    spec = dense_layer("fc1", in_features=784, out_features=256)
    assert layer_weight_count(spec) == 784 * 256


def test_layer_weight_count_conv() -> None:
    spec = conv_layer("conv1", out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
    assert layer_weight_count(spec) == 16 * 3 * 3 * 3


def test_layer_weight_count_single_dim() -> None:
    spec = LayerSpec(name="bias", shape=(10,))
    assert layer_weight_count(spec) == 10
