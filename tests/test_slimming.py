"""Tests for Network Slimming (BN-scale channel pruning, Liu et al. 2017)."""

from __future__ import annotations

import json

import pytest

from prune_kit import (
    slimming_channel_masks,
    slimming_l1_penalty,
    slimming_prune_model,
    slimming_prune_summary,
    slimming_subgradient,
    slimming_threshold,
)
from prune_kit.cli import main
from prune_kit.layers import conv_layer, dense_layer


def _net():
    specs = [
        conv_layer("conv1", 4, 1, 1, 2),   # 4 filters x 2 weights
        conv_layer("conv2", 2, 4, 1, 1),   # 2 filters x 4 inputs
        dense_layer("fc", 2, 3),           # (3, 2): fed by conv2's 2 channels
    ]
    model = {
        "conv1": [float(i + 1) for i in range(8)],
        "conv2": [float(i + 1) for i in range(8)],
        "fc": [float(i + 1) for i in range(6)],
    }
    return specs, model


def test_penalty_and_subgradient():
    scales = {"a": [0.5, -0.25, 0.0], "b": [2.0]}
    assert slimming_l1_penalty(scales, lam=0.1) == pytest.approx(0.1 * 2.75)
    assert slimming_subgradient(scales, lam=0.1) == {"a": [0.1, -0.1, 0.0], "b": [0.1]}
    with pytest.raises(ValueError, match="lam"):
        slimming_l1_penalty(scales, lam=-1.0)


def test_global_threshold_gives_layers_different_widths():
    scales = {
        "conv1": [0.9, 0.01, 0.8, 0.02],   # two near-zero channels
        "conv2": [0.7, 0.6],               # all important
    }
    masks = slimming_channel_masks(scales, prune_ratio=1 / 3)
    assert masks == {"conv1": [1, 0, 1, 0], "conv2": [1, 1]}
    assert slimming_threshold(scales, prune_ratio=1 / 3) == pytest.approx(0.02)
    assert slimming_threshold(scales, prune_ratio=0.0) is None


def test_uses_absolute_value_of_gamma():
    scales = {"conv1": [-0.9, 0.05, 0.3, -0.01]}
    assert slimming_channel_masks(scales, 0.5) == {"conv1": [1, 0, 1, 0]}


def test_min_channels_protects_each_layer():
    scales = {"conv1": [0.9, 0.8, 0.7, 0.6], "conv2": [0.01, 0.02]}
    masks = slimming_channel_masks(scales, prune_ratio=0.5, min_channels=1)
    assert masks["conv2"] == [0, 1]  # its largest scale survives
    assert sum(masks["conv1"]) + sum(masks["conv2"]) == 3
    # the protected slot is filled by the next-smallest channels elsewhere
    assert masks["conv1"] == [1, 1, 0, 0]
    unprotected = slimming_channel_masks(scales, prune_ratio=0.5, min_channels=0)
    assert unprotected["conv2"] == [0, 0]


def test_ties_are_deterministic():
    scales = {"b": [0.1, 0.1], "a": [0.1, 0.1]}
    assert slimming_channel_masks(scales, 0.5, min_channels=0) == {"b": [1, 1], "a": [0, 0]}


def test_prune_model_zeroes_filters_and_optional_next_inputs():
    specs, model = _net()
    scales = {"conv1": [0.9, 0.01, 0.8, 0.02], "conv2": [0.5, 0.001]}
    out = slimming_prune_model(specs, model, scales, prune_ratio=0.5)
    # conv1 filters 1 and 3 (weights 2..3 and 6..7) zeroed; conv2 filter 1 zeroed
    assert out["conv1"] == [1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
    assert out["conv2"] == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    assert out["fc"] == model["fc"]
    chained = slimming_prune_model(specs, model, scales, prune_ratio=0.5, prune_next_inputs=True)
    # conv2's input channels 1 and 3 follow conv1's pruned filters
    assert chained["conv2"] == [1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # fc column 1 follows conv2's pruned channel
    assert chained["fc"] == [1.0, 0.0, 3.0, 0.0, 5.0, 0.0]
    assert model["conv1"] == [float(i + 1) for i in range(8)]  # input untouched


def test_summary_rows():
    specs, model = _net()
    scales = {"conv1": [0.9, 0.01, 0.8, 0.02], "conv2": [0.5, 0.001]}
    summary = slimming_prune_summary(specs, model, scales, prune_ratio=0.5, prune_next_inputs=True)
    assert summary["total_channels"] == 6 and summary["kept_channels"] == 3
    assert summary["channel_survival"] == pytest.approx(0.5)
    rows = {row["layer"]: row for row in summary["layers"]}
    assert rows["conv1"]["kept_channels"] == 2
    assert rows["fc"]["channels"] is None and rows["fc"]["inputs_pruned"] is True
    assert rows["conv2"]["nonzero_weights"] == 2


@pytest.mark.parametrize("scales, match", [
    ({"nope": [1.0]}, "unknown layer"),
    ({"conv1": [1.0, 2.0]}, "output channels"),
    ({}, "at least one layer"),
    ({"conv1": [1.0, float("nan"), 1.0, 1.0]}, "non-finite"),
])
def test_validation(scales, match):
    specs, model = _net()
    with pytest.raises(ValueError, match=match):
        slimming_prune_model(specs, model, scales)


def test_ratio_validation():
    with pytest.raises(ValueError, match="prune_ratio"):
        slimming_channel_masks({"a": [1.0]}, prune_ratio=1.0)
    with pytest.raises(ValueError, match="min_channels"):
        slimming_channel_masks({"a": [1.0]}, min_channels=-1)


def test_cli_markdown_and_json(capsys):
    args = [
        "slim",
        "--specs", "conv1=conv:4x1x1x2,conv2=conv:2x4x1x1",
        "--weights", ",".join(str(float(i + 1)) for i in range(16)),
        "--bn-scales", "conv1=0.9:0.01:0.8:0.02,conv2=0.5:0.001",
        "--prune-ratio", "0.5",
        "--prune-next-inputs",
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "Network Slimming" in text and "| conv1 | conv | 2/4 |" in text
    assert main(args + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["masks"] == {"conv1": [1, 0, 1, 0], "conv2": [1, 0]}
    assert payload["threshold"] == pytest.approx(0.02)


def test_cli_errors(capsys):
    assert main([
        "slim", "--specs", "conv1=conv:2x1x1x1", "--weights", "1,2",
        "--bn-scales", "conv1=0.1",
    ]) == 2
    assert "output channels" in capsys.readouterr().err
    assert main([
        "slim", "--specs", "conv1=conv:2x1x1x1", "--weights", "1,2",
        "--bn-scales", "conv1",
    ]) == 2
    assert "name=g1:g2" in capsys.readouterr().err
