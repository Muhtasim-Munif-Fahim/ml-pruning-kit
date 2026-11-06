import pytest

from prune_kit.flops import dense_macs, estimate_layer_macs, flops_report
from prune_kit.layers import conv_layer, dense_layer
from prune_kit.masks import dense_mask


def test_dense_macs_and_unstructured_scaling():
    layer = dense_layer("fc", 10, 4)
    assert dense_macs(layer) == 40
    half = estimate_layer_macs(layer, density=0.5)
    assert half.macs_dense == 40
    assert half.macs_sparse == 20
    assert half.params_kept == 20
    assert half.mac_compression == pytest.approx(2.0)


def test_mask_path_matches_density():
    layer = dense_layer("fc", 8, 8)
    mask = dense_mask(layer.shape, 0.25, seed=0)
    row = estimate_layer_macs(layer, mask=mask)
    assert row.density == pytest.approx(0.25)
    assert row.macs_sparse == pytest.approx(round(64 * 0.25))


def test_structured_conv_channels():
    layer = conv_layer("c1", out_channels=16, in_channels=8, kernel_h=3, kernel_w=3)
    dense = dense_macs(layer, spatial_hw=(32, 32))
    assert dense == 16 * 8 * 9 * 32 * 32
    pruned = estimate_layer_macs(layer, kept_out=8, kept_in=8, spatial_hw=(32, 32))
    assert pruned.macs_sparse == 8 * 8 * 9 * 32 * 32
    assert pruned.mac_compression == pytest.approx(2.0)


def test_flops_report_aggregates():
    specs = [dense_layer("a", 10, 5), dense_layer("b", 5, 2)]
    report = flops_report(specs, densities={"a": 0.5, "b": 1.0})
    assert report.params_dense == 50 + 10
    assert report.params_kept == 25 + 10
    assert report.macs_dense == 50 + 10
    assert report.macs_sparse == 25 + 10
    d = report.to_dict()
    assert len(d["layers"]) == 2


def test_rejects_bad_density():
    layer = dense_layer("fc", 4, 4)
    with pytest.raises(ValueError):
        estimate_layer_macs(layer, density=1.5)
