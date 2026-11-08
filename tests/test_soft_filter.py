import math

import pytest

from prune_kit import (
    asfp_rate_schedule,
    constant_rate_schedule,
    conv_layer,
    dense_layer,
    soft_filter_prune_layer,
    soft_filter_pruning,
)
from prune_kit.structured import channel_groups, channel_norms


def _filters(values_per_filter, inner=4):
    w = []
    for v in values_per_filter:
        w.extend([float(v)] * inner)
    return w


def test_asfp_schedule_anchor_points():
    sched = asfp_rate_schedule(0.4, 40)
    assert len(sched) == 40
    assert sched[-1] == pytest.approx(0.4)
    assert sched[9] == pytest.approx(0.75 * 0.4)  # epoch E/4 = 10
    assert all(b > a for a, b in zip(sched, sched[1:]))
    # concave ramp: first increments are the largest
    assert sched[0] > sched[1] - sched[0] > 0
    lin_like = asfp_rate_schedule(0.4, 40, quarter_fraction=0.26)
    assert lin_like[9] == pytest.approx(0.26 * 0.4, rel=1e-6)
    with pytest.raises(ValueError):
        asfp_rate_schedule(0.4, 40, quarter_fraction=0.25)


def test_constant_schedule():
    assert constant_rate_schedule(0.3, 3) == [0.3, 0.3, 0.3]
    with pytest.raises(ValueError):
        constant_rate_schedule(1.0, 3)
    with pytest.raises(ValueError):
        constant_rate_schedule(0.3, 0)


def test_soft_prune_layer_zeroes_lowest_norm_filters():
    shape = (5, 1, 2, 2)
    w = _filters([3.0, 0.1, 2.0, 0.2, 1.0])
    new, zeroed = soft_filter_prune_layer(w, shape, 0.4)
    assert zeroed == [1, 3]
    norms = channel_norms(new, shape, norm="l2")
    assert norms[1] == 0.0 and norms[3] == 0.0
    assert norms[0] == pytest.approx(6.0)
    assert w[4] == pytest.approx(0.1)  # input not modified
    # never zero every filter
    _, all_but = soft_filter_prune_layer(w, shape, 0.99)
    assert len(all_but) == 4


def _setup():
    specs = [conv_layer("c1", 6, 1, 2, 2), dense_layer("fc", 6, 2)]
    c1 = _filters([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    model = {"c1": c1, "fc": [0.5] * 12}
    return specs, model


def test_sfp_lets_needed_filters_regrow():
    specs, model = _setup()
    groups = channel_groups((6, 1, 2, 2))

    def train_epoch(m, epoch):
        # From epoch 2 on the loss "needs" filter 5 (smallest at the start):
        # training regrows it past filter 3, so SFP brings it back.
        if epoch >= 2:
            for idx in groups[5]:
                m["c1"][idx] += 3.5
        return m

    res = soft_filter_pruning(specs, model, train_epoch, epochs=4, rate=1 / 3)
    assert res.history[0].zeroed["c1"] == [4, 5]
    assert res.history[1].zeroed["c1"] == [3, 4]
    assert res.history[1].revived["c1"] == [5]
    assert 5 in res.kept_filters["c1"]
    assert res.total_revived == 1
    assert len(res.pruned_filters["c1"]) == 2
    norms = channel_norms(res.model["c1"], (6, 1, 2, 2), norm="l2")
    assert all(norms[i] == 0.0 for i in res.pruned_filters["c1"])
    assert res.model["fc"] == model["fc"]
    assert res.filter_density()["c1"] == pytest.approx(4 / 6)


def test_asymptotic_rate_ramps_number_of_zeroed_filters():
    specs = [conv_layer("c", 20, 1, 1, 1)]
    model = {"c": [float(i + 1) for i in range(20)]}
    res = soft_filter_pruning(specs, model, lambda m, e: m, epochs=8, rate=0.5, schedule="asymptotic")
    counts = [len(h.zeroed["c"]) for h in res.history]
    assert counts[-1] == 10
    assert counts[0] < counts[-1]
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert counts == [min(10, round(r * 20)) for r in res.schedule]


def test_per_layer_rates_and_validation():
    specs = [conv_layer("a", 10, 1, 1, 1), conv_layer("b", 10, 1, 1, 1)]
    model = {"a": [float(i + 1) for i in range(10)], "b": [float(i + 1) for i in range(10)]}
    res = soft_filter_pruning(specs, model, lambda m, e: m, epochs=2, rate=0.5, per_layer={"b": 0.2})
    assert len(res.pruned_filters["a"]) == 5
    assert len(res.pruned_filters["b"]) == 2
    with pytest.raises(ValueError):
        soft_filter_pruning(specs, model, lambda m, e: m, epochs=2, rate=0.5, schedule="nope")
    with pytest.raises(ValueError):
        soft_filter_pruning([dense_layer("d", 2, 2)], {"d": [1.0] * 4}, lambda m, e: m, epochs=1, rate=0.5)
    with pytest.raises(ValueError):
        soft_filter_pruning(specs, model, lambda m, e: {"a": m["a"]}, epochs=1, rate=0.5)
    assert not math.isnan(res.schedule[0])
