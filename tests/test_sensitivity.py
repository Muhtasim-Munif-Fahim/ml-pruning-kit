import pytest

from prune_kit import (
    default_density_grid,
    layer_sensitivity,
    magnitude_prune_layer,
    model_density,
    sensitivity_analysis,
    sensitivity_densities,
    sensitivity_prune_model,
)


def _model():
    # "robust" has many tiny weights: pruning them barely changes the output.
    robust = [5.0, 4.0] + [0.001 * (i + 1) for i in range(18)]
    # "fragile" has evenly sized weights: every pruned weight hurts.
    fragile = [1.0 + 0.01 * i for i in range(20)]
    return {"robust": robust, "fragile": fragile}


def _evaluate(model):
    # Score = negative L1 distance of each layer's sum to the dense sum.
    ref = _model()
    return -sum(abs(sum(model[k]) - sum(ref[k])) for k in ref)


def test_default_grid():
    g = default_density_grid(10, 0.1)
    assert g[0] == 1.0 and g[-1] == pytest.approx(0.1) and len(g) == 10
    with pytest.raises(ValueError):
        default_density_grid(1)
    with pytest.raises(ValueError):
        default_density_grid(5, 0.0)


def test_layer_sensitivity_only_prunes_target_layer():
    seen = []

    def ev(m):
        seen.append(model_density({"fragile": m["fragile"]}))
        return _evaluate(m)

    curve = layer_sensitivity(_model(), "robust", ev, [1.0, 0.5, 0.1])
    assert all(d == 1.0 for d in seen)
    assert curve.densities == (1.0, 0.5, 0.1)
    assert curve.scores[0] == pytest.approx(curve.baseline)
    assert curve.degradation()[0] == pytest.approx(0.0)
    assert list(curve.degradation()) == sorted(curve.degradation())


def test_ranking_and_auc():
    a = sensitivity_analysis(_model(), _evaluate, default_density_grid(10, 0.1))
    assert a.ranking() == ["fragile", "robust"]
    assert a.curves["fragile"].auc_degradation() > 10 * a.curves["robust"].auc_degradation()


def test_densities_allocation():
    a = sensitivity_analysis(_model(), _evaluate, default_density_grid(10, 0.1))
    d = sensitivity_densities(a, tolerance=0.2)
    assert d["robust"] == pytest.approx(0.1)
    assert d["fragile"] == 1.0
    loose = sensitivity_densities(a, tolerance=5.0)
    assert loose["fragile"] < 1.0


def test_relative_tolerance_and_loss_direction():
    def loss(m):
        return 1.0 - _evaluate(m)  # loss with baseline 1.0

    a = sensitivity_analysis(_model(), loss, [1.0, 0.5, 0.1], higher_is_better=False)
    assert a.baseline == pytest.approx(1.0)
    assert all(x >= -1e-12 for c in a.curves.values() for x in c.degradation())
    d = sensitivity_densities(a, tolerance=0.2, relative=True)
    assert d["robust"] == pytest.approx(0.1)
    assert d["fragile"] == 1.0


def test_monotone_stops_at_first_violation():
    calls = iter([0.0, 0.0, -1.0, 0.0])  # baseline, d=1, d=0.5 fails, d=0.1 "recovers"

    a = sensitivity_analysis({"x": [1.0, 2.0, 3.0, 4.0]}, lambda m: next(calls), [1.0, 0.5, 0.1])
    assert sensitivity_densities(a, 0.1)["x"] == 1.0
    assert sensitivity_densities(a, 0.1, monotone=False)["x"] == pytest.approx(0.1)


def test_custom_prune_fn_and_length_check():
    used = []

    def pf(w, d):
        used.append(d)
        return magnitude_prune_layer(w, density=d)

    sensitivity_analysis(_model(), _evaluate, [1.0, 0.5], prune_fn=pf)
    assert used == [0.5, 0.5]
    with pytest.raises(ValueError):
        layer_sensitivity(_model(), "robust", _evaluate, [0.5], prune_fn=lambda w, d: [0.0])


def test_sensitivity_prune_model_end_to_end():
    m = _model()
    r = sensitivity_prune_model(m, _evaluate, tolerance=0.2, densities=default_density_grid(10, 0.1))
    assert r.densities["fragile"] == 1.0
    assert model_density({"robust": r.model["robust"]}) == pytest.approx(0.1)
    assert r.model["fragile"] == m["fragile"]
    assert r.overall_density == pytest.approx((2 + 20) / 40)
    assert r.final_score >= r.analysis.baseline - 0.2
    assert r.notes == []
    assert m == _model()  # input untouched


def test_validation():
    with pytest.raises(ValueError):
        layer_sensitivity(_model(), "nope", _evaluate)
    with pytest.raises(ValueError):
        sensitivity_analysis({}, _evaluate)
    with pytest.raises(ValueError):
        sensitivity_analysis(_model(), _evaluate, [1.5])
    with pytest.raises(ValueError):
        sensitivity_analysis(_model(), _evaluate, layers=["nope"])
    a = sensitivity_analysis(_model(), _evaluate, [1.0, 0.5])
    with pytest.raises(ValueError):
        sensitivity_densities(a, -1.0)
