"""Tests for Zhu & Gupta gradual / polynomial magnitude pruning."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from prune_kit import (
    gradual_magnitude_prune_model,
    magnitude_prune_model,
    model_density,
    polynomial_sparsity,
    polynomial_sparsity_schedule,
)
from prune_kit.cli import main


def test_polynomial_endpoints_and_cubic() -> None:
    # Before begin → 0; at begin → initial; at end → final.
    assert polynomial_sparsity(0, final_sparsity=0.8, begin_step=10, end_step=110) == 0.0
    assert polynomial_sparsity(
        10, final_sparsity=0.8, begin_step=10, end_step=110, initial_sparsity=0.1
    ) == pytest.approx(0.1)
    assert polynomial_sparsity(
        110, final_sparsity=0.8, begin_step=10, end_step=110
    ) == pytest.approx(0.8)
    # Midpoint of cubic: progress=0.5 → (1-0.5)^3 = 0.125 → s = 0.8 + (0-0.8)*0.125
    mid = polynomial_sparsity(
        60, final_sparsity=0.8, begin_step=10, end_step=110, frequency=1, exponent=3.0
    )
    assert mid == pytest.approx(0.8 - 0.8 * 0.125)


def test_polynomial_staircase_holds_between_events() -> None:
    a = polynomial_sparsity(
        20, final_sparsity=0.5, begin_step=0, end_step=100, frequency=20
    )
    b = polynomial_sparsity(
        39, final_sparsity=0.5, begin_step=0, end_step=100, frequency=20
    )
    assert a == pytest.approx(b)
    c = polynomial_sparsity(
        40, final_sparsity=0.5, begin_step=0, end_step=100, frequency=20
    )
    assert c >= a - 1e-12


def test_schedule_includes_end_and_is_nondecreasing() -> None:
    sched = polynomial_sparsity_schedule(
        final_sparsity=0.9, begin_step=0, end_step=100, frequency=25, exponent=3.0
    )
    steps = [s for s, _ in sched]
    assert steps[0] == 0
    assert steps[-1] == 100
    targets = [t for _, t in sched]
    assert targets[-1] == pytest.approx(0.9)
    assert all(targets[i] <= targets[i + 1] + 1e-12 for i in range(len(targets) - 1))


def test_gradual_without_update_matches_oneshot() -> None:
    model = {
        "fc1": [0.1 * i for i in range(1, 11)],
        "fc2": [0.05 * i for i in range(1, 11)],
    }
    result = gradual_magnitude_prune_model(
        model,
        final_sparsity=0.5,
        begin_step=0,
        end_step=50,
        frequency=10,
        scope="layer",
    )
    oneshot = magnitude_prune_model(model, density=0.5)
    for name in model:
        assert result.weights[name] == pytest.approx(oneshot[name])
    assert result.final_density() == pytest.approx(0.5)


def test_gradual_global_scope_oneshot() -> None:
    model = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [0.1, 0.2, 0.3, 0.4],
    }
    result = gradual_magnitude_prune_model(
        model,
        final_sparsity=0.5,
        begin_step=0,
        end_step=10,
        frequency=5,
        scope="global",
    )
    # Keep top 4 of 8 globally by |w|: 4,3,2,1 from a.
    assert result.weights["a"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result.weights["b"] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.final_density() == pytest.approx(0.5)


def test_masks_are_monotone_with_update_fn() -> None:
    model = {"fc": [float(i) for i in range(1, 9)]}  # 1..8

    def update_fn(step, weights):
        # Try to regrow pruned weights aggressively.
        return {name: [10.0] * len(layer) for name, layer in weights.items()}

    result = gradual_magnitude_prune_model(
        model,
        final_sparsity=0.5,
        begin_step=0,
        end_step=4,
        frequency=2,
        scope="layer",
        update_fn=update_fn,
    )
    # Once pruned, stays zero despite update_fn writing 10s.
    zeros = [i for i, v in enumerate(result.weights["fc"]) if v == 0.0]
    assert len(zeros) == 4
    assert all(result.masks["fc"][i] == 0 for i in zeros)
    # Surviving entries were overwritten to 10 then re-masked.
    assert all(v in (0.0, 10.0) for v in result.weights["fc"])


def test_gradual_does_not_modify_input() -> None:
    model = {"fc": [0.1, 0.2, 0.3, 0.4]}
    original = {"fc": list(model["fc"])}
    gradual_magnitude_prune_model(
        model, final_sparsity=0.5, begin_step=0, end_step=2, frequency=1
    )
    assert model == original


def test_target_and_sparsity_curves() -> None:
    model = {"fc": [float(i) for i in range(20)]}
    result = gradual_magnitude_prune_model(
        model,
        final_sparsity=0.8,
        begin_step=0,
        end_step=100,
        frequency=25,
        scope="layer",
    )
    assert len(result.steps) == len(result.target_curve())
    assert result.target_curve()[-1] == pytest.approx(0.8)
    assert result.sparsity_curve()[-1] == pytest.approx(0.8)
    assert abs(sum(result.density_curve()[-1] for _ in [0]) + result.sparsity_curve()[-1] - 1.0) < 1e-9


def test_rejects_bad_schedule_args() -> None:
    with pytest.raises(ValueError):
        polynomial_sparsity(0, final_sparsity=1.5, end_step=10)
    with pytest.raises(ValueError):
        polynomial_sparsity(0, final_sparsity=0.5, end_step=10, initial_sparsity=0.9)
    with pytest.raises(ValueError):
        polynomial_sparsity(0, final_sparsity=0.5, begin_step=20, end_step=10)
    with pytest.raises(ValueError):
        polynomial_sparsity(0, final_sparsity=0.5, end_step=10, frequency=0)
    with pytest.raises(ValueError):
        polynomial_sparsity(0, final_sparsity=0.5, end_step=10, exponent=0.0)
    with pytest.raises(ValueError):
        gradual_magnitude_prune_model({}, final_sparsity=0.5, end_step=1)
    with pytest.raises(ValueError):
        gradual_magnitude_prune_model(
            {"fc": [1.0]}, final_sparsity=0.5, end_step=5, scope="weird"
        )


def test_update_fn_must_preserve_structure() -> None:
    model = {"fc": [1.0, 2.0, 3.0, 4.0]}

    def bad_keys(step, weights):
        return {"other": [1.0, 2.0, 3.0, 4.0]}

    with pytest.raises(ValueError, match="same layer"):
        gradual_magnitude_prune_model(
            model,
            final_sparsity=0.5,
            end_step=2,
            frequency=1,
            update_fn=bad_keys,
        )


def test_cli_gradual_json() -> None:
    weights = ",".join(str(0.1 * i) for i in range(1, 21))
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = main([
            "gradual",
            "--specs", "fc1=dense:4x5",
            "--weights", weights,
            "--final-sparsity", "0.5",
            "--end-step", "40",
            "--frequency", "10",
            "--json",
        ])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["final_sparsity"] == 0.5
    assert payload["final_density"] == pytest.approx(0.5)
    assert payload["layers"][0]["kept"] == 10
    assert len(payload["steps"]) >= 2


def test_cli_gradual_markdown() -> None:
    weights = ",".join(str(float(i)) for i in range(1, 9))
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(io.StringIO()):
        code = main([
            "gradual",
            "--specs", "fc=dense:2x4",
            "--weights", weights,
            "--final-sparsity", "0.5",
            "--end-step", "10",
            "--frequency", "5",
        ])
    assert code == 0
    text = buf.getvalue()
    assert "Gradual magnitude pruning" in text
    assert "final_density" in text


def test_export_available() -> None:
    from prune_kit import (
        GradualPruneResult,
        GradualPruneStep,
        gradual_magnitude_prune_model as g,
        polynomial_sparsity as ps,
        polynomial_sparsity_schedule as pss,
    )

    assert callable(g) and callable(ps) and callable(pss)
    assert GradualPruneResult is not None
    assert GradualPruneStep is not None
