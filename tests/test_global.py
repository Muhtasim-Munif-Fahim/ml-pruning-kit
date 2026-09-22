"""Tests for global unstructured magnitude pruning."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from prune_kit import (
    GlobalPruneResult,
    global_magnitude_prune_model,
    iterative_global_magnitude_prune_model,
    magnitude_prune_layer,
    magnitude_prune_model,
)
from prune_kit.cli import main


def test_global_prune_zeros_smallest_magnitudes_across_layers() -> None:
    model = {
        "small": [0.1, 0.2, 0.3, 0.4],
        "large": [1.0, 2.0, 3.0, 4.0],
    }
    pruned = global_magnitude_prune_model(model, sparsity=0.5)
    assert pruned["small"] == [0.0, 0.0, 0.0, 0.0]
    assert pruned["large"] == [1.0, 2.0, 3.0, 4.0]
    per_layer = magnitude_prune_model(model, density=0.5)
    assert pruned != per_layer


def test_global_prune_uses_absolute_value() -> None:
    model = {"fc": [-0.5, 0.1, -0.9, 0.2]}
    pruned = global_magnitude_prune_model(model, sparsity=0.5)
    assert pruned["fc"] == [-0.5, 0.0, -0.9, 0.0]


def test_single_layer_matches_per_layer_magnitude() -> None:
    weights = [0.1, -0.7, 0.3, -0.2, 0.9, 0.05]
    pruned = global_magnitude_prune_model({"fc": weights}, sparsity=0.5)
    expected = magnitude_prune_layer(weights, density=0.5)
    assert pruned["fc"] == expected


def test_tie_keeps_earlier_layer_and_lower_index() -> None:
    model = {"a": [0.2, 0.5], "b": [0.2, 0.9]}
    pruned = global_magnitude_prune_model(model, sparsity=0.25)
    assert pruned == {"a": [0.2, 0.5], "b": [0.0, 0.9]}


def test_tie_within_layer_keeps_lower_index() -> None:
    model = {"fc": [0.4, 0.4, 0.1]}
    pruned = global_magnitude_prune_model(model, sparsity=2.0 / 3.0)
    assert pruned["fc"] == [0.4, 0.0, 0.0]
    assert magnitude_prune_layer(model["fc"], density=1.0 / 3.0) == [0.4, 0.0, 0.0]


def test_sparsity_zero_keeps_everything() -> None:
    model = {"fc": [0.1, 0.2], "out": [-0.3]}
    assert global_magnitude_prune_model(model, sparsity=0.0) == {
        "fc": [0.1, 0.2],
        "out": [-0.3],
    }


def test_sparsity_one_zeros_everything() -> None:
    model = {"fc": [0.1, 0.9], "out": [0.4]}
    assert global_magnitude_prune_model(model, sparsity=1.0) == {
        "fc": [0.0, 0.0],
        "out": [0.0],
    }


def test_already_sparse_model_is_not_filled_back_in() -> None:
    model = {"fc": [0.0, 0.0, 0.5, 1.0]}
    pruned = global_magnitude_prune_model(model, sparsity=0.25)
    assert pruned["fc"] == [0.0, 0.0, 0.5, 1.0]


def test_global_prune_does_not_modify_input() -> None:
    model = {"fc": [0.1, 0.9, 0.2], "out": [0.3, 0.8]}
    original = {name: list(weights) for name, weights in model.items()}
    global_magnitude_prune_model(model, sparsity=0.5)
    assert model == original


def test_global_prune_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        global_magnitude_prune_model({})


def test_global_prune_rejects_empty_layer() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        global_magnitude_prune_model({"fc": []})


def test_global_prune_rejects_invalid_sparsity() -> None:
    model = {"fc": [0.1, 0.2]}
    with pytest.raises(ValueError, match="sparsity"):
        global_magnitude_prune_model(model, sparsity=-0.1)
    with pytest.raises(ValueError, match="sparsity"):
        global_magnitude_prune_model(model, sparsity=1.1)


def test_iterative_final_mask_matches_one_shot() -> None:
    model = {
        "a": [0.1, 0.4, 0.2, 0.5],
        "b": [0.3, 0.9, 0.15, 0.8],
    }
    one_shot = global_magnitude_prune_model(model, sparsity=0.5)
    result = iterative_global_magnitude_prune_model(
        model, sparsity=0.5, rounds=3
    )
    assert isinstance(result, GlobalPruneResult)
    assert result.weights == one_shot
    assert result.rounds == 3
    assert len(result.steps) == 3
    assert result.density_curve() == [step.density for step in result.steps]
    assert result.final_density() == result.steps[-1].density
    densities = result.density_curve()
    assert densities == sorted(densities, reverse=True)
    # 8 weights, keep round(0.5*8)=4. Even split of 4 removals over 3 rounds.
    assert result.steps[0].kept == 6
    assert result.steps[1].kept == 5
    assert result.steps[2].kept == 4
    assert result.schedule == pytest.approx([2 / 8, 3 / 8, 4 / 8])


def test_iterative_rewind_restores_survivors_and_keeps_trained_mask() -> None:
    trained = {
        "small": [0.1, 0.2, 0.3, 0.4],
        "large": [1.0, 2.0, 3.0, 4.0],
    }
    initial = {
        "small": [0.5, 0.5, 0.5, 0.5],
        "large": [0.5, 0.5, 0.5, 0.5],
    }
    result = iterative_global_magnitude_prune_model(
        trained,
        sparsity=0.5,
        rounds=2,
        rewind=True,
        initial_weights=initial,
    )
    assert result.weights == {
        "small": [0.0, 0.0, 0.0, 0.0],
        "large": [0.5, 0.5, 0.5, 0.5],
    }
    assert result.steps[0].weights == {
        "small": [0.0, 0.0, 0.5, 0.5],
        "large": [0.5, 0.5, 0.5, 0.5],
    }
    assert result.rewind is True
    assert result.density_curve() == pytest.approx([0.75, 0.5])


def test_iterative_rewind_ranks_trained_magnitudes_across_rounds() -> None:
    trained = {"a": [0.4, 0.1, 0.35], "b": [0.3, 0.2]}
    initial = {"a": [0.01, 5.0, 0.02], "b": [4.0, 3.0]}
    result = iterative_global_magnitude_prune_model(
        trained,
        sparsity=0.4,
        rounds=2,
        rewind=True,
        initial_weights=initial,
    )
    # Smallest trained weights are a[1]=0.1 then b[1]=0.2, not the tiny
    # rewound values at a[0] and a[2].
    assert result.weights == {
        "a": [0.01, 0.0, 0.02],
        "b": [4.0, 0.0],
    }


def test_rewind_defaults_to_the_input_weights() -> None:
    model = {"fc": [0.1, 0.9, 0.2, 0.8]}
    result = iterative_global_magnitude_prune_model(
        model, sparsity=0.5, rounds=1, rewind=True
    )
    assert result.weights == {"fc": [0.0, 0.9, 0.0, 0.8]}


def test_custom_schedule_sets_per_round_sparsity() -> None:
    model = {
        "small": [0.1, 0.2, 0.3, 0.4],
        "large": [1.0, 2.0, 3.0, 4.0],
    }
    result = iterative_global_magnitude_prune_model(
        model, schedule=[0.25, 0.5]
    )
    assert result.rounds == 2
    assert result.sparsity == pytest.approx(0.5)
    assert result.steps[0].kept == 6
    assert result.steps[1].kept == 4
    assert result.weights == global_magnitude_prune_model(model, sparsity=0.5)
    # Front-loaded compared with an even 4-round split (1 removal each).
    even = iterative_global_magnitude_prune_model(
        model, sparsity=0.5, rounds=4
    )
    assert even.steps[0].kept == 7
    assert result.steps[0].kept != even.steps[0].kept


def test_schedule_must_be_non_decreasing() -> None:
    model = {"fc": [0.1, 0.2, 0.3, 0.4]}
    with pytest.raises(ValueError, match="non-decreasing"):
        iterative_global_magnitude_prune_model(model, schedule=[0.5, 0.25])


def test_schedule_must_match_sparsity_and_rounds() -> None:
    model = {"fc": [0.1, 0.2, 0.3, 0.4]}
    with pytest.raises(ValueError, match="final schedule"):
        iterative_global_magnitude_prune_model(
            model, sparsity=0.9, schedule=[0.25, 0.5]
        )
    with pytest.raises(ValueError, match="rounds must match"):
        iterative_global_magnitude_prune_model(
            model, rounds=3, schedule=[0.25, 0.5]
        )


def test_schedule_rejects_empty_and_string() -> None:
    model = {"fc": [0.1, 0.2]}
    with pytest.raises(ValueError, match="schedule"):
        iterative_global_magnitude_prune_model(model, schedule=[])
    with pytest.raises(ValueError, match="schedule"):
        iterative_global_magnitude_prune_model(model, schedule="0.5,0.9")


def test_iterative_does_not_modify_inputs() -> None:
    model = {"fc": [0.1, 0.9, 0.2, 0.8]}
    initial = {"fc": [1.0, 2.0, 3.0, 4.0]}
    original_model = {"fc": list(model["fc"])}
    original_initial = {"fc": list(initial["fc"])}
    iterative_global_magnitude_prune_model(
        model,
        sparsity=0.5,
        rounds=2,
        rewind=True,
        initial_weights=initial,
    )
    assert model == original_model
    assert initial == original_initial


def test_iterative_rejects_initial_without_rewind() -> None:
    with pytest.raises(ValueError, match="rewind"):
        iterative_global_magnitude_prune_model(
            {"fc": [0.1, 0.2]},
            initial_weights={"fc": [0.3, 0.4]},
            rewind=False,
        )


def test_iterative_rejects_initial_name_or_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same layer names"):
        iterative_global_magnitude_prune_model(
            {"fc": [0.1, 0.2]},
            rewind=True,
            initial_weights={"other": [0.3, 0.4]},
        )
    with pytest.raises(ValueError, match="initial_weights has"):
        iterative_global_magnitude_prune_model(
            {"fc": [0.1, 0.2]},
            rewind=True,
            initial_weights={"fc": [0.3]},
        )


def test_iterative_rejects_invalid_rounds() -> None:
    with pytest.raises(ValueError, match="rounds"):
        iterative_global_magnitude_prune_model({"fc": [0.1, 0.2]}, rounds=0)


def test_step_snapshots_are_independent_copies() -> None:
    model = {"fc": [0.1, 0.2, 0.3, 0.4]}
    result = iterative_global_magnitude_prune_model(model, sparsity=0.5, rounds=2)
    result.steps[0].weights["fc"][0] = 99.0
    assert result.weights["fc"][0] != 99.0
    assert model["fc"][0] == 0.1


def test_cli_global_json_reports_uneven_layer_density() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "global",
            "--specs", "small=dense:4x1,large=dense:4x1",
            "--weights", "0.1,0.2,0.3,0.4,1,2,3,4",
            "--sparsity", "0.5",
            "--rounds", "2",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["sparsity"] == pytest.approx(0.5)
    assert payload["rounds"] == 2
    assert payload["rewind"] is False
    assert payload["final_kept"] == 4
    assert payload["final_total"] == 8
    assert len(payload["steps"]) == 2
    by_name = {row["layer"]: row for row in payload["layers"]}
    assert by_name["small"]["kept"] == 0
    assert by_name["large"]["kept"] == 4


def test_cli_global_schedule_and_rewind_json() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "global",
            "--specs", "fc=dense:4x1",
            "--weights", "0.1,0.2,0.3,0.4",
            "--initial-weights", "1,2,3,4",
            "--schedule", "0.25,0.5",
            "--rewind",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["rewind"] is True
    assert payload["rounds"] == 2
    assert payload["steps"][0]["kept"] == 3
    assert payload["steps"][1]["kept"] == 2
    assert payload["schedule"] == pytest.approx([0.25, 0.5])


def test_cli_global_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "global.md"
    rc = main([
        "global",
        "--specs", "small=dense:4x1,large=dense:4x1",
        "--weights", "0.1,0.2,0.3,0.4,1,2,3,4",
        "--sparsity", "0.5",
        "--rounds", "2",
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Global unstructured magnitude pruning" in text
    assert "Rewind: no" in text
    assert "| small | 0 | 4 |" in text
    assert "| large | 4 | 4 |" in text


def test_cli_global_rejects_initial_without_rewind() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "global",
                "--specs", "fc=dense:2x2",
                "--weights", "0.1,0.2,0.3,0.4",
                "--initial-weights", "1,2,3,4",
            ])
    assert rc == 2
    assert "rewind" in buf_err.getvalue()


def test_cli_global_rejects_bad_sparsity() -> None:
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stderr(buf_err):
            rc = main([
                "global",
                "--specs", "fc=dense:2x2",
                "--weights", "0.1,0.2,0.3,0.4",
                "--sparsity", "1.5",
            ])
    assert rc == 2
    assert "sparsity" in buf_err.getvalue()
