from __future__ import annotations

import pytest

from proxygap.selection import project_selection_inputs, select_best_tested_coefficient


def row(weight: float, training_seed: int, proxy: float, **extra: object) -> dict[str, object]:
    return {
        "ctrl_cost_weight": weight,
        "training_seed": training_seed,
        "checkpoint_fraction": 1.0,
        "common_rescored_return": proxy,
        **extra,
    }


def test_selection_aggregates_episodes_within_training_seed() -> None:
    rows = [
        row(0.5, 11, 10.0),
        row(0.5, 11, 14.0),
        row(0.25, 11, 20.0),
        row(0.25, 11, 20.0),
        row(0.5, 12, 12.0),
        row(0.25, 12, 20.0),
    ]
    result = select_best_tested_coefficient(rows, candidates=[0.25, 0.5])
    assert result.selected_ctrl_cost_weight == 0.25
    assert result.candidates[0].training_seed_count == 2


def test_selection_rejects_protected_diagnostics() -> None:
    with pytest.raises(ValueError, match="protected diagnostics"):
        select_best_tested_coefficient(
            [row(0.5, 11, 10.0, torso_tilt_rms=0.1), row(0.25, 11, 11.0)],
            candidates=[0.25, 0.5],
        )


def test_projection_removes_protected_diagnostics_before_selection() -> None:
    projected = project_selection_inputs(
        [row(0.5, 11, 10.0, torso_tilt_rms=0.1)]
    )
    assert projected == [
        {
            "ctrl_cost_weight": 0.5,
            "training_seed": 11,
            "checkpoint_fraction": 1.0,
            "common_rescored_return": 10.0,
        }
    ]


def test_selection_requires_final_checkpoint() -> None:
    with pytest.raises(ValueError, match="final-checkpoint"):
        select_best_tested_coefficient(
            [row(0.5, 11, 10.0, checkpoint_fraction=0.5), row(0.25, 11, 11.0)],
            candidates=[0.25, 0.5],
        )


def test_selection_tie_is_predeclared_and_deterministic() -> None:
    rows = [row(0.25, 11, 10.0), row(0.5, 11, 10.0), row(1.0, 11, 10.0)]
    result = select_best_tested_coefficient(rows, candidates=[0.25, 0.5, 1.0])
    assert result.selected_ctrl_cost_weight == 0.5


def test_selection_requires_fixed_benchmark_candidate() -> None:
    with pytest.raises(ValueError, match="0.5 benchmark"):
        select_best_tested_coefficient(
            [row(0.25, 11, 10.0)],
            candidates=[0.25],
        )
