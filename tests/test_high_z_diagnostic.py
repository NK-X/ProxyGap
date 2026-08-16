from __future__ import annotations

import pytest

from proxygap.high_z_diagnostic import (
    finite_summary,
    select_common_high_z_seed,
    summarise_step_trace,
)


def test_selects_lowest_seed_shared_by_every_failed_policy() -> None:
    rows = [
        {"training_seed": 1, "seed": 10, "high_z_termination": True},
        {"training_seed": 2, "seed": 10, "high_z_termination": True},
        {"training_seed": 1, "seed": 9, "high_z_termination": True},
        {"training_seed": 2, "seed": 9, "high_z_termination": True},
        {"training_seed": 3, "seed": 9, "high_z_termination": False},
    ]
    selected = select_common_high_z_seed(rows, failing_training_seeds=[1, 2])
    assert selected["evaluation_seed"] == 9
    assert selected["eligible_evaluation_seeds"] == [9, 10]
    assert selected["unexpected_other_high_z_training_seeds"] == []


def test_step_trace_summary_derives_vertical_motion() -> None:
    rows = [
        {
            "step_index": index,
            "torso_height": height,
            "x_position": 0.1 * index,
            "torso_tilt_rad": 0.1 * index,
            "squared_action_step": float(index),
            "action_saturation_fraction_step": 0.0,
            "termination_category": "high_z_excursion" if index == 3 else "none",
            "terminated": index == 3,
            "truncated": False,
        }
        for index, height in enumerate((0.5, 0.7, 1.01), start=1)
    ]
    summary = summarise_step_trace(rows, dt=0.05)
    assert summary["episode_length"] == 3
    assert summary["terminal_torso_height"] == pytest.approx(1.01)
    assert summary["terminal_vertical_velocity"] == pytest.approx(6.2)
    assert summary["maximum_upward_velocity"] == pytest.approx(6.2)
    assert summary["termination_category"] == "high_z_excursion"
    assert summary["proportion_steps_torso_tilt_ge_90_deg"] == 0.0
    assert summary["longest_consecutive_inverted_steps"] == 0
    assert finite_summary(summary)


def test_step_trace_rejects_missing_step() -> None:
    rows = [
        {
            "step_index": 1,
            "torso_height": 0.5,
            "x_position": 0.0,
            "torso_tilt_rad": 0.0,
            "squared_action_step": 0.0,
            "action_saturation_fraction_step": 0.0,
            "termination_category": "none",
            "terminated": False,
            "truncated": False,
        },
        {
            "step_index": 3,
            "torso_height": 1.01,
            "x_position": 0.1,
            "torso_tilt_rad": 0.0,
            "squared_action_step": 0.0,
            "action_saturation_fraction_step": 0.0,
            "termination_category": "high_z_excursion",
            "terminated": True,
            "truncated": False,
        },
    ]
    with pytest.raises(ValueError, match="contiguous"):
        summarise_step_trace(rows, dt=0.05)
