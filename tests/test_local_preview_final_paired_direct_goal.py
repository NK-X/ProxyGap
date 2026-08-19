from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_local_preview_final_paired_direct_goal.py"
SPEC = importlib.util.spec_from_file_location("paired_direct_goal_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ArrivalDwellTracker = MODULE.ArrivalDwellTracker
DurationCorrectedSlipTracker = MODULE.DurationCorrectedSlipTracker


def test_hold_annulus_cannot_establish_success_before_arrival_entry() -> None:
    tracker = ArrivalDwellTracker(1.5, 2.0, 4)
    for step in range(1, 9):
        tracker.update(step=step, distance_m=1.75, stable=True)
    assert not tracker.goal_entered
    assert tracker.hold_run_steps == 0
    assert not tracker.spatial_success
    assert not tracker.strict_dwell_success

    tracker.update(step=9, distance_m=1.49, stable=True)
    for step in range(10, 13):
        tracker.update(step=step, distance_m=1.75, stable=True)
    assert tracker.goal_entered
    assert tracker.entry_step == 9
    assert tracker.spatial_success
    assert tracker.spatial_success_step == 12
    assert tracker.strict_dwell_success


def test_arrival_gated_hold_and_strict_hold_reset_independently() -> None:
    tracker = ArrivalDwellTracker(1.5, 2.0, 3)
    tracker.update(step=1, distance_m=1.4, stable=True)
    tracker.update(step=2, distance_m=1.8, stable=False)
    tracker.update(step=3, distance_m=1.8, stable=True)
    assert tracker.spatial_success
    assert not tracker.strict_dwell_success
    assert tracker.longest_hold_run_steps == 3
    assert tracker.longest_strict_run_steps == 1

    tracker.update(step=4, distance_m=2.1, stable=True)
    assert tracker.hold_run_steps == 0
    assert tracker.strict_run_steps == 0


def _slip_tracker() -> object:
    return DurationCorrectedSlipTracker(
        dt=0.05,
        speed_threshold=0.2,
        minimum_normal_force=1.0,
        landing_grace_seconds=0.1,
        minimum_sustained_seconds=0.2,
    )


def _update_first_foot(tracker: object, *, speed: float, force: float = 20.0) -> None:
    tracker.update(
        contact_mask=np.asarray([True, False, False, False]),
        tangential_speeds=np.asarray([speed, 0.0, 0.0, 0.0]),
        normal_forces=np.asarray([force, 0.0, 0.0, 0.0]),
    )


def test_landing_grace_and_duration_gate_reject_short_impact_run() -> None:
    tracker = _slip_tracker()
    for _ in range(5):
        _update_first_foot(tracker, speed=0.8)
    result = tracker.finalise()
    assert int(np.sum(result["raw"][:, 0])) == 5
    assert int(np.sum(result["candidate"][:, 0])) == 3
    assert not np.any(result["sustained"])
    assert result["events"] == []


def test_four_post_grace_steps_form_one_duration_corrected_event() -> None:
    tracker = _slip_tracker()
    for _ in range(6):
        _update_first_foot(tracker, speed=0.8)
    result = tracker.finalise()
    assert int(np.sum(result["candidate"][:, 0])) == 4
    assert int(np.sum(result["sustained"][:, 0])) == 4
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["start_step"] == 3
    assert event["end_step"] == 6
    assert event["duration_steps"] == 4
    assert np.isclose(event["duration_seconds"], 0.2)


def test_force_gate_breaks_duration_corrected_run() -> None:
    tracker = _slip_tracker()
    for _ in range(2):
        _update_first_foot(tracker, speed=0.8)
    for index in range(7):
        _update_first_foot(
            tracker,
            speed=0.8,
            force=0.5 if index == 3 else 20.0,
        )
    result = tracker.finalise()
    assert not np.any(result["sustained"])
    assert result["events"] == []


def test_frozen_evaluation_contract_loads_and_keeps_required_values() -> None:
    config, fixed = MODULE.validate_and_load_config(
        ROOT
        / "configs"
        / "fixed_map_local_preview_final_paired_direct_goal_v1_20260819.json"
    )
    assert config["evaluation_seeds"] == [74801, 74802, 74803, 74804, 74805]
    assert config["horizon_steps"] == 12000
    assert config["controller"]["slow_radius_m"] == 4.0
    assert fixed["approved_map"]["fixed_friction"] == [1.0, 0.5, 0.5]
