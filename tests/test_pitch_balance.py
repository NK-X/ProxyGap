from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.ant_wrapper import (  # noqa: E402
    PitchBalanceEventTracker,
    quaternion_pitch_angle,
)
from run_body_smoothness_gsde_matrix import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "pitch_balance_v2_20260817.json"
CALIBRATED_CONFIG = ROOT / "configs" / "pitch_balance_v3_calibrated_20260817.json"


def test_quaternion_pitch_angle_is_signed_and_normalisation_invariant() -> None:
    angle = math.radians(30.0)
    quaternion = np.array([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0])
    assert math.isclose(quaternion_pitch_angle(quaternion), angle)
    assert math.isclose(quaternion_pitch_angle(-3.0 * quaternion), angle)
    negative = np.array([math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0])
    assert math.isclose(quaternion_pitch_angle(negative), -angle)


def test_pitch_balance_event_rewards_equal_positive_and_negative_time() -> None:
    tracker = PitchBalanceEventTracker(foot_count=4)
    tracker.reset(initial_grounded=np.zeros(4, dtype=bool))

    first = tracker.update(np.array([1, 0, 0, 0], dtype=bool), 0.1)
    second = tracker.update(np.array([1, 1, 0, 0], dtype=bool), 0.2)
    third = tracker.update(np.array([1, 1, 1, 0], dtype=bool), -0.1)
    fourth = tracker.update(np.ones(4, dtype=bool), -0.2)

    assert first["started"] and first["active"]
    assert not second["completed"] and not third["completed"]
    assert fourth["completed"] and not fourth["active"]
    assert fourth["positive_steps"] == 2
    assert fourth["negative_steps"] == 2
    assert math.isclose(fourth["score"], 1.0)
    assert tracker.completed_event_count == 1
    assert math.isclose(tracker.balance_score_sum, 1.0)


def test_pitch_balance_event_uses_distinct_new_landings_and_scores_imbalance() -> None:
    tracker = PitchBalanceEventTracker(foot_count=4)
    tracker.reset(initial_grounded=np.ones(4, dtype=bool))
    assert not tracker.update(np.ones(4, dtype=bool), 0.1)["started"]
    tracker.update(np.zeros(4, dtype=bool), 0.1)
    tracker.update(np.array([1, 0, 0, 0], dtype=bool), 0.1)
    tracker.update(np.array([0, 0, 0, 0], dtype=bool), 0.1)
    repeated = tracker.update(np.array([1, 0, 0, 0], dtype=bool), 0.1)
    assert repeated["landed_count"] == 1
    tracker.update(np.array([1, 1, 0, 0], dtype=bool), 0.1)
    tracker.update(np.array([1, 1, 1, 0], dtype=bool), 0.1)
    completed = tracker.update(np.ones(4, dtype=bool), 0.1)
    assert completed["completed"]
    assert math.isclose(completed["score"], 0.0)


def test_pitch_balance_training_config_is_single_modified_condition() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["design_type"] == "pitch_balance_single"
    assert len(config["conditions"]) == 1
    assert config["conditions"][0]["condition_id"] == "P1__PITCH_BALANCE"
    assert config["conditions"][0]["foot_landing_enabled"] is True
    assert config["conditions"][0]["pitch_balance_enabled"] is True
    assert config["pitch_balance"]["shaping_weight"] == 0.1


def test_calibrated_pitch_balance_config_preserves_reward_and_raises_only_event_weight() -> None:
    original = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibrated = json.loads(CALIBRATED_CONFIG.read_text(encoding="utf-8"))
    validate_config(calibrated)
    assert calibrated["design_type"] == "pitch_balance_single"
    assert len(calibrated["conditions"]) == 1
    assert calibrated["conditions"][0]["pitch_balance_enabled"] is True
    assert calibrated["shared_reward"] == original["shared_reward"]
    assert calibrated["body_dynamics"] == original["body_dynamics"]
    for key, value in original["foot_landing"].items():
        assert calibrated["foot_landing"][key] == value
    assert calibrated["pitch_balance"]["shaping_weight"] == 5.0
