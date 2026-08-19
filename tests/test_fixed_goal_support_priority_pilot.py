from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_goal_support_priority_pilot import (  # noqa: E402
    recursive_json_differences,
    reward_config_with_airborne_weight,
    validate_config,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_support_pilot_is_a_paired_single_reward_field_intervention() -> None:
    config = load_config()
    preview, reward_control = validate_config(config)
    assert config["approved_map"] == preview["approved_map"]
    assert config["task_adapter"] == preview["task_adapter"]
    assert config["ppo"] == preview["ppo"]
    assert config["task_adapter"]["additional_task_reward"] == 0.0
    assert config["energy_boundary"]["ctrl_cost_weight_unchanged"] == 0.5
    candidate = reward_config_with_airborne_weight(reward_control, 12.0)
    differences = recursive_json_differences(reward_control, candidate)
    assert differences == [
        ("preserved_pre_pitch_reward.airborne_shaping_weight", 4, 12.0)
    ]


def test_support_pilot_rejects_a_friction_change() -> None:
    config = copy.deepcopy(load_config())
    config["approved_map"]["fixed_friction"][0] = 1.1
    with pytest.raises(ValueError, match="approved_map"):
        validate_config(config)


def test_support_pilot_rejects_a_planar_reward_change() -> None:
    config = load_config()
    _, reward_control = validate_config(config)
    candidate = reward_config_with_airborne_weight(reward_control, 12.0)
    candidate["reward"]["planar_velocity_tracking_weight"] = 11
    differences = recursive_json_differences(reward_control, candidate)
    assert [item[0] for item in differences] == [
        "preserved_pre_pitch_reward.airborne_shaping_weight",
        "reward.planar_velocity_tracking_weight",
    ]


def test_support_pilot_rejects_a_training_budget_change() -> None:
    config = copy.deepcopy(load_config())
    config["training"]["additional_target_timesteps"] += 2048
    with pytest.raises(ValueError, match="training budget"):
        validate_config(config)
