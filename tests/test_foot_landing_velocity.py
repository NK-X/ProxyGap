from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_body_smoothness_gsde_matrix import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "foot_landing_velocity_v1_20260817.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_foot_landing_config_is_valid_and_preserves_a_matched_ablation() -> None:
    config = load_config()
    validate_config(config)
    assert [condition["condition_id"] for condition in config["conditions"]] == [
        "F0__LOW_ROOT",
        "F1__FOOT_LANDING",
    ]
    assert {
        condition["foot_landing_enabled"] for condition in config["conditions"]
    } == {False, True}
    assert all(
        condition["body_dynamics_enabled"] for condition in config["conditions"]
    )


def test_torso_weights_are_lower_and_foot_penalty_budget_is_bounded() -> None:
    config = load_config()
    shared = config["shared_reward"]
    foot = config["foot_landing"]
    assert shared["forward_velocity_tracking_weight"] == 0.5
    assert shared["lateral_drift_shaping_weight"] == 0.025
    assert foot["height_threshold_m"] == 0.03
    assert len(foot["foot_geom_names"]) == 4
    assert foot["lateral_velocity_weight_per_foot"] == 0.025
    assert foot["vertical_velocity_weight_per_foot"] == 0.025
    assert foot["maximum_lateral_penalty_per_step"] == 0.1
    assert foot["maximum_vertical_penalty_per_step"] == 0.1
