from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hybrid_intent_constraint_design_v1_20260816.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_intended_behaviour_is_multidimensional_and_fixed_horizon() -> None:
    intent = load_config()["intended_behaviour"]
    assert intent["scalar_true_performance"] == "not_defined"
    assert intent["evaluation_horizon_steps"] == 1000
    assert math.isclose(
        intent["evaluation_horizon_steps"] * intent["environment_dt_seconds"],
        intent["full_horizon_seconds"],
    )
    assert intent["commands"] == {
        "forward_velocity_m_per_s": 1.0,
        "lateral_velocity_m_per_s": 0.0,
        "yaw_rate_rad_per_s": 0.0,
    }


def test_episode_compliance_thresholds_are_internally_valid() -> None:
    rule = load_config()["intended_behaviour"]["episode_compliance"]
    velocity_low, velocity_high = rule[
        "fixed_horizon_forward_velocity_interval_m_per_s"
    ]
    assert velocity_low < 1.0 < velocity_high
    assert rule["requires_full_1000_steps"] is True
    assert rule["unhealthy_termination_allowed"] is False
    assert 0.0 < rule["forward_path_efficiency_min"] <= 1.0
    assert 0.0 <= rule["action_saturation_fraction_max"] <= 1.0
    assert rule["normalised_action_roughness_max"] > 0.0


def test_external_constraint_is_bounded_and_not_mislabelled_as_physical() -> None:
    constraint = load_config()["external_constraint"]
    maximum_action_vector_change = 2.0 * math.sqrt(
        constraint["action_dimensions"]
    )
    assert 0.0 < constraint["delta_l2_per_control_step"] < maximum_action_vector_change
    assert constraint["physical_torque_rate_calibrated"] is False
    assert constraint["previous_applied_action_on_reset"] == "zeros"
    assert constraint["projection_uses_previous_applied_action"] is True


def test_observation_augmentation_is_fair_across_groups() -> None:
    observation = load_config()["external_constraint"]["observation_augmentation"]
    assert (
        observation["default_observation_dimensions"]
        + observation["previous_applied_action_dimensions"]
        == observation["augmented_observation_dimensions"]
    )
    assert observation["applied_to_all_comparison_groups"] is True
    assert observation["new_baseline_training_required"] is True


def test_calibration_is_development_only() -> None:
    calibration = load_config()["external_constraint"]["calibration"]
    assert calibration["training_seeds"] == [41201, 41202, 41203, 41204, 41205]
    assert calibration["transition_count"] == 79430
    assert math.isclose(calibration["pooled_p90_l2_action_change"], 1.430823)
    assert math.isclose(calibration["historical_fraction_above_threshold"], 0.113332)
    assert calibration["uses_held_out_formal_data"] is False


def test_no_training_is_authorised_by_the_freeze() -> None:
    execution = load_config()["execution"]
    assert execution == {
        "constraint_wrapper_implemented": False,
        "engineering_tests_passed": False,
        "development_training_authorised": False,
        "held_out_training_authorised": False,
        "formal_training_authorised": False,
    }


def test_candidate_and_formal_matrices_are_distinguished() -> None:
    strategy = load_config()["design_strategy"]
    assert "may exceed 2x2" in strategy["candidate_stage"]
    assert "2x2 ablation" in strategy["formal_stage"]
    assert strategy["maximum_development_rounds_before_new_revision"] == 2
