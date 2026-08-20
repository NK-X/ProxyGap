from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rq1_matched_baseline_v1_20260820.py"
CONFIG = ROOT / "configs" / "rq1_matched_baseline_v1_20260820.json"


def load_module():
    spec = importlib.util.spec_from_file_location("rq1_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_config_validates() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_config(config)


def test_default_condition_has_default_reward_and_matched_observation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    default, shaped = config["conditions"]
    assert default["condition_id"] == "D0_DEFAULT_REWARD"
    assert default["replace_forward_reward_with_tracking"] is False
    assert default["augment_previous_applied_action"] is True
    assert shaped["augment_previous_applied_action"] is True
    shaping_keys = [key for key in default if key.endswith("shaping_weight")]
    assert all(float(default[key]) == 0.0 for key in shaping_keys)


def test_training_and_evaluation_seeds_are_disjoint() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    training = set(config["training"]["training_seeds"])
    evaluation = set(config["evaluation"]["evaluation_seeds"])
    assert len(training) == 3
    assert len(evaluation) == 10
    assert training.isdisjoint(evaluation)


def test_policy_level_aggregation_keeps_training_seed_as_unit() -> None:
    module = load_module()
    rows = []
    for condition in module.EXPECTED_CONDITIONS:
        for seed in module.EXPECTED_TRAINING_SEEDS:
            for episode in range(2):
                rows.append(
                    {
                        "condition_id": condition,
                        "training_seed": str(seed),
                        "target_timesteps": "1000000",
                        "fixed_horizon_mean_forward_velocity": str(0.9 + 0.01 * episode),
                        "support_count_step_fractions_0_to_4": "[0.1, 0.2, 0.3, 0.3, 0.1]",
                        "foot_contact_duty_fraction_by_foot": "[0.4, 0.5, 0.6, 0.7]",
                        "net_displacement_direction_error_degrees": "2.0",
                        "forward_path_efficiency": "0.8",
                        "normalised_action_roughness": "0.01",
                        "action_saturation_rate": "0.0",
                        "unhealthy_termination": "0",
                        "full_horizon_completed": "1",
                        "torso_tilt_rms": "0.1",
                        "airborne_step_fraction": "0.2",
                        "common_rescored_return": "100",
                        "condition_objective_return": "101",
                    }
                )
    policy_rows = module.build_policy_level_rows(rows, primary_timesteps=1_000_000, target_speed=1.0)
    assert len(policy_rows) == 6
    assert {row["evaluation_episodes"] for row in policy_rows} == {2}
    paired = module.build_paired_effect_rows(policy_rows)
    assert len(paired) == 3
    assert {row["training_seed"] for row in paired} == set(module.EXPECTED_TRAINING_SEEDS)


def test_decision_summary_uses_paired_policy_effects() -> None:
    module = load_module()
    rows = []
    for seed in module.EXPECTED_TRAINING_SEEDS:
        rows.append(
            {
                "training_seed": seed,
                "shaped_minus_default__target_speed_abs_error_m_per_s": -0.1,
                "shaped_minus_default__direction_error_degrees": -1.0,
                "shaped_minus_default__forward_path_efficiency": 0.1,
                "shaped_minus_default__normalised_action_roughness": -0.01,
                "shaped_minus_default__unhealthy_termination_rate": 0.0,
            }
        )
    decision = module.decision_summary(rows)
    assert decision["joint_descriptive_gate_pass"] is True
    assert decision["independent_training_pairs"] == 3
