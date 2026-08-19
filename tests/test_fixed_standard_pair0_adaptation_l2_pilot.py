from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mujoco
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fixed_standard_pair0_adaptation_l2_pilot.py"
CONFIG = ROOT / "configs" / "fixed_standard_pair0_adaptation_l2_pilot_v1_20260819.json"
SPEC = importlib.util.spec_from_file_location("pair0_adaptation_l2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_validated():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return config, MODULE.validate_config(config)


def test_config_freezes_l2_pairing_budgets_and_scope() -> None:
    config, (protocol, reward, _, _) = load_validated()
    assert config["status"] == "frozen_l2_exploratory_single_training_seed"
    assert config["training"]["training_seed"] == 62805
    assert config["training"]["condition_run_order"] == [
        MODULE.DEFAULT_ID,
        MODULE.PAIR0_ID,
    ]
    assert config["training"]["additional_timesteps_per_condition"] == 65536
    assert config["training"]["checkpoint_additional_timesteps"] == [
        16384,
        32768,
        49152,
        65536,
    ]
    assert config["evaluation"]["seeds"] == [82801, 82802, 82803]
    assert config["evaluation"]["all_five_physics_substeps_required"] is True
    assert (
        config["evaluation"]["corrected_slip"]["primary_denominator"]
        == "physics_substeps_with_at_least_one_distal_foot_in_contact_and_normal_force_at_least_1N"
    )
    assert config["evaluation"]["corrected_slip"]["zero_primary_denominator"] == "non_evaluable"
    assert reward["preserved_pre_pitch_reward"]["ctrl_cost_weight"] == 0.5
    assert config["ppo"] == protocol["ppo"]
    assert config["execution"]["fixed_map_evaluation"] is False
    assert config["execution"]["video_rendering"] is False
    assert config["execution"]["promotion"] is False


def test_pair0_32768_early_stop_requires_both_progress_conditions() -> None:
    config, _ = load_validated()
    aggregate = {
        "nonfinite_episode_count": 0,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "corrected_sustained_slip_per_supported_fraction": 0.0,
        "corrected_slip_events_per_100_supported_substeps": 0.0,
        "force_qualified_slip_evaluable": True,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": 0.0,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": 0.0,
        "pooled_full_interval_zero_foot_fraction": 0.02,
        "qualified_slip_per_supported_fraction": 0.0,
        "qualified_slip_per_force_qualified_supported_fraction": 0.0,
        "mean_best_progress_m": 4.80,
    }
    stopped = MODULE.checkpoint_stop_decision(
        config,
        aggregate,
        condition_id=MODULE.PAIR0_ID,
        checkpoint_additional_timesteps=32768,
        pair0_checkpoint_16384_progress_m=4.75,
    )
    assert stopped["pair0_checkpoint_32768_rule"]["triggered"] is True
    assert stopped["early_stop_triggered"] is True
    aggregate["mean_best_progress_m"] = 4.85
    continued = MODULE.checkpoint_stop_decision(
        config,
        aggregate,
        condition_id=MODULE.PAIR0_ID,
        checkpoint_additional_timesteps=32768,
        pair0_checkpoint_16384_progress_m=4.70,
    )
    assert continued["pair0_checkpoint_32768_rule"]["triggered"] is False


def test_qualified_transient_is_warning_not_a_stop() -> None:
    config, _ = load_validated()
    aggregate = {
        "nonfinite_episode_count": 0,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "corrected_sustained_slip_per_supported_fraction": 0.0,
        "corrected_slip_events_per_100_supported_substeps": 0.0,
        "force_qualified_slip_evaluable": True,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": 0.0,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": 0.0,
        "pooled_full_interval_zero_foot_fraction": 0.02,
        "qualified_slip_per_supported_fraction": 0.11,
        "qualified_slip_per_force_qualified_supported_fraction": 0.11,
        "mean_best_progress_m": 6.0,
    }
    decision = MODULE.checkpoint_stop_decision(
        config,
        aggregate,
        condition_id=MODULE.PAIR0_ID,
        checkpoint_additional_timesteps=16384,
        pair0_checkpoint_16384_progress_m=None,
    )
    assert decision["qualified_transient_warning"] is True
    assert decision["early_stop_triggered"] is False


def test_zero_force_qualified_slip_denominator_fails_closed() -> None:
    config, _ = load_validated()
    aggregate = {
        "nonfinite_episode_count": 0,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "force_qualified_slip_evaluable": False,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": None,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": None,
        "pooled_full_interval_zero_foot_fraction": 0.02,
        "qualified_slip_per_force_qualified_supported_fraction": None,
        "mean_best_progress_m": 6.0,
    }
    decision = MODULE.checkpoint_stop_decision(
        config,
        aggregate,
        condition_id=MODULE.PAIR0_ID,
        checkpoint_additional_timesteps=16384,
        pair0_checkpoint_16384_progress_m=None,
    )
    assert decision["catastrophe_checks"]["force_qualified_slip_non_evaluable"] is True
    assert decision["early_stop_triggered"] is True


def test_smoke_nonfinite_result_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        MODULE.smoke_engineering_decision(
            {"nonfinite_episode_count": 1},
            condition_id=MODULE.DEFAULT_ID,
            checkpoint_additional_timesteps=2048,
        )


def test_generated_default_and_pair0_models_pass_compiled_worker_contract(
    tmp_path: Path,
) -> None:
    config, (protocol, _, _, _) = load_validated()
    scenes, _ = MODULE.prepare_condition_scenes(config, protocol, tmp_path)
    for condition, expected_pairs in (
        (MODULE.DEFAULT_ID, 0),
        (MODULE.PAIR0_ID, 4),
    ):
        scene = scenes[condition]["flat"]
        model = mujoco.MjModel.from_xml_path(scene["xml_path"])
        audit = MODULE.compiled_contract_audit(
            model,
            scene,
            condition,
            config,
            construction_seed=62805,
        )
        assert audit["passed"] is True
        assert audit["npair"] == expected_pairs
        assert audit["construction_seed"] == 62805
        assert audit["effective_first_reset_seed"] is None
        assert {record["margin"] for record in audit["geom_contracts"]} == {0.01}


def test_effective_vecenv_reset_seeds_are_recorded_separately() -> None:
    class FakeVecEnv:
        def seed(self, value: int):
            return [value + rank for rank in range(4)]

    audits = [
        {
            "passed": True,
            "construction_seed": 62805 + 1000 * rank,
            "effective_first_reset_seed": None,
        }
        for rank in range(4)
    ]
    corrected = MODULE.record_effective_reset_seeds(
        FakeVecEnv(), audits, training_seed=62805
    )
    assert [row["construction_seed"] for row in corrected] == [
        62805,
        63805,
        64805,
        65805,
    ]
    assert [row["effective_first_reset_seed"] for row in corrected] == [
        62805,
        62806,
        62807,
        62808,
    ]


def test_force_qualified_denominator_is_primary_and_zero_is_non_evaluable() -> None:
    row = {
        "control_steps": 10,
        "physics_substeps": 50,
        "supported_physics_substep_count": 40,
        "force_qualified_supported_physics_substep_count": 20,
        "finite": True,
        "fall": False,
        "fixed_goal_success": False,
        "torso_ground_any": False,
        "sustained_nonfoot_contact": False,
        "fixed_goal_best_progress_m": 2.0,
        "full_interval_zero_foot_count": 1,
        "support_count_sum_physics_substeps": 60,
        "qualified_slip_physics_substep_count": 4,
        "corrected_sustained_slip_physics_substep_count": 2,
        "corrected_slip_event_count": 1,
    }
    aggregate = MODULE.aggregate_episode_rows([row])
    assert aggregate["force_qualified_slip_evaluable"] is True
    assert aggregate["qualified_slip_per_force_qualified_supported_fraction"] == pytest.approx(0.2)
    assert aggregate["corrected_sustained_slip_per_force_qualified_supported_fraction"] == pytest.approx(0.1)
    assert aggregate["corrected_slip_events_per_100_force_qualified_supported_substeps"] == pytest.approx(5.0)
    assert aggregate["corrected_sustained_slip_per_supported_fraction"] == pytest.approx(0.05)

    zero_force_row = dict(row)
    zero_force_row["force_qualified_supported_physics_substep_count"] = 0
    zero_force_row["qualified_slip_physics_substep_count"] = 0
    zero_force_row["corrected_sustained_slip_physics_substep_count"] = 0
    zero_force_row["corrected_slip_event_count"] = 0
    zero_force = MODULE.aggregate_episode_rows([zero_force_row])
    assert zero_force["force_qualified_slip_evaluable"] is False
    assert zero_force["qualified_slip_per_force_qualified_supported_fraction"] is None
    assert zero_force["corrected_sustained_slip_per_force_qualified_supported_fraction"] is None
    assert zero_force["corrected_slip_events_per_100_force_qualified_supported_substeps"] is None


def test_smoke_nonfinite_is_fail_closed() -> None:
    passed = MODULE.smoke_engineering_decision(
        {"nonfinite_episode_count": 0},
        condition_id=MODULE.DEFAULT_ID,
        checkpoint_additional_timesteps=128,
    )
    assert passed["nonfinite_fail_closed_passed"] is True
    with pytest.raises(RuntimeError, match="non-finite"):
        MODULE.smoke_engineering_decision(
            {"nonfinite_episode_count": 1},
            condition_id=MODULE.DEFAULT_ID,
            checkpoint_additional_timesteps=128,
        )


def test_existing_attempt_root_is_never_overwritten(tmp_path: Path) -> None:
    config, (protocol, reward, _, _) = load_validated()
    with pytest.raises(FileExistsError, match="Refusing to overwrite attempt root"):
        MODULE.run_pilot(
            CONFIG,
            config,
            protocol,
            reward,
            tmp_path,
            smoke=True,
            attempt=0,
        )
