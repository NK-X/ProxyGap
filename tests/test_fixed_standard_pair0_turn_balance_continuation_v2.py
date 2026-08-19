from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS))

import run_fixed_standard_pair0_turn_balance_continuation as design  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v2 as runner  # noqa: E402
from proxygap.paired_turn_balance import (  # noqa: E402
    BALANCED_CONDITION_ID,
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v2_20260819.json"
)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def passing_turn_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for condition in runner.expected_turn_conditions():
        name = str(condition["condition_name"])
        results[name] = {
            "safety_passed": True,
            "safety_checks": {
                "force_qualified_denominator_evaluable": True,
            },
            "mean_actual_cumulative_yaw_change_rad": (
                0.0 if name == "straight_055" else float(condition["target_yaw_rate_rad_per_s"]) * 30.0
            ),
            "mean_yaw_change_target_ratio": (
                None if name == "straight_055" else 1.0
            ),
            "same_sign_episode_count": (
                None if name == "straight_055" else 5
            ),
        }
    return results


def passing_slope_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for _ in range(20):
        row: dict[str, Any] = {
            "force_qualified_supported_physics_substep_count": 100,
        }
        for key in runner.turn.ENERGY_KEYS:
            row[key] = 1.0
        rows.append(row)
    aggregate = {
        "total_control_steps": 12_000,
        "total_physics_substeps": 60_000,
        "nonfinite_episode_count": 0,
        "mean_best_progress_m": 8.0,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "pooled_full_interval_zero_foot_fraction": 0.01,
        "force_qualified_slip_evaluable": True,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": 0.0,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": 0.0,
        "per_scene": {
            "uphill_8deg": {"mean_best_progress_m": 7.0},
            "downhill_8deg": {"mean_best_progress_m": 9.0},
        },
    }
    return aggregate, rows


def test_v2_freezes_matched_budgets_seed_threads_and_last_round_boundaries() -> None:
    config = load_config()
    v1_path = ROOT / config["design_source"]["configuration"]
    assert runner.sha256(v1_path) == config["design_source"]["configuration_sha256"]
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    assert v1["training"]["master_seed"] == 63806
    assert v1["training"]["worker_effective_first_reset_seeds"] == list(
        range(63806, 63814)
    )
    assert v1["training"]["training_seed_count"] == 1
    assert v1["training"]["multi_seed_training_robustness_claim_permitted"] is False
    assert v1["ppo"]["torch_num_threads"] == 2
    assert config["smoke"]["additional_timesteps_per_condition"] == 8192
    assert config["smoke"]["complete_episodes_per_worker"] == 2
    assert config["formal"]["additional_timesteps_per_condition"] == 65536
    assert config["formal"]["absolute_final_checkpoint_timesteps"] == 2793472
    assert config["hard_stop_and_archive"]["last_locomotion_optimisation_round"] is True
    assert config["hard_stop_and_archive"]["hard_stop_after_formal_pass_or_fail"] is True


def test_smoke_and_formal_exposure_are_exact_and_smoke_covers_both_signs() -> None:
    for rank in (2, 3, 4, 5, 6, 7):
        smoke = runner.expected_worker_exposure(
            BALANCED_CONDITION_ID, rank, smoke=True
        )
        assert len(smoke) == 2
        assert sorted(smoke.values()) == [512, 512]
        assert any(name.startswith("left_") for name in smoke)
        assert any(name.startswith("right_") for name in smoke)
        formal = runner.expected_worker_exposure(
            BALANCED_CONDITION_ID, rank, smoke=False
        )
        assert sorted(formal.values()) == [4096, 4096]
    assert runner.expected_worker_exposure(
        CONTROL_CONDITION_ID, 0, smoke=True
    ) == {"straight_000": 1024}


def test_loader_contract_explicitly_overrides_seed_and_threads_before_load() -> None:
    source = __import__("inspect").getsource(design.load_continuation_model)
    assert "torch.set_num_threads" in source
    assert "seed=master_seed" in source
    assert "n_steps=int(config[\"ppo\"][\"n_steps\"])" in source
    worker_source = __import__("inspect").getsource(runner.train_condition_worker)
    assert '"observed_first_reset_seeds"' in worker_source
    assert "actual first reset seeds consumed by workers" in worker_source


def test_parent_sets_two_torch_threads_before_any_canonical_root_creation() -> None:
    assert runner.configure_torch_threads(2) == 2
    parent_source = __import__("inspect").getsource(runner.run_parent_attempt)
    assert parent_source.index("configure_torch_threads") < parent_source.index(
        "output_root.exists()"
    )
    main_source = __import__("inspect").getsource(runner.main)
    assert main_source.index("configure_torch_threads(2)") < main_source.index(
        "validate_config(config)"
    )


def test_runtime_contract_is_exact_transitive_membership_without_placeholder() -> None:
    config = load_config()
    expected = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    assert tuple(expected) == runner.EXPECTED_RUNTIME_PATHS
    assert len(expected) == 25
    assert all(value != "<TO_FREEZE_RUNNER>" for value in expected.values())
    assert runner.validate_runtime_dependencies(config) == expected
    mutated = copy.deepcopy(config)
    mutated["runtime_dependency_contract"]["exact_relative_path_sha256"].pop(
        "src/proxygap/protocol.py"
    )
    with pytest.raises(ValueError, match="runtime exact membership/order"):
        runner.validate_runtime_dependencies(mutated)


def test_minimal_forged_smoke_manifest_cannot_authorise_formal(tmp_path: Path) -> None:
    config = load_config()
    fake_root = tmp_path / "attempt_0"
    fake_root.mkdir()
    (fake_root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "engineering_smoke_complete_no_science",
                "configuration_sha256": runner.sha256(CONFIG_PATH),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="smoke manifest schema"):
        runner.validate_smoke_manifest_and_inventory(
            config, runner.sha256(CONFIG_PATH), fake_root
        )


def test_turn_gate_pass_fail_and_zero_denominator_non_evaluable() -> None:
    v1 = json.loads(
        (ROOT / load_config()["design_source"]["configuration"]).read_text(
            encoding="utf-8"
        )
    )
    results = passing_turn_results()
    assert runner.apply_turn_gate(v1, results)["passed"] is True
    asymmetric = copy.deepcopy(results)
    asymmetric["curve_left_010"]["mean_yaw_change_target_ratio"] = 0.2
    failed = runner.apply_turn_gate(v1, asymmetric)
    assert failed["evaluable"] is True
    assert failed["passed"] is False
    denominator_zero = copy.deepcopy(results)
    denominator_zero["curve_left_010"]["safety_checks"][
        "force_qualified_denominator_evaluable"
    ] = False
    withheld = runner.apply_turn_gate(v1, denominator_zero)
    assert withheld["evaluable"] is False
    assert withheld["passed"] is False


def test_slope_gate_pass_fail_and_zero_denominator_non_evaluable() -> None:
    v1 = json.loads(
        (ROOT / load_config()["design_source"]["configuration"]).read_text(
            encoding="utf-8"
        )
    )
    aggregate, rows = passing_slope_inputs()
    assert runner.apply_slope_gate(v1, aggregate, rows)["passed"] is True
    slow = copy.deepcopy(aggregate)
    slow["per_scene"]["uphill_8deg"]["mean_best_progress_m"] = 1.0
    failed = runner.apply_slope_gate(v1, slow, rows)
    assert failed["evaluable"] is True
    assert failed["passed"] is False
    denominator_zero = copy.deepcopy(aggregate)
    denominator_zero["force_qualified_slip_evaluable"] = False
    denominator_zero[
        "corrected_sustained_slip_per_force_qualified_supported_fraction"
    ] = None
    denominator_zero[
        "corrected_slip_events_per_100_force_qualified_supported_substeps"
    ] = None
    withheld = runner.apply_slope_gate(v1, denominator_zero, rows)
    assert withheld["evaluable"] is False
    assert withheld["passed"] is False


def test_combined_gate_requires_both_evaluable_turn_and_slope_sets() -> None:
    branches = {
        branch: {
            "turn_gate": {"evaluable": True, "passed": True},
            "slope_gate": {"evaluable": True, "passed": True},
        }
        for branch in CONDITION_IDS
    }
    passed = runner.combined_final_gate(branches)
    assert passed["evaluable"] is True
    assert passed["passed"] is True
    branches[BALANCED_CONDITION_ID]["turn_gate"] = {
        "evaluable": False,
        "passed": False,
    }
    withheld = runner.combined_final_gate(branches)
    assert withheld["evaluable"] is False
    assert withheld["further_optimisation_authorised"] is False
    assert "withheld" in withheld["decision"]


def test_execution_surface_has_one_training_and_one_final_save_site_only() -> None:
    config = load_config()
    source = (ROOT / runner.RUNTIME_SELF).read_text(encoding="utf-8")
    assert source.count("model.learn(") == 1
    assert source.count("model.save(") == 1
    assert config["formal"]["save_intermediate_checkpoints"] is False
    assert config["execution"]["fixed_map"] is False
    assert config["execution"]["training_video"] is False
    assert config["execution"]["promotion"] is False
    assert config["hard_stop_and_archive"][
        "post_result_video_seed_predeclared_before_training"
    ] == 96131
    assert config["hard_stop_and_archive"]["post_result_video_turn_conditions"] == [
        "curve_left_020",
        "curve_right_020",
    ]


def test_failure_after_attempt_root_creation_is_recorded_and_non_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config()
    v1_path = ROOT / config["design_source"]["configuration"]
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        (ROOT / v1["source"]["standard_protocol"]).read_text(encoding="utf-8")
    )
    reward = json.loads(
        (ROOT / v1["source"]["reward_configuration"]).read_text(encoding="utf-8")
    )
    checkpoint = ROOT / v1["source"]["checkpoint"]

    def injected_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected pre-training snapshot failure")

    monkeypatch.setattr(runner, "snapshot_runtime_dependencies", injected_failure)
    attempt = tmp_path / "attempt_0"
    with pytest.raises(RuntimeError, match="injected pre-training"):
        runner.run_parent_attempt(
            CONFIG_PATH,
            config,
            v1,
            protocol,
            reward,
            checkpoint,
            attempt,
            smoke=True,
        )
    failure = json.loads(
        (attempt / "FAILURE_RECORD.json").read_text(encoding="utf-8")
    )
    assert failure["failed_stage"] == "snapshot_runtime"
    assert failure["scientifically_evaluable"] is False
    assert failure["all_decisions_withheld"] is True
    assert failure["retry_permitted"] is False
    assert failure["partial_root_permanently_reserved"] is True


def test_energy_is_measurement_only_and_final_video_is_read_only_postprocessing() -> None:
    config = load_config()
    v1 = json.loads(
        (ROOT / config["design_source"]["configuration"]).read_text(encoding="utf-8")
    )
    assert v1["energy_boundary"]["status"] == "measurement_only_not_reward_or_gate"
    assert v1["energy_boundary"]["reward_weight"] == 0.0
    assert config["hard_stop_and_archive"]["post_result_read_only_video_contract_required"] is True
    assert config["hard_stop_and_archive"]["video_participates_in_scientific_gate"] is False
