from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "evaluate_fixed_standard_pair0_flat_turn_diagnostic.py"
CONFIG = ROOT / "configs" / "fixed_standard_pair0_flat_turn_diagnostic_v1_20260819.json"
OUTPUT = ROOT / "artifacts" / "dev" / "fixed_standard_pair0_flat_turn_diagnostic_v1_20260819" / "attempt_0"
SPEC = importlib.util.spec_from_file_location("pair0_flat_turn", RUNNER)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validated(config: dict) -> tuple[dict, dict, Path]:
    return module.validate_config(config)


def test_canonical_configuration_validates_without_mutating_attempt(config: dict, validated: tuple[dict, dict, Path]) -> None:
    assert validated[2].is_file()
    manifest = OUTPUT / "manifest.json"
    before = module.slope.sha256(manifest) if manifest.is_file() else None
    module.validate_config(config)
    after = module.slope.sha256(manifest) if manifest.is_file() else None
    assert after == before
    if OUTPUT.exists():
        assert manifest.is_file()
        assert not (OUTPUT / "FAILURE_RECORD.json").exists()
    assert config["source"]["observation_dimension"] == 135
    assert config["evaluation"]["local_terrain_preview_dimension"] == 13


def test_matrix_is_predeclared_45_episode_design(config: dict) -> None:
    specs = module.condition_specs(config)
    assert len(specs) == 9
    assert len(module.EXPECTED_SEEDS) == 5
    assert len(specs) * len(module.EXPECTED_SEEDS) == 45
    assert len(set(module.EXPECTED_SEEDS)) == 5
    assert set(module.EXPECTED_SEEDS).isdisjoint(module.slope.EXPECTED_SEEDS)


def test_constant_curvature_yaw_rates_and_left_right_signs(config: dict) -> None:
    for row in module.condition_specs(config):
        assert row["speed_m_per_s"] > 0.0
        assert row["target_yaw_rate_rad_per_s"] == pytest.approx(
            row["speed_m_per_s"] * row["target_curvature_per_m"]
        )
        if "left" in row["condition_name"]:
            assert row["target_yaw_rate_rad_per_s"] > 0.0
        elif "right" in row["condition_name"]:
            assert row["target_yaw_rate_rad_per_s"] < 0.0
    straight = module.condition_specs(config)[0]
    assert straight["target_yaw_rate_rad_per_s"] == 0.0


def test_low_speed_probe_is_not_mislabelled_as_in_place(config: dict) -> None:
    low = [row for row in module.condition_specs(config) if row["condition_name"].startswith("low_speed")]
    assert len(low) == 2
    assert all(row["speed_m_per_s"] == 0.1 for row in low)
    assert all(abs(row["target_yaw_rate_rad_per_s"]) == 0.1 for row in low)
    assert all(row["out_of_training_command_envelope"] for row in low)
    assert all("not_in_place_turn" in row["kind"] for row in low)


def test_turn_effectiveness_is_descriptive_and_safety_only_is_gated(config: dict) -> None:
    turn = config["turn_effectiveness"]
    assert turn["decision_status"] == "descriptive_only_no_pass_fail"
    assert turn["formal_gate_available"] is False
    assert turn["post_hoc_threshold_selection_permitted"] is False
    assert "minimum_yaw" not in json.dumps(turn)
    assert "maximum_yaw" not in json.dumps(turn)


def test_reference_trajectory_matches_straight_and_circular_geometry() -> None:
    start = np.asarray([1.0, -2.0])
    straight = module.reference_xy(start, 0.0, 0.55, 0.0, 2.0)
    np.testing.assert_allclose(straight, [2.1, -2.0], atol=1e-12)
    quarter = module.reference_xy(start, 0.0, 1.0, 1.0, math.pi / 2.0)
    np.testing.assert_allclose(quarter, [2.0, -1.0], atol=1e-12)
    right = module.reference_xy(start, 0.0, 1.0, -1.0, math.pi / 2.0)
    np.testing.assert_allclose(right, [2.0, -3.0], atol=1e-12)
    assert module.wrapped_angle_difference(-math.pi + 0.01, math.pi - 0.01) == pytest.approx(0.02)
    assert module.wrapped_angle_difference(math.pi - 0.01, -math.pi + 0.01) == pytest.approx(-0.02)


def test_all_nine_reference_paths_retain_predeclared_boundary_margin(config: dict) -> None:
    audit = module.reference_envelope_audit(config)
    assert audit["all_conditions_passed"] is True
    assert set(audit["conditions"]) == {
        row["condition_name"] for row in module.condition_specs(config)
    }
    assert all(
        row["continuous_boundary_margin_lower_bound_m"] >= 3.0
        for row in audit["conditions"].values()
    )
    assert audit["conditions"]["straight_055"][
        "minimum_sampled_boundary_margin_m"
    ] == pytest.approx(3.5)


def test_expanded_flat_preserves_source_physical_grid_spacing(config: dict, validated: tuple[dict, dict, Path]) -> None:
    protocol, _, _ = validated
    source = protocol["standard_scenes"]
    expanded = module.diagnostic_flat_protocol(config, protocol)["standard_scenes"]
    source_spacing = 2.0 * source["map_half_extent_m"] / (source["grid_rows"] - 1)
    expanded_spacing = 2.0 * expanded["map_half_extent_m"] / (expanded["grid_rows"] - 1)
    assert source_spacing == pytest.approx(0.078125)
    assert expanded_spacing == source_spacing
    assert expanded["grid_rows"] == expanded["grid_cols"] == 513
    assert expanded["start_xy_m"] == [0.0, 0.0]


def test_terrain_health_audit_detects_low_clearance_at_exact_grace() -> None:
    state = module.new_terrain_health_audit()
    qpos = np.zeros(15, dtype=np.float64)
    qpos[2] = 0.1
    qpos[3] = 1.0
    qvel = np.zeros(14, dtype=np.float64)
    for step in range(1, 5):
        module.update_terrain_health_audit(
            state,
            qpos=qpos,
            qvel=qvel,
            terrain_height_m=0.0,
            map_half_extent_m=20.0,
            healthy_clearance_m=(0.18, 1.4),
            maximum_healthy_tilt_rad=math.radians(80.0),
            unhealthy_grace_steps=5,
            control_step=step,
        )
        assert state["terrain_relative_fall"] is False
    module.update_terrain_health_audit(
        state,
        qpos=qpos,
        qvel=qvel,
        terrain_height_m=0.0,
        map_half_extent_m=20.0,
        healthy_clearance_m=(0.18, 1.4),
        maximum_healthy_tilt_rad=math.radians(80.0),
        unhealthy_grace_steps=5,
        control_step=5,
    )
    assert state["terrain_relative_fall"] is True
    assert state["first_fall_control_step"] == 5
    assert state["fall_reason"] == "terrain_relative_torso_clearance"
    assert state["maximum_unhealthy_run_steps"] == 5


def _episode_row(seed: int, *, zero_count: int = 0, force_count: int = 2500) -> dict:
    return {
        "evaluation_seed": seed,
        "control_steps": 600,
        "physics_substeps": 3000,
        "finite": True,
        "fall": False,
        "fixed_goal_success": False,
        "fixed_goal_best_progress_m": 5.0,
        "fixed_goal_net_progress_m": 4.0,
        "torso_ground_any": False,
        "sustained_nonfoot_contact": False,
        "full_interval_zero_foot_count": zero_count,
        "support_count_sum_physics_substeps": 7000,
        "supported_physics_substep_count": 2700,
        "force_qualified_supported_physics_substep_count": force_count,
        "qualified_slip_physics_substep_count": 0,
        "corrected_sustained_slip_physics_substep_count": 0,
        "corrected_slip_event_count": 0,
        "target_cumulative_yaw_change_rad": 1.65,
        "actual_cumulative_yaw_change_rad": 1.5,
        "yaw_change_target_ratio": 1.5 / 1.65,
        "yaw_change_same_sign_as_target": True,
        "cumulative_yaw_error_rad": -0.15,
        "yaw_rate_rmse_rad_per_s": 0.02,
        "actual_path_integrated_curvature_per_m": 0.09,
        "planar_path_length_m": 16.0,
        "signed_initial_heading_progress_m": 10.0,
        "final_com_displacement_m": 10.0,
        "maximum_com_displacement_m": 10.5,
        "final_com_reference_error_m": 1.0,
        "maximum_com_reference_error_m": 1.2,
        "cumulative_squared_action": 100.0,
        "actuator_abs_torque_time_integral_total_n_m_s": 200.0,
        "actuator_positive_mechanical_work_total_j": 300.0,
        "actuator_abs_mechanical_work_total_j": 400.0,
    }


def test_safety_gate_requires_each_seed_force_denominator(config: dict) -> None:
    condition = module.condition_specs(config)[1]
    rows = [_episode_row(seed) for seed in module.EXPECTED_SEEDS]
    result = module.aggregate_condition(config, condition, rows)
    assert result["safety_passed"] is True
    assert result["turn_effectiveness_passed"] is None
    broken = copy.deepcopy(rows)
    broken[2]["force_qualified_supported_physics_substep_count"] = 0
    result = module.aggregate_condition(config, condition, broken)
    assert result["safety_passed"] is False
    assert "force_qualified_denominator_evaluable" in result["failed_safety_checks"]
    assert result["per_seed_safety_failures"][str(module.EXPECTED_SEEDS[2])] == [
        "force_qualified_denominator_evaluable"
    ]


def test_zero_foot_gate_is_pooled_and_predeclared(config: dict) -> None:
    condition = module.condition_specs(config)[0]
    allowed = int(math.floor(3000 * config["safety_gates"]["maximum_pooled_full_interval_zero_foot_fraction"]))
    rows = [_episode_row(seed) for seed in module.EXPECTED_SEEDS]
    for row in rows:
        row["target_cumulative_yaw_change_rad"] = 0.0
        row["yaw_change_target_ratio"] = None
        row["yaw_change_same_sign_as_target"] = None
    rows[0]["full_interval_zero_foot_count"] = allowed
    passing = module.aggregate_condition(config, condition, rows)
    assert passing["safety_passed"] is True
    assert passing["mean_yaw_change_target_ratio"] is None
    assert passing["same_sign_episode_count"] is None
    rows = [_episode_row(seed, zero_count=100) for seed in module.EXPECTED_SEEDS]
    for row in rows:
        row["target_cumulative_yaw_change_rad"] = 0.0
        row["yaw_change_target_ratio"] = None
        row["yaw_change_same_sign_as_target"] = None
    result = module.aggregate_condition(config, condition, rows)
    assert result["safety_passed"] is False
    assert "zero_foot_within_gate" in result["failed_safety_checks"]


def test_runtime_contract_rejects_missing_and_extra_paths(config: dict) -> None:
    missing = copy.deepcopy(config)
    missing["runtime_dependency_contract"]["exact_relative_path_sha256"].pop(
        "src/proxygap/two_experiment_protocol.py"
    )
    with pytest.raises(ValueError, match="membership"):
        module.validate_runtime_dependencies(missing)
    extra = copy.deepcopy(config)
    extra["runtime_dependency_contract"]["exact_relative_path_sha256"]["README.md"] = "0" * 64
    with pytest.raises(ValueError, match="membership"):
        module.validate_runtime_dependencies(extra)


def test_runtime_snapshot_rejects_hash_mutation_and_extra(config: dict, tmp_path: Path) -> None:
    output = tmp_path / "attempt"
    output.mkdir()
    module.snapshot_runtime(config, output)
    snapshot = output / "runtime_snapshot"
    module.validate_runtime_snapshot(config, snapshot)
    target = snapshot / "src" / "proxygap" / "selection.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        module.validate_runtime_snapshot(config, snapshot)
    target.write_bytes((module.ROOT / "src/proxygap/selection.py").read_bytes())
    (snapshot / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership"):
        module.validate_runtime_snapshot(config, snapshot)


def test_failure_after_root_creation_is_fail_closed_and_non_retryable(
    config: dict,
    validated: tuple[dict, dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, reward, checkpoint = validated
    output = tmp_path / "attempt_0"

    def injected_failure(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("injected scene failure")

    monkeypatch.setattr(module, "prepare_flat_scene", injected_failure)
    with pytest.raises(RuntimeError, match="injected scene failure"):
        module.run(CONFIG, config, protocol, reward, checkpoint, output)
    failure = json.loads((output / "FAILURE_RECORD.json").read_text(encoding="utf-8"))
    assert failure["scientifically_evaluable"] is False
    assert failure["all_safety_decisions_withheld"] is True
    assert failure["all_turn_tracking_interpretations_withheld"] is True
    assert failure["retry_permitted"] is False
    assert failure["failed_stage"] == "prepare_and_audit_flat_scene"
    with pytest.raises(FileExistsError):
        module.run(CONFIG, config, protocol, reward, checkpoint, output)


def test_runner_has_no_training_checkpoint_write_video_or_promotion_calls() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert "render(" not in source
    assert "Video" not in source
    assert "candidate_promoted\": True" not in source
    assert "fixed_map_evaluated\": True" not in source


def test_commanded_observation_preserves_135d_contract(config: dict, validated: tuple[dict, dict, Path], tmp_path: Path) -> None:
    protocol, reward, _ = validated
    scene = module.prepare_flat_scene(config, protocol, tmp_path / "scene")
    assert scene["map_half_extent_m"] == 20.0
    assert scene["start_xy_m"] == [0.0, 0.0]
    assert np.load(scene["heights_path"], allow_pickle=False).shape == (513, 513)
    local_protocol = copy.deepcopy(protocol)
    local_protocol["task_adapter"]["maximum_abs_curvature_per_m"] = 1.0
    env = module.slope.l2.make_standard_env(
        local_protocol,
        reward,
        scene,
        condition_id=module.PAIR0_ID,
        seed=module.EXPECTED_SEEDS[0],
        max_episode_steps=2,
        cruise_speed=0.1,
    )
    observation, _ = env.reset(seed=module.EXPECTED_SEEDS[0])
    observed_com = module.whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
    torso_id = module.mujoco.mj_name2id(
        env.unwrapped.model,
        module.mujoco.mjtObj.mjOBJ_BODY,
        "torso",
    )
    np.testing.assert_allclose(
        observed_com,
        np.asarray(env.unwrapped.data.subtree_com[torso_id, :2]),
        atol=0.0,
    )
    yaw = module.quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
    commanded = module.commanded_observation(
        env,
        observation[:122],
        target_heading=yaw,
        yaw_rate=0.1,
        speed=0.1,
    )
    assert commanded[115] == pytest.approx(0.1)
    assert env.env._external_yaw_rate_command == pytest.approx(0.1)
    assert env.env._current_curvature == pytest.approx(1.0)
    commanded_right = module.commanded_observation(
        env,
        observation[:122],
        target_heading=yaw,
        yaw_rate=-0.1,
        speed=0.1,
    )
    assert commanded_right[115] == pytest.approx(-0.1)
    assert env.env._external_yaw_rate_command == pytest.approx(-0.1)
    assert env.env._current_curvature == pytest.approx(-1.0)
    env.close()
    assert commanded.shape == (135,)
    assert np.all(np.isfinite(commanded))
    assert commanded[-13:].shape == (13,)


def test_two_step_read_only_episode_path_records_tracking_contact_and_energy(
    config: dict,
    validated: tuple[dict, dict, Path],
    tmp_path: Path,
) -> None:
    protocol, reward, checkpoint = validated
    local_config = copy.deepcopy(config)
    local_config["evaluation"]["max_episode_steps"] = 2
    scene = module.prepare_flat_scene(local_config, protocol, tmp_path / "two_step")
    model = module.PPO.load(checkpoint, device="cpu")
    condition = next(
        row
        for row in module.condition_specs(local_config)
        if row["condition_name"] == "curve_left_010"
    )
    row, events = module.evaluate_episode(
        model,
        local_config,
        protocol,
        reward,
        scene,
        condition,
        module.EXPECTED_SEEDS[0],
    )
    assert row["control_steps"] == 2
    assert row["physics_substeps"] == 10
    assert row["target_cumulative_yaw_change_rad"] == pytest.approx(0.055 * 0.1)
    assert row["turn_effectiveness_decision"] == "descriptive_only_no_pass_fail"
    assert row["checkpoint_timesteps"] == 2_727_936
    assert row["terrain_relative_fall"] is False
    assert math.isfinite(row["terrain_relative_minimum_torso_clearance_m"])
    assert module.energy_is_finite(row)
    assert isinstance(events, list)
