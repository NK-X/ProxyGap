"""Read-only flat-ground forward and turning diagnostic for final PAIR0.

The frozen checkpoint is evaluated under nine predeclared constant-command
conditions.  Safety has a formal gate; turn tracking is descriptive because no
directly transferable, pre-existing constant-yaw-rate threshold was found.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_standard_pair0_slope_capability_boundary as slope  # noqa: E402
from proxygap.curved_gait import quaternion_yaw_angle, wrapped_angle_difference  # noqa: E402
from proxygap.fixed_goal_terrain import quaternion_tilt_angle  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "fixed_standard_pair0_flat_turn_diagnostic_v1_20260819.json"
PAIR0_ID = "PAIR0_ADAPT"
EXPECTED_SEEDS = (95131, 95137, 95149, 95153, 95177)
ENERGY_KEYS = slope.ENERGY_KEYS
RUNTIME_SELF_RELATIVE_PATH = "scripts/evaluate_fixed_standard_pair0_flat_turn_diagnostic.py"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF_RELATIVE_PATH,
    "scripts/evaluate_fixed_standard_pair0_slope_capability_boundary.py",
    *slope.EXPECTED_RUNTIME_PATHS[1:],
)
FROZEN_NORMALISED_RUNTIME_CONTRACT_SHA256 = "1ad57a916bd108251469d149499d46c0d0e9586c31865d56ed5766ff99442ad3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser.parse_args()


def _equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen field changed: {label}: {observed!r} != {expected!r}")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = slope.sha256(path)
    if observed != str(record[hash_key]):
        raise ValueError(f"SHA-256 changed for {path_key}: {observed}")
    return path


def condition_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in config["evaluation"]["conditions"]]


def validate_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    contract = config["runtime_dependency_contract"]
    _equal(
        tuple(contract),
        ("copy_preserving_relative_paths", "verify_before_and_after", "exact_relative_path_sha256"),
        "runtime contract keys/order",
    )
    expected = contract["exact_relative_path_sha256"]
    _equal(tuple(expected), EXPECTED_RUNTIME_PATHS, "runtime exact membership/order")
    normalised = copy.deepcopy(contract)
    normalised["exact_relative_path_sha256"][RUNTIME_SELF_RELATIVE_PATH] = "<RUNNER_SELF_SHA256>"
    _equal(
        _canonical_json_sha256(normalised),
        FROZEN_NORMALISED_RUNTIME_CONTRACT_SHA256,
        "normalised runtime contract digest",
    )
    observed: dict[str, str] = {}
    for relative_path, digest in expected.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = slope.sha256(path)
        if actual != digest:
            raise ValueError(f"Runtime dependency changed: {relative_path}")
        observed[relative_path] = actual
    return observed


def _expected_conditions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"condition_name": "straight_055", "kind": "straight", "speed_m_per_s": 0.55, "target_curvature_per_m": 0.0, "target_yaw_rate_rad_per_s": 0.0, "out_of_training_command_envelope": False},
    ]
    for curvature in (0.1, 0.2, 0.35):
        suffix = f"{int(round(curvature * 100)):03d}"
        for side, sign in (("left", 1.0), ("right", -1.0)):
            rows.append(
                {
                    "condition_name": f"curve_{side}_{suffix}",
                    "kind": "constant_curvature",
                    "speed_m_per_s": 0.55,
                    "target_curvature_per_m": sign * curvature,
                    "target_yaw_rate_rad_per_s": sign * round(0.55 * curvature, 4),
                    "out_of_training_command_envelope": False,
                }
            )
    rows.extend(
        [
            {"condition_name": "low_speed_yaw_left", "kind": "positive_speed_yaw_rate_probe_not_in_place_turn", "speed_m_per_s": 0.1, "target_curvature_per_m": 1.0, "target_yaw_rate_rad_per_s": 0.1, "out_of_training_command_envelope": True},
            {"condition_name": "low_speed_yaw_right", "kind": "positive_speed_yaw_rate_probe_not_in_place_turn", "speed_m_per_s": 0.1, "target_curvature_per_m": -1.0, "target_yaw_rate_rad_per_s": -0.1, "out_of_training_command_envelope": True},
        ]
    )
    return rows


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _equal(config.get("schema_version"), "proxygap-pair0-flat-turn-diagnostic-v1", "schema")
    _equal(config.get("config_id"), "fixed_standard_pair0_flat_turn_diagnostic_v1_20260819", "config id")
    _equal(config.get("status"), "frozen_read_only_predeclared_final_pair0_flat_turn_diagnostic", "status")
    source = config["source"]
    v3_config_path = verify_file(source, "v3_configuration", "v3_configuration_sha256")
    verify_file(source, "v3_frozen_configuration", "v3_frozen_configuration_sha256")
    verify_file(source, "v3_manifest", "v3_manifest_sha256")
    gate_path = verify_file(source, "v3_final_gate", "v3_final_gate_sha256")
    checkpoint = verify_file(source, "checkpoint", "checkpoint_sha256")
    _equal(int(source["checkpoint_timesteps"]), 2_727_936, "checkpoint timesteps")
    _equal(int(source["checkpoint_additional_timesteps"]), 65_536, "additional timesteps")
    _equal(int(source["observation_dimension"]), 135, "observation dimension")
    _equal(int(source["action_dimension"]), 8, "action dimension")
    _equal(source["observation_contract"], "122D locomotion/contact/command observation plus 13D local terrain preview", "observation contract")
    v3_config = json.loads(v3_config_path.read_text(encoding="utf-8"))
    protocol, reward = slope.l2b.validate_config(v3_config)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or int(gate.get("absolute_final_checkpoint", -1)) != 2_727_936:
        raise ValueError("Source V3 final gate is not satisfied")
    model = PPO.load(checkpoint, device="cpu")
    _equal(int(model.num_timesteps), 2_727_936, "loaded checkpoint timesteps")
    _equal(tuple(model.observation_space.shape), (135,), "checkpoint observation shape")
    _equal(tuple(model.action_space.shape), (8,), "checkpoint action shape")

    evidence = config["prior_protocol_evidence"]
    verify_file(evidence, "v4_configuration", "v4_configuration_sha256")
    verify_file(evidence, "v22_configuration", "v22_configuration_sha256")
    verify_file(evidence, "low_speed_diagnostic_configuration", "low_speed_diagnostic_configuration_sha256")
    _equal(evidence["transfer_decision"], "no_directly_transferable_constant_yaw_rate_effectiveness_gate", "threshold transfer decision")
    _equal(evidence["reason"], "V4 thresholds were selection criteria for a different policy, speed and curvature profiles; V22 had no numerical turn-effectiveness gate; the earlier low-speed diagnostic targeted fixed-angle completion rather than constant yaw-rate tracking.", "threshold transfer rationale")

    evaluation = config["evaluation"]
    _equal(tuple(evaluation["heldout_seeds"]), EXPECTED_SEEDS, "held-out seeds")
    _equal(evaluation["seed_status"], "predeclared_new_heldout_not_used_by_v3_slope_or_prior_configured_evaluations", "seed status")
    excluded = set(v3_config["evaluation"]["intermediate_safety_audit"]["seeds"])
    excluded.update(v3_config["evaluation"]["final_heldout"]["seeds"])
    excluded.update(slope.EXPECTED_SEEDS)
    if set(EXPECTED_SEEDS) & excluded:
        raise ValueError("Flat-turn seeds overlap V3 or slope evaluation seeds")
    _equal(evaluation["condition_order"], [row["condition_name"] for row in _expected_conditions()], "condition order")
    _equal(condition_specs(config), _expected_conditions(), "condition definitions")
    _equal(int(evaluation["max_episode_steps"]), 600, "horizon")
    _equal(float(evaluation["control_timestep_seconds"]), 0.05, "control timestep")
    _equal(float(evaluation["physics_timestep_seconds"]), 0.01, "physics timestep")
    _equal(int(evaluation["physics_substeps_per_control_step"]), 5, "substeps")
    _equal(bool(evaluation["all_five_physics_substeps_required"]), True, "all substeps")
    _equal(bool(evaluation["deterministic_policy"]), True, "deterministic policy")
    _equal(bool(evaluation["flat_scene_only"]), True, "flat only")
    _equal(int(evaluation["local_terrain_preview_dimension"]), 13, "terrain preview dimension")
    _equal(float(evaluation["policy_training_maximum_abs_curvature_per_m"]), 0.35, "training curvature")
    _equal(float(protocol["task_adapter"]["maximum_abs_curvature_per_m"]), 0.35, "source training curvature")
    _equal(float(evaluation["diagnostic_command_adapter_maximum_abs_curvature_per_m"]), 1.0, "diagnostic adapter curvature")
    _equal(evaluation["diagnostic_adapter_change"], "command_admissibility_only_no_reward_or_policy_change", "diagnostic adapter scope")
    _equal(evaluation["target_heading_definition"], "initial_torso_yaw_plus_target_yaw_rate_times_elapsed_control_time", "target heading definition")
    _equal(evaluation["com_definition"], "mujoco_torso_subtree_com_representing_whole_ant", "COM definition")
    _equal(
        evaluation["terrain_relative_fall_audit"],
        {
            "definition": "mirror_fixed_goal_terrain_clearance_tilt_bounds_finite_and_grace_without_early_termination",
            "terrain_relative_healthy_clearance_m": [0.18, 1.4],
            "maximum_healthy_tilt_degrees": 80.0,
            "unhealthy_grace_steps": 5,
            "continue_after_detection_to_complete_horizon": True,
        },
        "terrain-relative fall audit",
    )
    _equal(
        evaluation["terrain_relative_fall_audit"]["terrain_relative_healthy_clearance_m"],
        protocol["task_adapter"]["terrain_relative_healthy_clearance_m"],
        "source healthy clearance",
    )
    _equal(
        float(evaluation["terrain_relative_fall_audit"]["maximum_healthy_tilt_degrees"]),
        float(protocol["task_adapter"]["maximum_healthy_tilt_degrees"]),
        "source maximum tilt",
    )
    _equal(
        int(evaluation["terrain_relative_fall_audit"]["unhealthy_grace_steps"]),
        int(protocol["task_adapter"]["unhealthy_grace_steps"]),
        "source unhealthy grace",
    )
    diagnostic_scene = evaluation["diagnostic_flat_scene"]
    _equal(
        diagnostic_scene,
        {
            "source_grid_rows": 257,
            "source_grid_cols": 257,
            "source_map_half_extent_m": 10.0,
            "source_grid_spacing_m": 0.078125,
            "grid_rows": 513,
            "grid_cols": 513,
            "map_half_extent_m": 20.0,
            "grid_spacing_m": 0.078125,
            "start_xy_m": [0.0, 0.0],
            "goal_marker_xy_m": [6.0, 0.0],
            "required_continuous_reference_boundary_margin_m": 3.0,
            "reference_envelope_audit_timestep_seconds": 0.01,
            "continuous_margin_lower_bound": "minimum_sampled_margin_minus_maximum_commanded_speed_times_audit_timestep",
            "purpose": "expanded_numerical_flat_heightfield_for_turn_diagnostic_not_fixed_delivery_map",
        },
        "diagnostic flat scene",
    )
    source_scene = protocol["standard_scenes"]
    _equal(int(source_scene["grid_rows"]), int(diagnostic_scene["source_grid_rows"]), "source rows")
    _equal(int(source_scene["grid_cols"]), int(diagnostic_scene["source_grid_cols"]), "source cols")
    _equal(float(source_scene["map_half_extent_m"]), float(diagnostic_scene["source_map_half_extent_m"]), "source extent")
    source_spacing = 2.0 * float(source_scene["map_half_extent_m"]) / (int(source_scene["grid_rows"]) - 1)
    expanded_spacing = 2.0 * float(diagnostic_scene["map_half_extent_m"]) / (int(diagnostic_scene["grid_rows"]) - 1)
    _equal(source_spacing, float(diagnostic_scene["source_grid_spacing_m"]), "declared source spacing")
    _equal(expanded_spacing, float(diagnostic_scene["grid_spacing_m"]), "declared expanded spacing")
    _equal(expanded_spacing, source_spacing, "preserved physical grid spacing")
    _equal(evaluation["corrected_slip"], v3_config["evaluation"]["corrected_slip"], "slip definition")
    _equal(config["contact_contract"], v3_config["contact_contract"], "PAIR0 contact contract")

    gates = config["safety_gates"]
    for key in ("maximum_fall_count", "maximum_torso_ground_episode_count", "maximum_sustained_nonfoot_contact_episode_count", "maximum_corrected_sustained_slip_substep_count", "maximum_corrected_slip_event_count"):
        _equal(int(gates[key]), 0, key)
    _equal(float(gates["nonfoot_minimum_sustained_seconds"]), 0.2, "nonfoot duration")
    _equal(bool(gates["force_qualified_denominator_required_for_every_seed"]), True, "denominator gate")
    _equal(float(gates["maximum_pooled_full_interval_zero_foot_fraction"]), 0.0580555556, "zero-foot gate")
    _equal(bool(gates["complete_five_seed_matrix_required"]), True, "five-seed matrix")
    _equal(bool(gates["required_all_checks"]), True, "all safety checks")
    turn = config["turn_effectiveness"]
    _equal(turn["decision_status"], "descriptive_only_no_pass_fail", "turn decision status")
    _equal(bool(turn["post_hoc_threshold_selection_permitted"]), False, "post-hoc threshold ban")
    _equal(bool(turn["formal_gate_available"]), False, "turn gate availability")
    _equal(
        tuple(turn["required_measurements"]),
        (
            "target_cumulative_yaw_change_rad",
            "actual_cumulative_yaw_change_rad",
            "yaw_change_target_ratio",
            "yaw_change_same_sign_as_target",
            "cumulative_yaw_error_rad",
            "yaw_rate_rmse_rad_per_s",
            "actual_path_integrated_curvature_per_m",
            "curvature_error_per_m",
            "planar_path_length_m",
            "signed_initial_heading_progress_m",
            "final_com_displacement_m",
            "maximum_com_displacement_m",
            "final_com_reference_error_m",
            "maximum_com_reference_error_m",
        ),
        "tracking measurements",
    )
    _equal(turn["straight_ratio_status"], "not_applicable_zero_target_yaw", "straight ratio status")
    _equal(turn["low_speed_label_constraint"], "positive_speed_yaw_rate_probe_not_in_place_turn", "low-speed claim")
    _equal(turn["claim_permitted"], "observed_tracking_diagnostics_only_not_turn_capability_certification", "turn claim boundary")
    _equal(tuple(config["energy_measurement"]["raw_components_required"]), ENERGY_KEYS, "energy fields")
    _equal(config["energy_measurement"]["status"], "measurement_only_not_reward_or_gate", "energy status")
    _equal(bool(config["energy_measurement"]["nonfinite_is_safety_failure"]), True, "energy finiteness")
    _equal(bool(config["energy_measurement"]["electrical_battery_energy_claim_permitted"]), False, "battery-energy claim")
    _equal(config["checkpoint_early_stopping"], {"nonfoot_contact_minimum_sustained_seconds": 0.2}, "nonfoot contract")
    _equal(
        config["invariants"],
        {"training_performed": False, "checkpoint_modified": False, "reward_changed": False, "friction_changed": False, "energy_formula_changed": False, "energy_status": "measurement_only_not_reward_or_gate", "energy_reward_weight": 0.0, "fixed_map_evaluated": False, "video_rendered": False, "candidate_promoted": False, "observation_dimension": 135, "action_dimension": 8, "explicit_pair_count": 4},
        "invariants",
    )
    execution = config["execution"]
    _equal(execution["formal_output_root"], "artifacts/dev/fixed_standard_pair0_flat_turn_diagnostic_v1_20260819/attempt_0", "output root")
    _equal(bool(execution["fail_if_output_root_exists"]), True, "overwrite refusal")
    for key in ("training", "checkpoint_write", "fixed_map", "video", "promotion"):
        _equal(bool(execution[key]), False, f"execution {key}")
    _equal(bool(config["runtime_dependency_contract"]["copy_preserving_relative_paths"]), True, "runtime copy")
    _equal(bool(config["runtime_dependency_contract"]["verify_before_and_after"]), True, "runtime verification")
    _equal(config["claim_boundary"], "This protocol can diagnose flat-ground tracking and safety of one frozen policy under predeclared commands. It does not certify optimal turning, true in-place rotation, fixed-map readiness, or random-map generalisation.", "claim boundary")
    reference_envelope_audit(config)
    validate_runtime_dependencies(config)
    return protocol, reward, checkpoint


def snapshot_runtime(config: dict[str, Any], output_root: Path) -> dict[str, str]:
    observed = validate_runtime_dependencies(config)
    snapshot_root = output_root / "runtime_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for relative_path, digest in observed.items():
        target = snapshot_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
        if slope.sha256(target) != digest:
            raise RuntimeError(f"Runtime snapshot changed: {relative_path}")
    return observed


def validate_runtime_snapshot(config: dict[str, Any], snapshot_root: Path) -> dict[str, str]:
    actual_paths = tuple(sorted(path.relative_to(snapshot_root).as_posix() for path in snapshot_root.rglob("*") if path.is_file()))
    if actual_paths != tuple(sorted(EXPECTED_RUNTIME_PATHS)):
        raise RuntimeError("Runtime snapshot exact membership changed")
    expected = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    observed: dict[str, str] = {}
    for relative_path in EXPECTED_RUNTIME_PATHS:
        actual = slope.sha256(snapshot_root / relative_path)
        if actual != expected[relative_path]:
            raise RuntimeError(f"Runtime snapshot changed: {relative_path}")
        observed[relative_path] = actual
    return observed


def diagnostic_flat_protocol(config: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    local_protocol = copy.deepcopy(protocol)
    scene = config["evaluation"]["diagnostic_flat_scene"]
    standard = local_protocol["standard_scenes"]
    standard["scene_order"] = ["flat"]
    standard["grid_rows"] = int(scene["grid_rows"])
    standard["grid_cols"] = int(scene["grid_cols"])
    standard["map_half_extent_m"] = float(scene["map_half_extent_m"])
    standard["start_xy_m"] = list(scene["start_xy_m"])
    standard["goal_xy_m"] = list(scene["goal_marker_xy_m"])
    return local_protocol


def prepare_flat_scene(config: dict[str, Any], protocol: dict[str, Any], output_root: Path) -> dict[str, Any]:
    local_protocol = diagnostic_flat_protocol(config, protocol)
    controls, generation = slope.prepare_standard_scenes(local_protocol, output_root / "scene_source")
    source = dict(controls["flat"])
    observed = np.load(source["heights_path"], allow_pickle=False)
    expected = slope.build_standard_heights(local_protocol["standard_scenes"])["flat"]
    if not np.array_equal(observed, expected):
        raise RuntimeError("Flat analytic height recipe changed")
    source.update({"scene_name": "flat_turn_diagnostic", "direction": "flat", "angle_degrees": 0})
    pair, audit = slope.prepare_pair(source, output_root / "condition_assets", "flat_turn_diagnostic", slope._pair_contract(config))
    candidate = dict(pair[slope.CANDIDATE_ID])
    candidate.update({"condition_id": PAIR0_ID, "scene_name": "flat_turn_diagnostic", "direction": "flat", "angle_degrees": 0})
    model = mujoco.MjModel.from_xml_path(candidate["xml_path"])
    slope.l2.compiled_contract_audit(model, candidate, PAIR0_ID, config, construction_seed=EXPECTED_SEEDS[0])
    diagnostic_scene = config["evaluation"]["diagnostic_flat_scene"]
    _equal(float(candidate["map_half_extent_m"]), float(diagnostic_scene["map_half_extent_m"]), "compiled diagnostic extent")
    _equal(candidate["start_xy_m"], diagnostic_scene["start_xy_m"], "compiled diagnostic start")
    _equal(candidate["goal_xy_m"], diagnostic_scene["goal_marker_xy_m"], "compiled diagnostic goal marker")
    envelope = reference_envelope_audit(config)
    slope.write_json(output_root / "prepared_scene.json", candidate)
    slope.write_json(output_root / "scene_contract_audit.json", {"generation": generation, "explicit_pair_audit": audit, "analytic_height_match": True, "candidate_xml_sha256": candidate["xml_sha256"], "reference_envelope_audit": envelope})
    return candidate


def reference_xy(initial_xy: np.ndarray, initial_yaw: float, speed: float, yaw_rate: float, elapsed_seconds: float) -> np.ndarray:
    if abs(yaw_rate) <= 1e-12:
        delta = speed * elapsed_seconds * np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)])
    else:
        final_yaw = initial_yaw + yaw_rate * elapsed_seconds
        delta = np.asarray([
            speed / yaw_rate * (math.sin(final_yaw) - math.sin(initial_yaw)),
            -speed / yaw_rate * (math.cos(final_yaw) - math.cos(initial_yaw)),
        ])
    return np.asarray(initial_xy, dtype=np.float64) + delta


def reference_envelope_audit(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    scene = evaluation["diagnostic_flat_scene"]
    extent = float(scene["map_half_extent_m"])
    start = np.asarray(scene["start_xy_m"], dtype=np.float64)
    audit_dt = float(scene["reference_envelope_audit_timestep_seconds"])
    duration = int(evaluation["max_episode_steps"]) * float(
        evaluation["control_timestep_seconds"]
    )
    sample_count = int(round(duration / audit_dt)) + 1
    times = np.linspace(0.0, duration, sample_count, dtype=np.float64)
    results: dict[str, Any] = {}
    maximum_speed = max(float(row["speed_m_per_s"]) for row in condition_specs(config))
    required = float(scene["required_continuous_reference_boundary_margin_m"])
    for condition in condition_specs(config):
        points = np.asarray(
            [
                reference_xy(
                    start,
                    0.0,
                    float(condition["speed_m_per_s"]),
                    float(condition["target_yaw_rate_rad_per_s"]),
                    float(value),
                )
                for value in times
            ],
            dtype=np.float64,
        )
        sampled_margin = float(
            np.min(extent - np.max(np.abs(points), axis=1))
        )
        continuous_lower_bound = sampled_margin - maximum_speed * audit_dt
        if continuous_lower_bound < required:
            raise ValueError(
                f"Reference path lacks boundary margin: {condition['condition_name']}: "
                f"{continuous_lower_bound} < {required}"
            )
        results[condition["condition_name"]] = {
            "minimum_sampled_boundary_margin_m": sampled_margin,
            "continuous_boundary_margin_lower_bound_m": continuous_lower_bound,
            "required_continuous_boundary_margin_m": required,
            "sample_count": sample_count,
            "audit_timestep_seconds": audit_dt,
            "passed": True,
        }
    return {
        "map_half_extent_m": extent,
        "start_xy_m": start.tolist(),
        "duration_seconds": duration,
        "maximum_commanded_speed_m_per_s": maximum_speed,
        "all_conditions_passed": True,
        "conditions": results,
    }


def whole_robot_com_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the planar centre of mass of the torso subtree (the whole Ant)."""

    torso_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso"))
    if torso_body_id <= 0:
        raise RuntimeError("Cannot identify the Ant torso subtree for COM measurement")
    value = np.asarray(data.subtree_com[torso_body_id, :2], dtype=np.float64).copy()
    if value.shape != (2,):
        raise RuntimeError("Whole-robot planar COM changed shape")
    return value


def new_terrain_health_audit() -> dict[str, Any]:
    return {
        "unhealthy_run_steps": 0,
        "maximum_unhealthy_run_steps": 0,
        "terrain_relative_fall": False,
        "first_fall_control_step": None,
        "fall_reason": None,
        "minimum_torso_clearance_m": math.inf,
        "maximum_torso_clearance_m": -math.inf,
        "maximum_torso_tilt_rad": 0.0,
    }


def update_terrain_health_audit(
    state: dict[str, Any],
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    terrain_height_m: float,
    map_half_extent_m: float,
    healthy_clearance_m: tuple[float, float],
    maximum_healthy_tilt_rad: float,
    unhealthy_grace_steps: int,
    control_step: int,
) -> dict[str, Any]:
    """Mirror FixedGoalTerrainWrapper health semantics without early truncation."""

    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    position = qpos[:2]
    clearance = float(qpos[2] - terrain_height_m)
    tilt = float(quaternion_tilt_angle(qpos[3:7]))
    finite = bool(
        np.all(np.isfinite(qpos))
        and np.all(np.isfinite(qvel))
        and np.isfinite(clearance)
        and np.isfinite(tilt)
    )
    out_of_bounds = bool(np.any(np.abs(position) > float(map_half_extent_m)))
    clearance_healthy = bool(
        float(healthy_clearance_m[0]) <= clearance <= float(healthy_clearance_m[1])
    )
    tilt_healthy = bool(tilt <= float(maximum_healthy_tilt_rad))
    healthy = finite and not out_of_bounds and clearance_healthy and tilt_healthy
    state["minimum_torso_clearance_m"] = min(
        float(state["minimum_torso_clearance_m"]), clearance
    )
    state["maximum_torso_clearance_m"] = max(
        float(state["maximum_torso_clearance_m"]), clearance
    )
    state["maximum_torso_tilt_rad"] = max(
        float(state["maximum_torso_tilt_rad"]), tilt
    )
    if healthy:
        state["unhealthy_run_steps"] = 0
    else:
        state["unhealthy_run_steps"] = int(state["unhealthy_run_steps"]) + 1
        if not finite:
            reason = "non_finite"
        elif out_of_bounds:
            reason = "out_of_bounds"
        elif not clearance_healthy:
            reason = "terrain_relative_torso_clearance"
        else:
            reason = "torso_tilt"
        if (
            int(state["unhealthy_run_steps"]) >= int(unhealthy_grace_steps)
            and not bool(state["terrain_relative_fall"])
        ):
            state["terrain_relative_fall"] = True
            state["first_fall_control_step"] = int(control_step)
            state["fall_reason"] = reason
    state["maximum_unhealthy_run_steps"] = max(
        int(state["maximum_unhealthy_run_steps"]), int(state["unhealthy_run_steps"])
    )
    return state


def commanded_observation(env: Any, observation_122: np.ndarray, *, target_heading: float, yaw_rate: float, speed: float) -> np.ndarray:
    curve_observation = env.env.set_external_curve_command(
        np.asarray(observation_122), target_heading=target_heading, yaw_rate=yaw_rate, speed=speed, lateral_speed=0.0
    )
    result = env._append_local_terrain_observation(curve_observation, env._position(), target_heading)
    if result.shape != (135,):
        raise RuntimeError(f"Commanded observation changed shape: {result.shape}")
    return result


def _vector_sum(summary: dict[str, Any], key: str) -> float:
    return float(np.sum(np.asarray(summary.get(key, []), dtype=np.float64)))


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def evaluate_episode(model: PPO, config: dict[str, Any], protocol: dict[str, Any], reward: dict[str, Any], scene: dict[str, Any], condition: dict[str, Any], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_protocol = copy.deepcopy(protocol)
    local_protocol["task_adapter"]["maximum_abs_curvature_per_m"] = float(config["evaluation"]["diagnostic_command_adapter_maximum_abs_curvature_per_m"])
    speed = float(condition["speed_m_per_s"])
    target_yaw_rate = float(condition["target_yaw_rate_rad_per_s"])
    target_curvature = float(condition["target_curvature_per_m"])
    horizon = int(config["evaluation"]["max_episode_steps"])
    control_dt = float(config["evaluation"]["control_timestep_seconds"])
    env = slope.l2.make_standard_env(local_protocol, reward, scene, condition_id=PAIR0_ID, seed=seed, max_episode_steps=horizon, cruise_speed=speed)
    slope.l2.compiled_contract_audit(env.unwrapped.model, scene, PAIR0_ID, config, construction_seed=seed)
    reset_observation, _ = env.reset(seed=seed)
    if reset_observation.shape != (135,):
        env.close()
        raise RuntimeError("Reset observation is not 135D")
    audit_state = slope.l2.install_five_substep_audit(env)
    initial_xy = whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
    initial_yaw = quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
    observation = commanded_observation(env, reset_observation[:122], target_heading=initial_yaw, yaw_rate=target_yaw_rate, speed=speed)
    slip = config["evaluation"]["corrected_slip"]
    tracker = slope.l2.DurationCorrectedSlipTracker(
        dt=float(config["evaluation"]["physics_timestep_seconds"]),
        speed_threshold=float(slip["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip["minimum_sustained_seconds"]),
    )
    contact_rows: list[np.ndarray] = []
    force_support_rows: list[bool] = []
    nonfoot_rows: list[bool] = []
    torso_rows: list[bool] = []
    control_fullzero: list[bool] = []
    actual_yaw_change = 0.0
    yaw_rate_error_squared_sum = 0.0
    previous_yaw = initial_yaw
    previous_xy = initial_xy.copy()
    path_length = 0.0
    maximum_displacement = 0.0
    maximum_reference_error = 0.0
    maximum_signed_progress = 0.0
    terrain_health = new_terrain_health_audit()
    finite = True
    terminated = truncated = False
    control_step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        raw_observation, reward_value, terminated, truncated, _ = env.env.step(action)
        control_step += 1
        rows = audit_state.get("last")
        if not isinstance(rows, list) or len(rows) != 5:
            env.close()
            raise RuntimeError("Five-substep audit did not return five rows")
        local_contacts: list[np.ndarray] = []
        for substep in rows:
            contacts = np.asarray(substep["contacts"], dtype=bool)
            speeds = np.asarray(substep["speeds"], dtype=np.float64)
            forces = np.asarray(substep["forces"], dtype=np.float64)
            if contacts.shape != (4,) or speeds.shape != (4,) or forces.shape != (4,):
                env.close()
                raise RuntimeError("Substep foot vector shape changed")
            tracker.update(contact_mask=contacts, tangential_speeds=speeds, normal_forces=forces)
            contact_rows.append(contacts.copy())
            force_support_rows.append(bool(np.any(contacts & (forces >= float(slip["minimum_normal_force_n"])))))
            nonfoot_rows.append(bool(substep["nonfoot"]))
            torso_rows.append(bool(substep["torso"]))
            local_contacts.append(contacts)
        control_fullzero.append(not np.any(np.asarray(local_contacts, dtype=bool)))
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
        xy = whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
        yaw = quaternion_yaw_angle(qpos[3:7])
        yaw_delta = wrapped_angle_difference(yaw, previous_yaw)
        actual_yaw_change += yaw_delta
        yaw_rate_error_squared_sum += (yaw_delta / control_dt - target_yaw_rate) ** 2
        path_length += float(np.linalg.norm(xy - previous_xy))
        displacement = float(np.linalg.norm(xy - initial_xy))
        maximum_displacement = max(maximum_displacement, displacement)
        elapsed = control_step * control_dt
        reference_error = float(np.linalg.norm(xy - reference_xy(initial_xy, initial_yaw, speed, target_yaw_rate, elapsed)))
        maximum_reference_error = max(maximum_reference_error, reference_error)
        signed_progress = float(np.dot(xy - initial_xy, np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)])))
        maximum_signed_progress = max(maximum_signed_progress, signed_progress)
        update_terrain_health_audit(
            terrain_health,
            qpos=qpos,
            qvel=qvel,
            terrain_height_m=float(env._terrain_height(float(qpos[0]), float(qpos[1]))),
            map_half_extent_m=float(env.map_half_extent_m),
            healthy_clearance_m=tuple(float(value) for value in env.healthy_clearance),
            maximum_healthy_tilt_rad=float(env.maximum_healthy_tilt),
            unhealthy_grace_steps=int(env.unhealthy_grace_steps),
            control_step=control_step,
        )
        finite = finite and bool(np.all(np.isfinite(raw_observation)) and np.all(np.isfinite(action)) and np.isfinite(reward_value) and np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        previous_xy = xy
        previous_yaw = yaw
        if not (terminated or truncated):
            target_heading = initial_yaw + target_yaw_rate * elapsed
            observation = commanded_observation(env, raw_observation, target_heading=target_heading, yaw_rate=target_yaw_rate, speed=speed)
        if control_step > horizon:
            env.close()
            raise RuntimeError("Evaluation exceeded frozen horizon")
    corrected = tracker.finalise()
    contacts = np.asarray(contact_rows, dtype=bool)
    candidate = np.asarray(corrected["candidate"], dtype=bool)
    sustained = np.asarray(corrected["sustained"], dtype=bool)
    if contacts.shape != candidate.shape or contacts.shape != sustained.shape:
        env.close()
        raise RuntimeError("Corrected-slip output shape changed")
    summary = env.env.episode_summary()
    final_xy = whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
    env.close()
    supported = np.any(contacts, axis=1)
    force_supported = np.asarray(force_support_rows, dtype=bool)
    target_yaw_change = target_yaw_rate * control_step * control_dt
    ratio = actual_yaw_change / target_yaw_change if abs(target_yaw_change) > 1e-12 else None
    same_sign = bool(actual_yaw_change * target_yaw_change > 0.0) if ratio is not None else None
    actual_curvature = actual_yaw_change / path_length if path_length > 1e-12 else None
    final_reference = reference_xy(initial_xy, initial_yaw, speed, target_yaw_rate, control_step * control_dt)
    events = [{"condition_id": PAIR0_ID, "condition_name": condition["condition_name"], "checkpoint_additional_timesteps": 65_536, "evaluation_seed": seed, **event} for event in corrected["events"]]
    dt = float(config["evaluation"]["physics_timestep_seconds"])
    row = {
        "condition_id": PAIR0_ID,
        "condition_name": condition["condition_name"],
        "condition_kind": condition["kind"],
        "evaluation_seed": seed,
        "checkpoint_additional_timesteps": 65_536,
        "checkpoint_timesteps": int(model.num_timesteps),
        "control_steps": control_step,
        "physics_substeps": int(contacts.shape[0]),
        "finite": finite,
        "fall": bool(
            summary.get("fall", False)
            or summary.get("inner_absolute_z_fall", False)
            or terrain_health["terrain_relative_fall"]
        ),
        "terrain_relative_fall": bool(terrain_health["terrain_relative_fall"]),
        "terrain_relative_first_fall_control_step": terrain_health["first_fall_control_step"],
        "terrain_relative_fall_reason": terrain_health["fall_reason"],
        "terrain_relative_maximum_unhealthy_run_steps": int(terrain_health["maximum_unhealthy_run_steps"]),
        "terrain_relative_minimum_torso_clearance_m": float(terrain_health["minimum_torso_clearance_m"]),
        "terrain_relative_maximum_torso_clearance_m": float(terrain_health["maximum_torso_clearance_m"]),
        "terrain_relative_maximum_torso_tilt_rad": float(terrain_health["maximum_torso_tilt_rad"]),
        "torso_ground_any": bool(np.any(torso_rows)),
        "nonfoot_ground_longest_run_seconds": _longest_true_run(nonfoot_rows) * dt,
        "sustained_nonfoot_contact": _longest_true_run(nonfoot_rows) * dt >= float(config["checkpoint_early_stopping"]["nonfoot_contact_minimum_sustained_seconds"]),
        "full_interval_zero_foot_count": int(np.sum(control_fullzero)),
        "support_count_sum_physics_substeps": int(np.sum(contacts)),
        "supported_physics_substep_count": int(np.sum(supported)),
        "force_qualified_supported_physics_substep_count": int(np.sum(force_supported)),
        "qualified_slip_physics_substep_count": int(np.sum(np.any(candidate, axis=1))),
        "corrected_sustained_slip_physics_substep_count": int(np.sum(np.any(sustained, axis=1))),
        "corrected_slip_event_count": len(events),
        "target_speed_m_per_s": speed,
        "target_yaw_rate_rad_per_s": target_yaw_rate,
        "target_curvature_per_m": target_curvature,
        "target_cumulative_yaw_change_rad": target_yaw_change,
        "actual_cumulative_yaw_change_rad": actual_yaw_change,
        "yaw_change_target_ratio": ratio,
        "yaw_change_same_sign_as_target": same_sign,
        "cumulative_yaw_error_rad": actual_yaw_change - target_yaw_change,
        "yaw_rate_rmse_rad_per_s": math.sqrt(yaw_rate_error_squared_sum / max(1, control_step)),
        "actual_path_integrated_curvature_per_m": actual_curvature,
        "curvature_error_per_m": actual_curvature - target_curvature if actual_curvature is not None else None,
        "planar_path_length_m": path_length,
        "signed_initial_heading_progress_m": float(np.dot(final_xy - initial_xy, np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)]))),
        "maximum_signed_initial_heading_progress_m": maximum_signed_progress,
        "final_com_displacement_m": float(np.linalg.norm(final_xy - initial_xy)),
        "maximum_com_displacement_m": maximum_displacement,
        "final_com_reference_error_m": float(np.linalg.norm(final_xy - final_reference)),
        "maximum_com_reference_error_m": maximum_reference_error,
        "out_of_training_command_envelope": bool(condition["out_of_training_command_envelope"]),
        "turn_effectiveness_decision": "descriptive_only_no_pass_fail",
        "fixed_goal_success": False,
        "fixed_goal_best_progress_m": maximum_signed_progress,
        "fixed_goal_net_progress_m": float(np.dot(final_xy - initial_xy, np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)]))),
        "cumulative_squared_action": float(summary.get("cumulative_squared_action", 0.0)),
        "actuator_abs_torque_time_integral_total_n_m_s": _vector_sum(summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator"),
        "actuator_positive_mechanical_work_total_j": _vector_sum(summary, "actuator_positive_mechanical_work_j_by_actuator"),
        "actuator_abs_mechanical_work_total_j": _vector_sum(summary, "actuator_abs_mechanical_work_j_by_actuator"),
    }
    return row, events


def energy_is_finite(row: dict[str, Any]) -> bool:
    return all(math.isfinite(float(row[key])) for key in ENERGY_KEYS)


def aggregate_condition(config: dict[str, Any], condition: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = slope.l2.aggregate_episode_rows(rows)
    gates = config["safety_gates"]
    energy_finite = all(energy_is_finite(row) for row in rows)
    sustained_count = sum(int(row["corrected_sustained_slip_physics_substep_count"]) for row in rows)
    checks = {
        "five_complete_episodes": len(rows) == 5 and aggregate["total_control_steps"] == 3000 and aggregate["total_physics_substeps"] == 15000,
        "finite_state_and_reward": aggregate["nonfinite_episode_count"] == 0,
        "finite_energy_measurements": energy_finite,
        "no_falls": aggregate["fall_count"] <= int(gates["maximum_fall_count"]),
        "no_torso_ground": aggregate["torso_ground_episode_count"] <= int(gates["maximum_torso_ground_episode_count"]),
        "no_sustained_nonfoot": aggregate["sustained_nonfoot_contact_episode_count"] <= int(gates["maximum_sustained_nonfoot_contact_episode_count"]),
        "force_qualified_denominator_evaluable": bool(aggregate["force_qualified_slip_evaluable"]) and int(aggregate["force_qualified_supported_physics_substep_count"]) > 0 and all(int(row["force_qualified_supported_physics_substep_count"]) > 0 for row in rows),
        "zero_corrected_sustained_slip": sustained_count <= int(gates["maximum_corrected_sustained_slip_substep_count"]),
        "zero_corrected_slip_events": aggregate["corrected_slip_event_count"] <= int(gates["maximum_corrected_slip_event_count"]),
        "zero_foot_within_gate": aggregate["pooled_full_interval_zero_foot_fraction"] <= float(gates["maximum_pooled_full_interval_zero_foot_fraction"]),
    }
    numeric_keys = (
        "target_cumulative_yaw_change_rad", "actual_cumulative_yaw_change_rad", "cumulative_yaw_error_rad", "yaw_rate_rmse_rad_per_s", "planar_path_length_m", "signed_initial_heading_progress_m", "final_com_displacement_m", "maximum_com_displacement_m", "final_com_reference_error_m", "maximum_com_reference_error_m"
    )
    aggregate.update({f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys})
    ratio_values = [float(row["yaw_change_target_ratio"]) for row in rows if row["yaw_change_target_ratio"] is not None]
    curvature_values = [float(row["actual_path_integrated_curvature_per_m"]) for row in rows if row["actual_path_integrated_curvature_per_m"] is not None]
    aggregate.update(
        {
            "condition_name": condition["condition_name"],
            "condition_kind": condition["kind"],
            "target_speed_m_per_s": condition["speed_m_per_s"],
            "target_yaw_rate_rad_per_s": condition["target_yaw_rate_rad_per_s"],
            "target_curvature_per_m": condition["target_curvature_per_m"],
            "out_of_training_command_envelope": condition["out_of_training_command_envelope"],
            "mean_yaw_change_target_ratio": float(np.mean(ratio_values)) if ratio_values else None,
            "same_sign_episode_count": (
                sum(row["yaw_change_same_sign_as_target"] is True for row in rows)
                if abs(float(condition["target_yaw_rate_rad_per_s"])) > 1e-12
                else None
            ),
            "mean_actual_path_integrated_curvature_per_m": float(np.mean(curvature_values)) if curvature_values else None,
            "energy_components_finite": energy_finite,
            "corrected_sustained_slip_physics_substep_count": sustained_count,
            "safety_checks": checks,
            "failed_safety_checks": [name for name, passed in checks.items() if not passed],
            "per_seed_safety_failures": {
                str(row["evaluation_seed"]): [name for name, passed in {
                    "complete_horizon": int(row["control_steps"]) == 600 and int(row["physics_substeps"]) == 3000,
                    "finite_state_and_reward": bool(row["finite"]),
                    "finite_energy_measurements": energy_is_finite(row),
                    "no_fall": not bool(row["fall"]),
                    "no_torso_ground": not bool(row["torso_ground_any"]),
                    "no_sustained_nonfoot": not bool(row["sustained_nonfoot_contact"]),
                    "force_qualified_denominator_evaluable": int(row["force_qualified_supported_physics_substep_count"]) > 0,
                    "zero_corrected_sustained_slip": int(row["corrected_sustained_slip_physics_substep_count"]) == 0,
                    "zero_corrected_slip_events": int(row["corrected_slip_event_count"]) == 0,
                }.items() if not passed]
                for row in rows
            },
            "safety_passed": all(checks.values()),
            "turn_effectiveness_decision": "descriptive_only_no_pass_fail",
            "turn_effectiveness_passed": None,
        }
    )
    return aggregate


def write_report(output_root: Path, config: dict[str, Any], results: dict[str, dict[str, Any]]) -> Path:
    lines = [
        "# Final PAIR0 flat forward-and-turn diagnostic",
        "",
        "This is a read-only evaluation of one frozen checkpoint. Safety has a predeclared PASS/FAIL gate. Turn effectiveness is descriptive only because no directly transferable constant-yaw-rate gate existed before seeing these results.",
        "",
        "The two low-speed conditions command 0.10 m/s and ±0.10 rad/s. They are positive-speed, out-of-training-envelope yaw-rate probes, not in-place rotation.",
        "",
        "| Condition | Target yaw change (rad) | Actual yaw change (rad) | Ratio | Same-sign episodes | Yaw-rate RMSE | Path curvature | Final reference error (m) | Zero-foot fraction | Safety | Failed safety checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for condition in condition_specs(config):
        row = results[condition["condition_name"]]
        ratio = "n/a" if row["mean_yaw_change_target_ratio"] is None else f"{row['mean_yaw_change_target_ratio']:.6f}"
        curvature = "n/a" if row["mean_actual_path_integrated_curvature_per_m"] is None else f"{row['mean_actual_path_integrated_curvature_per_m']:.6f}"
        same_sign = "n/a" if row["same_sign_episode_count"] is None else f"{row['same_sign_episode_count']}/5"
        lines.append(f"| {condition['condition_name']} | {row['mean_target_cumulative_yaw_change_rad']:.6f} | {row['mean_actual_cumulative_yaw_change_rad']:.6f} | {ratio} | {same_sign} | {row['mean_yaw_rate_rmse_rad_per_s']:.6f} | {curvature} | {row['mean_final_com_reference_error_m']:.6f} | {row['pooled_full_interval_zero_foot_fraction']:.6f} | {'PASS' if row['safety_passed'] else 'FAIL'} | {', '.join(row['failed_safety_checks']) or 'none'} |")
    lines.extend(["", "No row contains a turn-effectiveness PASS/FAIL decision. Energy is measurement-only and does not enter safety or tracking decisions. These observations alone cannot certify fixed-map readiness or random-map generalisation."])
    path = output_root / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def artifact_inventory(output_root: Path) -> list[dict[str, Any]]:
    return [{"relative_path": path.relative_to(output_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": slope.sha256(path)} for path in sorted(output_root.rglob("*")) if path.is_file() and path.name != "manifest.json"]


def run(config_path: Path, config: dict[str, Any], protocol: dict[str, Any], reward: dict[str, Any], checkpoint: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    stage = "attempt_root_created"
    runtime_before: dict[str, str] | None = None
    snapshot_before: dict[str, str] | None = None
    try:
        stage = "freeze_configuration"
        shutil.copy2(config_path, output_root / "frozen_config.json")
        if slope.sha256(output_root / "frozen_config.json") != slope.sha256(config_path):
            raise RuntimeError("Frozen configuration copy changed")
        stage = "snapshot_runtime_dependencies"
        runtime_before = snapshot_runtime(config, output_root)
        snapshot_before = validate_runtime_snapshot(config, output_root / "runtime_snapshot")
        if snapshot_before != runtime_before:
            raise RuntimeError("Live and snapshotted runtime closures differ")
        stage = "prepare_and_audit_flat_scene"
        scene = prepare_flat_scene(config, protocol, output_root)
        stage = "load_read_only_checkpoint"
        model = PPO.load(checkpoint, device="cpu")
        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        stage = "evaluate_45_episodes"
        for condition in condition_specs(config):
            for seed in EXPECTED_SEEDS:
                row, local_events = evaluate_episode(model, config, protocol, reward, scene, condition, seed)
                if int(row["checkpoint_timesteps"]) != 2_727_936:
                    raise RuntimeError("Checkpoint timestep metadata changed during evaluation")
                rows.append(row)
                events.extend(local_events)
        if len(rows) != 45:
            raise RuntimeError("Evaluation matrix is incomplete")
        stage = "write_raw_metrics"
        slope.write_rows(output_root / "episode_metrics.csv", rows)
        slope.l2.write_event_rows(output_root / "corrected_slip_events.csv", events)
        stage = "apply_safety_gates_and_descriptive_tracking"
        results = {condition["condition_name"]: aggregate_condition(config, condition, [row for row in rows if row["condition_name"] == condition["condition_name"]]) for condition in condition_specs(config)}
        summary = {
            "schema_version": "proxygap-pair0-flat-turn-diagnostic-result-v1",
            "checkpoint_timesteps": 2_727_936,
            "evaluation_seeds": list(EXPECTED_SEEDS),
            "condition_results": results,
            "all_safety_decisions_reported": True,
            "turn_effectiveness_decision": "descriptive_only_no_pass_fail",
            "fixed_map_readiness_certified": False,
        }
        slope.write_json(output_root / "summary.json", summary)
        report = write_report(output_root, config, results)
        stage = "post_run_provenance_verification"
        runtime_after = validate_runtime_dependencies(config)
        snapshot_after = validate_runtime_snapshot(config, output_root / "runtime_snapshot")
        if runtime_after != runtime_before or snapshot_after != snapshot_before:
            raise RuntimeError("Runtime dependencies changed during evaluation")
        if slope.sha256(checkpoint) != config["source"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
        stage = "write_success_manifest"
        manifest = {
            "schema_version": "proxygap-pair0-flat-turn-diagnostic-artifact-v1",
            "status": "read_only_flat_turn_diagnostic_formally_evaluated",
            "training_performed": False,
            "checkpoint_modified": False,
            "reward_changed": False,
            "friction_changed": False,
            "energy_formula_changed": False,
            "energy_status": "measurement_only_not_reward_or_gate",
            "fixed_map_evaluated": False,
            "video_rendered": False,
            "candidate_promoted": False,
            "turn_effectiveness_formally_gated": False,
            "runtime_dependency_sha256_before": runtime_before,
            "runtime_snapshot_sha256_before": snapshot_before,
            "runtime_dependency_sha256_after": runtime_after,
            "runtime_snapshot_sha256_after": snapshot_after,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": slope.sha256(checkpoint),
            "report": report.name,
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "mujoco": mujoco.__version__, "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "numpy": np.__version__},
            "artifact_inventory_excludes_manifest_itself": artifact_inventory(output_root),
        }
        slope.write_json(output_root / "manifest.json", manifest)
        return summary
    except BaseException as error:
        failure = {
            "schema_version": "proxygap-pair0-flat-turn-diagnostic-failure-v1",
            "status": "formal_attempt_failed_closed_non_evaluable",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "scientifically_evaluable": False,
            "all_safety_decisions_withheld": True,
            "all_turn_tracking_interpretations_withheld": True,
            "retry_permitted": False,
            "canonical_attempt_root_permanently_reserved": True,
            "training_performed": False,
            "checkpoint_write_performed": False,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_expected_sha256": config["source"]["checkpoint_sha256"],
            "checkpoint_observed_sha256": slope.sha256(checkpoint) if checkpoint.is_file() else None,
            "configuration": str(config_path.resolve()),
            "configuration_sha256": slope.sha256(config_path),
            "runtime_dependency_sha256_before": runtime_before,
            "runtime_snapshot_sha256_before": snapshot_before,
            "elapsed_seconds": time.perf_counter() - started,
        }
        slope.write_json(output_root / "FAILURE_RECORD.json", failure)
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("Only the canonical frozen configuration may be executed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward, checkpoint = validate_config(config)
    if args.validate_only:
        print("VALIDATION_OK")
        return
    expected_root = (ROOT / config["execution"]["formal_output_root"]).resolve()
    output_root = args.output_root.resolve() if args.output_root else expected_root
    if output_root != expected_root:
        raise ValueError("Formal evaluation must use the canonical output root")
    summary = run(config_path, config, protocol, reward, checkpoint, output_root)
    print(json.dumps({"status": "FORMAL_EVALUATION_COMPLETE", "turn_effectiveness_decision": summary["turn_effectiveness_decision"], "safety": {name: row["safety_passed"] for name, row in summary["condition_results"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
