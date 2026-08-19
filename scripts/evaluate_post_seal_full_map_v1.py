"""Read-only post-seal full-map evaluation for the final PAIR0 policy.

The default mode validates the frozen contract only.  ``--mode smoke`` runs a
20-control-step engineering prefix in a separate smoke root.  The 12,000-step
evaluation is reachable only through the explicit ``--mode formal`` option.
No code path trains, selects, modifies, or re-serialises a policy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any

import mujoco
import numpy as np
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fixed_standard_distal_margin0_paired as pair_tools  # noqa: E402
import evaluate_local_preview_final_paired_direct_goal as direct  # noqa: E402
import run_fixed_goal_terrain_training as fixed_task  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "post_seal_full_map_eval_v1_20260819.json"
FORMAL_MODE = "formal"
SMOKE_MODE = "smoke"
VALIDATE_MODE = "validate"
FOOT_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(VALIDATE_MODE, SMOKE_MODE, FORMAL_MODE),
        default=VALIDATE_MODE,
        help="Default is contract validation only; formal must be explicit.",
    )
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def pair_contract_for_injection(config: dict[str, Any]) -> dict[str, Any]:
    contact = config["contact_contract"]
    return {
        "margin_m": float(contact["explicit_pair_margin_m"]),
        "gap_m": float(contact["explicit_pair_gap_m"]),
        "condim": int(contact["condim"]),
        "friction": [float(value) for value in contact["explicit_pair_friction"]],
        "solref": [float(value) for value in contact["solref"]],
        "solreffriction": [float(value) for value in contact["solreffriction"]],
        "solimp": [float(value) for value in contact["solimp"]],
        "adhesion": float(contact["adhesion"]),
    }


def live_runtime_dependency_map(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in config["runtime_dependencies"]:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe runtime dependency path: {relative_path}")
        resolved = ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(f"Runtime dependency missing: {relative_path}")
        result[path.as_posix()] = sha256(resolved)
    return result


def loaded_project_runtime_paths() -> set[str]:
    """Return the actually imported project-code closure, excluding tests/configs."""

    result: set[str] = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        candidate = Path(filename)
        if not candidate.is_absolute() or not candidate.is_file():
            continue
        try:
            relative = candidate.resolve().relative_to(ROOT)
        except (OSError, ValueError):
            continue
        relative_posix = relative.as_posix()
        if relative_posix.startswith("scripts/") or relative_posix.startswith(
            "src/proxygap/"
        ):
            if relative.suffix == ".py":
                result.add(relative_posix)
    return result


def copy_runtime_snapshot(
    config: dict[str, Any], output_root: Path
) -> tuple[Path, dict[str, str]]:
    snapshot_root = output_root / "runtime_snapshot"
    if snapshot_root.exists():
        raise RuntimeError("Runtime snapshot root already exists")
    for relative_path in config["runtime_dependencies"]:
        relative = Path(relative_path)
        source = ROOT / relative
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    snapshot_map = {
        path.relative_to(snapshot_root).as_posix(): sha256(path)
        for path in sorted(snapshot_root.rglob("*"))
        if path.is_file()
    }
    expected_paths = {Path(path).as_posix() for path in config["runtime_dependencies"]}
    if set(snapshot_map) != expected_paths:
        raise RuntimeError("Runtime snapshot membership differs from frozen closure")
    return snapshot_root, snapshot_map


def snapshot_dependency_map(snapshot_root: Path) -> dict[str, str]:
    return {
        path.relative_to(snapshot_root).as_posix(): sha256(path)
        for path in sorted(snapshot_root.rglob("*"))
        if path.is_file()
    }


def source_asset_hash_map(config: dict[str, Any]) -> dict[str, str]:
    fixed = config["fixed_map"]
    keys = ("approved_xml", "approved_heights", "approved_hfield", "approved_texture")
    return {str(fixed[key]): sha256(ROOT / fixed[key]) for key in keys}


def validate_and_load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    _equal(config.get("schema_version"), "proxygap-post-seal-full-map-eval-v1", "schema")
    _equal(config.get("config_id"), "post_seal_full_map_eval_v1_20260819", "config id")
    _equal(
        config.get("status"),
        "frozen_read_only_post_seal_single_full_map_evaluation",
        "status",
    )
    source = config["source"]
    fixed_decl = config["fixed_map"]
    for key, hash_key in (
        ("sealed_checkpoint", "sealed_checkpoint_sha256"),
        ("policy_configuration", "policy_configuration_sha256"),
        ("final_pair0_configuration", "final_pair0_configuration_sha256"),
    ):
        actual = sha256(ROOT / source[key])
        _equal(actual, source[hash_key], f"source SHA-256 for {key}")
    actual_fixed_sha = sha256(ROOT / fixed_decl["configuration"])
    _equal(actual_fixed_sha, fixed_decl["configuration_sha256"], "fixed-map config SHA-256")
    _equal(sha256(ROOT / fixed_decl["approved_xml"]), fixed_decl["approved_xml_sha256"], "map XML SHA-256")
    _equal(sha256(ROOT / fixed_decl["approved_heights"]), fixed_decl["approved_heights_sha256"], "height array SHA-256")
    _equal(sha256(ROOT / fixed_decl["approved_hfield"]), fixed_decl["approved_hfield_sha256"], "hfield SHA-256")
    _equal(sha256(ROOT / fixed_decl["approved_texture"]), fixed_decl["approved_texture_sha256"], "texture SHA-256")
    fixed = json.loads((ROOT / fixed_decl["configuration"]).read_text(encoding="utf-8"))
    approved = fixed["approved_map"]
    _equal(approved["xml_path"], fixed_decl["approved_xml"], "approved XML path")
    _equal(approved["xml_sha256"], fixed_decl["approved_xml_sha256"], "approved XML declaration")
    _equal(approved["heights_path"], fixed_decl["approved_heights"], "approved heights path")
    _equal(approved["heights_sha256"], fixed_decl["approved_heights_sha256"], "approved heights declaration")
    _equal(approved["hfield_sha256"], fixed_decl["approved_hfield_sha256"], "approved hfield declaration")
    _equal(approved["start_xy_m"], fixed_decl["start_xy_m"], "start")
    _equal(approved["goal_xy_m"], fixed_decl["goal_xy_m"], "goal")
    _close(approved["map_half_extent_m"], fixed_decl["map_half_extent_m"], "map extent")
    _equal(approved["fixed_friction"], [1.0, 0.5, 0.5], "frozen friction")
    _equal(int(approved["condim"]), 3, "frozen condim")

    final_config = json.loads((ROOT / source["final_pair0_configuration"]).read_text(encoding="utf-8"))
    expected_contact = copy.deepcopy(final_config["contact_contract"])
    declared_contact = copy.deepcopy(config["contact_contract"])
    _equal(declared_contact["terrain_geom"], expected_contact["terrain_geom"], "terrain geom")
    _equal(declared_contact["distal_geoms"], expected_contact["distal_geoms"], "distal geoms")
    for declared_key, final_key in (
        ("all_geom_margins_m", "all_geom_margins_m"),
        ("explicit_pair_margin_m", "explicit_pair_margin_m"),
        ("explicit_pair_gap_m", "explicit_pair_gap_m"),
        ("condim", "condim"),
        ("geom_friction", "geom_friction"),
        ("explicit_pair_friction", "explicit_pair_friction"),
        ("solref", "solref"),
        ("solreffriction", "solreffriction"),
        ("solimp", "solimp"),
        ("adhesion", "adhesion"),
        ("all_ground_pairs_included", "all_ground_pairs_included"),
    ):
        _equal(declared_contact[declared_key], expected_contact[final_key], f"contact {declared_key}")
    _equal(int(declared_contact["explicit_pair_count"]), 4, "explicit pair count")
    _equal(tuple(declared_contact["distal_geoms"]), tuple(pair_tools.FOOT_NAMES), "PAIR0 foot order")

    observation = config["observation_contract"]
    _equal(int(observation["dimension"]), 135, "observation dimension")
    _equal(int(observation["base_observation_dimension"]), 122, "base observation dimension")
    _equal(int(observation["local_terrain_preview_dimension"]), 13, "preview dimension")
    _equal(bool(observation["augment_local_terrain_observation"]), True, "preview enabled")
    _equal(bool(observation["policy_observes_global_position"]), False, "policy global position")
    _equal(bool(observation["policy_observes_goal_coordinates"]), False, "policy goal")
    _equal(bool(observation["high_level_controller_uses_global_position"]), True, "controller global position")

    evaluation = config["evaluation"]
    derivation = evaluation["formal_seed_derivation"]
    derived_hash = hashlib.sha256(
        str(derivation["utf8_material"]).encode("utf-8")
    ).hexdigest()
    _equal(derived_hash, derivation["sha256"], "formal seed derivation SHA-256")
    derived_seed = int(derived_hash[:8], 16) % (2**31 - 1)
    _equal(derivation["rule"], "int(first_8_hex,16) mod (2^31-1)", "formal seed rule")
    _equal(derived_seed, int(derivation["derived_seed"]), "derived formal seed")
    _equal(int(evaluation["formal_seed"]), derived_seed, "formal seed")
    _equal(bool(derivation["outcome_based_seed_selection_permitted"]), False, "seed selection permission")
    _equal(int(evaluation["horizon_control_steps"]), 12000, "formal horizon")
    _equal(bool(evaluation["deterministic_policy"]), True, "deterministic policy")
    _close(evaluation["physics_timestep_seconds"], 0.01, "physics dt")
    _equal(int(evaluation["physics_substeps_per_control_step"]), 5, "physics substeps")
    _equal(bool(evaluation["all_five_physics_substeps_required"]), True, "all substeps")
    _close(evaluation["spawn_fraction"], 0.0, "spawn fraction")
    _equal(bool(evaluation["terminate_on_strict_stable_dwell_success"]), True, "strict success termination")
    _equal(bool(evaluation["terminate_on_spatial_hold_only"]), False, "spatial-only termination")

    success = config["independent_success"]
    _close(success["arrival_radius_m"], 1.5, "arrival radius")
    _close(success["hold_radius_m"], 2.0, "hold radius")
    _close(success["hold_seconds"], 2.0, "hold duration")
    _equal(bool(success["require_arrival_entry_before_hold"]), True, "arrival gate")

    slip = config["duration_corrected_slip"]
    _equal(slip["sampling"], "all_five_physics_substeps", "slip sampling")
    _close(slip["tangential_speed_threshold_m_per_s"], 0.2, "slip speed")
    _close(slip["minimum_normal_force_n"], 1.0, "slip force")
    _close(slip["landing_grace_seconds"], 0.1, "landing grace")
    _close(slip["minimum_sustained_seconds"], 0.2, "sustained slip")

    execution = config["execution"]
    _equal(execution["default_mode"], "validate_only", "default mode")
    _equal(bool(execution["training_permitted"]), False, "training permission")
    _equal(bool(execution["model_save_permitted"]), False, "model save permission")
    _equal(bool(execution["checkpoint_selection_permitted"]), False, "selection permission")
    _equal(bool(execution["map_source_mutation_permitted"]), False, "map mutation permission")
    if execution["formal_output_root"] == execution["smoke_output_root"]:
        raise ValueError("Smoke and formal output roots must be different")
    live_runtime = live_runtime_dependency_map(config)
    _equal(live_runtime, config["runtime_dependencies"], "complete live runtime closure")
    _equal(
        set(live_runtime),
        loaded_project_runtime_paths(),
        "actual imported project-code closure",
    )
    return config, fixed


def load_sealed_model(config: dict[str, Any]) -> PPO:
    source = config["source"]
    model = PPO.load(ROOT / source["sealed_checkpoint"], device="cpu")
    _equal(tuple(model.observation_space.shape), (int(source["observation_dimension"]),), "loaded observation shape")
    _equal(tuple(model.action_space.shape), (int(source["action_dimension"]),), "loaded action shape")
    _equal(int(model.num_timesteps), int(source["checkpoint_timesteps"]), "loaded checkpoint timesteps")
    model.policy.set_training_mode(False)
    return model


def prepare_pair0_scene(
    config: dict[str, Any], fixed: dict[str, Any], output_root: Path
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    source_scenes, spawn_metadata = fixed_task.prepare_task_scenes(fixed, output_root, [0.0])
    if len(source_scenes) != 1 or len(spawn_metadata) != 1:
        raise RuntimeError("Expected one full-map start scene")
    source_scene = source_scenes[0]
    candidate_scene = source_scene.with_name("spawn_0_0.000_pair0.xml")
    contract = pair_contract_for_injection(config)
    injected = pair_tools.inject_explicit_pairs(
        source_scene.read_text(encoding="utf-8"), contract
    )
    # Write canonical LF bytes so the precompiled PAIR0 scene hash is
    # platform-independent instead of depending on Windows newline translation.
    candidate_scene.write_text(injected, encoding="utf-8", newline="")
    _equal(
        sha256(source_scene),
        config["fixed_map"]["prepared_spawn_scene_expected_sha256"],
        "prepared full-map spawn scene SHA-256",
    )
    _equal(
        sha256(candidate_scene),
        config["fixed_map"]["pair0_artifact_scene_expected_sha256"],
        "canonical LF PAIR0 scene SHA-256",
    )
    audit = pair_tools.audit_compiled_pair(source_scene, candidate_scene, contract)
    _equal(int(audit["explicit_pair_count"]), 4, "compiled explicit pair count")
    _equal(audit["friction"], [1.0, 0.5, 0.5], "compiled floor friction")
    _equal(int(audit["condim"]), 3, "compiled floor condim")
    audit.update(
        {
            "source_scene": str(source_scene.relative_to(output_root)),
            "source_scene_sha256": sha256(source_scene),
            "pair0_scene": str(candidate_scene.relative_to(output_root)),
            "pair0_scene_sha256": sha256(candidate_scene),
            "approved_source_assets_unchanged": True,
        }
    )
    return candidate_scene, audit, spawn_metadata


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _sum_vector(summary: dict[str, Any], key: str) -> float:
    values = np.asarray(summary.get(key, []), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite or malformed energy component: {key}")
    return float(np.sum(values))


def _make_condition(config: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    condition = copy.deepcopy(fixed)
    controller = config["controller"]
    success = config["independent_success"]
    observation = config["observation_contract"]
    condition["task_adapter"].update(
        {
            "yaw_gain_per_second": float(controller["yaw_gain_per_second"]),
            "yaw_deadband_degrees": float(controller["yaw_deadband_degrees"]),
            "slow_radius_m": float(controller["slow_radius_m"]),
            "maximum_abs_curvature_per_m": float(controller["maximum_abs_curvature_per_m"]),
            "curvature_speed_reduction_gain": float(controller["curvature_speed_reduction_gain"]),
            "minimum_turn_speed_fraction": float(controller["minimum_turn_speed_fraction"]),
            "arrival_radius_m": float(success["arrival_radius_m"]),
            "hold_radius_m": float(success["hold_radius_m"]),
            "hold_seconds": float(success["hold_seconds"]),
            "augment_local_terrain_observation": True,
            "terrain_preview_longitudinal_m": observation["terrain_preview_longitudinal_m"],
            "terrain_preview_lateral_m": observation["terrain_preview_lateral_m"],
        }
    )
    return condition


def evaluate_episode(
    *,
    config: dict[str, Any],
    fixed: dict[str, Any],
    model: PPO,
    scene: Path,
    seed: int,
    horizon: int,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy_config = json.loads(
        (ROOT / config["source"]["policy_configuration"]).read_text(encoding="utf-8")
    )
    condition = _make_condition(config, fixed)
    controller = config["controller"]
    env = fixed_task.make_task_env(
        condition,
        policy_config,
        xml_path=scene,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=horizon,
        cruise_speed=float(controller["cruise_speed_m_per_s"]),
        terminate_on_success=False,
    )
    try:
        observation, _ = env.reset(seed=seed)
        _equal(tuple(observation.shape), (135,), "environment observation shape")
        _equal(tuple(observation.shape), tuple(model.observation_space.shape), "model/environment observation")
        if int(env.unwrapped.frame_skip) != 5:
            raise RuntimeError("Full-map evaluation requires frame_skip=5")
        if not math.isclose(float(env.unwrapped.model.opt.timestep), 0.01, abs_tol=1e-12):
            raise RuntimeError("Full-map evaluation requires physics dt=0.01 s")
        audit_state = l2.install_five_substep_audit(env)
        slip_config = config["duration_corrected_slip"]
        slip_tracker = l2.DurationCorrectedSlipTracker(
            dt=float(config["evaluation"]["physics_timestep_seconds"]),
            speed_threshold=float(slip_config["tangential_speed_threshold_m_per_s"]),
            minimum_normal_force=float(slip_config["minimum_normal_force_n"]),
            landing_grace_seconds=float(slip_config["landing_grace_seconds"]),
            minimum_sustained_seconds=float(slip_config["minimum_sustained_seconds"]),
        )
        control_dt = float(env.unwrapped.dt)
        required_hold_steps = int(
            math.ceil(float(config["independent_success"]["hold_seconds"]) / control_dt)
        )
        arrival = direct.ArrivalDwellTracker(
            arrival_radius_m=float(config["independent_success"]["arrival_radius_m"]),
            hold_radius_m=float(config["independent_success"]["hold_radius_m"]),
            required_hold_steps=required_hold_steps,
        )
        control_trace: list[dict[str, Any]] = []
        substep_trace: list[dict[str, Any]] = []
        all_contacts: list[np.ndarray] = []
        force_supported: list[bool] = []
        full_interval_zero: list[bool] = []
        nonfoot: list[bool] = []
        torso: list[bool] = []
        previous_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        initial_distance = float(np.linalg.norm(env.goal_xy - previous_xy))
        minimum_distance = initial_distance
        path_length = 0.0
        total_reward = 0.0
        cumulative_squared_action = 0.0
        finite = True
        terminated = truncated = False
        termination_reason = "horizon"
        start_wall = time.perf_counter()

        for control_step in range(1, horizon + 1):
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float64)
            observation, reward, terminated, truncated, info = env.step(action_array)
            total_reward += float(reward)
            cumulative_squared_action += float(np.sum(np.square(action_array)))
            substeps = audit_state.get("last")
            if not isinstance(substeps, list) or len(substeps) != 5:
                raise RuntimeError("Five-substep audit did not return exactly five rows")
            interval_contacts: list[np.ndarray] = []
            interval_qualified_slip = False
            for physics_substep, substep in enumerate(substeps, start=1):
                contacts = np.asarray(substep["contacts"], dtype=bool)
                speeds = np.asarray(substep["speeds"], dtype=np.float64)
                forces = np.asarray(substep["forces"], dtype=np.float64)
                if contacts.shape != (FOOT_COUNT,) or speeds.shape != (FOOT_COUNT,) or forces.shape != (FOOT_COUNT,):
                    raise RuntimeError("Physics-substep foot vector shape changed")
                if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(forces)):
                    raise RuntimeError("Non-finite physics-substep contact diagnostic")
                raw, qualified = slip_tracker.update(
                    contact_mask=contacts,
                    tangential_speeds=speeds,
                    normal_forces=forces,
                )
                supported_by_force = bool(
                    np.any(contacts & (forces >= float(slip_config["minimum_normal_force_n"])))
                )
                all_contacts.append(contacts.copy())
                force_supported.append(supported_by_force)
                interval_contacts.append(contacts.copy())
                interval_qualified_slip = interval_qualified_slip or bool(np.any(qualified))
                nonfoot.append(bool(substep["nonfoot"]))
                torso.append(bool(substep["torso"]))
                substep_trace.append(
                    {
                        "mode": mode,
                        "evaluation_seed": seed,
                        "control_step": control_step,
                        "physics_substep": physics_substep,
                        "physics_time_seconds": ((control_step - 1) * 5 + physics_substep) * 0.01,
                        "contact_mask": json.dumps(contacts.astype(int).tolist()),
                        "support_count": int(np.sum(contacts)),
                        "tangential_speeds_m_per_s": json.dumps(speeds.tolist(), separators=(",", ":")),
                        "normal_forces_n": json.dumps(forces.tolist(), separators=(",", ":")),
                        "force_qualified_supported": int(supported_by_force),
                        "raw_slip_any": int(np.any(raw)),
                        "force_qualified_slip_candidate_any": int(np.any(qualified)),
                        "duration_corrected_sustained_slip_any": 0,
                        "nonfoot_ground": int(bool(substep["nonfoot"])),
                        "torso_ground": int(bool(substep["torso"])),
                    }
                )
            interval_matrix = np.asarray(interval_contacts, dtype=bool)
            full_zero = not bool(np.any(interval_matrix))
            full_interval_zero.append(full_zero)
            endpoint_support = int(np.sum(interval_matrix[-1]))
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64).copy()
            xy = qpos[:2]
            path_length += float(np.linalg.norm(xy - previous_xy))
            previous_xy = xy.copy()
            distance = float(np.linalg.norm(env.goal_xy - xy))
            minimum_distance = min(minimum_distance, distance)
            terrain_height = float(env._terrain_height(float(xy[0]), float(xy[1])))
            terrain_tilt = direct.terrain_relative_tilt_rad(env, qpos)
            finite_step = bool(
                np.all(np.isfinite(observation))
                and np.all(np.isfinite(action_array))
                and math.isfinite(float(reward))
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
                and math.isfinite(terrain_tilt)
            )
            finite = finite and finite_step
            stable = direct._stable_step(
                qpos=qpos,
                qvel=qvel,
                terrain_tilt=terrain_tilt,
                support_count=endpoint_support,
                corrected_slip_candidate=interval_qualified_slip,
                settings={
                    "require_finite_state": bool(config["strict_stable_dwell"]["require_finite_state"]),
                    "maximum_planar_speed_m_per_s": float(config["strict_stable_dwell"]["maximum_planar_speed_m_per_s"]),
                    "maximum_terrain_relative_torso_tilt_degrees": float(config["strict_stable_dwell"]["maximum_terrain_relative_torso_tilt_degrees"]),
                    "minimum_foot_support_count": int(config["strict_stable_dwell"]["minimum_foot_support_count_at_control_endpoint"]),
                    "require_no_duration_corrected_slip_candidate": bool(config["strict_stable_dwell"]["require_no_force_qualified_slip_candidate_in_control_interval"]),
                },
            )
            arrival.update(step=control_step, distance_m=distance, stable=stable)
            control_trace.append(
                {
                    "mode": mode,
                    "evaluation_seed": seed,
                    "control_step": control_step,
                    "time_seconds": control_step * control_dt,
                    "x_m": float(qpos[0]),
                    "y_m": float(qpos[1]),
                    "torso_z_m": float(qpos[2]),
                    "terrain_z_m": terrain_height,
                    "torso_clearance_m": float(qpos[2] - terrain_height),
                    "goal_distance_m": distance,
                    "minimum_goal_distance_so_far_m": minimum_distance,
                    "net_progress_so_far_m": initial_distance - distance,
                    "path_length_so_far_m": path_length,
                    "world_vx_m_per_s": float(qvel[0]),
                    "world_vy_m_per_s": float(qvel[1]),
                    "planar_speed_m_per_s": float(np.linalg.norm(qvel[:2])),
                    "terrain_relative_torso_tilt_rad": terrain_tilt,
                    "endpoint_support_count": endpoint_support,
                    "mean_support_count_this_control_interval": float(np.mean(np.sum(interval_matrix, axis=1))),
                    "full_control_interval_zero_foot": int(full_zero),
                    "force_qualified_slip_candidate_in_interval": int(interval_qualified_slip),
                    "goal_entered": int(arrival.goal_entered),
                    "spatial_hold_run_steps": arrival.hold_run_steps,
                    "strict_stable_step": int(stable),
                    "strict_stable_hold_run_steps": arrival.strict_run_steps,
                    "spatial_hold_success": int(arrival.spatial_success),
                    "strict_stable_dwell_success": int(arrival.strict_dwell_success),
                    "action": json.dumps(action_array.tolist(), separators=(",", ":")),
                    "reward_step": float(reward),
                    "finite_step": int(finite_step),
                    "environment_terminated": int(terminated),
                    "environment_truncated": int(truncated),
                }
            )
            if not finite_step:
                termination_reason = "nonfinite_state_fail_closed"
                break
            if arrival.strict_dwell_success and bool(
                config["evaluation"]["terminate_on_strict_stable_dwell_success"]
            ):
                termination_reason = "independent_strict_stable_dwell_success"
                break
            if terminated:
                termination_reason = "environment_terminated"
                break
            if truncated:
                termination_reason = "horizon_truncated"
                break

        wall_seconds = time.perf_counter() - start_wall
        corrected = slip_tracker.finalise()
        sustained = np.asarray(corrected["sustained"], dtype=bool)
        candidate = np.asarray(corrected["candidate"], dtype=bool)
        contacts_matrix = np.asarray(all_contacts, dtype=bool)
        if sustained.shape != contacts_matrix.shape or candidate.shape != contacts_matrix.shape:
            raise RuntimeError("Corrected-slip result shape differs from contact trace")
        for index, row in enumerate(substep_trace):
            row["duration_corrected_sustained_slip_any"] = int(np.any(sustained[index]))
        summary = env.episode_summary()
        completed_steps = len(control_trace)
        if completed_steps <= 0:
            raise RuntimeError("No control step completed")
        supported_any = np.any(contacts_matrix, axis=1)
        force_supported_array = np.asarray(force_supported, dtype=bool)
        sustained_any = np.any(sustained, axis=1)
        force_denominator = int(np.sum(force_supported_array))
        nonfoot_longest = _longest_true_run(nonfoot)
        zero_substeps = ~supported_any
        zero_substep_longest = _longest_true_run(zero_substeps.tolist())
        fall = bool(summary.get("fall", False) or summary.get("inner_absolute_z_fall", False))
        torso_any = bool(np.any(torso))
        sustained_nonfoot = bool(
            nonfoot_longest * 0.01
            >= float(config["safety_and_qualification"]["nonfoot_contact_minimum_sustained_seconds"])
        )
        events = [
            {"mode": mode, "evaluation_seed": seed, **event}
            for event in corrected["events"]
        ]
        positive_work = _sum_vector(summary, "actuator_positive_mechanical_work_j_by_actuator")
        absolute_work = _sum_vector(summary, "actuator_abs_mechanical_work_j_by_actuator")
        torque_integral = _sum_vector(summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator")
        energy_components = [
            cumulative_squared_action,
            torque_integral,
            positive_work,
            absolute_work,
        ]
        energy_finite = bool(np.all(np.isfinite(energy_components)))
        safety_checks = {
            "finite_state": finite,
            "no_fall": not fall,
            "no_torso_ground_contact": not torso_any,
            "no_sustained_nonfoot_ground_contact": not sustained_nonfoot,
            "zero_full_control_intervals_with_all_four_distal_feet_airborne": int(np.sum(full_interval_zero)) == 0,
            "zero_duration_corrected_slip_events": len(events) == 0,
            "force_qualified_slip_denominator_nonzero": force_denominator > 0,
            "energy_components_finite": energy_finite,
        }
        qualified = bool(
            arrival.strict_dwell_success
            and all(safety_checks.values())
        )
        result = {
            "schema_version": "proxygap-post-seal-full-map-episode-result-v1",
            "status": "engineering_smoke_complete_not_scientific" if mode == SMOKE_MODE else "formal_episode_complete",
            "mode": mode,
            "evaluation_seed": seed,
            "deterministic_policy": True,
            "horizon_control_steps": horizon,
            "completed_control_steps": completed_steps,
            "completed_physics_substeps": int(contacts_matrix.shape[0]),
            "elapsed_physical_seconds": completed_steps * control_dt,
            "wall_seconds": wall_seconds,
            "termination_reason": termination_reason,
            "environment_terminated": bool(terminated),
            "environment_truncated": bool(truncated),
            "goal_entered": arrival.goal_entered,
            "goal_entry_control_step": arrival.entry_step,
            "spatial_hold_success": arrival.spatial_success,
            "spatial_hold_success_control_step": arrival.spatial_success_step,
            "strict_stable_dwell_success": arrival.strict_dwell_success,
            "strict_stable_dwell_success_control_step": arrival.strict_dwell_success_step,
            "longest_spatial_hold_seconds": arrival.longest_hold_run_steps * control_dt,
            "longest_strict_stable_hold_seconds": arrival.longest_strict_run_steps * control_dt,
            "safety_qualified_success": qualified if mode == FORMAL_MODE else None,
            "scientifically_evaluable": bool(mode == FORMAL_MODE and finite and force_denominator > 0 and energy_finite),
            "safety_checks": safety_checks,
            "fall": fall,
            "torso_ground_any": torso_any,
            "sustained_nonfoot_ground_contact": sustained_nonfoot,
            "nonfoot_ground_longest_run_seconds": nonfoot_longest * 0.01,
            "initial_distance_m": initial_distance,
            "final_distance_m": float(control_trace[-1]["goal_distance_m"]),
            "minimum_distance_m": minimum_distance,
            "net_progress_m": initial_distance - float(control_trace[-1]["goal_distance_m"]),
            "best_progress_m": initial_distance - minimum_distance,
            "path_length_m": path_length,
            "path_efficiency_net_progress_over_path": (
                (initial_distance - float(control_trace[-1]["goal_distance_m"])) / path_length
                if path_length > 0.0
                else None
            ),
            "cumulative_reward": total_reward,
            "full_control_interval_zero_foot_count": int(np.sum(full_interval_zero)),
            "full_control_interval_zero_foot_fraction": float(np.mean(full_interval_zero)),
            "zero_foot_physics_substep_count": int(np.sum(zero_substeps)),
            "zero_foot_physics_substep_fraction": float(np.mean(zero_substeps)),
            "zero_foot_physics_substep_longest_run_seconds": zero_substep_longest * 0.01,
            "mean_distal_support_count_per_physics_substep": float(np.mean(np.sum(contacts_matrix, axis=1))),
            "force_qualified_supported_physics_substep_count": force_denominator,
            "force_qualified_slip_denominator_evaluable": force_denominator > 0,
            "force_qualified_slip_candidate_physics_substep_count": int(np.sum(np.any(candidate, axis=1))),
            "duration_corrected_sustained_slip_physics_substep_count": int(np.sum(sustained_any)),
            "duration_corrected_sustained_slip_per_force_qualified_supported_fraction": (
                float(np.sum(sustained_any)) / force_denominator
                if force_denominator > 0
                else None
            ),
            "duration_corrected_slip_event_count": len(events),
            "duration_corrected_slip_distance_proxy_m": float(
                sum(float(event["slip_distance_proxy_m"]) for event in events)
            ),
            "cumulative_squared_action": cumulative_squared_action,
            "actuator_abs_torque_time_integral_total_n_m_s": torque_integral,
            "actuator_positive_mechanical_work_total_j": positive_work,
            "actuator_abs_mechanical_work_total_j": absolute_work,
            "positive_mechanical_work_proxy_per_path_m_j_per_m": positive_work / path_length if path_length > 0.0 else None,
            "absolute_mechanical_work_proxy_per_path_m_j_per_m": absolute_work / path_length if path_length > 0.0 else None,
            "energy_status": config["energy_boundary"]["status"],
            "wrapper_fixed_goal_success_diagnostic": bool(summary.get("fixed_goal_success", False)),
            "inner_episode_summary": summary,
            "claim_boundary": config["claim_boundary"],
        }
        return result, control_trace, substep_trace, events
    finally:
        env.close()


def write_manifest(output_root: Path, status: str) -> str:
    excluded = {"manifest.json", "manifest.sha256"}
    inventory = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            inventory.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": "proxygap-post-seal-full-map-artifact-manifest-v1",
        "status": status,
        "artifact_root": str(output_root),
        "inventory_count": len(inventory),
        "inventory": inventory,
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    digest = sha256(manifest_path)
    (output_root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8"
    )
    return digest


def execute_run(config_path: Path, mode: str) -> Path:
    if mode not in (SMOKE_MODE, FORMAL_MODE):
        raise ValueError("execute_run accepts smoke or formal mode only")
    config, fixed = validate_and_load_config(config_path)
    output_key = "smoke_output_root" if mode == SMOKE_MODE else "formal_output_root"
    output_root = (ROOT / config["execution"][output_key]).resolve()
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse any existing output root: {output_root}")
    output_root.mkdir(parents=True)
    stage = "freeze_configuration"
    started = time.time()
    try:
        frozen_config = output_root / "frozen_config.json"
        frozen_config.write_bytes(config_path.read_bytes())
        stage = "freeze_runtime_snapshot"
        checkpoint_hash_before = sha256(ROOT / config["source"]["sealed_checkpoint"])
        source_assets_before = source_asset_hash_map(config)
        live_runtime_before = live_runtime_dependency_map(config)
        runtime_snapshot_root, snapshot_runtime_before = copy_runtime_snapshot(
            config, output_root
        )
        _equal(live_runtime_before, config["runtime_dependencies"], "live runtime before")
        _equal(snapshot_runtime_before, config["runtime_dependencies"], "snapshot runtime before")
        stage = "prepare_pair0_full_map_scene"
        pair0_scene, contact_audit, spawn_metadata = prepare_pair0_scene(
            config, fixed, output_root
        )
        write_json(output_root / "compiled_pair0_contact_audit.json", contact_audit)
        stage = "load_sealed_checkpoint"
        model = load_sealed_model(config)
        seed = int(
            config["engineering_smoke"]["seed"]
            if mode == SMOKE_MODE
            else config["evaluation"]["formal_seed"]
        )
        horizon = int(
            config["engineering_smoke"]["horizon_control_steps"]
            if mode == SMOKE_MODE
            else config["evaluation"]["horizon_control_steps"]
        )
        stage = "evaluate_episode"
        result, control_trace, substep_trace, events = evaluate_episode(
            config=config,
            fixed=fixed,
            model=model,
            scene=pair0_scene,
            seed=seed,
            horizon=horizon,
            mode=mode,
        )
        stage = "verify_immutable_inputs_after_evaluation"
        checkpoint_hash_after = sha256(ROOT / config["source"]["sealed_checkpoint"])
        source_assets_after = source_asset_hash_map(config)
        live_runtime_after = live_runtime_dependency_map(config)
        snapshot_runtime_after = snapshot_dependency_map(runtime_snapshot_root)
        _equal(checkpoint_hash_after, checkpoint_hash_before, "checkpoint before/after")
        _equal(source_assets_after, source_assets_before, "source assets before/after")
        _equal(live_runtime_after, live_runtime_before, "live runtime before/after")
        _equal(snapshot_runtime_after, snapshot_runtime_before, "snapshot runtime before/after")
        _equal(set(snapshot_runtime_after), set(config["runtime_dependencies"]), "snapshot exact membership")
        stage = "write_evidence"
        write_csv(output_root / "control_step_trace.csv", control_trace)
        write_csv(output_root / "physics_substep_trace.csv", substep_trace)
        if events:
            write_csv(output_root / "duration_corrected_slip_events.csv", events)
        else:
            (output_root / "duration_corrected_slip_events.csv").write_text(
                "mode,evaluation_seed,foot_index,start_step,end_step,duration_steps,duration_seconds,maximum_tangential_speed_m_per_s,mean_tangential_speed_m_per_s,minimum_normal_force_n,mean_normal_force_n,slip_distance_proxy_m\n",
                encoding="utf-8",
            )
        write_json(output_root / "episode_result.json", result)
        execution = {
            "schema_version": "proxygap-post-seal-full-map-execution-v1",
            "status": result["status"],
            "mode": mode,
            "configuration": str(config_path),
            "configuration_sha256": sha256(config_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "sealed_checkpoint": str((ROOT / config["source"]["sealed_checkpoint"]).resolve()),
            "sealed_checkpoint_sha256_before": checkpoint_hash_before,
            "sealed_checkpoint_sha256_after": checkpoint_hash_after,
            "sealed_checkpoint_sha256_expected": config["source"]["sealed_checkpoint_sha256"],
            "model_num_timesteps_before_and_after": [
                int(config["source"]["checkpoint_timesteps"]),
                int(model.num_timesteps),
            ],
            "checkpoint_modified": False,
            "training_performed": False,
            "model_serialised": False,
            "checkpoint_selected": False,
            "source_map_modified": False,
            "source_asset_sha256_before": source_assets_before,
            "source_asset_sha256_after": source_assets_after,
            "pair0_injected_into_artifact_copy_only": True,
            "live_runtime_sha256_before": live_runtime_before,
            "live_runtime_sha256_after": live_runtime_after,
            "snapshot_runtime_sha256_before": snapshot_runtime_before,
            "snapshot_runtime_sha256_after": snapshot_runtime_after,
            "runtime_snapshot_exact_membership": sorted(snapshot_runtime_after),
            "runtime_live_and_snapshot_match_frozen_contract": bool(
                live_runtime_before
                == live_runtime_after
                == snapshot_runtime_before
                == snapshot_runtime_after
                == config["runtime_dependencies"]
            ),
            "formal_output_root": config["execution"]["formal_output_root"],
            "formal_output_root_remained_absent_during_smoke": (
                not (ROOT / config["execution"]["formal_output_root"]).exists()
                if mode == SMOKE_MODE
                else None
            ),
            "spawn_metadata": spawn_metadata,
            "contact_audit_sha256": sha256(output_root / "compiled_pair0_contact_audit.json"),
            "control_trace_sha256": sha256(output_root / "control_step_trace.csv"),
            "physics_substep_trace_sha256": sha256(output_root / "physics_substep_trace.csv"),
            "slip_events_sha256": sha256(output_root / "duration_corrected_slip_events.csv"),
            "episode_result_sha256": sha256(output_root / "episode_result.json"),
            "wall_seconds_total": time.time() - started,
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "output_checkpoint_file_count": len(list(output_root.rglob("*.zip"))),
            "immutability_claim_evidence": {
                "checkpoint_before_after_equal": checkpoint_hash_before == checkpoint_hash_after,
                "four_source_assets_before_after_equal": source_assets_before == source_assets_after,
                "runtime_live_before_after_equal": live_runtime_before == live_runtime_after,
                "runtime_snapshot_before_after_equal": snapshot_runtime_before == snapshot_runtime_after,
                "runtime_snapshot_exact_membership": set(snapshot_runtime_after) == set(config["runtime_dependencies"]),
                "model_num_timesteps_unchanged": int(model.num_timesteps) == int(config["source"]["checkpoint_timesteps"]),
                "no_checkpoint_file_created_in_output": not any(output_root.rglob("*.zip")),
            },
            "claim_boundary": config["claim_boundary"],
        }
        _equal(
            execution["sealed_checkpoint_sha256_before"],
            execution["sealed_checkpoint_sha256_expected"],
            "checkpoint post-run hash",
        )
        write_json(output_root / "execution_record.json", execution)
        write_manifest(output_root, result["status"])
        return output_root
    except BaseException as exc:
        failure = {
            "schema_version": "proxygap-post-seal-full-map-failure-v1",
            "status": "failed_closed",
            "mode": mode,
            "stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "training_performed": False,
            "model_serialised": False,
            "checkpoint_modified": False,
            "source_map_modified": False,
            "scientific_interpretation_permitted": False,
            "wall_seconds_total": time.time() - started,
        }
        write_json(output_root / config["execution"]["failure_record_name"], failure)
        write_manifest(output_root, "failed_closed")
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.mode == VALIDATE_MODE:
        config, _ = validate_and_load_config(config_path)
        model = load_sealed_model(config)
        print(
            json.dumps(
                {
                    "status": "validated_no_evaluation_run",
                    "configuration": str(config_path),
                    "configuration_sha256": sha256(config_path),
                    "checkpoint_sha256": sha256(ROOT / config["source"]["sealed_checkpoint"]),
                    "observation_dimension": int(model.observation_space.shape[0]),
                    "action_dimension": int(model.action_space.shape[0]),
                    "num_timesteps": int(model.num_timesteps),
                    "formal_output_root_exists": (ROOT / config["execution"]["formal_output_root"]).exists(),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    output_root = execute_run(config_path, args.mode)
    print(str(output_root), flush=True)


if __name__ == "__main__":
    main()
