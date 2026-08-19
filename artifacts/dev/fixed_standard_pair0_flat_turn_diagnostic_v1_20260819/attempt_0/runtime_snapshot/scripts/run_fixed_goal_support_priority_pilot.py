"""Run a paired support-priority continuation from the 135D preview policy.

The formal comparison starts both continuations from the same checkpoint and
uses the same seeds, environments, PPO budget and controller.  The only
between-branch change is ``airborne_shaping_weight`` (4 versus 12).  Energy
terms are diagnostics only; neither the relative-energy V2 design nor a new
energy reward is introduced here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.experiment import write_rows  # noqa: E402
from run_fixed_goal_terrain_training import (  # noqa: E402
    keep_windows_awake,
    make_task_env,
    prepare_task_scenes,
    sha256,
    vector_env,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed.lower() != str(expected_sha256).lower():
        raise ValueError(f"SHA-256 mismatch for {path}: {observed}")
    return json.loads(path.read_text(encoding="utf-8"))


def recursive_json_differences(
    left: Any,
    right: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, Any, Any]]:
    """Return leaf-level differences between two JSON-compatible objects."""

    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[str, Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = path + (str(key),)
            if key not in left:
                differences.append((".".join(child), "<missing>", right[key]))
            elif key not in right:
                differences.append((".".join(child), left[key], "<missing>"))
            else:
                differences.extend(
                    recursive_json_differences(left[key], right[key], child)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append((".".join(path + ("length",)), len(left), len(right)))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                recursive_json_differences(
                    left_item,
                    right_item,
                    path + (str(index),),
                )
            )
        return differences
    if left != right:
        return [(".".join(path), left, right)]
    return []


def reward_config_with_airborne_weight(
    base_reward_config: dict[str, Any],
    weight: float,
) -> dict[str, Any]:
    result = copy.deepcopy(base_reward_config)
    result["preserved_pre_pitch_reward"]["airborne_shaping_weight"] = float(weight)
    return result


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the declared pair differs only in airborne weight."""

    if config.get("status") != "frozen_support_priority_paired_pilot":
        raise ValueError("Support-priority pilot configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("The fixed-map pilot must prohibit generalisation claims")

    comparison = config["comparison_control"]
    if comparison.get("permitted_intervention") != "airborne_shaping_weight_only":
        raise ValueError("Unexpected intervention declaration")
    permitted_path = "preserved_pre_pitch_reward.airborne_shaping_weight"
    if comparison.get("permitted_reward_path") != permitted_path:
        raise ValueError("The permitted reward path is not fail-closed")

    preview = _load_verified_json(
        ROOT / comparison["local_preview_configuration"],
        comparison["local_preview_configuration_sha256"],
    )
    source_run = _load_verified_json(
        ROOT / comparison["source_run_configuration"],
        comparison["source_run_configuration_sha256"],
    )
    if source_run != preview:
        raise ValueError("Source frozen configuration differs from preview control")
    for key in ("approved_map", "task_adapter", "ppo"):
        if config[key] != preview[key]:
            raise ValueError(f"Support pilot changed frozen control field: {key}")

    preview_training = preview["training"]
    training = config["training"]
    for key in ("parallel_environments", "spawn_fractions", "max_episode_steps"):
        if training[key] != preview_training[key]:
            raise ValueError(f"Support pilot changed training control field: {key}")
    first_stage = preview_training["stages"][0]
    if int(training["additional_target_timesteps"]) != int(
        first_stage["additional_target_timesteps"]
    ):
        raise ValueError("Support pilot training budget differs from preview stage 1")
    if float(training["cruise_speed_m_per_s"]) != float(
        first_stage["cruise_speed_m_per_s"]
    ):
        raise ValueError("Support pilot speed differs from preview stage 1")
    rollout_size = int(training["parallel_environments"]) * int(
        config["ppo"]["n_steps"]
    )
    if int(training["additional_target_timesteps"]) % rollout_size:
        raise ValueError("Training budget is not divisible by the rollout size")

    evaluation = config["evaluation"]
    preview_evaluation = preview["evaluation"]
    expected_evaluation = {
        "max_episode_steps": int(preview_evaluation["full_route_max_episode_steps"]),
        "cruise_speed_m_per_s": float(preview_evaluation["cruise_speed_m_per_s"]),
        "validation_seeds": list(preview_evaluation["validation_seeds"]),
        "deterministic_policy": bool(preview_evaluation["deterministic_policy"]),
    }
    for key, expected in expected_evaluation.items():
        if evaluation[key] != expected:
            raise ValueError(f"Support pilot changed evaluation control field: {key}")
    if int(evaluation["representative_trace_seed"]) not in [
        int(value) for value in evaluation["validation_seeds"]
    ]:
        raise ValueError("Representative trace seed must be predeclared validation seed")
    dt = 0.05
    transient_seconds = float(evaluation["slip_transient_minimum_seconds"])
    if transient_seconds <= 0.0 or not math.isclose(
        transient_seconds / dt,
        round(transient_seconds / dt),
        abs_tol=1e-12,
    ):
        raise ValueError("Slip transient duration must be a positive multiple of 0.05 s")

    base = config["base_policy"]
    reward_config = _load_verified_json(
        ROOT / base["reward_configuration"],
        base["reward_configuration_sha256"],
    )
    control_weight = float(
        reward_config["preserved_pre_pitch_reward"]["airborne_shaping_weight"]
    )
    if control_weight != float(comparison["control_airborne_shaping_weight"]):
        raise ValueError("Base reward airborne weight differs from declared control")
    variants = list(training["variants"])
    if [float(item["airborne_shaping_weight"]) for item in variants] != [4.0, 12.0]:
        raise ValueError("The paired pilot must contain the predeclared W4 and W12 variants")
    if len({str(item["condition_id"]) for item in variants}) != len(variants):
        raise ValueError("Variant condition identifiers must be unique")
    for variant in variants:
        candidate = reward_config_with_airborne_weight(
            reward_config,
            float(variant["airborne_shaping_weight"]),
        )
        differences = recursive_json_differences(reward_config, candidate)
        expected_differences = [] if float(variant["airborne_shaping_weight"]) == 4.0 else [permitted_path]
        if [item[0] for item in differences] != expected_differences:
            raise ValueError(
                f"Variant {variant['condition_id']} changes fields outside {permitted_path}: "
                f"{differences}"
            )

    source_path = ROOT / base["model_path"]
    if not source_path.is_file() or sha256(source_path) != base["model_sha256"]:
        raise ValueError("The 135D source checkpoint is missing or has changed")
    source_model = PPO.load(source_path, device="cpu")
    if int(source_model.num_timesteps) != int(base["source_timesteps"]):
        raise ValueError("Source checkpoint timestep count mismatch")
    if int(source_model.observation_space.shape[0]) != int(
        base["observation_dimension"]
    ):
        raise ValueError("Source observation dimension mismatch")
    if int(source_model.action_space.shape[0]) != int(base["action_dimension"]):
        raise ValueError("Source action dimension mismatch")

    energy = config["energy_boundary"]
    ctrl_weight = float(
        reward_config["preserved_pre_pitch_reward"]["ctrl_cost_weight"]
    )
    if ctrl_weight != float(energy["ctrl_cost_weight_unchanged"]):
        raise ValueError("ctrl_cost_weight changed during the support pilot")
    if energy["relative_mission_energy_v2_status"] != (
        "measurement_only_not_implemented_as_reward"
    ):
        raise ValueError("Energy V2 must remain measurement-only")
    return preview, reward_config


def _configure_continuation_model(
    source_path: Path,
    env: Any,
    ppo_config: dict[str, Any],
    *,
    training_seed: int,
    smoke: bool,
) -> PPO:
    model = PPO.load(source_path, env=env, device=str(ppo_config["device"]))
    model.learning_rate = float(ppo_config["learning_rate"])
    model._setup_lr_schedule()
    model.set_random_seed(training_seed)
    for name in (
        "n_steps",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "normalize_advantage",
    ):
        if float(getattr(model, name)) != float(ppo_config[name]):
            raise ValueError(f"Continuation PPO parameter mismatch: {name}")
    clip_value = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    if float(clip_value) != float(ppo_config["clip_range"]):
        raise ValueError("Continuation PPO clip_range mismatch")
    if smoke:
        model.batch_size = int(ppo_config["n_steps"])
        model.n_epochs = 1
    else:
        for name in ("batch_size", "n_epochs"):
            if int(getattr(model, name)) != int(ppo_config[name]):
                raise ValueError(f"Continuation PPO parameter mismatch: {name}")
    return model


def _vector_sum(summary: dict[str, Any], key: str) -> float:
    value = np.asarray(summary[key], dtype=np.float64)
    return float(np.sum(value))


def evaluate_episode(
    model: PPO,
    config: dict[str, Any],
    reward_config: dict[str, Any],
    *,
    start_scene: Path,
    condition_id: str,
    checkpoint_timesteps: int,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one episode and add sustained contact-transient diagnostics."""

    env = make_task_env(
        config,
        reward_config,
        xml_path=start_scene,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=max_episode_steps,
        cruise_speed=cruise_speed,
        terminate_on_success=True,
    )
    observation, _ = env.reset(seed=seed)
    deterministic = bool(config["evaluation"]["deterministic_policy"])
    dt = float(env.unwrapped.dt)
    threshold = float(config["task_adapter"]["slip_speed_threshold_m_per_s"])
    sustained_seconds = float(
        config["evaluation"]["slip_transient_minimum_seconds"]
    )
    sustained_steps = int(round(sustained_seconds / dt))
    initial_position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
    goal = np.asarray(config["approved_map"]["goal_xy_m"], dtype=np.float64)
    initial_distance = float(np.linalg.norm(goal - initial_position))
    transient_runs: list[int] = []
    current_run = 0
    trace_rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        contact_mask = np.asarray(
            info.get("proxygap_foot_contact_mask_step", np.zeros(4)),
            dtype=bool,
        )
        contact_speeds = np.asarray(
            info.get(
                "proxygap_foot_contact_tangential_speeds_m_per_s_step",
                np.zeros(4),
            ),
            dtype=np.float64,
        )
        active_speeds = (
            contact_speeds[contact_mask]
            if contact_mask.shape == (4,) and contact_speeds.shape == (4,)
            else np.asarray([], dtype=np.float64)
        )
        step_max_contact_speed = (
            float(np.max(active_speeds)) if active_speeds.size else 0.0
        )
        transient_flag = bool(step_max_contact_speed > threshold)
        if transient_flag:
            current_run += 1
        elif current_run:
            transient_runs.append(current_run)
            current_run = 0

        if trace_path is not None:
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
            position = qpos[:2]
            distance = float(np.linalg.norm(goal - position))
            powers = np.asarray(
                info.get("proxygap_actuator_mechanical_powers_w_step", np.zeros(8)),
                dtype=np.float64,
            )
            applied_action = np.asarray(
                info.get("proxygap_applied_action", action),
                dtype=np.float64,
            )
            trace_rows.append(
                {
                    "condition_id": condition_id,
                    "evaluation_seed": seed,
                    "step": step,
                    "time_seconds": step * dt,
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "terrain_height_m": float(
                        env._terrain_height(float(position[0]), float(position[1]))
                    ),
                    "torso_z_m": float(qpos[2]),
                    "distance_to_goal_m": distance,
                    "best_progress_upper_bound_m": initial_distance - distance,
                    "support_count": int(np.sum(contact_mask)),
                    "airborne": bool(not np.any(contact_mask)),
                    "maximum_contact_tangential_speed_m_per_s": step_max_contact_speed,
                    "contact_speed_threshold_exceeded": transient_flag,
                    "squared_applied_action_step": float(np.sum(applied_action**2)),
                    "applied_action": json.dumps(applied_action.tolist(), separators=(",", ":")),
                    "joint_torque_n_m": json.dumps(
                        np.asarray(
                            info.get(
                                "proxygap_actuator_joint_torques_n_m_step",
                                np.zeros(8),
                            ),
                            dtype=np.float64,
                        ).tolist(),
                        separators=(",", ":"),
                    ),
                    "joint_velocity_rad_per_s": json.dumps(
                        np.asarray(
                            info.get(
                                "proxygap_actuator_joint_velocities_rad_per_s_step",
                                np.zeros(8),
                            ),
                            dtype=np.float64,
                        ).tolist(),
                        separators=(",", ":"),
                    ),
                    "mechanical_power_w": json.dumps(
                        powers.tolist(), separators=(",", ":")
                    ),
                    "positive_mechanical_work_step_j": float(
                        np.maximum(powers, 0.0).sum() * dt
                    ),
                    "negative_mechanical_work_abs_step_j": float(
                        np.maximum(-powers, 0.0).sum() * dt
                    ),
                    "absolute_mechanical_work_step_j": float(
                        np.abs(powers).sum() * dt
                    ),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    if current_run:
        transient_runs.append(current_run)

    summary = env.episode_summary()
    env.close()
    sustained_runs = [run for run in transient_runs if run >= sustained_steps]
    elapsed_steps = max(1, int(summary["episode_length"]))
    summary.update(
        {
            "condition_id": condition_id,
            "checkpoint_timesteps": checkpoint_timesteps,
            "evaluation_seed": seed,
            "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"])
            - float(summary["fixed_goal_minimum_distance_m"]),
            "contact_transient_run_count": len(transient_runs),
            "contact_transient_longest_run_steps": max(transient_runs, default=0),
            "contact_transient_longest_run_seconds": max(transient_runs, default=0)
            * dt,
            "sustained_contact_transient_minimum_steps": sustained_steps,
            "sustained_contact_transient_minimum_seconds": sustained_seconds,
            "sustained_contact_transient_run_count": len(sustained_runs),
            "sustained_contact_transient_step_count": int(sum(sustained_runs)),
            "sustained_contact_transient_step_fraction": float(sum(sustained_runs))
            / elapsed_steps,
            "actuator_abs_torque_time_integral_total_n_m_s": _vector_sum(
                summary,
                "actuator_abs_torque_time_integral_n_m_s_by_actuator",
            ),
            "actuator_positive_mechanical_work_total_j": _vector_sum(
                summary,
                "actuator_positive_mechanical_work_j_by_actuator",
            ),
            "actuator_negative_mechanical_work_abs_total_j": _vector_sum(
                summary,
                "actuator_negative_mechanical_work_abs_j_by_actuator",
            ),
            "actuator_abs_mechanical_work_total_j": _vector_sum(
                summary,
                "actuator_abs_mechanical_work_j_by_actuator",
            ),
        }
    )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(trace_path, trace_rows)
    return summary


def evaluate_model(
    model: PPO,
    config: dict[str, Any],
    reward_config: dict[str, Any],
    *,
    start_scene: Path,
    condition_id: str,
    output_root: Path,
    seeds: list[int],
    max_episode_steps: int,
    cruise_speed: float,
    write_representative_trace: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    trace_record: dict[str, Any] | None = None
    trace_seed = int(config["evaluation"]["representative_trace_seed"])
    for seed in seeds:
        trace_path = (
            output_root
            / "traces"
            / f"{condition_id.lower()}_seed_{seed}_trace.csv"
            if write_representative_trace and seed == trace_seed
            else None
        )
        row = evaluate_episode(
            model,
            config,
            reward_config,
            start_scene=start_scene,
            condition_id=condition_id,
            checkpoint_timesteps=int(model.num_timesteps),
            seed=seed,
            max_episode_steps=max_episode_steps,
            cruise_speed=cruise_speed,
            trace_path=trace_path,
        )
        rows.append(row)
        if trace_path is not None:
            trace_record = {
                "condition_id": condition_id,
                "evaluation_seed": seed,
                "path": str(trace_path),
                "sha256": sha256(trace_path),
                "rows": int(row["episode_length"]),
            }
    return rows, trace_record


AGGREGATE_FIELDS = (
    "fixed_goal_best_progress_m",
    "fixed_goal_net_progress_m",
    "task_airborne_step_fraction",
    "longest_airborne_run_seconds",
    "task_slip_violation_step_fraction",
    "contact_transient_longest_run_seconds",
    "sustained_contact_transient_step_fraction",
    "cumulative_squared_action",
    "control_effort",
    "actuator_abs_torque_time_integral_total_n_m_s",
    "actuator_positive_mechanical_work_total_j",
    "actuator_negative_mechanical_work_abs_total_j",
    "actuator_abs_mechanical_work_total_j",
)


def aggregate_condition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episode_count": len(rows),
        "fall_count": int(sum(bool(row["fall"]) for row in rows)),
        "spatial_success_count": int(
            sum(bool(row["fixed_goal_success"]) for row in rows)
        ),
        "qualified_success_count": int(
            sum(
                bool(row["fixed_goal_qualified_no_fall_no_airborne_no_slip"])
                for row in rows
            )
        ),
    }
    for field in AGGREGATE_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_std_population"] = float(np.std(values))
    result["sustained_contact_transient_run_count_total"] = int(
        sum(int(row["sustained_contact_transient_run_count"]) for row in rows)
    )
    return result


def build_comparison_summary(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    aggregates = {
        condition: aggregate_condition_rows(condition_rows)
        for condition, condition_rows in by_condition.items()
    }
    control_id = "W4_MATCHED_CONTINUATION_CONTROL"
    intervention_id = "W12_SUPPORT_PRIORITY_INTERVENTION"
    source_id = "SOURCE_STAGE1_W4"
    control = aggregates[control_id]
    intervention = aggregates[intervention_id]
    gate = config["evaluation"]["retention_gate"]
    airborne_reduction = (
        control["task_airborne_step_fraction_mean"]
        - intervention["task_airborne_step_fraction_mean"]
    )
    control_progress = control["fixed_goal_best_progress_m_mean"]
    intervention_progress = intervention["fixed_goal_best_progress_m_mean"]
    progress_ratio = (
        intervention_progress / control_progress
        if control_progress > 1e-12
        else float("nan")
    )
    fall_delta = intervention["fall_count"] - control["fall_count"]
    retention_passed = bool(
        airborne_reduction
        >= float(gate["minimum_absolute_airborne_fraction_reduction"])
        and np.isfinite(progress_ratio)
        and progress_ratio >= float(gate["minimum_progress_ratio_to_matched_control"])
        and fall_delta <= int(gate["maximum_additional_falls"])
    )
    paired_deltas: list[dict[str, Any]] = []
    control_by_seed = {
        int(row["evaluation_seed"]): row for row in by_condition[control_id]
    }
    intervention_by_seed = {
        int(row["evaluation_seed"]): row for row in by_condition[intervention_id]
    }
    for seed in sorted(set(control_by_seed) & set(intervention_by_seed)):
        control_row = control_by_seed[seed]
        intervention_row = intervention_by_seed[seed]
        paired_deltas.append(
            {
                "evaluation_seed": seed,
                **{
                    f"delta_{field}_w12_minus_w4": float(intervention_row[field])
                    - float(control_row[field])
                    for field in AGGREGATE_FIELDS
                },
                "delta_fall_w12_minus_w4": int(bool(intervention_row["fall"]))
                - int(bool(control_row["fall"])),
            }
        )
    return {
        "schema_version": "proxygap-support-priority-comparison-v1",
        "condition_aggregates": aggregates,
        "paired_deltas": paired_deltas,
        "retention_gate": {
            "predeclared_rule": gate,
            "observed_absolute_airborne_fraction_reduction": airborne_reduction,
            "observed_best_progress_ratio_to_matched_control": progress_ratio,
            "observed_additional_falls": fall_delta,
            "passed": retention_passed,
            "retained_condition": intervention_id if retention_passed else source_id,
            "interpretation": (
                "W12 is retained for further screening"
                if retention_passed
                else "W12 is not retained; this pilot does not justify another support-weight stage"
            ),
        },
        "energy_boundary": (
            "All action, torque and mechanical-work values are diagnostics. "
            "ctrl_cost_weight remains 0.5 and relative-energy V2 is not a reward."
        ),
        "claim_boundary": config["evaluation"]["claim_boundary"],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, base_reward_config = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config": str(config_path)}))
        return
    keep_windows_awake()

    if args.output_root:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = ROOT / "artifacts" / "smoke" / config["config_id"]
    else:
        output_root = (
            ROOT
            / config["execution"]["output_root"]
            / f"seed_{config['training']['training_seed']}"
        )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    (output_root / "logs").mkdir(parents=True)
    (output_root / "models").mkdir(parents=True)
    (output_root / "traces").mkdir(parents=True)
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())

    training = config["training"]
    evaluation = config["evaluation"]
    ppo_config = config["ppo"]
    training_seed = int(training["training_seed"])
    spawn_fractions = [float(value) for value in training["spawn_fractions"]]
    if args.smoke:
        spawn_fractions = spawn_fractions[:1]
    scene_paths, spawn_metadata = prepare_task_scenes(
        config,
        output_root,
        spawn_fractions,
    )
    (output_root / "spawn_manifest.json").write_text(
        json.dumps(spawn_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    source_path = ROOT / config["base_policy"]["model_path"]
    torch.set_num_threads(int(ppo_config["torch_num_threads"]))
    eval_seeds = (
        [int(evaluation["validation_seeds"][0])]
        if args.smoke
        else [int(value) for value in evaluation["validation_seeds"]]
    )
    eval_max_steps = 160 if args.smoke else int(evaluation["max_episode_steps"])
    training_max_steps = 160 if args.smoke else int(training["max_episode_steps"])
    additional_steps = int(ppo_config["n_steps"]) if args.smoke else int(
        training["additional_target_timesteps"]
    )
    trace_seed_original = int(evaluation["representative_trace_seed"])
    if args.smoke:
        evaluation["representative_trace_seed"] = eval_seeds[0]

    execution: dict[str, Any] = {
        "schema_version": "proxygap-fixed-goal-support-priority-pilot-v1",
        "status": "started",
        "smoke": bool(args.smoke),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_model": str(source_path),
        "source_model_sha256": sha256(source_path),
        "source_model_timesteps": int(config["base_policy"]["source_timesteps"]),
        "approved_height_sha256": config["approved_map"]["heights_sha256"],
        "approved_xml_sha256": config["approved_map"]["xml_sha256"],
        "fixed_friction": config["approved_map"]["fixed_friction"],
        "condim": config["approved_map"]["condim"],
        "observation_dimension": config["base_policy"]["observation_dimension"],
        "action_dimension": config["base_policy"]["action_dimension"],
        "training_seed": training_seed,
        "evaluation_seeds": eval_seeds,
        "training_budget_per_variant": additional_steps,
        "smoke_overrides": (
            {"parallel_environments": 1, "batch_size": 512, "n_epochs": 1}
            if args.smoke
            else {}
        ),
        "energy_v2_used_as_reward": False,
        "ctrl_cost_weight": 0.5,
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    try:
        source_model = PPO.load(source_path, device="cpu")
        source_rows, _ = evaluate_model(
            source_model,
            config,
            base_reward_config,
            start_scene=scene_paths[0],
            condition_id="SOURCE_STAGE1_W4",
            output_root=output_root,
            seeds=eval_seeds,
            max_episode_steps=eval_max_steps,
            cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
            write_representative_trace=False,
        )
        all_rows.extend(source_rows)
        write_rows(output_root / "logs" / "evaluation_episodes.csv", all_rows)

        for variant in training["variants"]:
            condition_id = str(variant["condition_id"])
            weight = float(variant["airborne_shaping_weight"])
            reward_config = reward_config_with_airborne_weight(
                base_reward_config,
                weight,
            )
            differences = recursive_json_differences(
                base_reward_config,
                reward_config,
            )
            monitor_path = output_root / "logs" / f"{condition_id.lower()}_vecmonitor.csv"
            env = vector_env(
                config,
                reward_config,
                scene_paths=scene_paths,
                spawn_fractions=spawn_fractions,
                seed=training_seed,
                max_episode_steps=training_max_steps,
                cruise_speed=float(training["cruise_speed_m_per_s"]),
                monitor_path=monitor_path,
            )
            try:
                model = _configure_continuation_model(
                    source_path,
                    env,
                    ppo_config,
                    training_seed=training_seed,
                    smoke=bool(args.smoke),
                )
                restored_optimizer_entries = len(model.policy.optimizer.state)
                source_timesteps = int(model.num_timesteps)
                started = time.perf_counter()
                model.learn(total_timesteps=additional_steps, reset_num_timesteps=False)
                elapsed = time.perf_counter() - started
                variant_model_dir = output_root / "models" / condition_id.lower()
                variant_model_dir.mkdir(parents=True)
                checkpoint_path = (
                    variant_model_dir / f"checkpoint_{int(model.num_timesteps)}.zip"
                )
                model.save(checkpoint_path)
                runtime_rows.append(
                    {
                        "condition_id": condition_id,
                        "airborne_shaping_weight": weight,
                        "source_timesteps": source_timesteps,
                        "additional_training_timesteps": additional_steps,
                        "final_timesteps": int(model.num_timesteps),
                        "train_elapsed_seconds": elapsed,
                        "train_steps_per_second": additional_steps / max(elapsed, 1e-12),
                        "restored_optimizer_state_entries": restored_optimizer_entries,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256(checkpoint_path),
                        "reward_json_differences_from_w4": json.dumps(
                            differences, separators=(",", ":")
                        ),
                    }
                )
                model_records.append(
                    {
                        "condition_id": condition_id,
                        "airborne_shaping_weight": weight,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256(checkpoint_path),
                        "checkpoint_timesteps": int(model.num_timesteps),
                        "reward_json_differences_from_w4": differences,
                    }
                )
                write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)
                rows, trace_record = evaluate_model(
                    model,
                    config,
                    reward_config,
                    start_scene=scene_paths[0],
                    condition_id=condition_id,
                    output_root=output_root,
                    seeds=eval_seeds,
                    max_episode_steps=eval_max_steps,
                    cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
                    write_representative_trace=True,
                )
                all_rows.extend(rows)
                if trace_record is not None:
                    trace_records.append(trace_record)
                write_rows(output_root / "logs" / "evaluation_episodes.csv", all_rows)
                print(
                    json.dumps(
                        {
                            "condition_id": condition_id,
                            "timesteps": int(model.num_timesteps),
                            "steps_per_second": runtime_rows[-1]["train_steps_per_second"],
                        }
                    ),
                    flush=True,
                )
            finally:
                env.close()

        comparison_summary = build_comparison_summary(all_rows, config)
        summary_path = output_root / "comparison_summary.json"
        summary_path.write_text(
            json.dumps(comparison_summary, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "proxygap-support-priority-manifest-v1",
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_model_path": str(source_path),
            "source_model_sha256": sha256(source_path),
            "source_run_configuration": str(
                ROOT / config["comparison_control"]["source_run_configuration"]
            ),
            "source_run_configuration_sha256": config["comparison_control"]
            ["source_run_configuration_sha256"],
            "approved_height_sha256": config["approved_map"]["heights_sha256"],
            "approved_xml_sha256": config["approved_map"]["xml_sha256"],
            "fixed_friction": config["approved_map"]["fixed_friction"],
            "condim": config["approved_map"]["condim"],
            "models": model_records,
            "traces": trace_records,
            "evaluation_csv": str(output_root / "logs" / "evaluation_episodes.csv"),
            "evaluation_csv_sha256": sha256(
                output_root / "logs" / "evaluation_episodes.csv"
            ),
            "comparison_summary": str(summary_path),
            "comparison_summary_sha256": sha256(summary_path),
            "energy_boundary": config["energy_boundary"],
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        execution.update(
            {
                "status": "complete",
                "model_records": model_records,
                "trace_records": trace_records,
                "evaluation_episode_rows": len(all_rows),
                "comparison_summary": str(summary_path),
                "comparison_summary_sha256": sha256(summary_path),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "representative_trace_seed_configured": trace_seed_original,
            }
        )
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        raise
    finally:
        record_path.write_text(
            json.dumps(execution, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
