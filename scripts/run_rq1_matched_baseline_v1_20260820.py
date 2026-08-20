from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


DEFAULT_CONFIG = ROOT / "configs" / "rq1_matched_baseline_v1_20260820.json"
EXPECTED_CONDITIONS = ("D0_DEFAULT_REWARD", "S1_STAGE1_SHAPED")
EXPECTED_TRAINING_SEEDS = (62401, 62402, 62403)
EXPECTED_EVALUATION_SEEDS = tuple(range(72401, 72411))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep_windows_awake() -> None:
    if sys.platform != "win32":
        return
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        0x80000000 | 0x00000001
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_authorised_formal_resource_limited":
        raise ValueError("The RQ1 configuration is not frozen for the authorised run")
    if config.get("experiment_stage") != "formal_resource_limited_descriptive":
        raise ValueError("The experiment stage changed")
    if tuple(config["training"]["training_seeds"]) != EXPECTED_TRAINING_SEEDS:
        raise ValueError("Frozen training seeds changed")
    if tuple(config["evaluation"]["evaluation_seeds"]) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Frozen evaluation seeds changed")
    if set(EXPECTED_TRAINING_SEEDS) & set(EXPECTED_EVALUATION_SEEDS):
        raise ValueError("Training and evaluation seed spaces overlap")
    if config["training"]["timesteps_per_policy"] != 1_000_000:
        raise ValueError("Frozen training budget changed")
    if config["training"]["checkpoint_timesteps"] != [250000, 500000, 750000, 1000000]:
        raise ValueError("Frozen checkpoint schedule changed")
    if config["evaluation"]["episodes_per_policy_checkpoint"] != 10:
        raise ValueError("Frozen evaluation episode count changed")
    conditions = {item["condition_id"]: item for item in config["conditions"]}
    if tuple(item["condition_id"] for item in config["conditions"]) != EXPECTED_CONDITIONS:
        raise ValueError("Frozen condition order or identity changed")
    default = conditions["D0_DEFAULT_REWARD"]
    shaped = conditions["S1_STAGE1_SHAPED"]
    shaping_keys = (
        "orientation_shaping_weight",
        "lateral_drift_shaping_weight",
        "action_rate_shaping_weight",
        "vertical_velocity_shaping_weight",
        "roll_pitch_angular_velocity_shaping_weight",
        "foot_lateral_velocity_shaping_weight",
        "foot_vertical_velocity_shaping_weight",
    )
    if default["replace_forward_reward_with_tracking"]:
        raise ValueError("Default comparator no longer uses the default forward reward")
    if any(float(default[key]) != 0.0 for key in shaping_keys):
        raise ValueError("Default comparator contains a non-zero shaping term")
    expected_shaped = {
        "replace_forward_reward_with_tracking": True,
        "forward_velocity_tracking_weight": 0.5,
        "orientation_shaping_weight": 0.1,
        "lateral_drift_shaping_weight": 0.025,
        "action_rate_shaping_weight": 0.2,
        "vertical_velocity_shaping_weight": 0.05,
        "roll_pitch_angular_velocity_shaping_weight": 0.05,
        "foot_lateral_velocity_shaping_weight": 0.025,
        "foot_vertical_velocity_shaping_weight": 0.025,
    }
    for key, value in expected_shaped.items():
        if shaped[key] != value:
            raise ValueError(f"Frozen shaped term changed: {key}")
    matched_keys = (
        "ctrl_cost_weight",
        "augment_previous_applied_action",
        "action_slew_l2_limit",
        "forward_velocity_target",
        "forward_velocity_tracking_scale",
        "lateral_drift_shaping_scale",
        "lateral_shaping_signal",
        "lateral_velocity_target",
        "orientation_shaping_function",
        "orientation_shaping_scale",
        "vertical_velocity_shaping_scale",
        "roll_pitch_angular_velocity_shaping_scale",
        "foot_landing_height_threshold",
        "foot_lateral_velocity_shaping_scale",
        "foot_vertical_velocity_shaping_scale",
    )
    for key in matched_keys:
        if default[key] != shaped[key]:
            raise ValueError(f"Matched non-reward setting differs: {key}")
    if not default["augment_previous_applied_action"]:
        raise ValueError("The matched observation augmentation was disabled")
    if config["environment"]["observation_dimensions"] != 113:
        raise ValueError("Frozen observation dimension changed")
    if config["ppo"]["policy_kwargs"] != {"net_arch": [64, 64], "activation_fn": "Tanh"}:
        raise ValueError("Frozen policy architecture changed")


def condition_kwargs(condition: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ctrl_cost_weight": float(condition["ctrl_cost_weight"]),
        "forward_progress_shaping_weight": 0.0,
        "lateral_drift_shaping_weight": float(condition["lateral_drift_shaping_weight"]),
        "lateral_drift_shaping_scale": float(condition["lateral_drift_shaping_scale"]),
        "lateral_shaping_signal": str(condition["lateral_shaping_signal"]),
        "lateral_velocity_target": float(condition["lateral_velocity_target"]),
        "effort_shaping_weight": 0.0,
        "effort_shaping_scale": 1.0,
        "orientation_shaping_weight": float(condition["orientation_shaping_weight"]),
        "orientation_shaping_scale": float(condition["orientation_shaping_scale"]),
        "orientation_shaping_function": str(condition["orientation_shaping_function"]),
        "replace_forward_reward_with_tracking": bool(condition["replace_forward_reward_with_tracking"]),
        "forward_velocity_target": float(condition["forward_velocity_target"]),
        "forward_velocity_tracking_scale": float(condition["forward_velocity_tracking_scale"]),
        "forward_velocity_tracking_weight": float(condition["forward_velocity_tracking_weight"]),
        "action_rate_shaping_weight": float(condition["action_rate_shaping_weight"]),
        "vertical_velocity_shaping_weight": float(condition["vertical_velocity_shaping_weight"]),
        "vertical_velocity_shaping_scale": float(condition["vertical_velocity_shaping_scale"]),
        "roll_pitch_angular_velocity_shaping_weight": float(condition["roll_pitch_angular_velocity_shaping_weight"]),
        "roll_pitch_angular_velocity_shaping_scale": float(condition["roll_pitch_angular_velocity_shaping_scale"]),
        "foot_landing_height_threshold": float(condition["foot_landing_height_threshold"]),
        "foot_lateral_velocity_shaping_weight": float(condition["foot_lateral_velocity_shaping_weight"]),
        "foot_lateral_velocity_shaping_scale": float(condition["foot_lateral_velocity_shaping_scale"]),
        "foot_vertical_velocity_shaping_weight": float(condition["foot_vertical_velocity_shaping_weight"]),
        "foot_vertical_velocity_shaping_scale": float(condition["foot_vertical_velocity_shaping_scale"]),
        "pitch_balance_shaping_weight": 0.0,
        "foot_geom_names": tuple(config["foot_geom_names"]),
        "common_rescore_ctrl_cost_weight": float(config["metrics"]["common_rescore_ctrl_cost_weight"]),
        "augment_previous_applied_action": bool(condition["augment_previous_applied_action"]),
        "action_slew_l2_limit": condition["action_slew_l2_limit"],
    }


def write_full_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    condition = task["condition"]
    ppo = config["ppo"]
    output_root = Path(task["output_root"])
    runtime_rows, eval_rows = train_condition(
        output_root=output_root,
        condition_id=str(condition["condition_id"]),
        total_timesteps=int(task["timesteps"]),
        checkpoint_timesteps=task["checkpoints"],
        seed=int(task["training_seed"]),
        evaluation_seed_base=int(config["evaluation"]["evaluation_seeds"][0]),
        eval_episodes=int(task["eval_episodes"]),
        eval_max_episode_steps=int(config["environment"]["max_episode_steps"]),
        record_evaluation_steps=bool(config["execution"]["record_evaluation_steps"]),
        ppo_n_steps=int(ppo["n_steps"]),
        ppo_batch_size=int(ppo["batch_size"]),
        ppo_n_epochs=int(ppo["n_epochs"]),
        ppo_learning_rate=float(ppo["learning_rate"]),
        ppo_gamma=float(ppo["gamma"]),
        ppo_gae_lambda=float(ppo["gae_lambda"]),
        ppo_clip_range=float(ppo["clip_range"]),
        ppo_ent_coef=float(ppo["ent_coef"]),
        ppo_vf_coef=float(ppo["vf_coef"]),
        ppo_max_grad_norm=float(ppo["max_grad_norm"]),
        ppo_normalize_advantage=bool(ppo["normalize_advantage"]),
        ppo_policy=str(ppo["policy"]),
        ppo_policy_kwargs=dict(ppo["policy_kwargs"]),
        ppo_device=str(config["device"]),
        ppo_torch_num_threads=int(ppo["torch_num_threads"]),
        ppo_use_sde=bool(ppo["use_sde"]),
        ppo_sde_sample_freq=int(ppo["sde_sample_freq"]),
        **condition_kwargs(condition, config),
    )
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=eval_rows,
        summary_rows=summarise_evaluation(eval_rows),
    )
    write_full_rows(output_root / "logs" / "evaluation_metrics_full.csv", eval_rows)
    return {
        "condition_id": condition["condition_id"],
        "training_seed": task["training_seed"],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(eval_rows),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_mean(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def json_vector(value: str, expected: int) -> list[float]:
    raw = json.loads(value)
    if len(raw) != expected:
        raise ValueError(f"Expected vector length {expected}, got {len(raw)}")
    return [float(item) for item in raw]


def build_policy_level_rows(
    rows: list[dict[str, str]], *, primary_timesteps: int, target_speed: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        if int(row["target_timesteps"]) != primary_timesteps:
            continue
        key = (str(row["condition_id"]), int(row["training_seed"]))
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (condition_id, training_seed), group in sorted(groups.items()):
        velocities = [float(row["fixed_horizon_mean_forward_velocity"]) for row in group]
        support_means = []
        duty_by_episode = []
        for row in group:
            fractions = json_vector(row["support_count_step_fractions_0_to_4"], 5)
            support_means.append(sum(index * value for index, value in enumerate(fractions)))
            duty_by_episode.append(json_vector(row["foot_contact_duty_fraction_by_foot"], 4))
        duty_array = np.asarray(duty_by_episode, dtype=np.float64)
        result.append(
            {
                "condition_id": condition_id,
                "training_seed": training_seed,
                "evaluation_episodes": len(group),
                "target_speed_abs_error_m_per_s": float(np.mean([abs(value - target_speed) for value in velocities])),
                "fixed_horizon_mean_forward_velocity_m_per_s": float(np.mean(velocities)),
                "direction_error_degrees": finite_mean(group, "net_displacement_direction_error_degrees"),
                "forward_path_efficiency": finite_mean(group, "forward_path_efficiency"),
                "normalised_action_roughness": finite_mean(group, "normalised_action_roughness"),
                "action_saturation_rate": finite_mean(group, "action_saturation_rate"),
                "unhealthy_termination_rate": finite_mean(group, "unhealthy_termination"),
                "full_horizon_completion_rate": finite_mean(group, "full_horizon_completed"),
                "torso_tilt_rms_degrees": math.degrees(finite_mean(group, "torso_tilt_rms")),
                "airborne_step_fraction": finite_mean(group, "airborne_step_fraction"),
                "mean_distal_support_count": float(np.mean(support_means)),
                "foot_1_duty_fraction": float(np.mean(duty_array[:, 0])),
                "foot_2_duty_fraction": float(np.mean(duty_array[:, 1])),
                "foot_3_duty_fraction": float(np.mean(duty_array[:, 2])),
                "foot_4_duty_fraction": float(np.mean(duty_array[:, 3])),
                "common_default_reward_rescore": finite_mean(group, "common_rescored_return"),
                "condition_objective_return": finite_mean(group, "condition_objective_return"),
            }
        )
    return result


def build_paired_effect_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["condition_id"], int(row["training_seed"])): row for row in policy_rows}
    metrics = (
        "target_speed_abs_error_m_per_s",
        "fixed_horizon_mean_forward_velocity_m_per_s",
        "direction_error_degrees",
        "forward_path_efficiency",
        "normalised_action_roughness",
        "action_saturation_rate",
        "unhealthy_termination_rate",
        "full_horizon_completion_rate",
        "torso_tilt_rms_degrees",
        "airborne_step_fraction",
        "mean_distal_support_count",
        "common_default_reward_rescore",
    )
    rows = []
    for seed in EXPECTED_TRAINING_SEEDS:
        default = indexed[("D0_DEFAULT_REWARD", seed)]
        shaped = indexed[("S1_STAGE1_SHAPED", seed)]
        row: dict[str, Any] = {"training_seed": seed}
        for metric in metrics:
            row[f"default__{metric}"] = default[metric]
            row[f"shaped__{metric}"] = shaped[metric]
            row[f"shaped_minus_default__{metric}"] = shaped[metric] - default[metric]
        rows.append(row)
    return rows


def decision_summary(paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvement = {
        "target_speed_abs_error_m_per_s": "lower",
        "direction_error_degrees": "lower",
        "forward_path_efficiency": "higher",
        "normalised_action_roughness": "lower",
    }
    counts: dict[str, int] = {}
    for metric, direction in improvement.items():
        key = f"shaped_minus_default__{metric}"
        if direction == "lower":
            counts[metric] = sum(float(row[key]) < 0 for row in paired_rows)
        else:
            counts[metric] = sum(float(row[key]) > 0 for row in paired_rows)
    safety_non_worse_pairs = sum(
        float(row["shaped_minus_default__unhealthy_termination_rate"]) <= 0
        for row in paired_rows
    )
    quality_metrics_passing = sum(value >= 2 for value in counts.values())
    safety_gate = safety_non_worse_pairs >= 2
    quality_gate = quality_metrics_passing >= 3
    return {
        "independent_training_pairs": len(paired_rows),
        "improved_pair_counts_out_of_3": counts,
        "safety_non_worse_pair_count_out_of_3": safety_non_worse_pairs,
        "quality_metrics_passing_2_of_3_rule_out_of_4": quality_metrics_passing,
        "safety_gate_pass": safety_gate,
        "quality_gate_pass": quality_gate,
        "joint_descriptive_gate_pass": safety_gate and quality_gate,
        "inferential_status": "descriptive_only_n_3_minimum_two_sided_sign_test_p_0p25",
    }


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("gymnasium", "mujoco", "numpy", "stable-baselines3", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }


def build_manifest(output_root: Path, excluded_name: str = "manifest.json") -> dict[str, Any]:
    files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == excluded_name:
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"file_count": len(files), "files": files}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config_sha256": sha256(config_path)}, indent=2))
        return
    keep_windows_awake()
    smoke = bool(args.smoke)
    output_root = ROOT / (
        config["execution"]["smoke_output_root"]
        if smoke
        else config["execution"]["output_root"]
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    timesteps = 4096 if smoke else int(config["training"]["timesteps_per_policy"])
    checkpoints = [4096] if smoke else [int(value) for value in config["training"]["checkpoint_timesteps"]]
    seeds = [EXPECTED_TRAINING_SEEDS[0]] if smoke else list(EXPECTED_TRAINING_SEEDS)
    eval_episodes = 2 if smoke else int(config["evaluation"]["episodes_per_policy_checkpoint"])
    tasks = []
    for seed in seeds:
        for condition in config["conditions"]:
            tasks.append(
                {
                    "config": config,
                    "condition": condition,
                    "training_seed": seed,
                    "timesteps": timesteps,
                    "checkpoints": checkpoints,
                    "eval_episodes": eval_episodes,
                    "output_root": str(output_root / "runs" / f"seed_{seed}" / condition["condition_id"]),
                }
            )
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    configured_workers = int(config["execution"]["max_workers"])
    max_workers = int(args.max_workers) if args.max_workers is not None else configured_workers
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
        ROOT / "src" / "proxygap" / "experiment.py",
        ROOT / "src" / "proxygap" / "metrics.py",
        ROOT / "protocols" / "RQ1_MATCHED_BASELINE_PROTOCOL_V1_20260820.md",
    ]
    execution = {
        "status": "started",
        "scientific_stage": "engineering_smoke" if smoke else config["experiment_stage"],
        "smoke": smoke,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "environment": environment_snapshot(),
        "tasks": len(tasks),
        "max_workers": min(max_workers, len(tasks)),
        "task_order": [
            {"condition_id": task["condition"]["condition_id"], "training_seed": task["training_seed"]}
            for task in tasks
        ],
    }
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    failures = []
    with ProcessPoolExecutor(max_workers=execution["max_workers"]) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(json.dumps({"completed": future.result()}), flush=True)
            except Exception as error:
                failure = {
                    "condition_id": task["condition"]["condition_id"],
                    "training_seed": task["training_seed"],
                    "error": repr(error),
                }
                failures.append(failure)
                print(json.dumps({"failed": failure}), flush=True)
    runtime_rows: list[dict[str, str]] = []
    full_eval_rows: list[dict[str, str]] = []
    for task in tasks:
        task_root = Path(task["output_root"])
        runtime_rows.extend(read_csv(task_root / "logs" / "training_runtime.csv"))
        full_eval_rows.extend(read_csv(task_root / "logs" / "evaluation_metrics_full.csv"))
    write_full_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)
    write_full_rows(output_root / "logs" / "evaluation_metrics_full.csv", full_eval_rows)
    scientific_status = "engineering_smoke_complete" if smoke else "formally_evaluated_resource_limited"
    if failures:
        scientific_status = "failed_incomplete"
    if not failures and not smoke:
        policy_rows = build_policy_level_rows(
            full_eval_rows,
            primary_timesteps=int(config["training"]["primary_checkpoint_timesteps"]),
            target_speed=1.0,
        )
        paired_rows = build_paired_effect_rows(policy_rows)
        write_full_rows(output_root / "analysis" / "policy_level_metrics.csv", policy_rows)
        write_full_rows(output_root / "analysis" / "paired_training_seed_effects.csv", paired_rows)
        formal_summary = {
            "status": scientific_status,
            "decision": decision_summary(paired_rows),
            "claim_boundary": config["claim_boundary"],
            "analysis_unit": "independent training seed",
            "evaluation_episodes_are_nested": True,
        }
        (output_root / "analysis" / "formal_summary.json").write_text(
            json.dumps(formal_summary, indent=2) + "\n", encoding="utf-8"
        )
    execution.update(
        {
            "status": "failed" if failures else "complete",
            "scientific_status": scientific_status,
            "failures": failures,
            "runtime_rows": len(runtime_rows),
            "evaluation_rows": len(full_eval_rows),
        }
    )
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    manifest = build_manifest(output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} RQ1 tasks failed")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
