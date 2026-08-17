from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap.experiment import checkpoint_targets, write_rows  # noqa: E402
from proxygap.planar_transition import (  # noqa: E402
    QUARTER_TURN_ACTION_PERMUTATION,
    distill_quarter_turn_command_adapter,
    make_planar_transition_env,
    make_ppo_from_config,
    transfer_pretrained_policy,
)


DEFAULT_CONFIG = ROOT / "configs" / "planar_translation_transition_v1_20260818.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Train the first seed to 250k steps before the full matrix.",
    )
    parser.add_argument("--max-workers", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep_windows_awake() -> None:
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        0x80000000 | 0x00000001
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")


def validate_config(config: dict[str, Any], *, require_local_base: bool) -> None:
    if config.get("status") != "frozen_user_requested_development":
        raise ValueError("planar transition configuration is not frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("formal launch must remain prohibited")
    if config["checkpoint_timesteps"] != [250000, 500000, 750000, 1000000]:
        raise ValueError("checkpoint schedule changed")
    if config["commands"]["initial_velocity_m_per_s"] != [1.0, 0.0]:
        raise ValueError("initial command changed")
    if config["commands"]["lateral_velocity_m_per_s"] != [0.0, 1.0]:
        raise ValueError("lateral command changed")
    expected_permutation = QUARTER_TURN_ACTION_PERMUTATION.tolist()
    if config["symmetry"]["positive_quarter_turn_action_permutation"] != expected_permutation:
        raise ValueError("quarter-turn motor mapping changed")
    if config["base_policy"]["pitch_balance_reward_enabled"] is not False:
        raise ValueError("base policy must predate pitch-balance shaping")
    if float(config["preserved_pre_pitch_reward"]["pitch_balance_shaping_weight"]) != 0.0:
        raise ValueError("pitch-balance reward must remain disabled")
    if set(config["training_seeds"]) & set(config["reserved_formal_training_seeds"]):
        raise ValueError("development and reserved seeds overlap")
    expected_eval = list(
        range(
            int(config["evaluation_seed_base"]),
            int(config["evaluation_seed_base"])
            + int(config["eval_episodes_per_checkpoint"]),
        )
    )
    if config["evaluation_seeds"] != expected_eval:
        raise ValueError("evaluation seed sequence is inconsistent")
    base_path = ROOT / config["base_policy"]["model_path"]
    if require_local_base:
        if not base_path.exists():
            raise FileNotFoundError(
                "The ignored pre-pitch base model is required locally: "
                f"{base_path}"
            )
        if sha256(base_path) != config["base_policy"]["model_sha256"]:
            raise ValueError("pre-pitch base model SHA-256 mismatch")


def environment_kwargs(config: dict[str, Any], *, evaluation: bool) -> dict[str, Any]:
    commands = config["commands"]
    braking = config["braking"]
    reward = config["reward"]
    preserved = config["preserved_pre_pitch_reward"]
    if evaluation:
        switch_min = switch_max = int(commands["evaluation_switch_step"])
    else:
        switch_min, switch_max = (
            int(value) for value in commands["training_switch_step_interval"]
        )
    return {
        "ctrl_cost_weight": float(preserved["ctrl_cost_weight"]),
        "orientation_shaping_weight": float(
            preserved["orientation_shaping_weight"]
        ),
        "orientation_shaping_scale": float(
            preserved["orientation_shaping_scale"]
        ),
        "orientation_shaping_function": str(
            preserved["orientation_shaping_function"]
        ),
        "action_rate_shaping_weight": float(
            preserved["action_rate_shaping_weight"]
        ),
        "vertical_velocity_shaping_weight": float(
            preserved["vertical_velocity_shaping_weight"]
        ),
        "vertical_velocity_shaping_scale": float(
            preserved["vertical_velocity_shaping_scale"]
        ),
        "roll_pitch_angular_velocity_shaping_weight": float(
            preserved["roll_pitch_angular_velocity_shaping_weight"]
        ),
        "roll_pitch_angular_velocity_shaping_scale": float(
            preserved["roll_pitch_angular_velocity_shaping_scale"]
        ),
        "foot_landing_height_threshold": float(
            preserved["foot_landing_height_threshold_m"]
        ),
        "foot_lateral_velocity_shaping_weight": float(
            preserved["foot_lateral_velocity_shaping_weight_per_foot"]
        ),
        "foot_lateral_velocity_shaping_scale": float(
            preserved["foot_lateral_velocity_shaping_scale_m_per_s"]
        ),
        "foot_vertical_velocity_shaping_weight": float(
            preserved["foot_vertical_velocity_shaping_weight_per_foot"]
        ),
        "foot_vertical_velocity_shaping_scale": float(
            preserved["foot_vertical_velocity_shaping_scale_m_per_s"]
        ),
        "augment_previous_applied_action": bool(
            preserved["augment_previous_applied_action"]
        ),
        "initial_command_xy": tuple(commands["initial_velocity_m_per_s"]),
        "lateral_command_xy": tuple(commands["lateral_velocity_m_per_s"]),
        "switch_step_min": switch_min,
        "switch_step_max": switch_max,
        "brake_min_steps": int(braking["minimum_steps"]),
        "brake_max_steps": int(braking["maximum_steps"]),
        "stop_speed_threshold": float(braking["stop_speed_threshold_m_per_s"]),
        "stop_consecutive_steps": int(braking["required_consecutive_steps"]),
        "planar_tracking_weight": float(
            reward["planar_velocity_tracking_weight"]
        ),
        "planar_tracking_scale": float(
            reward["planar_velocity_tracking_scale_m_per_s"]
        ),
        "planar_tracking_function": str(
            reward.get("planar_velocity_tracking_function", "exponential")
        ),
        "cross_axis_velocity_weight": float(reward["cross_axis_velocity_weight"]),
        "cross_axis_velocity_scale": float(
            reward["cross_axis_velocity_scale_m_per_s"]
        ),
        "yaw_shaping_weight": float(reward["yaw_shaping_weight"]),
        "yaw_shaping_scale": math.radians(
            float(reward["yaw_shaping_scale_degrees"])
        ),
        "yaw_shaping_function": str(
            reward.get("yaw_shaping_function", "bounded_squared")
        ),
        "brake_speed_weight": float(reward["brake_speed_weight"]),
        "brake_speed_scale": float(reward["brake_speed_scale_m_per_s"]),
    }


def evaluate(
    model: PPO,
    config: dict[str, Any],
    *,
    training_seed: int,
    target_timesteps: int,
    actual_timesteps: int,
    episodes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        evaluation_seed = int(config["evaluation_seed_base"]) + episode
        env = make_planar_transition_env(
            condition_id="T1__STOP_TO_POSITIVE_Y",
            seed=evaluation_seed,
            max_episode_steps=int(config["eval_max_episode_steps"]),
            **environment_kwargs(config, evaluation=True),
        )
        observation, _ = env.reset(seed=evaluation_seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
        summary = env.episode_summary()
        rows.append(
            {
                "condition_id": "T1__STOP_TO_POSITIVE_Y",
                "training_seed": training_seed,
                "seed": evaluation_seed,
                "target_timesteps": target_timesteps,
                "actual_model_timesteps": actual_timesteps,
                **summary,
            }
        )
        env.close()
    return rows


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    seed = int(task["training_seed"])
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    run_root = Path(task["run_root"])
    env = make_planar_transition_env(
        condition_id="T1__STOP_TO_POSITIVE_Y",
        seed=seed,
        **environment_kwargs(config, evaluation=False),
    )
    observation, _ = env.reset(seed=seed)
    monitor_path = run_root / "logs" / "training.monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    monitored_env = Monitor(env, filename=str(monitor_path))
    target_model = make_ppo_from_config(monitored_env, config["ppo"], seed=seed)
    source_path = ROOT / config["base_policy"]["model_path"]
    source_model = PPO.load(source_path, device="cpu")
    transfer_manifest = transfer_pretrained_policy(source_model, target_model)
    source_action, _ = source_model.predict(observation[:-2], deterministic=True)
    target_action, _ = target_model.predict(observation, deterministic=True)
    parity_error = float(np.max(np.abs(source_action - target_action)))
    tolerance = float(config["transfer"]["required_initial_action_parity_tolerance"])
    if parity_error > tolerance:
        raise RuntimeError(
            f"Transferred policy action mismatch {parity_error} exceeds {tolerance}"
        )
    transfer_manifest.update(
        {
            "source_model": str(source_path),
            "source_model_sha256": sha256(source_path),
            "initial_action_max_abs_error": parity_error,
            "training_seed": seed,
        }
    )
    distillation = config.get("symmetry_distillation", {})
    if bool(distillation.get("enabled", False)):
        source_env = make_planar_transition_env(
            condition_id="symmetry_distillation_source",
            seed=seed + 900000,
            **environment_kwargs(config, evaluation=True),
        ).env
        transfer_manifest["symmetry_distillation"] = (
            distill_quarter_turn_command_adapter(
                source_model,
                target_model,
                source_env,
                rollout_steps=int(distillation["rollout_steps"]),
                epochs=int(distillation["epochs"]),
                batch_size=int(distillation["batch_size"]),
                learning_rate=float(distillation["learning_rate"]),
                seed=seed + int(distillation["seed_offset"]),
                forward_weight=float(distillation["forward_weight"]),
                brake_weight=float(distillation["brake_weight"]),
                lateral_weight=float(distillation["lateral_weight"]),
            )
        )
        source_env.close()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "transfer_manifest.json").write_text(
        json.dumps(transfer_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for fraction, target in checkpoint_targets(
        int(task["timesteps"]),
        task["checkpoints"],
    ):
        chunk = max(1, target - int(target_model.num_timesteps))
        start = time.perf_counter()
        target_model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        train_elapsed = time.perf_counter() - start
        actual = int(target_model.num_timesteps)
        model_path = run_root / "models" / f"checkpoint_{target:07d}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(model_path)
        eval_start = time.perf_counter()
        rows = evaluate(
            target_model,
            config,
            training_seed=seed,
            target_timesteps=target,
            actual_timesteps=actual,
            episodes=int(task["eval_episodes"]),
        )
        eval_elapsed = time.perf_counter() - eval_start
        evaluation_rows.extend(rows)
        runtime_rows.append(
            {
                "condition_id": "T1__STOP_TO_POSITIVE_Y",
                "training_seed": seed,
                "checkpoint_fraction": fraction,
                "target_timesteps": target,
                "actual_model_timesteps": actual,
                "chunk_timesteps_requested": chunk,
                "train_elapsed_sec": round(train_elapsed, 3),
                "train_steps_per_sec": round(chunk / max(train_elapsed, 1e-12), 2),
                "eval_episodes": len(rows),
                "eval_elapsed_sec": round(eval_elapsed, 3),
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "initial_action_max_abs_error": parity_error,
                "torch_num_threads": torch.get_num_threads(),
            }
        )
        # Persist each completed checkpoint immediately so long full-matrix
        # runs can be inspected without waiting for the final checkpoint.
        write_rows(run_root / "logs" / "training_runtime.csv", runtime_rows)
        write_rows(run_root / "logs" / "evaluation_metrics.csv", evaluation_rows)
    write_rows(run_root / "logs" / "training_runtime.csv", runtime_rows)
    write_rows(run_root / "logs" / "evaluation_metrics.csv", evaluation_rows)
    monitored_env.close()
    return {
        "training_seed": seed,
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(evaluation_rows),
        "initial_action_max_abs_error": parity_error,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=True)
    keep_windows_awake()
    smoke = bool(args.smoke)
    pilot = bool(args.pilot)
    output_root = (
        ROOT / "artifacts" / "smoke" / str(config["config_id"])
        if smoke
        else (
            ROOT / "artifacts" / "pilot" / str(config["config_id"])
            if pilot
            else ROOT / config["execution"]["output_root"]
        )
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True)
    seeds = (
        [int(config["training_seeds"][0])]
        if smoke or pilot
        else [int(value) for value in config["training_seeds"]]
    )
    timesteps = (
        4096 if smoke else 250000 if pilot else int(config["timesteps_per_policy"])
    )
    checkpoints = (
        [4096]
        if smoke
        else [250000]
        if pilot
        else [int(v) for v in config["checkpoint_timesteps"]]
    )
    eval_episodes = (
        2
        if smoke
        else min(5, int(config["eval_episodes_per_checkpoint"]))
        if pilot
        else int(config["eval_episodes_per_checkpoint"])
    )
    max_workers = (
        1
        if smoke or pilot
        else int(args.max_workers or config["execution"]["max_workers"])
    )
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())
    tasks = [
        {
            "config": config,
            "training_seed": seed,
            "run_root": str(output_root / "runs" / f"seed_{seed}"),
            "timesteps": timesteps,
            "checkpoints": checkpoints,
            "eval_episodes": eval_episodes,
        }
        for seed in seeds
    ]
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    sources = [
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
        ROOT / "src" / "proxygap" / "planar_transition.py",
        ROOT / "src" / "proxygap" / "experiment.py",
        Path(__file__).resolve(),
    ]
    execution: dict[str, Any] = {
        "status": "started",
        "smoke": smoke,
        "pilot": pilot,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "base_model_sha256": sha256(ROOT / config["base_policy"]["model_path"]),
        "tasks": len(tasks),
        "max_workers": max_workers,
        "task_order": [task["training_seed"] for task in tasks],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    record = output_root / "execution_record.json"
    record.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                failure = {
                    "training_seed": task["training_seed"],
                    "error": repr(error),
                }
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
    runtime_rows: list[dict[str, str]] = []
    evaluation_rows: list[dict[str, str]] = []
    for seed in seeds:
        run_root = output_root / "runs" / f"seed_{seed}" / "logs"
        runtime_rows.extend(read_rows(run_root / "training_runtime.csv"))
        evaluation_rows.extend(read_rows(run_root / "evaluation_metrics.csv"))
    write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)
    write_rows(output_root / "logs" / "evaluation_metrics.csv", evaluation_rows)
    execution.update(
        {
            "status": "failed" if failures else "complete",
            "failures": failures,
            "runtime_rows": len(runtime_rows),
            "evaluation_rows": len(evaluation_rows),
        }
    )
    record.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} planar transition tasks failed")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
