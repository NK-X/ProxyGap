"""Train curve-conditioned tangent-aligned gait from the selected planar model."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap.curved_gait import (  # noqa: E402
    CURVE_PROFILES,
    make_curved_gait_env,
    transfer_curved_policy_with_contact_observation,
    transfer_planar_policy_to_curved_gait,
)
from proxygap.experiment import write_rows  # noqa: E402
from proxygap.planar_transition import make_ppo_from_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "curved_gait_tangent_v1_20260818.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-root", type=Path)
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
        raise ValueError("curved gait configuration is not frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("formal launch must remain prohibited")
    commands = config["commands"]
    command_frame = str(commands.get("command_frame", "world_tangent"))
    if command_frame not in {"world_tangent", "body_tangent"}:
        raise ValueError("unsupported curve command frame")
    observation_frame = str(commands.get("observation_frame", "world"))
    if observation_frame not in {"world", "target_tangent"}:
        raise ValueError("unsupported curve observation frame")
    planar_names = (
        ["vx_command", "vy_command"]
        if command_frame == "world_tangent"
        else ["v_forward_command", "v_lateral_command"]
    )
    expected_order = planar_names + [
        "yaw_rate_command",
        "sin_heading_error",
        "cos_heading_error",
    ]
    contact_observation_enabled = bool(
        commands.get("augment_foot_contact_mask", False)
    )
    if contact_observation_enabled:
        expected_order += [
            "left_ankle_contact",
            "right_ankle_contact",
            "third_ankle_contact",
            "fourth_ankle_contact",
        ]
    if commands["observation_append_order"] != expected_order:
        raise ValueError("curve command observation order changed")
    expected_dimension = 122 if contact_observation_enabled else 118
    if int(commands["target_observation_dimension"]) != expected_dimension:
        raise ValueError(
            f"curved gait observation dimension must be {expected_dimension}"
        )
    if commands["global_path_position_in_observation"] is not False:
        raise ValueError("global path position must not enter the policy")
    if commands["global_path_position_reward_enabled"] is not False:
        raise ValueError("global path position reward must remain disabled")
    if config["base_policy"]["pitch_balance_reward_enabled"] is not False:
        raise ValueError("the selected model must predate pitch shaping")
    preserved = config["preserved_pre_pitch_reward"]
    if float(preserved["pitch_balance_shaping_weight"]) != 0.0:
        raise ValueError("pitch-balance shaping must remain disabled")
    curriculum = list(config["curriculum"])
    targets = [int(stage["target_timesteps"]) for stage in curriculum]
    if targets != sorted(targets) or targets[-1] != int(
        config["timesteps_per_policy"]
    ):
        raise ValueError("curriculum targets must increase to the total budget")
    rollout = int(config["execution"]["parallel_environments_per_seed"]) * int(
        config["ppo"]["n_steps"]
    )
    if any(target % rollout for target in targets):
        raise ValueError("each curriculum target must align with one PPO rollout")
    for profile in config["evaluation_profiles"]:
        if profile["profile"] not in CURVE_PROFILES:
            raise ValueError(f"unsupported evaluation profile: {profile}")
    base_path = ROOT / config["base_policy"]["model_path"]
    if require_local_base:
        if not base_path.is_file():
            raise FileNotFoundError(f"selected planar model is missing: {base_path}")
        if sha256(base_path) != config["base_policy"]["model_sha256"]:
            raise ValueError("selected planar model SHA-256 mismatch")


def common_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    commands = config["commands"]
    reward = config["reward"]
    preserved = config["preserved_pre_pitch_reward"]
    segment_min, segment_max = (
        int(value) for value in commands["segment_steps_interval"]
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
        "airborne_shaping_weight": float(
            preserved.get("airborne_shaping_weight", 0.0)
        ),
        "foot_contact_gap_shaping_weight": float(
            preserved.get("foot_contact_gap_shaping_weight", 0.0)
        ),
        "foot_contact_gap_grace_seconds": float(
            preserved.get("foot_contact_gap_grace_seconds", 0.5)
        ),
        "foot_contact_gap_scale_seconds": float(
            preserved.get("foot_contact_gap_scale_seconds", 0.5)
        ),
        "augment_previous_applied_action": bool(
            preserved["augment_previous_applied_action"]
        ),
        "command_frame": str(commands.get("command_frame", "world_tangent")),
        "observation_frame": str(commands.get("observation_frame", "world")),
        "augment_foot_contact_mask": bool(
            commands.get("augment_foot_contact_mask", False)
        ),
        "curvature_slew_rate": float(
            commands["curvature_slew_rate_per_m_per_s"]
        ),
        "lateral_speed_slew_rate": float(
            commands.get("lateral_speed_slew_rate_m_per_s2", 0.40)
        ),
        "segment_steps_min": segment_min,
        "segment_steps_max": segment_max,
        "warmup_steps": int(commands["warmup_steps"]),
        "s_curve_period_steps": int(commands["s_curve_period_steps"]),
        "planar_tracking_weight": float(
            reward["planar_velocity_tracking_weight"]
        ),
        "planar_tracking_scale": float(
            reward["planar_velocity_tracking_scale_m_per_s"]
        ),
        "planar_tracking_function": str(
            reward["planar_velocity_tracking_function"]
        ),
        "cross_axis_velocity_weight": float(
            reward["cross_axis_velocity_weight"]
        ),
        "cross_axis_velocity_scale": float(
            reward["cross_axis_velocity_scale_m_per_s"]
        ),
        "heading_alignment_weight": float(
            reward["heading_alignment_weight"]
        ),
        "heading_alignment_scale": math.radians(
            float(reward["heading_alignment_scale_degrees"])
        ),
        "heading_alignment_function": str(
            reward.get("heading_alignment_function", "pseudo_huber")
        ),
        "yaw_rate_tracking_weight": float(
            reward["yaw_rate_tracking_weight"]
        ),
        "yaw_rate_tracking_scale": float(
            reward["yaw_rate_tracking_scale_rad_per_s"]
        ),
        "yaw_rate_tracking_function": str(
            reward.get("yaw_rate_tracking_function", "pseudo_huber")
        ),
        "heading_tolerance": math.radians(
            float(reward["heading_tolerance_degrees"])
        ),
        "heading_termination_threshold": math.radians(
            float(reward["heading_termination_threshold_degrees"])
        ),
        "heading_termination_consecutive_steps": int(
            reward["heading_termination_consecutive_steps"]
        ),
    }


def make_training_factory(
    config: dict[str, Any],
    *,
    seed: int,
    rank: int,
) -> Callable[[], gym.Env]:
    def factory() -> gym.Env:
        stage = config["curriculum"][0]
        speed_min, speed_max = (
            float(value) for value in stage["speed_interval_m_per_s"]
        )
        env = make_curved_gait_env(
            condition_id="C1__TANGENT_GAIT_TRAIN",
            seed=seed + rank,
            max_episode_steps=int(config["evaluation_max_episode_steps"]),
            profile="random",
            speed_min=speed_min,
            speed_max=speed_max,
            max_abs_curvature=float(stage["max_abs_curvature_per_m"]),
            max_abs_lateral_speed=float(
                stage.get("max_abs_lateral_speed_m_per_s", 0.0)
            ),
            heading_termination_enabled=bool(
                config["reward"].get(
                    "training_heading_termination_enabled",
                    True,
                )
            ),
            **common_env_kwargs(config),
        )
        return env

    return factory


def make_vector_env(
    config: dict[str, Any],
    *,
    seed: int,
    n_envs: int,
    monitor_path: Path,
):
    factories = [
        make_training_factory(config, seed=seed, rank=rank * 1000)
        for rank in range(n_envs)
    ]
    if n_envs == 1:
        vector_env = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(
            factories,
            start_method=str(config["execution"]["subprocess_start_method"]),
        )
    return VecMonitor(vector_env, filename=str(monitor_path))


def evaluate(
    model: PPO,
    config: dict[str, Any],
    *,
    training_seed: int,
    checkpoint_timesteps: int,
    smoke: bool,
    evaluation_seed_base: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = list(config["evaluation_profiles"])
    if smoke:
        profiles = profiles[:3]
    max_steps = 160 if smoke else int(config["evaluation_max_episode_steps"])
    speed = float(config["evaluation_speed_m_per_s"])
    for index, profile in enumerate(profiles):
        seed_base = (
            int(config["evaluation_seed_base"])
            if evaluation_seed_base is None
            else int(evaluation_seed_base)
        )
        evaluation_seed = seed_base + index
        env = make_curved_gait_env(
            condition_id=f"C1__EVAL_{profile['name'].upper()}",
            seed=evaluation_seed,
            max_episode_steps=max_steps,
            profile=str(profile["profile"]),
            speed_min=speed,
            speed_max=speed,
            max_abs_curvature=float(profile["max_abs_curvature_per_m"]),
            max_abs_lateral_speed=abs(
                float(profile.get("fixed_lateral_speed_m_per_s", 0.0))
            ),
            fixed_lateral_speed=float(
                profile.get("fixed_lateral_speed_m_per_s", 0.0)
            ),
            heading_termination_enabled=bool(
                config["reward"].get(
                    "evaluation_heading_termination_enabled",
                    True,
                )
            ),
            **common_env_kwargs(config),
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
                "condition_id": f"C1__EVAL_{profile['name'].upper()}",
                "profile_name": profile["name"],
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "checkpoint_timesteps": checkpoint_timesteps,
                **summary,
            }
        )
        env.close()
    return rows


def final_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    curved = [row for row in rows if row["profile_name"] != "straight"]
    source = curved or rows

    def mean(key: str) -> float:
        values = [float(row[key]) for row in source if np.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "profiles": [row["profile_name"] for row in rows],
        "mean_non_straight_heading_error_rms_rad": mean(
            "curve_heading_error_rms_rad"
        ),
        "mean_non_straight_heading_within_tolerance_fraction": mean(
            "curve_heading_within_tolerance_fraction"
        ),
        "mean_non_straight_cross_axis_velocity_rms_m_per_s": mean(
            "curve_cross_axis_velocity_rms_m_per_s"
        ),
        "heading_constraint_terminations": int(
            sum(bool(row["curve_heading_constraint_terminated"]) for row in source)
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=True)
    keep_windows_awake()
    seed = int(args.seed or config["training_seeds"][0])
    if seed not in [int(value) for value in config["training_seeds"]]:
        raise ValueError("seed is not declared in the frozen configuration")
    n_envs = int(
        args.n_envs or config["execution"]["parallel_environments_per_seed"]
    )
    if args.smoke and args.n_envs is None:
        n_envs = 4
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    ppo_config = dict(config["ppo"])
    if args.device:
        ppo_config["device"] = args.device
    rollout_size = n_envs * int(ppo_config["n_steps"])
    if args.smoke:
        smoke_offset = (
            int(config["base_policy"]["checkpoint_timesteps"])
            if str(
                config.get("transfer", {}).get("initialisation", "")
            )
            == "continue_same_observation_policy"
            else 0
        )
        curriculum = [
            {
                "name": "smoke_straight",
                "target_timesteps": smoke_offset + 4096,
                "max_abs_curvature_per_m": 0.0,
                "speed_interval_m_per_s": [0.8, 1.0],
            },
            {
                "name": "smoke_gentle_curve",
                "target_timesteps": smoke_offset + 8192,
                "max_abs_curvature_per_m": 0.15,
                "speed_interval_m_per_s": [0.7, 1.0],
            },
        ]
    else:
        curriculum = list(config["curriculum"])
    if any(int(stage["target_timesteps"]) % rollout_size for stage in curriculum):
        raise ValueError(
            f"curriculum targets must be divisible by rollout size {rollout_size}"
        )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (
            ROOT
            / (
                Path("artifacts") / "smoke" / config["config_id"]
                if args.smoke
                else Path(config["execution"]["output_root"]) / "runs"
            )
            / f"seed_{seed}"
        )
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    (output_root / "logs").mkdir(parents=True)
    (output_root / "models").mkdir(parents=True)
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())
    torch.set_num_threads(int(ppo_config["torch_num_threads"]))
    monitor_path = output_root / "logs" / "training_vecmonitor.csv"
    env = make_vector_env(
        config,
        seed=seed,
        n_envs=n_envs,
        monitor_path=monitor_path,
    )
    source_path = ROOT / config["base_policy"]["model_path"]
    initialisation = str(
        config.get("transfer", {}).get(
            "initialisation",
            "expand_planar_observation",
        )
    )
    source_model = PPO.load(source_path, device="cpu")
    if initialisation == "continue_same_observation_policy":
        source_dimension = int(source_model.observation_space.shape[0])
        if source_dimension != int(config["commands"]["target_observation_dimension"]):
            raise ValueError("continuation source and target observations must match")
        model = PPO.load(
            source_path,
            env=env,
            device=str(ppo_config["device"]),
        )
        for name in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "normalize_advantage",
        ):
            configured = ppo_config[name]
            live = getattr(model, name)
            live_value = live(1.0) if callable(live) and name == "clip_range" else live
            if float(live_value) != float(configured):
                raise ValueError(
                    f"continuation PPO parameter mismatch for {name}: "
                    f"{live_value} != {configured}"
                )
        model.learning_rate = float(ppo_config["learning_rate"])
        model._setup_lr_schedule()
        model.set_random_seed(seed)
        transfer_manifest = {
            "initialisation": initialisation,
            "source_observation_dimension": source_dimension,
            "target_observation_dimension": int(model.observation_space.shape[0]),
            "action_dimension": int(model.action_space.shape[0]),
            "policy_parameter_tensors_restored": True,
            "optimizer_state_restored": True,
        }
    elif initialisation == "expand_planar_observation":
        model = make_ppo_from_config(env, ppo_config, seed=seed)
        transfer_manifest = transfer_planar_policy_to_curved_gait(source_model, model)
        transfer_manifest["initialisation"] = initialisation
    elif initialisation == "append_contact_observation":
        model = make_ppo_from_config(env, ppo_config, seed=seed)
        transfer_manifest = transfer_curved_policy_with_contact_observation(
            source_model,
            model,
        )
        model.num_timesteps = int(source_model.num_timesteps)
        transfer_manifest["initialisation"] = initialisation
        transfer_manifest["optimizer_state_restored"] = False
        transfer_manifest["source_num_timesteps_restored"] = int(
            model.num_timesteps
        )
    else:
        raise ValueError(f"unsupported policy initialisation: {initialisation}")
    initial_observations = env.reset()
    if initialisation == "continue_same_observation_policy":
        source_input = initial_observations[0]
    elif initialisation == "expand_planar_observation":
        source_input = initial_observations[0, :-3]
    else:
        source_input = initial_observations[0, :-4]
    source_action, _ = source_model.predict(source_input, deterministic=True)
    target_action, _ = model.predict(initial_observations[0], deterministic=True)
    parity_error = float(np.max(np.abs(source_action - target_action)))
    configured_tolerance = float(
        config["transfer"]["required_initial_action_parity_tolerance"]
    )
    tolerance = max(configured_tolerance, float(np.finfo(np.float32).eps))
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
    (output_root / "transfer_manifest.json").write_text(
        json.dumps(transfer_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    execution = {
        "status": "started",
        "smoke": bool(args.smoke),
        "training_seed": seed,
        "parallel_environments": n_envs,
        "rollout_size": rollout_size,
        "device": str(model.device),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and str(model.device).startswith("cuda")
            else None
        ),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_model_sha256": sha256(source_path),
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    runtime_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for stage in curriculum:
        target = int(stage["target_timesteps"])
        speed_min, speed_max = (
            float(value) for value in stage["speed_interval_m_per_s"]
        )
        env.env_method(
            "set_curriculum",
            float(stage["max_abs_curvature_per_m"]),
            speed_min,
            speed_max,
            float(stage.get("max_abs_lateral_speed_m_per_s", 0.0)),
        )
        requested = target - int(model.num_timesteps)
        start = time.perf_counter()
        model.learn(total_timesteps=requested, reset_num_timesteps=False)
        elapsed = time.perf_counter() - start
        actual = int(model.num_timesteps)
        model_path = output_root / "models" / f"checkpoint_{target:07d}.zip"
        model.save(model_path)
        rows = evaluate(
            model,
            config,
            training_seed=seed,
            checkpoint_timesteps=target,
            smoke=bool(args.smoke),
        )
        evaluation_rows.extend(rows)
        final_rows = rows
        runtime_rows.append(
            {
                "training_seed": seed,
                "stage_name": stage["name"],
                "target_timesteps": target,
                "actual_model_timesteps": actual,
                "requested_timesteps": requested,
                "max_abs_curvature_per_m": stage["max_abs_curvature_per_m"],
                "max_abs_lateral_speed_m_per_s": stage.get(
                    "max_abs_lateral_speed_m_per_s",
                    0.0,
                ),
                "speed_min_m_per_s": speed_min,
                "speed_max_m_per_s": speed_max,
                "parallel_environments": n_envs,
                "device": str(model.device),
                "train_elapsed_sec": round(elapsed, 3),
                "train_steps_per_sec": round(requested / max(elapsed, 1e-12), 2),
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
            }
        )
        write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)
        write_rows(output_root / "logs" / "evaluation_metrics.csv", evaluation_rows)
        print(
            json.dumps(
                {
                    "stage": stage["name"],
                    "timesteps": actual,
                    "steps_per_sec": runtime_rows[-1]["train_steps_per_sec"],
                    "selection_preview": final_selection_summary(rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    execution.update(
        {
            "status": "complete",
            "completed_timesteps": int(model.num_timesteps),
            "final_model": runtime_rows[-1]["model_path"],
            "final_model_sha256": runtime_rows[-1]["model_sha256"],
            "selection_summary": final_selection_summary(final_rows),
        }
    )
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    env.close()
    print(json.dumps(execution, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
