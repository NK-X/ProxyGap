"""Training, checkpoint and evaluation helpers for ProxyGap."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch import nn

from .ant_wrapper import make_proxygap_ant_env
from .metrics import CSV_SCHEMA


CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)

DEFAULT_PPO_CONFIG: dict[str, Any] = {
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "policy_kwargs": {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": "Tanh",
    },
    "device": "cpu",
}


def resolve_ppo_config(
    ppo_config: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Resolve and validate every PPO parameter used by this project."""
    missing = sorted(set(DEFAULT_PPO_CONFIG) - set(ppo_config))
    if require_complete and missing:
        raise ValueError(f"Prospective config is missing PPO keys: {missing}")
    resolved = {**DEFAULT_PPO_CONFIG, **dict(ppo_config)}
    policy_kwargs = {
        **DEFAULT_PPO_CONFIG["policy_kwargs"],
        **dict(resolved.get("policy_kwargs", {})),
    }
    if policy_kwargs.get("activation_fn") != "Tanh":
        raise ValueError("Only the audited Tanh activation is currently supported")
    resolved["policy_kwargs"] = policy_kwargs
    if int(resolved["n_steps"]) <= 0 or int(resolved["batch_size"]) <= 0:
        raise ValueError("n_steps and batch_size must be positive")
    if int(resolved["n_steps"]) % int(resolved["batch_size"]) != 0:
        raise ValueError("n_steps must be divisible by batch_size for one environment")
    if str(resolved["device"]) != "cpu":
        raise ValueError("The locked ProxyGap protocol is CPU-only")
    return resolved


def checkpoint_targets(
    total_timesteps: int,
    checkpoint_timesteps: Sequence[int] | None = None,
) -> list[tuple[float, int]]:
    """Return validated checkpoint fractions and exact timestep targets."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if checkpoint_timesteps is not None:
        requested = [int(value) for value in checkpoint_timesteps]
        if not requested or requested != sorted(set(requested)):
            raise ValueError("checkpoint_timesteps must be non-empty, unique and increasing")
        if requested[0] <= 0 or requested[-1] != total_timesteps:
            raise ValueError("checkpoint_timesteps must be positive and end at total_timesteps")
        if any(value > total_timesteps for value in requested):
            raise ValueError("checkpoint timestep exceeds total_timesteps")
        return [(target / total_timesteps, target) for target in requested]

    targets: list[tuple[float, int]] = []
    last = 0
    for fraction in CHECKPOINT_FRACTIONS:
        target = int(round(total_timesteps * fraction))
        target = max(target, last + 1)
        targets.append((fraction, target))
        last = target
    return targets


def make_run_id(prefix: str = "pilot") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: PPO,
    *,
    condition_id: str,
    ctrl_cost_weight: float,
    checkpoint_fraction: float,
    seed: int,
    episodes: int,
    forward_progress_shaping_weight: float = 0.0,
    lateral_drift_shaping_weight: float = 0.0,
    effort_shaping_weight: float = 0.0,
    effort_shaping_scale: float = 1.0,
    stability_shaping_weight: float = 0.0,
    stability_shaping_scale: float = 1.0,
    common_rescore_ctrl_cost_weight: float = 0.5,
    effort_distance_min: float = 1e-8,
    action_saturation_threshold: float = 0.95,
    target_timesteps: int = 0,
    actual_model_timesteps: int = 0,
    training_seed: int | None = None,
    max_episode_steps: int | None = None,
    step_log_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Evaluate deterministic policy and return rows plus elapsed seconds."""
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for episode in range(episodes):
        evaluation_seed = seed + episode
        step_log_path = None
        if step_log_dir is not None:
            step_log_path = step_log_dir / (
                f"{condition_id}__train_{training_seed if training_seed is not None else seed}"
                f"__target_{target_timesteps}__eval_{evaluation_seed}.csv.gz"
            )
        env = make_proxygap_ant_env(
            ctrl_cost_weight=ctrl_cost_weight,
            condition_id=condition_id,
            seed=evaluation_seed,
            max_episode_steps=max_episode_steps,
            forward_progress_shaping_weight=forward_progress_shaping_weight,
            lateral_drift_shaping_weight=lateral_drift_shaping_weight,
            effort_shaping_weight=effort_shaping_weight,
            effort_shaping_scale=effort_shaping_scale,
            stability_shaping_weight=stability_shaping_weight,
            stability_shaping_scale=stability_shaping_scale,
            common_rescore_ctrl_cost_weight=common_rescore_ctrl_cost_weight,
            effort_distance_min=effort_distance_min,
            action_saturation_threshold=action_saturation_threshold,
            step_log_path=step_log_path,
        )
        observation, _ = env.reset(seed=evaluation_seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)

        summary = env.episode_summary()
        row = {
            "episode": episode + 1,
            "checkpoint_fraction": checkpoint_fraction,
            "target_timesteps": target_timesteps,
            "actual_model_timesteps": actual_model_timesteps,
            "condition_id": condition_id,
            "ctrl_cost_weight": ctrl_cost_weight,
            "forward_progress_shaping_weight": forward_progress_shaping_weight,
            "lateral_drift_shaping_weight": lateral_drift_shaping_weight,
            "training_seed": training_seed if training_seed is not None else seed,
            "seed": evaluation_seed,
            **summary,
        }
        rows.append(row)
        env.close()
    return rows, time.perf_counter() - start


def train_condition(
    *,
    output_root: Path,
    condition_id: str,
    ctrl_cost_weight: float,
    total_timesteps: int,
    seed: int,
    eval_episodes: int,
    eval_max_episode_steps: int | None,
    ppo_n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_learning_rate: float = 3e-4,
    ppo_gamma: float = 0.99,
    ppo_gae_lambda: float = 0.95,
    ppo_clip_range: float = 0.2,
    ppo_ent_coef: float = 0.0,
    ppo_vf_coef: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_normalize_advantage: bool = True,
    ppo_policy: str = "MlpPolicy",
    ppo_policy_kwargs: Mapping[str, Any] | None = None,
    ppo_device: str = "cpu",
    checkpoint_timesteps: Sequence[int] | None = None,
    forward_progress_shaping_weight: float = 0.0,
    lateral_drift_shaping_weight: float = 0.0,
    effort_shaping_weight: float = 0.0,
    effort_shaping_scale: float = 1.0,
    stability_shaping_weight: float = 0.0,
    stability_shaping_scale: float = 1.0,
    common_rescore_ctrl_cost_weight: float = 0.5,
    effort_distance_min: float = 1e-8,
    action_saturation_threshold: float = 0.95,
    record_evaluation_steps: bool = False,
    evaluation_seed_base: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train one condition and save models at predeclared checkpoints."""
    raw_env = make_proxygap_ant_env(
        ctrl_cost_weight=ctrl_cost_weight,
        condition_id=condition_id,
        seed=seed,
        forward_progress_shaping_weight=forward_progress_shaping_weight,
        lateral_drift_shaping_weight=lateral_drift_shaping_weight,
        effort_shaping_weight=effort_shaping_weight,
        effort_shaping_scale=effort_shaping_scale,
        stability_shaping_weight=stability_shaping_weight,
        stability_shaping_scale=stability_shaping_scale,
        common_rescore_ctrl_cost_weight=common_rescore_ctrl_cost_weight,
        effort_distance_min=effort_distance_min,
        action_saturation_threshold=action_saturation_threshold,
    )
    monitor_path = output_root / "logs" / "training.monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(raw_env, filename=str(monitor_path))
    policy_kwargs = dict(ppo_policy_kwargs or {})
    activation_name = policy_kwargs.pop("activation_fn", "Tanh")
    if activation_name != "Tanh":
        raise ValueError("Only the audited Tanh activation is currently supported")
    policy_kwargs["activation_fn"] = nn.Tanh
    model = PPO(
        ppo_policy,
        env,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        learning_rate=ppo_learning_rate,
        gamma=ppo_gamma,
        gae_lambda=ppo_gae_lambda,
        clip_range=ppo_clip_range,
        ent_coef=ppo_ent_coef,
        vf_coef=ppo_vf_coef,
        max_grad_norm=ppo_max_grad_norm,
        normalize_advantage=ppo_normalize_advantage,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=ppo_device,
        verbose=0,
    )

    runtime_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    condition_model_dir = output_root / "models" / condition_id

    for checkpoint_fraction, target in checkpoint_targets(
        total_timesteps,
        checkpoint_timesteps,
    ):
        chunk = max(1, target - int(model.num_timesteps))
        start = time.perf_counter()
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        train_elapsed = time.perf_counter() - start
        actual_timesteps = int(model.num_timesteps)

        checkpoint_label = f"{target:06d}"
        model_path = condition_model_dir / f"checkpoint_{checkpoint_label}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        checkpoint_eval_rows, eval_elapsed = evaluate_model(
            model,
            condition_id=condition_id,
            ctrl_cost_weight=ctrl_cost_weight,
            forward_progress_shaping_weight=forward_progress_shaping_weight,
            lateral_drift_shaping_weight=lateral_drift_shaping_weight,
            effort_shaping_weight=effort_shaping_weight,
            effort_shaping_scale=effort_shaping_scale,
            stability_shaping_weight=stability_shaping_weight,
            stability_shaping_scale=stability_shaping_scale,
            common_rescore_ctrl_cost_weight=common_rescore_ctrl_cost_weight,
            effort_distance_min=effort_distance_min,
            action_saturation_threshold=action_saturation_threshold,
            checkpoint_fraction=checkpoint_fraction,
            target_timesteps=target,
            actual_model_timesteps=actual_timesteps,
            training_seed=seed,
            seed=evaluation_seed_base if evaluation_seed_base is not None else seed + 10_000,
            episodes=eval_episodes,
            max_episode_steps=eval_max_episode_steps,
            step_log_dir=(
                output_root / "logs" / "evaluation_steps"
                if record_evaluation_steps
                else None
            ),
        )
        eval_rows.extend(checkpoint_eval_rows)
        runtime_rows.append(
            {
                "condition_id": condition_id,
                "ctrl_cost_weight": ctrl_cost_weight,
                "forward_progress_shaping_weight": forward_progress_shaping_weight,
                "lateral_drift_shaping_weight": lateral_drift_shaping_weight,
                "effort_shaping_weight": effort_shaping_weight,
                "stability_shaping_weight": stability_shaping_weight,
                "common_rescore_ctrl_cost_weight": common_rescore_ctrl_cost_weight,
                "training_seed": seed,
                "checkpoint_fraction": checkpoint_fraction,
                "target_timesteps": target,
                "actual_model_timesteps": actual_timesteps,
                "chunk_timesteps_requested": chunk,
                "train_elapsed_sec": round(train_elapsed, 3),
                "train_steps_per_sec": round(chunk / max(train_elapsed, 1e-8), 2),
                "eval_episodes": eval_episodes,
                "eval_elapsed_sec": round(eval_elapsed, 3),
                "model_path": str(model_path),
            }
        )

    env.close()
    return runtime_rows, eval_rows


def summarise_evaluation(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create simple pilot summaries without claiming formal conclusions."""
    grouped: dict[tuple[str, float, float, float, float, int], list[dict[str, Any]]] = {}
    for row in eval_rows:
        key = (
            str(row["condition_id"]),
            float(row["ctrl_cost_weight"]),
            float(row.get("forward_progress_shaping_weight", 0.0)),
            float(row.get("lateral_drift_shaping_weight", 0.0)),
            float(row["checkpoint_fraction"]),
            int(row.get("target_timesteps", 0)),
        )
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "condition_objective_return",
        "common_rescored_return",
        "proxy_return",
        "base_proxy_return",
        "reward_shaping_sum",
        "reward_forward_shaping_sum",
        "reward_lateral_shaping_sum",
        "net_forward_progress",
        "net_forward_progress_per_step",
        "cumulative_squared_action",
        "mean_squared_action_per_step",
        "action_saturation_rate",
        "effort_per_distance_defined",
        "cumulative_squared_action_per_unit_distance",
        "control_effort_per_unit_distance",
        "unhealthy_termination",
        "fall",  # Legacy formal-v1 alias; prospective reports use unhealthy_termination.
        "low_z_termination",
        "high_z_termination",
        "non_finite_termination",
        "time_limit_truncation",
        "lateral_drift_final_abs",
        "lateral_drift_mean_abs",
        "lateral_drift_max_abs",
        "cumulative_lateral_path",
        "torso_tilt_mean",
        "torso_tilt_std",
        "torso_tilt_rms",
        "torso_tilt_p95",
        "torso_tilt_max",
        "episode_length",
    ]
    summary_rows: list[dict[str, Any]] = []
    for (
        condition_id,
        ctrl_cost_weight,
        forward_progress_shaping_weight,
        lateral_drift_shaping_weight,
        checkpoint_fraction,
        target_timesteps,
    ), rows in grouped.items():
        summary: dict[str, Any] = {
            "condition_id": condition_id,
            "ctrl_cost_weight": ctrl_cost_weight,
            "forward_progress_shaping_weight": forward_progress_shaping_weight,
            "lateral_drift_shaping_weight": lateral_drift_shaping_weight,
            "checkpoint_fraction": checkpoint_fraction,
            "target_timesteps": target_timesteps,
            "episodes": len(rows),
        }
        for metric in metric_names:
            values = [
                metric_value_as_float(row[metric])
                for row in rows
                if row.get(metric) not in (None, "")
            ]
            finite_values = [value for value in values if np.isfinite(value)]
            summary[f"{metric}_mean"] = (
                float(np.mean(finite_values)) if finite_values else float("nan")
            )
        summary_rows.append(summary)
    return sorted(
        summary_rows,
        key=lambda row: (row["target_timesteps"], row["condition_id"]),
    )


def metric_value_as_float(value: Any) -> float:
    """Convert in-memory values and CSV round-tripped booleans consistently."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def select_representative_evaluation_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    final_target_timesteps: int,
) -> dict[str, Any]:
    """Select a final-checkpoint median policy and median-progress episode.

    The selected evaluation seed can then be reused for early, middle and final
    checkpoint videos.  Numeric seed ordering resolves exact ties deterministically.
    """
    final_rows = [
        row for row in rows if int(row["target_timesteps"]) == final_target_timesteps
    ]
    if not final_rows:
        raise ValueError("No rows exist at final_target_timesteps")
    condition_ids = {str(row["condition_id"]) for row in final_rows}
    if len(condition_ids) != 1:
        raise ValueError("Representative selection requires exactly one condition")
    by_training_seed: dict[int, list[Mapping[str, Any]]] = {}
    for row in final_rows:
        by_training_seed.setdefault(int(row["training_seed"]), []).append(row)
    policy_means = {
        seed: float(np.mean([float(row["net_forward_progress"]) for row in seed_rows]))
        for seed, seed_rows in by_training_seed.items()
    }
    median_policy_progress = float(np.median(list(policy_means.values())))
    selected_training_seed = min(
        policy_means,
        key=lambda seed: (abs(policy_means[seed] - median_policy_progress), seed),
    )
    policy_rows = by_training_seed[selected_training_seed]
    median_episode_progress = float(
        np.median([float(row["net_forward_progress"]) for row in policy_rows])
    )
    selected_row = min(
        policy_rows,
        key=lambda row: (
            abs(float(row["net_forward_progress"]) - median_episode_progress),
            int(row["seed"]),
        ),
    )
    return {
        "condition_id": str(selected_row["condition_id"]),
        "training_seed": selected_training_seed,
        "evaluation_seed": int(selected_row["seed"]),
        "policy_mean_net_forward_progress": policy_means[selected_training_seed],
        "episode_net_forward_progress": float(selected_row["net_forward_progress"]),
        "selection_rule": "median-policy then median-progress episode; lower seed breaks ties",
    }


def save_run_config(output_root: Path, config: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def write_standard_outputs(
    output_root: Path,
    *,
    runtime_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)
    write_rows(output_root / "logs" / "evaluation_metrics.csv", eval_rows, CSV_SCHEMA)
    write_rows(output_root / "logs" / "evaluation_summary.csv", summary_rows)
