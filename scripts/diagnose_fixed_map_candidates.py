"""Screen existing local locomotion checkpoints on the approved fixed map."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_goal_terrain_training import make_task_env, prepare_task_scenes  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "fixed_map_candidate_screen_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("candidate screen produced no rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_episode(
    *,
    fixed_config: dict[str, Any],
    policy_config: dict[str, Any],
    model: PPO,
    scene: Path,
    seed: int,
    horizon_steps: int,
    cruise_speed: float,
    trace_stride_steps: int,
    terminate_on_success: bool,
) -> tuple[int, bool, bool, dict[str, Any], list[dict[str, Any]]]:
    env = make_task_env(
        fixed_config,
        policy_config,
        xml_path=scene,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=horizon_steps,
        cruise_speed=cruise_speed,
        terminate_on_success=terminate_on_success,
    )
    try:
        observation, _ = env.reset(seed=seed)
        if observation.shape != model.observation_space.shape:
            raise ValueError(
                f"observation mismatch: environment {observation.shape}, "
                f"model {model.observation_space.shape}"
            )
        terminated = False
        truncated = False
        completed_steps = 0
        trace: list[dict[str, Any]] = []
        for step in range(horizon_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            completed_steps = step + 1
            if (
                completed_steps == 1
                or completed_steps % trace_stride_steps == 0
                or terminated
                or truncated
            ):
                qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
                qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
                w, x, y, z = (float(value) for value in qpos[3:7])
                yaw = math.atan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z),
                )
                terrain_height = env._terrain_height(float(qpos[0]), float(qpos[1]))
                trace.append(
                    {
                        "step": completed_steps,
                        "time_seconds": completed_steps * float(env.unwrapped.dt),
                        "x_m": float(qpos[0]),
                        "y_m": float(qpos[1]),
                        "torso_z_m": float(qpos[2]),
                        "terrain_z_m": float(terrain_height),
                        "torso_clearance_m": float(qpos[2] - terrain_height),
                        "yaw_rad": yaw,
                        "world_vx_m_per_s": float(qvel[0]),
                        "world_vy_m_per_s": float(qvel[1]),
                        "goal_distance_m": float(
                            info["proxygap_fixed_goal_distance_m"]
                        ),
                        "action_l2": float(np.linalg.norm(action)),
                    }
                )
            if terminated or truncated:
                break
        return completed_steps, terminated, truncated, env.episode_summary(), trace
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_path = ROOT / config["fixed_map_config"]
    fixed_base = json.loads(fixed_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["output_root"]).resolve()
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "frozen_screen_config.json").write_bytes(config_path.read_bytes())
    scenes, spawn_metadata = prepare_task_scenes(fixed_base, output_root, [0.0])

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    horizon_steps = int(config["horizon_steps"])
    cruise_speed = float(config["cruise_speed_m_per_s"])
    trace_stride_steps = int(config.get("trace_stride_steps", 10))
    terminate_on_success = bool(config.get("terminate_on_success", False))
    if trace_stride_steps <= 0:
        raise ValueError("trace_stride_steps must be positive")
    for candidate in config["candidates"]:
        model_path = ROOT / candidate["model_path"]
        if sha256(model_path) != candidate["model_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {candidate['name']}")
        policy_config = json.loads(
            (ROOT / candidate["configuration"]).read_text(encoding="utf-8")
        )
        model = PPO.load(model_path, device="cpu")
        for controller in config["controller_conditions"]:
            condition_config = copy.deepcopy(fixed_base)
            condition_config["task_adapter"].update(controller)
            for seed in config["evaluation_seeds"]:
                completed, terminated, truncated, summary, trace = evaluate_episode(
                    fixed_config=condition_config,
                    policy_config=policy_config,
                    model=model,
                    scene=scenes[0],
                    seed=int(seed),
                    horizon_steps=horizon_steps,
                    cruise_speed=cruise_speed,
                    trace_stride_steps=trace_stride_steps,
                    terminate_on_success=terminate_on_success,
                )
                trace_rows.extend(
                    {
                        "candidate": candidate["name"],
                        "controller": controller["name"],
                        "evaluation_seed": int(seed),
                        **entry,
                    }
                    for entry in trace
                )
                row = {
                    "candidate": candidate["name"],
                    "controller": controller["name"],
                    "evaluation_seed": int(seed),
                    "completed_steps": completed,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "success": int(bool(summary["fixed_goal_success"])),
                    "fall": int(bool(summary["fall"])),
                    "net_progress_m": scalar(summary, "fixed_goal_net_progress_m"),
                    "best_progress_m": (
                        float(summary["fixed_goal_initial_distance_m"])
                        - float(summary["fixed_goal_minimum_distance_m"])
                    ),
                    "final_distance_m": scalar(summary, "fixed_goal_final_distance_m"),
                    "airborne_fraction": scalar(summary, "task_airborne_step_fraction"),
                    "maximum_torso_tilt_rad": scalar(
                        summary,
                        "terrain_relative_maximum_torso_tilt_rad",
                    ),
                    "heading_rmse_rad": scalar(summary, "curve_heading_error_rms_rad"),
                    "heading_within_tolerance_fraction": scalar(
                        summary,
                        "curve_heading_within_tolerance_fraction",
                    ),
                    "contact_speed_exceedance_fraction": scalar(
                        summary,
                        "task_slip_violation_step_fraction",
                    ),
                }
                rows.append(row)
                print(
                    candidate["name"],
                    controller["name"],
                    seed,
                    f"progress={float(row['net_progress_m']):.3f} m",
                    f"airborne={float(row['airborne_fraction']):.3f}",
                    f"fall={row['fall']}",
                    flush=True,
                )

    write_csv(output_root / "candidate_episode_rows.csv", rows)
    if bool(config.get("save_step_traces", False)):
        write_csv(output_root / "candidate_step_traces.csv", trace_rows)
    record = {
        "schema_version": "proxygap-fixed-map-candidate-screen-v1",
        "configuration": str(config_path),
        "configuration_sha256": sha256(config_path),
        "fixed_map_height_sha256": fixed_base["approved_map"]["heights_sha256"],
        "spawn_metadata": spawn_metadata,
        "episodes": len(rows),
        "claim_boundary": config["claim_boundary"],
    }
    (output_root / "execution_record.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
