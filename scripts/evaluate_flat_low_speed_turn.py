"""Evaluate low-speed turning with an existing checkpoint and no training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.planar_transition import quaternion_yaw_angle  # noqa: E402
from run_curved_gait_training import common_env_kwargs  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "flat_low_speed_turn_diagnostic_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def commanded_yaw_rate(
    current_positive_yaw_change_rad: float,
    target_positive_yaw_change_rad: float,
    *,
    gain_per_second: float,
    maximum_abs_yaw_rate_rad_per_s: float,
    tolerance_rad: float,
) -> float:
    error = target_positive_yaw_change_rad - current_positive_yaw_change_rad
    if abs(error) <= tolerance_rad:
        return 0.0
    return float(
        np.clip(
            gain_per_second * error,
            -maximum_abs_yaw_rate_rad_per_s,
            maximum_abs_yaw_rate_rad_per_s,
        )
    )


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    if config.get("status") != "read_only_checkpoint_diagnostic_not_training_not_formal":
        raise ValueError("Unexpected diagnostic status")
    if bool(config["training_performed"]) or bool(config["core_source_modified"]):
        raise ValueError("This diagnostic must not train or modify core source")
    policy = config["source_policy"]
    if int(policy["observation_dimension"]) != int(source["commands"]["target_observation_dimension"]):
        raise ValueError("Source observation dimension mismatch")
    speeds = [float(value) for value in config["diagnostic_commands"]["positive_crawl_speeds_m_per_s"]]
    if any(value <= 0.0 for value in speeds):
        raise ValueError("The existing adapter diagnostic requires positive speed")
    maximum_yaw = float(config["diagnostic_commands"]["maximum_abs_yaw_rate_rad_per_s"])
    diagnostic_curvature = float(
        config["diagnostic_commands"]["diagnostic_maximum_abs_curvature_per_m"]
    )
    if maximum_yaw / min(speeds) > diagnostic_curvature + 1e-12:
        raise ValueError("Diagnostic curvature guard cannot admit the requested yaw rate")
    if int(config["evaluation"]["horizon_steps"]) <= 0:
        raise ValueError("Evaluation horizon must be positive")


def build_env(source: dict[str, Any], config: dict[str, Any], *, seed: int) -> Any:
    kwargs = common_env_kwargs(source)
    kwargs.update(
        {
            "profile": "external",
            "speed_min": min(config["diagnostic_commands"]["positive_crawl_speeds_m_per_s"]),
            "speed_max": max(config["diagnostic_commands"]["positive_crawl_speeds_m_per_s"]),
            "fixed_lateral_speed": 0.0,
            "max_abs_curvature": float(
                config["diagnostic_commands"]["diagnostic_maximum_abs_curvature_per_m"]
            ),
            "heading_termination_enabled": False,
        }
    )
    return make_curved_gait_env(
        condition_id="FLAT_LOW_SPEED_TURN_DIAGNOSTIC",
        seed=seed,
        max_episode_steps=int(config["evaluation"]["horizon_steps"]),
        terminate_when_unhealthy=bool(config["evaluation"]["terminate_when_unhealthy"]),
        **kwargs,
    )


def evaluate_episode(
    model: PPO,
    source: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
    speed: float,
    target_degrees: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = build_env(source, config, seed=seed)
    try:
        observation, _ = env.reset(seed=seed)
        initial_position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        previous_yaw = quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
        target_heading = wrapped_angle(previous_yaw + math.radians(target_degrees))
        unwrapped_change = 0.0
        maximum_drift = 0.0
        airborne_steps = 0
        support_sum = 0
        minimum_support = 4
        dwell_steps = 0
        longest_dwell = 0
        target_first_step: int | None = None
        completed_steps = 0
        terminated = False
        truncated = False
        traces: list[dict[str, Any]] = []
        tolerance = math.radians(float(config["evaluation"]["turn_tolerance_degrees"]))
        required_dwell = int(
            math.ceil(
                float(config["evaluation"]["target_dwell_seconds"])
                / float(config["evaluation"]["environment_dt_seconds"])
            )
        )
        for step in range(1, int(config["evaluation"]["horizon_steps"]) + 1):
            yaw_rate = commanded_yaw_rate(
                unwrapped_change,
                math.radians(target_degrees),
                gain_per_second=float(config["diagnostic_commands"]["yaw_rate_gain_per_second"]),
                maximum_abs_yaw_rate_rad_per_s=float(
                    config["diagnostic_commands"]["maximum_abs_yaw_rate_rad_per_s"]
                ),
                tolerance_rad=tolerance,
            )
            observation = env.set_external_curve_command(
                observation,
                target_heading=target_heading,
                yaw_rate=yaw_rate,
                speed=float(speed),
                lateral_speed=0.0,
            )
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            completed_steps = step
            yaw = quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
            unwrapped_change += wrapped_angle(yaw - previous_yaw)
            previous_yaw = yaw
            position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
            drift = float(np.linalg.norm(position - initial_position))
            maximum_drift = max(maximum_drift, drift)
            mask = np.asarray(info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool)
            support = int(np.sum(mask))
            support_sum += support
            minimum_support = min(minimum_support, support)
            airborne = support == 0
            airborne_steps += int(airborne)
            error_degrees = target_degrees - math.degrees(unwrapped_change)
            if abs(error_degrees) <= float(config["evaluation"]["turn_tolerance_degrees"]):
                dwell_steps += 1
                if target_first_step is None:
                    target_first_step = step
            else:
                dwell_steps = 0
            longest_dwell = max(longest_dwell, dwell_steps)
            traces.append(
                {
                    "step": step,
                    "time_seconds": step * float(env.unwrapped.dt),
                    "yaw_change_degrees": math.degrees(unwrapped_change),
                    "target_error_degrees": error_degrees,
                    "commanded_yaw_rate_rad_per_s": yaw_rate,
                    "centre_drift_m": drift,
                    "support_count": support,
                    "airborne": int(airborne),
                    "reward": float(reward),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                }
            )
            if dwell_steps >= required_dwell or terminated or truncated:
                break
        summary = env.episode_summary()
        fall = bool(summary.get("fall", False))
        kinematic = longest_dwell >= required_dwell
        drift_limited = bool(
            kinematic
            and maximum_drift <= float(config["evaluation"]["maximum_centre_drift_m"])
            and not fall
        )
        safety_qualified = bool(drift_limited and airborne_steps == 0)
        row = {
            "seed": seed,
            "crawl_speed_m_per_s": speed,
            "target_yaw_change_degrees": target_degrees,
            "completed_steps": completed_steps,
            "elapsed_seconds": completed_steps * float(env.unwrapped.dt),
            "final_yaw_change_degrees": math.degrees(unwrapped_change),
            "absolute_final_target_error_degrees": abs(
                target_degrees - math.degrees(unwrapped_change)
            ),
            "target_first_step": target_first_step,
            "longest_target_dwell_steps": longest_dwell,
            "kinematic_target_reached": int(kinematic),
            "maximum_centre_drift_m": maximum_drift,
            "final_centre_displacement_m": float(
                np.linalg.norm(np.asarray(env.unwrapped.data.qpos[:2]) - initial_position)
            ),
            "fall": int(fall),
            "mean_support_count": support_sum / max(completed_steps, 1),
            "minimum_support_count": minimum_support,
            "airborne_step_fraction": airborne_steps / max(completed_steps, 1),
            "drift_limited_turn": int(drift_limited),
            "safety_qualified_turn": int(safety_qualified),
            "terminated": int(terminated),
            "truncated": int(truncated),
            "termination_category": summary.get("termination_category", "unknown"),
        }
        return row, traces
    finally:
        env.close()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    conditions = sorted(
        {(float(row["crawl_speed_m_per_s"]), float(row["target_yaw_change_degrees"])) for row in rows}
    )
    for speed, target in conditions:
        group = [
            row
            for row in rows
            if float(row["crawl_speed_m_per_s"]) == speed
            and float(row["target_yaw_change_degrees"]) == target
        ]
        result.append(
            {
                "crawl_speed_m_per_s": speed,
                "target_yaw_change_degrees": target,
                "episodes": len(group),
                "kinematic_target_reached_count": sum(
                    int(row["kinematic_target_reached"]) for row in group
                ),
                "drift_limited_turn_count": sum(int(row["drift_limited_turn"]) for row in group),
                "safety_qualified_turn_count": sum(
                    int(row["safety_qualified_turn"]) for row in group
                ),
                "fall_count": sum(int(row["fall"]) for row in group),
                "mean_final_yaw_change_degrees": float(
                    np.mean([float(row["final_yaw_change_degrees"]) for row in group])
                ),
                "mean_maximum_centre_drift_m": float(
                    np.mean([float(row["maximum_centre_drift_m"]) for row in group])
                ),
                "mean_airborne_step_fraction": float(
                    np.mean([float(row["airborne_step_fraction"]) for row in group])
                ),
                "mean_support_count": float(
                    np.mean([float(row["mean_support_count"]) for row in group])
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_policy"]["config"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_config(config, source)
    model_path = ROOT / config["source_policy"]["checkpoint"]
    if sha256(model_path) != config["source_policy"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    output = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["output_root"]).resolve()
    )
    if output.exists() and any(output.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    (output / "traces").mkdir(parents=True)
    (output / "frozen_config.json").write_bytes(config_path.read_bytes())
    shutil.copy2(Path(__file__).resolve(), output / "frozen_evaluation_script.py")
    model = PPO.load(model_path, device="cpu")
    rows: list[dict[str, Any]] = []
    for speed in config["diagnostic_commands"]["positive_crawl_speeds_m_per_s"]:
        for target in config["diagnostic_commands"]["target_positive_yaw_changes_degrees"]:
            for seed in config["evaluation"]["seeds"]:
                row, trace = evaluate_episode(
                    model,
                    source,
                    config,
                    seed=int(seed),
                    speed=float(speed),
                    target_degrees=float(target),
                )
                rows.append(row)
                write_csv(
                    output
                    / "traces"
                    / f"speed_{float(speed):.2f}_target_{int(target)}_seed_{int(seed)}.csv",
                    trace,
                )
                print(json.dumps(row, ensure_ascii=False), flush=True)
    write_csv(output / "episode_rows.csv", rows)
    aggregates = aggregate(rows)
    with (output / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    execution = {
        "schema_version": "proxygap-flat-low-speed-turn-diagnostic-v1",
        "status": "complete",
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(model_path),
        "episode_count": len(rows),
        "training_performed": False,
        "core_source_modified": False,
        "true_zero_speed_tested": False,
        "diagnostic_is_out_of_training_distribution": True,
        "claim_boundary": config["claim_boundary"],
    }
    with (output / "execution_record.json").open("w", encoding="utf-8") as handle:
        json.dump(execution, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
