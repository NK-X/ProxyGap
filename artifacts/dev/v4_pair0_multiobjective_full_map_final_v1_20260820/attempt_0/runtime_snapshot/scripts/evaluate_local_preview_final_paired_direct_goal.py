"""Run the audited paired direct-goal evaluation for the 135-D pilot.

The evaluator is deliberately independent of the task wrapper's success flag.
It also separates the historical instantaneous foot-contact speed exceedance
from a per-foot, landing-grace and duration-corrected slip definition.  Frozen
training outputs, map assets, friction, reward and energy formulae are inputs
only and are never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_goal_terrain_training import (  # noqa: E402
    make_task_env,
    prepare_task_scenes,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_map_local_preview_final_paired_direct_goal_v1_20260819.json"
)
FOOT_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidate_names",
        help="Run only a named candidate; may be repeated.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Run only a named fixed seed; may be repeated.",
    )
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _vector(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (FOOT_COUNT,):
        raise ValueError(f"{name} must contain exactly four foot values")
    return result


def _sum_summary_vector(summary: dict[str, Any], key: str) -> float:
    value = np.asarray(summary[key], dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid mechanical proxy vector: {key}")
    return float(np.sum(value))


def _body_up_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(item) for item in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        return np.full(3, float("nan"))
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return np.asarray(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def terrain_relative_tilt_rad(env: Any, qpos: np.ndarray) -> float:
    x, y = float(qpos[0]), float(qpos[1])
    gradient = np.asarray(
        [
            env._terrain_value(env._terrain_dz_dx, x, y),
            env._terrain_value(env._terrain_dz_dy, x, y),
        ],
        dtype=np.float64,
    )
    normal = np.asarray([-gradient[0], -gradient[1], 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    body_up = _body_up_from_wxyz(qpos[3:7])
    if not np.all(np.isfinite(body_up)):
        return float("nan")
    return float(math.acos(float(np.clip(np.dot(body_up, normal), -1.0, 1.0))))


@dataclass
class ArrivalDwellTracker:
    """Arrival-gated spatial and strict-stability dwell tracker."""

    arrival_radius_m: float
    hold_radius_m: float
    required_hold_steps: int
    goal_entered: bool = False
    entry_step: int | None = None
    hold_run_steps: int = 0
    longest_hold_run_steps: int = 0
    spatial_success: bool = False
    spatial_success_step: int | None = None
    strict_run_steps: int = 0
    longest_strict_run_steps: int = 0
    strict_dwell_success: bool = False
    strict_dwell_success_step: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.arrival_radius_m <= self.hold_radius_m:
            raise ValueError("arrival radius must be positive and no larger than hold radius")
        if self.required_hold_steps <= 0:
            raise ValueError("required_hold_steps must be positive")

    def update(self, *, step: int, distance_m: float, stable: bool) -> None:
        if not self.goal_entered and distance_m <= self.arrival_radius_m:
            self.goal_entered = True
            self.entry_step = int(step)
        inside_gated_hold = self.goal_entered and distance_m <= self.hold_radius_m
        self.hold_run_steps = self.hold_run_steps + 1 if inside_gated_hold else 0
        self.longest_hold_run_steps = max(
            self.longest_hold_run_steps,
            self.hold_run_steps,
        )
        strict = bool(inside_gated_hold and stable)
        self.strict_run_steps = self.strict_run_steps + 1 if strict else 0
        self.longest_strict_run_steps = max(
            self.longest_strict_run_steps,
            self.strict_run_steps,
        )
        if not self.spatial_success and self.hold_run_steps >= self.required_hold_steps:
            self.spatial_success = True
            self.spatial_success_step = int(step)
        if (
            not self.strict_dwell_success
            and self.strict_run_steps >= self.required_hold_steps
        ):
            self.strict_dwell_success = True
            self.strict_dwell_success_step = int(step)


class DurationCorrectedSlipTracker:
    """Collect foot-level raw and persistence-corrected slip evidence."""

    def __init__(
        self,
        *,
        dt: float,
        speed_threshold: float,
        minimum_normal_force: float,
        landing_grace_seconds: float,
        minimum_sustained_seconds: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if speed_threshold <= 0.0 or minimum_normal_force < 0.0:
            raise ValueError("invalid slip thresholds")
        self.dt = float(dt)
        self.speed_threshold = float(speed_threshold)
        self.minimum_normal_force = float(minimum_normal_force)
        self.grace_steps = int(math.ceil(float(landing_grace_seconds) / self.dt))
        self.minimum_steps = max(
            1,
            int(math.ceil(float(minimum_sustained_seconds) / self.dt)),
        )
        self._contact_age = np.zeros(FOOT_COUNT, dtype=np.int64)
        self._contacts: list[np.ndarray] = []
        self._speeds: list[np.ndarray] = []
        self._forces: list[np.ndarray] = []
        self._raw: list[np.ndarray] = []
        self._candidates: list[np.ndarray] = []

    def update(
        self,
        *,
        contact_mask: np.ndarray,
        tangential_speeds: np.ndarray,
        normal_forces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        contacts = _vector(contact_mask, name="contact_mask", dtype=bool)
        speeds = _vector(
            tangential_speeds,
            name="tangential_speeds",
            dtype=np.float64,
        )
        forces = _vector(normal_forces, name="normal_forces", dtype=np.float64)
        if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(forces)):
            raise ValueError("slip inputs must be finite")
        self._contact_age = np.where(contacts, self._contact_age + 1, 0)
        raw = contacts & (speeds > self.speed_threshold)
        candidate = (
            raw
            & (self._contact_age > self.grace_steps)
            & (forces >= self.minimum_normal_force)
        )
        self._contacts.append(contacts.copy())
        self._speeds.append(speeds.copy())
        self._forces.append(forces.copy())
        self._raw.append(raw.copy())
        self._candidates.append(candidate.copy())
        return raw, candidate

    def finalise(self) -> dict[str, Any]:
        if not self._candidates:
            empty = np.zeros((0, FOOT_COUNT), dtype=bool)
            return {
                "raw": empty,
                "candidate": empty,
                "sustained": empty,
                "events": [],
            }
        candidates = np.asarray(self._candidates, dtype=bool)
        raw = np.asarray(self._raw, dtype=bool)
        speeds = np.asarray(self._speeds, dtype=np.float64)
        forces = np.asarray(self._forces, dtype=np.float64)
        sustained = np.zeros_like(candidates)
        events: list[dict[str, Any]] = []
        for foot in range(FOOT_COUNT):
            start: int | None = None
            for index in range(candidates.shape[0] + 1):
                active = bool(candidates[index, foot]) if index < candidates.shape[0] else False
                if active and start is None:
                    start = index
                if not active and start is not None:
                    length = index - start
                    if length >= self.minimum_steps:
                        sustained[start:index, foot] = True
                        event_speeds = speeds[start:index, foot]
                        event_forces = forces[start:index, foot]
                        events.append(
                            {
                                "foot_index": foot,
                                "start_step": start + 1,
                                "end_step": index,
                                "duration_steps": length,
                                "duration_seconds": length * self.dt,
                                "maximum_tangential_speed_m_per_s": float(
                                    np.max(event_speeds)
                                ),
                                "mean_tangential_speed_m_per_s": float(
                                    np.mean(event_speeds)
                                ),
                                "minimum_normal_force_n": float(np.min(event_forces)),
                                "mean_normal_force_n": float(np.mean(event_forces)),
                                "slip_distance_proxy_m": float(
                                    np.sum(event_speeds) * self.dt
                                ),
                            }
                        )
                    start = None
        return {
            "raw": raw,
            "candidate": candidates,
            "sustained": sustained,
            "events": events,
        }


def validate_and_load_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("status") != "predeclared_paired_development_evaluation":
        raise ValueError("Evaluation configuration is not predeclared and frozen")
    fixed_path = ROOT / config["fixed_map_configuration"]
    if sha256(fixed_path) != config["fixed_map_configuration_sha256"]:
        raise ValueError("Fixed-map pilot configuration SHA-256 mismatch")
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    expected_friction = np.asarray([1.0, 0.5, 0.5], dtype=np.float64)
    observed_friction = np.asarray(
        fixed["approved_map"]["fixed_friction"],
        dtype=np.float64,
    )
    if not np.array_equal(observed_friction, expected_friction):
        raise ValueError("Frozen friction differs from [1.0, 0.5, 0.5]")
    controller = config["controller"]
    required_controller = {
        "cruise_speed_m_per_s": 0.5,
        "yaw_gain_per_second": 0.75,
        "slow_radius_m": 4.0,
        "maximum_abs_curvature_per_m": 0.35,
    }
    for key, expected in required_controller.items():
        if not math.isclose(float(controller[key]), expected, abs_tol=1e-12):
            raise ValueError(f"Unexpected controller setting: {key}")
    if list(config["evaluation_seeds"]) != [74801, 74802, 74803, 74804, 74805]:
        raise ValueError("Fixed paired seed set changed")
    if int(config["horizon_steps"]) != 12000:
        raise ValueError("Paired horizon must be 12000 steps")
    success = config["independent_success"]
    if not bool(success["require_arrival_entry_before_hold"]):
        raise ValueError("Arrival entry gate must be enabled")
    if not math.isclose(float(success["arrival_radius_m"]), 1.5, abs_tol=1e-12):
        raise ValueError("Arrival radius changed")
    if not math.isclose(float(success["hold_radius_m"]), 2.0, abs_tol=1e-12):
        raise ValueError("Hold radius changed")
    if not math.isclose(float(success["hold_seconds"]), 2.0, abs_tol=1e-12):
        raise ValueError("Hold duration changed")
    return config, fixed


def _stable_step(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    terrain_tilt: float,
    support_count: int,
    corrected_slip_candidate: bool,
    settings: dict[str, Any],
) -> bool:
    finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
    return bool(
        (finite or not bool(settings["require_finite_state"]))
        and float(np.linalg.norm(qvel[:2]))
        <= float(settings["maximum_planar_speed_m_per_s"])
        and terrain_tilt
        <= math.radians(
            float(settings["maximum_terrain_relative_torso_tilt_degrees"])
        )
        and support_count >= int(settings["minimum_foot_support_count"])
        and (
            not corrected_slip_candidate
            or not bool(settings["require_no_duration_corrected_slip_candidate"])
        )
    )


def evaluate_episode(
    *,
    config: dict[str, Any],
    fixed_config: dict[str, Any],
    candidate: dict[str, Any],
    policy_config: dict[str, Any],
    model: PPO,
    scene: Path,
    seed: int,
    trace_path: Path,
    event_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    condition = copy.deepcopy(fixed_config)
    controller = config["controller"]
    condition["task_adapter"].update(
        {
            "yaw_gain_per_second": float(controller["yaw_gain_per_second"]),
            "yaw_deadband_degrees": float(controller["yaw_deadband_degrees"]),
            "slow_radius_m": float(controller["slow_radius_m"]),
            "maximum_abs_curvature_per_m": float(
                controller["maximum_abs_curvature_per_m"]
            ),
            "curvature_speed_reduction_gain": float(
                controller["curvature_speed_reduction_gain"]
            ),
            "minimum_turn_speed_fraction": float(
                controller["minimum_turn_speed_fraction"]
            ),
            "augment_local_terrain_observation": bool(
                candidate["augment_local_terrain_observation"]
            ),
        }
    )
    independent = config["independent_success"]
    condition["task_adapter"].update(
        {
            "arrival_radius_m": float(independent["arrival_radius_m"]),
            "hold_radius_m": float(independent["hold_radius_m"]),
            "hold_seconds": float(independent["hold_seconds"]),
        }
    )
    horizon = int(config["horizon_steps"])
    env = make_task_env(
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
        if tuple(observation.shape) != tuple(model.observation_space.shape):
            raise ValueError(
                f"Observation mismatch for {candidate['name']}: "
                f"env={observation.shape}, model={model.observation_space.shape}"
            )
        if int(observation.shape[0]) != int(candidate["observation_dimension"]):
            raise ValueError("Candidate observation dimension declaration mismatch")
        dt = float(env.unwrapped.dt)
        required_steps = int(math.ceil(float(independent["hold_seconds"]) / dt))
        arrival = ArrivalDwellTracker(
            arrival_radius_m=float(independent["arrival_radius_m"]),
            hold_radius_m=float(independent["hold_radius_m"]),
            required_hold_steps=required_steps,
        )
        slip_cfg = config["duration_corrected_slip"]
        slip = DurationCorrectedSlipTracker(
            dt=dt,
            speed_threshold=float(slip_cfg["tangential_speed_threshold_m_per_s"]),
            minimum_normal_force=float(slip_cfg["minimum_normal_force_n"]),
            landing_grace_seconds=float(slip_cfg["landing_grace_seconds"]),
            minimum_sustained_seconds=float(slip_cfg["minimum_sustained_seconds"]),
        )
        trace: list[dict[str, Any]] = []
        previous_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        minimum_distance_so_far = float(np.linalg.norm(env.goal_xy - previous_xy))
        path_length = 0.0
        raw_airborne_steps = 0
        current_airborne_run = 0
        longest_airborne_run = 0
        first_airborne_step: int | None = None
        terminated = False
        truncated = False
        termination_reason = "horizon"
        total_reward = 0.0
        start_wall = time.perf_counter()

        for step_index in range(1, horizon + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64).copy()
            xy = qpos[:2]
            path_length += float(np.linalg.norm(xy - previous_xy))
            previous_xy = xy.copy()
            distance = float(np.linalg.norm(env.goal_xy - xy))
            minimum_distance_so_far = min(minimum_distance_so_far, distance)
            terrain_height = float(env._terrain_height(float(xy[0]), float(xy[1])))
            terrain_tilt = terrain_relative_tilt_rad(env, qpos)
            contacts = _vector(
                info.get("proxygap_foot_contact_mask_step", np.zeros(FOOT_COUNT)),
                name="proxygap_foot_contact_mask_step",
                dtype=bool,
            )
            speeds = _vector(
                info.get(
                    "proxygap_foot_contact_tangential_speeds_m_per_s_step",
                    np.zeros(FOOT_COUNT),
                ),
                name="proxygap_foot_contact_tangential_speeds_m_per_s_step",
                dtype=np.float64,
            )
            forces = _vector(
                info.get("proxygap_foot_normal_forces_n_step", np.zeros(FOOT_COUNT)),
                name="proxygap_foot_normal_forces_n_step",
                dtype=np.float64,
            )
            raw_slip, corrected_candidate = slip.update(
                contact_mask=contacts,
                tangential_speeds=speeds,
                normal_forces=forces,
            )
            support_count = int(np.sum(contacts))
            airborne = support_count == 0
            if airborne:
                raw_airborne_steps += 1
                current_airborne_run += 1
                longest_airborne_run = max(longest_airborne_run, current_airborne_run)
                if first_airborne_step is None:
                    first_airborne_step = step_index
            else:
                current_airborne_run = 0
            stable = _stable_step(
                qpos=qpos,
                qvel=qvel,
                terrain_tilt=terrain_tilt,
                support_count=support_count,
                corrected_slip_candidate=bool(np.any(corrected_candidate)),
                settings=config["strict_stable_dwell"],
            )
            arrival.update(step=step_index, distance_m=distance, stable=stable)
            powers = np.asarray(
                info.get("proxygap_actuator_mechanical_powers_w_step", np.zeros(8)),
                dtype=np.float64,
            )
            trace.append(
                {
                    "candidate": candidate["name"],
                    "evaluation_seed": seed,
                    "step": step_index,
                    "time_seconds": step_index * dt,
                    "x_m": float(qpos[0]),
                    "y_m": float(qpos[1]),
                    "torso_z_m": float(qpos[2]),
                    "terrain_z_m": terrain_height,
                    "torso_clearance_m": float(qpos[2] - terrain_height),
                    "goal_distance_m": distance,
                    "minimum_goal_distance_so_far_m": minimum_distance_so_far,
                    "world_vx_m_per_s": float(qvel[0]),
                    "world_vy_m_per_s": float(qvel[1]),
                    "planar_speed_m_per_s": float(np.linalg.norm(qvel[:2])),
                    "terrain_relative_torso_tilt_rad": terrain_tilt,
                    "support_count": support_count,
                    "four_foot_airborne": int(airborne),
                    "foot_contact_mask": json.dumps(contacts.astype(int).tolist()),
                    "foot_tangential_speeds_m_per_s": json.dumps(speeds.tolist()),
                    "foot_normal_forces_n": json.dumps(forces.tolist()),
                    "raw_contact_speed_exceedance": int(np.any(raw_slip)),
                    "duration_corrected_slip_candidate": int(
                        np.any(corrected_candidate)
                    ),
                    "duration_corrected_sustained_slip": 0,
                    "duration_corrected_sustained_slip_foot_mask": "[0, 0, 0, 0]",
                    "goal_entered": int(arrival.goal_entered),
                    "spatial_hold_run_steps": arrival.hold_run_steps,
                    "strict_stable_step": int(stable),
                    "strict_stable_hold_run_steps": arrival.strict_run_steps,
                    "spatial_success": int(arrival.spatial_success),
                    "strict_stable_dwell_success": int(arrival.strict_dwell_success),
                    "reward_step": float(reward),
                    "positive_mechanical_power_proxy_w": float(
                        np.sum(np.maximum(powers, 0.0))
                    ),
                    "absolute_mechanical_power_proxy_w": float(
                        np.sum(np.abs(powers))
                    ),
                    "terminated_by_environment": int(terminated),
                    "truncated_by_environment": int(truncated),
                }
            )
            if arrival.spatial_success and bool(
                independent["terminate_evaluation_on_spatial_success"]
            ):
                termination_reason = "independent_spatial_success"
                break
            if terminated:
                termination_reason = "environment_terminated"
                break
            if truncated:
                termination_reason = "horizon_truncated"
                break

        wall_seconds = time.perf_counter() - start_wall
        slip_result = slip.finalise()
        sustained = np.asarray(slip_result["sustained"], dtype=bool)
        for index, row in enumerate(trace):
            if index < sustained.shape[0]:
                row["duration_corrected_sustained_slip"] = int(
                    np.any(sustained[index])
                )
                row["duration_corrected_sustained_slip_foot_mask"] = json.dumps(
                    sustained[index].astype(int).tolist()
                )
        write_csv(trace_path, trace)
        events = [
            {
                "candidate": candidate["name"],
                "evaluation_seed": seed,
                **event,
            }
            for event in slip_result["events"]
        ]
        if events:
            write_csv(event_path, events)
        else:
            event_path.write_text(
                "candidate,evaluation_seed,foot_index,start_step,end_step,duration_steps,duration_seconds,maximum_tangential_speed_m_per_s,mean_tangential_speed_m_per_s,minimum_normal_force_n,mean_normal_force_n,slip_distance_proxy_m\n",
                encoding="utf-8",
            )

        summary = env.episode_summary()
        completed_steps = len(trace)
        raw_any = np.any(np.asarray(slip_result["raw"], dtype=bool), axis=1)
        candidate_any = np.any(
            np.asarray(slip_result["candidate"], dtype=bool),
            axis=1,
        )
        sustained_any = np.any(sustained, axis=1)
        final_distance = float(trace[-1]["goal_distance_m"])
        minimum_distance = float(min(row["goal_distance_m"] for row in trace))
        fall = bool(summary.get("fall", False))
        qualified = bool(
            arrival.spatial_success
            and arrival.strict_dwell_success
            and not fall
            and raw_airborne_steps == 0
            and not events
        )
        positive_work = _sum_summary_vector(
            summary,
            "actuator_positive_mechanical_work_j_by_actuator",
        )
        negative_work_abs = _sum_summary_vector(
            summary,
            "actuator_negative_mechanical_work_abs_j_by_actuator",
        )
        absolute_work = _sum_summary_vector(
            summary,
            "actuator_abs_mechanical_work_j_by_actuator",
        )
        torque_time_integral = _sum_summary_vector(
            summary,
            "actuator_abs_torque_time_integral_n_m_s_by_actuator",
        )
        row = {
            "candidate": candidate["name"],
            "role": candidate["role"],
            "evaluation_seed": seed,
            "completed_steps": completed_steps,
            "elapsed_seconds": completed_steps * dt,
            "wall_seconds": wall_seconds,
            "termination_reason": termination_reason,
            "environment_terminated": int(terminated),
            "environment_truncated": int(truncated),
            "fall": int(fall),
            "termination_category": summary.get("termination_category", "unknown"),
            "independent_goal_entered": int(arrival.goal_entered),
            "independent_goal_entry_step": arrival.entry_step,
            "independent_spatial_success": int(arrival.spatial_success),
            "independent_spatial_success_step": arrival.spatial_success_step,
            "independent_spatial_success_time_seconds": (
                arrival.spatial_success_step * dt
                if arrival.spatial_success_step is not None
                else None
            ),
            "wrapper_spatial_success_diagnostic": int(
                bool(summary.get("fixed_goal_success", False))
            ),
            "wrapper_independent_success_mismatch": int(
                bool(summary.get("fixed_goal_success", False))
                != arrival.spatial_success
            ),
            "strict_stable_dwell_success": int(arrival.strict_dwell_success),
            "strict_stable_dwell_success_step": arrival.strict_dwell_success_step,
            "longest_spatial_hold_seconds": arrival.longest_hold_run_steps * dt,
            "longest_strict_stable_hold_seconds": arrival.longest_strict_run_steps * dt,
            "qualified": int(qualified),
            "initial_distance_m": float(summary["fixed_goal_initial_distance_m"]),
            "final_distance_m": final_distance,
            "minimum_distance_m": minimum_distance,
            "net_progress_m": float(summary["fixed_goal_initial_distance_m"])
            - final_distance,
            "path_length_m": path_length,
            "cumulative_reward": total_reward,
            "airborne_step_count": raw_airborne_steps,
            "airborne_step_fraction": raw_airborne_steps / completed_steps,
            "first_airborne_step": first_airborne_step,
            "longest_airborne_run_seconds": longest_airborne_run * dt,
            "raw_contact_speed_exceedance_step_count": int(np.sum(raw_any)),
            "raw_contact_speed_exceedance_step_fraction": float(np.mean(raw_any)),
            "post_landing_force_gated_candidate_step_count": int(
                np.sum(candidate_any)
            ),
            "post_landing_force_gated_candidate_step_fraction": float(
                np.mean(candidate_any)
            ),
            "duration_corrected_slip_event_count": len(events),
            "duration_corrected_slip_step_count": int(np.sum(sustained_any)),
            "duration_corrected_slip_step_fraction": float(
                np.mean(sustained_any)
            ),
            "duration_corrected_slip_longest_event_seconds": (
                max(event["duration_seconds"] for event in events) if events else 0.0
            ),
            "duration_corrected_slip_distance_proxy_m": float(
                sum(event["slip_distance_proxy_m"] for event in events)
            ),
            "positive_mechanical_work_proxy_j": positive_work,
            "negative_mechanical_work_abs_proxy_j": negative_work_abs,
            "absolute_mechanical_work_proxy_j": absolute_work,
            "absolute_torque_time_integral_proxy_n_m_s": torque_time_integral,
            "positive_mechanical_work_proxy_per_path_m_j_per_m": (
                positive_work / path_length if path_length > 0.0 else None
            ),
            "absolute_mechanical_work_proxy_per_path_m_j_per_m": (
                absolute_work / path_length if path_length > 0.0 else None
            ),
            "model_sha256": candidate["model_sha256"],
            "trace_path": str(trace_path.relative_to(trace_path.parents[1])),
            "trace_sha256": sha256(trace_path),
            "slip_events_path": str(event_path.relative_to(event_path.parents[1])),
            "slip_events_sha256": sha256(event_path),
        }
        detail = {
            "episode": row,
            "duration_corrected_slip_events": events,
            "inner_summary": summary,
            "operational_definitions": {
                "arrival": config["independent_success"],
                "strict_stable_dwell": config["strict_stable_dwell"],
                "duration_corrected_slip": config["duration_corrected_slip"],
                "qualification": config["qualification"],
                "mechanical_work_boundary": (
                    "tau*qdot integrated in the existing simulation instrumentation; "
                    "a mechanical proxy, not battery electrical energy"
                ),
            },
        }
        return row, detail
    finally:
        env.close()


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, Any] = {}
    for candidate in sorted({str(row["candidate"]) for row in rows}):
        subset = [row for row in rows if row["candidate"] == candidate]
        by_candidate[candidate] = {
            "episodes": len(subset),
            "spatial_successes": sum(int(row["independent_spatial_success"]) for row in subset),
            "strict_stable_dwell_successes": sum(
                int(row["strict_stable_dwell_success"]) for row in subset
            ),
            "qualified_episodes": sum(int(row["qualified"]) for row in subset),
            "falls": sum(int(row["fall"]) for row in subset),
            "mean_final_distance_m": _mean(subset, "final_distance_m"),
            "mean_minimum_distance_m": _mean(subset, "minimum_distance_m"),
            "mean_path_length_m": _mean(subset, "path_length_m"),
            "mean_airborne_step_fraction": _mean(subset, "airborne_step_fraction"),
            "mean_raw_contact_speed_exceedance_step_fraction": _mean(
                subset,
                "raw_contact_speed_exceedance_step_fraction",
            ),
            "total_duration_corrected_slip_events": sum(
                int(row["duration_corrected_slip_event_count"]) for row in subset
            ),
            "mean_duration_corrected_slip_step_fraction": _mean(
                subset,
                "duration_corrected_slip_step_fraction",
            ),
            "mean_positive_mechanical_work_proxy_j": _mean(
                subset,
                "positive_mechanical_work_proxy_j",
            ),
            "mean_absolute_mechanical_work_proxy_j": _mean(
                subset,
                "absolute_mechanical_work_proxy_j",
            ),
        }
    paired: list[dict[str, Any]] = []
    roles = {str(row["role"]): str(row["candidate"]) for row in rows}
    if "paired_control" in roles and "local_preview_intervention" in roles:
        control_name = roles["paired_control"]
        preview_name = roles["local_preview_intervention"]
        for seed in sorted({int(row["evaluation_seed"]) for row in rows}):
            control = next(
                (
                    row
                    for row in rows
                    if row["candidate"] == control_name
                    and int(row["evaluation_seed"]) == seed
                ),
                None,
            )
            preview = next(
                (
                    row
                    for row in rows
                    if row["candidate"] == preview_name
                    and int(row["evaluation_seed"]) == seed
                ),
                None,
            )
            if control is None or preview is None:
                continue
            paired.append(
                {
                    "evaluation_seed": seed,
                    "spatial_success_delta": int(preview["independent_spatial_success"])
                    - int(control["independent_spatial_success"]),
                    "strict_stable_dwell_success_delta": int(
                        preview["strict_stable_dwell_success"]
                    )
                    - int(control["strict_stable_dwell_success"]),
                    "qualified_delta": int(preview["qualified"])
                    - int(control["qualified"]),
                    "fall_delta": int(preview["fall"]) - int(control["fall"]),
                    "final_distance_delta_m": float(preview["final_distance_m"])
                    - float(control["final_distance_m"]),
                    "minimum_distance_delta_m": float(preview["minimum_distance_m"])
                    - float(control["minimum_distance_m"]),
                    "airborne_fraction_delta": float(
                        preview["airborne_step_fraction"]
                    )
                    - float(control["airborne_step_fraction"]),
                    "duration_corrected_slip_fraction_delta": float(
                        preview["duration_corrected_slip_step_fraction"]
                    )
                    - float(control["duration_corrected_slip_step_fraction"]),
                    "positive_mechanical_work_proxy_delta_j": float(
                        preview["positive_mechanical_work_proxy_j"]
                    )
                    - float(control["positive_mechanical_work_proxy_j"]),
                    "absolute_mechanical_work_proxy_delta_j": float(
                        preview["absolute_mechanical_work_proxy_j"]
                    )
                    - float(control["absolute_mechanical_work_proxy_j"]),
                }
            )
    paired_mean_delta = {
        key: _mean(paired, key)
        for key in (
            "final_distance_delta_m",
            "minimum_distance_delta_m",
            "airborne_fraction_delta",
            "duration_corrected_slip_fraction_delta",
            "positive_mechanical_work_proxy_delta_j",
            "absolute_mechanical_work_proxy_delta_j",
        )
    }
    return {
        "by_candidate": by_candidate,
        "paired_rows": paired,
        "paired_mean_delta_local_preview_minus_control": paired_mean_delta,
    }


def chinese_report(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# 135维局部地形预瞄最终策略：配对直达终点复验",
        "",
        "## 结论边界",
        "",
        "本报告是同一封存地图、五个固定种子的开发诊断。它不能证明未见地图泛化，也不能把机械功代理解释为电池电能。所有结果均保留，包括失败轮。",
        "",
        "## 预先固定的复验条件",
        "",
        f"- 种子：{', '.join(str(seed) for seed in config['evaluation_seeds'])}。",
        f"- 每轮上限：{config['horizon_steps']}步；速度0.50 m/s；yaw gain 0.75 s^-1；slow radius 4.0 m；最大绝对曲率0.35 m^-1。",
        "- 空间成功：必须先真正进入1.5 m到达圈，之后才允许累计在2.0 m圈内连续2 s；仅停在1.5–2.0 m环带不能成功。",
        "- 严格稳定dwell：上述位置门控同时要求实际平面速度不超过0.20 m/s、躯干相对局部地形法向倾角不超过30度、至少一个指定足端支撑、状态有限且不存在校正后滑移候选。",
        "- 校正后滑移：逐足端计算；落脚后0.10 s为宽限，法向力至少1 N，切向接触速度连续至少0.20 s超过0.20 m/s才形成事件。该指标不覆盖小腿或躯干接触滑动。",
        "- 合格：空间成功、严格稳定dwell、无摔倒、全程零四足端同时离地、零校正后持续滑移事件。",
        "",
        "## 汇总观察",
        "",
    ]
    for candidate, values in aggregate["by_candidate"].items():
        lines.extend(
            [
                f"### {candidate}",
                "",
                f"- 空间成功 {values['spatial_successes']}/{values['episodes']}；严格稳定dwell {values['strict_stable_dwell_successes']}/{values['episodes']}；合格 {values['qualified_episodes']}/{values['episodes']}；摔倒 {values['falls']}/{values['episodes']}。",
                f"- 平均最终/最小终点距离：{values['mean_final_distance_m']:.3f} / {values['mean_minimum_distance_m']:.3f} m。",
                f"- 平均四足端同时离地比例：{100.0 * values['mean_airborne_step_fraction']:.2f}%。",
                f"- 原始接触速度超限比例均值：{100.0 * values['mean_raw_contact_speed_exceedance_step_fraction']:.2f}%；校正后持续滑移比例均值：{100.0 * values['mean_duration_corrected_slip_step_fraction']:.2f}%；事件总数：{values['total_duration_corrected_slip_events']}。",
                f"- 正机械功/绝对机械功代理均值：{values['mean_positive_mechanical_work_proxy_j']:.3f} / {values['mean_absolute_mechanical_work_proxy_j']:.3f} J。",
                "",
            ]
        )
    lines.extend(["## 每个种子的审计表", ""])
    lines.append(
        "| 候选 | seed | 空间成功 | 稳定dwell | 合格 | 摔倒 | 最终距离(m) | 最小距离(m) | 腾空(%) | 校正滑移(%) | 正机械功代理(J) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['evaluation_seed']} | {row['independent_spatial_success']} | {row['strict_stable_dwell_success']} | {row['qualified']} | {row['fall']} | {row['final_distance_m']:.3f} | {row['minimum_distance_m']:.3f} | {100.0 * row['airborne_step_fraction']:.2f} | {100.0 * row['duration_corrected_slip_step_fraction']:.2f} | {row['positive_mechanical_work_proxy_j']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- 表中数据是观察事实；模型是否因局部预瞄而改善，需要看同seed差值是否方向一致，而不能用单个成功seed下因果结论。",
            "- 原始速度超限包含大量落脚冲击；校正指标通过落脚宽限、法向力门控和持续时间门控降低这一混淆，但仍不是库仑摩擦锥失效的直接测量。",
            "- 正/负/绝对机械功由既有tau*qdot积分得到；本次只读取并汇总，没有改动能耗公式或奖励。",
            "- 四足端同时离地按项目规则仍是违规，即使小腿或躯干与地面接触。",
            "",
        ]
    )
    mismatches = [row for row in rows if row["wrapper_independent_success_mismatch"]]
    lines.append(
        f"独立成功判定与包装器诊断不一致：{len(mismatches)}/{len(rows)}轮。最终结论以本评估器的独立门控为准。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config, fixed = validate_and_load_config(config_path)
    selected_candidates = list(config["candidates"])
    if args.candidate_names:
        requested = set(args.candidate_names)
        selected_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate["name"] in requested
        ]
        missing = requested - {candidate["name"] for candidate in selected_candidates}
        if missing:
            raise ValueError(f"Unknown candidates: {sorted(missing)}")
    seeds = list(config["evaluation_seeds"])
    if args.seeds:
        requested_seeds = set(args.seeds)
        if not requested_seeds.issubset(set(seeds)):
            raise ValueError("Requested seed is outside the frozen paired seed set")
        seeds = [seed for seed in seeds if seed in requested_seeds]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["output_root"]).resolve()
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    trace_root = output_root / "traces"
    event_root = output_root / "slip_events"
    detail_root = output_root / "episode_details"
    for directory in (output_root, trace_root, event_root, detail_root):
        directory.mkdir(parents=True, exist_ok=True)
    (output_root / "frozen_evaluation_config.json").write_bytes(
        config_path.read_bytes()
    )
    scenes, spawn_metadata = prepare_task_scenes(fixed, output_root, [0.0])
    scene = scenes[0]
    compiled = mujoco.MjModel.from_xml_path(str(scene))
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_friction = compiled.geom_friction[floor_id].astype(float).tolist()
    if floor_friction != [1.0, 0.5, 0.5]:
        raise RuntimeError("Compiled scene friction changed")

    rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    started = time.time()
    for candidate in selected_candidates:
        model_path = (ROOT / candidate["model_path"]).resolve()
        policy_path = (ROOT / candidate["policy_configuration"]).resolve()
        if sha256(model_path) != candidate["model_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {candidate['name']}")
        if sha256(policy_path) != candidate["policy_configuration_sha256"]:
            raise ValueError(f"Policy configuration SHA-256 mismatch: {candidate['name']}")
        policy_config = json.loads(policy_path.read_text(encoding="utf-8"))
        model = PPO.load(model_path, device="cpu")
        if int(model.observation_space.shape[0]) != int(
            candidate["observation_dimension"]
        ):
            raise ValueError(f"Loaded observation dimension mismatch: {candidate['name']}")
        candidate_records.append(
            {
                "name": candidate["name"],
                "role": candidate["role"],
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "policy_configuration": str(policy_path),
                "policy_configuration_sha256": sha256(policy_path),
                "observation_dimension": int(model.observation_space.shape[0]),
                "action_dimension": int(model.action_space.shape[0]),
            }
        )
        for seed in seeds:
            stem = f"{candidate['name']}_seed_{seed}"
            trace_path = trace_root / f"{stem}_trace.csv"
            event_path = event_root / f"{stem}_slip_events.csv"
            row, detail = evaluate_episode(
                config=config,
                fixed_config=fixed,
                candidate=candidate,
                policy_config=policy_config,
                model=model,
                scene=scene,
                seed=int(seed),
                trace_path=trace_path,
                event_path=event_path,
            )
            rows.append(row)
            write_json(detail_root / f"{stem}.json", detail)
            write_csv(output_root / "episode_results_partial.csv", rows)
            print(
                candidate["name"],
                f"seed={seed}",
                f"steps={row['completed_steps']}",
                f"success={row['independent_spatial_success']}",
                f"stable={row['strict_stable_dwell_success']}",
                f"fall={row['fall']}",
                f"final={row['final_distance_m']:.3f}m",
                f"minimum={row['minimum_distance_m']:.3f}m",
                f"airborne={100.0 * row['airborne_step_fraction']:.2f}%",
                f"corrected_slip={100.0 * row['duration_corrected_slip_step_fraction']:.2f}%",
                flush=True,
            )

    write_csv(output_root / "episode_results.csv", rows)
    aggregate = aggregate_results(rows)
    write_json(output_root / "aggregate_results.json", aggregate)
    report_path = output_root / "PAIRED_DIRECT_GOAL_EVALUATION_REPORT_CN.md"
    report_path.write_text(
        chinese_report(config=config, rows=rows, aggregate=aggregate),
        encoding="utf-8",
    )
    output_hashes = {
        str(path.relative_to(output_root)): sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "execution_record.json"
    }
    execution = {
        "schema_version": "proxygap-local-preview-final-paired-direct-goal-v1",
        "status": "complete",
        "configuration": str(config_path),
        "configuration_sha256": sha256(config_path),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "fixed_map_configuration": str(
            (ROOT / config["fixed_map_configuration"]).resolve()
        ),
        "fixed_map_configuration_sha256": config[
            "fixed_map_configuration_sha256"
        ],
        "approved_height_sha256": fixed["approved_map"]["heights_sha256"],
        "approved_xml_sha256": fixed["approved_map"]["xml_sha256"],
        "compiled_scene_sha256": sha256(scene),
        "compiled_floor_friction": floor_friction,
        "compiled_floor_condim": int(compiled.geom_condim[floor_id]),
        "spawn_metadata": spawn_metadata,
        "candidates": candidate_records,
        "evaluation_seeds": seeds,
        "episodes_completed": len(rows),
        "wall_seconds": time.time() - started,
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "output_sha256": output_hashes,
        "representative_video": config["representative_video"],
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output_root / "execution_record.json", execution)
    print(str(output_root), flush=True)


if __name__ == "__main__":
    main()
