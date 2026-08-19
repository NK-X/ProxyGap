"""Engineering evaluation of three V4+PAIR0 full-map route objectives.

The route objectives differ only in route cost and speed schedule.  Arrival
and whole-episode safety are hard constraints shared by all three regimes.
Mechanical-work quantities are measurements, not battery-energy claims.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_map_waypoint_route as route_eval  # noqa: E402
import evaluate_post_seal_full_map_v1 as full_map  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/v4_pair0_multiobjective_routes_engineering_v1_20260819"
PAIR0_SCENE = ROOT / "artifacts/dev/post_seal_full_map_eval_v1_20260819/attempt_0/task_scenes/spawn_0_0.000_pair0.xml"
V4_CHECKPOINT = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"

REGIMES = (
    {"id": "time_priority", "weights": [0.8, 0.2], "slope_weight": 0.25, "turn_weight": 0.35, "speed": 0.50, "minimum_speed": 0.30},
    {"id": "balanced", "weights": [0.5, 0.5], "slope_weight": 1.00, "turn_weight": 1.00, "speed": 0.45, "minimum_speed": 0.28},
    {"id": "energy_priority", "weights": [0.2, 0.8], "slope_weight": 3.00, "turn_weight": 1.50, "speed": 0.38, "minimum_speed": 0.25},
)


class V4RoutePolicy:
    def __init__(self) -> None:
        self.model = PPO.load(V4_CHECKPOINT, device="cpu")
        self.model.policy.set_training_mode(False)
        self.num_timesteps = int(self.model.num_timesteps)
        self.observation_space = Box(-np.inf, np.inf, shape=(135,), dtype=np.float32)
        self.action_space = self.model.action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        vector = np.asarray(observation, dtype=np.float32)
        return self.model.predict(vector[:118], deterministic=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def longest_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def evaluate_route(
    *,
    canonical_config: dict[str, Any],
    fixed: dict[str, Any],
    route: route_eval.Polyline,
    regime: dict[str, Any],
    heights: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    condition = full_map._make_condition(canonical_config, fixed)
    condition["task_adapter"].update(
        {
            "maximum_abs_curvature_per_m": 0.2,
            "yaw_gain_per_second": 0.75,
            "slow_radius_m": 4.0,
        }
    )
    policy_config = json.loads((ROOT / canonical_config["source"]["policy_configuration"]).read_text(encoding="utf-8"))
    env = full_map.fixed_task.make_task_env(
        condition,
        policy_config,
        xml_path=PAIR0_SCENE,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=int(canonical_config["evaluation"]["horizon_control_steps"]),
        cruise_speed=float(regime["speed"]),
        terminate_on_success=False,
    )
    model = V4RoutePolicy()
    observation, _ = env.reset(seed=seed)
    audit_state = l2.install_five_substep_audit(env)
    slip_settings = canonical_config["duration_corrected_slip"]
    tracker = l2.DurationCorrectedSlipTracker(
        dt=0.01,
        speed_threshold=float(slip_settings["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip_settings["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip_settings["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip_settings["minimum_sustained_seconds"]),
    )
    final_goal = np.asarray(fixed["approved_map"]["goal_xy_m"], dtype=np.float64)
    half_extent = float(fixed["approved_map"]["map_half_extent_m"])
    required_hold = int(math.ceil(float(canonical_config["independent_success"]["hold_seconds"]) / float(env.unwrapped.dt)))
    arrival = full_map.direct.ArrivalDwellTracker(
        arrival_radius_m=float(canonical_config["independent_success"]["arrival_radius_m"]),
        hold_radius_m=float(canonical_config["independent_success"]["hold_radius_m"]),
        required_hold_steps=required_hold,
    )
    contacts_rows: list[np.ndarray] = []
    force_supported: list[bool] = []
    nonfoot: list[bool] = []
    torso: list[bool] = []
    full_zero: list[bool] = []
    control_trace: list[dict[str, Any]] = []
    substep_trace: list[dict[str, Any]] = []
    progress = 0.0
    lookahead = 3.0
    previous_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
    initial_distance = float(np.linalg.norm(final_goal - previous_xy))
    minimum_distance = initial_distance
    path_length = 0.0
    terminated = truncated = False
    finite = True
    reason = "horizon"
    for step in range(1, int(canonical_config["evaluation"]["horizon_control_steps"]) + 1):
        position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
        progress, cross_track = route.project(position, progress, search_ahead_m=15.0)
        target_distance = min(route.total_length, progress + lookahead)
        env.goal_xy = route.point_at(target_distance)
        gx_now = float(route_eval.terrain_values(gradient_x, position[0], position[1], half_extent))
        gy_now = float(route_eval.terrain_values(gradient_y, position[0], position[1], half_extent))
        gx_target = float(route_eval.terrain_values(gradient_x, env.goal_xy[0], env.goal_xy[1], half_extent))
        gy_target = float(route_eval.terrain_values(gradient_y, env.goal_xy[0], env.goal_xy[1], half_extent))
        local_slope_deg = math.degrees(math.atan(max(math.hypot(gx_now, gy_now), math.hypot(gx_target, gy_target))))
        fraction = float(np.clip((local_slope_deg - 6.0) / 10.0, 0.0, 1.0))
        scheduled_speed = float(regime["speed"] + fraction * (regime["minimum_speed"] - regime["speed"]))
        env.set_task_speed(scheduled_speed)
        env.slow_radius = 4.0 if target_distance >= route.total_length - 1e-9 else env.arrival_radius
        observation = env._command_observation(np.asarray(observation)[:122])
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        substeps = audit_state.get("last")
        if not isinstance(substeps, list) or len(substeps) != 5:
            raise RuntimeError("Five-substep audit missing")
        interval_contacts = []
        for subindex, item in enumerate(substeps, start=1):
            contacts = np.asarray(item["contacts"], dtype=bool)
            speeds = np.asarray(item["speeds"], dtype=np.float64)
            forces = np.asarray(item["forces"], dtype=np.float64)
            raw, qualified = tracker.update(contact_mask=contacts, tangential_speeds=speeds, normal_forces=forces)
            contacts_rows.append(contacts.copy())
            supported = bool(np.any(contacts & (forces >= float(slip_settings["minimum_normal_force_n"]))))
            force_supported.append(supported)
            nonfoot.append(bool(item["nonfoot"]))
            torso.append(bool(item["torso"]))
            interval_contacts.append(contacts.copy())
            substep_trace.append({
                "control_step": step, "physics_substep": subindex,
                "contact_mask": json.dumps(contacts.astype(int).tolist()),
                "normal_forces_n": json.dumps(forces.tolist()),
                "tangential_speeds_m_per_s": json.dumps(speeds.tolist()),
                "force_qualified_supported": int(supported),
                "raw_slip_any": int(np.any(raw)), "qualified_slip_any": int(np.any(qualified)),
                "nonfoot_ground": int(bool(item["nonfoot"])), "torso_ground": int(bool(item["torso"])),
            })
        interval_matrix = np.asarray(interval_contacts, dtype=bool)
        full_zero.append(not bool(np.any(interval_matrix)))
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
        xy = qpos[:2].copy()
        path_length += float(np.linalg.norm(xy - previous_xy))
        previous_xy = xy
        distance = float(np.linalg.norm(final_goal - xy))
        minimum_distance = min(minimum_distance, distance)
        finite_step = bool(np.all(np.isfinite(observation)) and np.all(np.isfinite(action)) and np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)) and math.isfinite(float(reward)))
        finite = finite and finite_step
        arrival.update(step=step, distance_m=distance, stable=False)
        control_trace.append({
            "control_step": step, "time_seconds": step * float(env.unwrapped.dt),
            "x_m": float(xy[0]), "y_m": float(xy[1]), "goal_distance_m": distance,
            "route_progress_m": progress, "route_cross_track_m": cross_track,
            "target_x_m": float(env.goal_xy[0]), "target_y_m": float(env.goal_xy[1]),
            "scheduled_speed_m_per_s": scheduled_speed, "local_planning_slope_degrees": local_slope_deg,
            "full_control_interval_zero_foot": int(full_zero[-1]),
            "spatial_hold_run_steps": arrival.hold_run_steps,
            "spatial_hold_success": int(arrival.spatial_success),
            "action": json.dumps(np.asarray(action).tolist()),
        })
        if not finite_step:
            reason = "nonfinite"
            break
        if arrival.spatial_success:
            reason = "arrival_and_two_second_spatial_hold"
            break
        if terminated:
            reason = "environment_terminated"
            break
        if truncated:
            reason = "horizon_truncated"
            break
    corrected = tracker.finalise()
    sustained = np.asarray(corrected["sustained"], dtype=bool)
    candidate = np.asarray(corrected["candidate"], dtype=bool)
    contacts = np.asarray(contacts_rows, dtype=bool)
    summary = env.env.episode_summary()
    env.close()
    slip_events = len(corrected["events"])
    fall = bool(summary.get("fall", False) or summary.get("inner_absolute_z_fall", False))
    safe = bool(finite and not fall and not any(torso) and longest_run(nonfoot) * 0.01 < 0.2 and slip_events == 0)
    completion = bool(arrival.spatial_success and safe)
    energy = {
        "cumulative_squared_action": float(summary.get("cumulative_squared_action", 0.0)),
        "actuator_abs_torque_time_integral_total_n_m_s": float(np.sum(summary.get("actuator_abs_torque_time_integral_n_m_s_by_actuator", []))),
        "actuator_positive_mechanical_work_total_j": float(np.sum(summary.get("actuator_positive_mechanical_work_j_by_actuator", []))),
        "actuator_abs_mechanical_work_total_j": float(np.sum(summary.get("actuator_abs_mechanical_work_j_by_actuator", []))),
    }
    result = {
        "regime": regime["id"], "weights_time_energy": regime["weights"],
        "termination_reason": reason, "control_steps": len(control_trace),
        "elapsed_seconds": len(control_trace) * float(condition["task_adapter"].get("environment_dt_seconds", 0.05)),
        "arrival_entered": arrival.goal_entered, "spatial_hold_success": arrival.spatial_success,
        "safety_qualified_completion": completion, "finite": finite, "fall": fall,
        "torso_ground_any": bool(any(torso)),
        "sustained_nonfoot_contact": longest_run(nonfoot) * 0.01 >= 0.2,
        "duration_corrected_slip_event_count": slip_events,
        "duration_corrected_sustained_slip_substep_count": int(np.sum(np.any(sustained, axis=1))),
        "qualified_slip_candidate_substep_count": int(np.sum(np.any(candidate, axis=1))),
        "full_control_zero_foot_fraction": float(np.mean(full_zero)),
        "zero_foot_substep_fraction": float(np.mean(~np.any(contacts, axis=1))),
        "mean_support_count": float(np.mean(np.sum(contacts, axis=1))),
        "force_qualified_supported_substep_count": int(np.sum(force_supported)),
        "initial_goal_distance_m": initial_distance, "minimum_goal_distance_m": minimum_distance,
        "final_goal_distance_m": float(control_trace[-1]["goal_distance_m"]),
        "path_length_m": path_length, "route_progress_m": progress,
        **energy,
        "energy_status": "measurement_only_not_electrical_battery_energy",
    }
    return result, control_trace, substep_trace


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    heights = np.load(ROOT / fixed["approved_map"]["heights_path"], allow_pickle=False)
    spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    seed = int(config["evaluation"]["formal_seed"])
    results = []
    for regime in REGIMES:
        settings = json.loads(route_eval.DEFAULT_CONFIG.read_text(encoding="utf-8"))["route_reconstruction"]
        settings.update({
            "maximum_abs_directional_slope_degrees": 16.0,
            "slope_cost_weight": regime["slope_weight"],
            "turn_cost_m_per_rad": regime["turn_weight"],
            "start_heading_degrees": 45.0,
            "line_of_sight_max_span_m": 8.0,
        })
        planner = route_eval.RouteReconstructor(heights, half_extent=float(fixed["approved_map"]["map_half_extent_m"]), settings=settings)
        points, route_metrics = planner.plan(np.asarray(fixed["approved_map"]["start_xy_m"]), np.asarray(fixed["approved_map"]["goal_xy_m"]))
        route = route_eval.Polyline(points)
        result, controls, substeps = evaluate_route(
            canonical_config=config, fixed=fixed, route=route, regime=regime,
            heights=heights, gradient_x=gradient_x, gradient_y=gradient_y, seed=seed,
        )
        result["route_metrics"] = route_metrics
        results.append(result)
        root = OUTPUT / regime["id"]
        root.mkdir(parents=True, exist_ok=False)
        write_csv(root / "route_waypoints.csv", [{"index": i, "x_m": p[0], "y_m": p[1]} for i, p in enumerate(points)])
        write_csv(root / "control_trace.csv", controls)
        write_csv(root / "substep_trace.csv", substeps)
        (root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    (OUTPUT / "summary.json").write_text(json.dumps({"status": "engineering_complete", "results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
