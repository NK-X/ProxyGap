"""Exploratory fixed-map waypoint-route evaluation without policy training.

The script reconstructs a 16-degree slope-constrained polyline from the frozen
heightfield, then changes only the outer task wrapper's ``goal_xy`` to a
look-ahead point.  The locomotion policy, reward, energy instrumentation,
termination implementation, robot XML, terrain and friction remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.metrics import quaternion_tilt_angle  # noqa: E402
from proxygap.planar_transition import quaternion_yaw_angle  # noqa: E402
from run_fixed_goal_terrain_training import (  # noqa: E402
    make_task_env,
    prepare_task_scenes,
)


DEFAULT_CONFIG = ROOT / "configs" / "fixed_map_waypoint_route_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("route-only", "smoke", "screen", "paired"),
        default="smoke",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def terrain_values(
    values: np.ndarray,
    x: np.ndarray | float,
    y: np.ndarray | float,
    half_extent: float,
) -> np.ndarray:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    rows, cols = values.shape
    col_f = np.clip((xs + half_extent) / (2.0 * half_extent) * (cols - 1), 0, cols - 1)
    row_f = np.clip((ys + half_extent) / (2.0 * half_extent) * (rows - 1), 0, rows - 1)
    col0 = np.minimum(np.floor(col_f).astype(np.int64), cols - 2)
    row0 = np.minimum(np.floor(row_f).astype(np.int64), rows - 2)
    tx = col_f - col0
    ty = row_f - row0
    return (1.0 - ty) * (
        (1.0 - tx) * values[row0, col0] + tx * values[row0, col0 + 1]
    ) + ty * (
        (1.0 - tx) * values[row0 + 1, col0] + tx * values[row0 + 1, col0 + 1]
    )


class RouteReconstructor:
    """Heading-state A* with slope-checked finite-width motion edges."""

    DIRECTIONS = np.asarray(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
        ],
        dtype=np.int64,
    )

    def __init__(
        self,
        heights: np.ndarray,
        *,
        half_extent: float,
        settings: dict[str, Any],
    ) -> None:
        self.heights = np.asarray(heights, dtype=np.float64)
        self.half_extent = float(half_extent)
        self.settings = settings
        spacing = 2.0 * self.half_extent / (self.heights.shape[0] - 1)
        self.dz_dy, self.dz_dx = np.gradient(self.heights, spacing, spacing)
        self.grid_spacing = float(settings["grid_spacing_m"])
        self.boundary = float(settings["grid_boundary_m"])
        count = int(round(2.0 * self.boundary / self.grid_spacing)) + 1
        self.coordinates = np.linspace(-self.boundary, self.boundary, count)
        self.max_slope_rad = math.radians(
            float(settings["maximum_abs_directional_slope_degrees"])
        )
        self.corridor_offsets = np.asarray(
            settings["corridor_offsets_m"], dtype=np.float64
        )
        self.sample_spacing = float(settings["edge_sample_spacing_m"])
        self.slope_cost_weight = float(settings["slope_cost_weight"])
        self.turn_cost = float(settings["turn_cost_m_per_rad"])
        self._edge_cache: dict[tuple[int, int, int, int], tuple[bool, float, float]] = {}

    def _index(self, point: np.ndarray) -> tuple[int, int]:
        indices = np.rint((point + self.boundary) / self.grid_spacing).astype(int)
        reconstructed = -self.boundary + indices * self.grid_spacing
        if not np.allclose(point, reconstructed, atol=1e-9, rtol=0.0):
            raise ValueError(f"Route endpoint {point.tolist()} is not on the planning grid")
        return int(indices[0]), int(indices[1])

    def _point(self, ix: int, iy: int) -> np.ndarray:
        return np.asarray([self.coordinates[ix], self.coordinates[iy]], dtype=np.float64)

    def _segment_metrics(self, p0: np.ndarray, p1: np.ndarray) -> tuple[bool, float, float]:
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            return False, float("inf"), float("inf")
        direction = delta / length
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        sample_count = max(2, int(math.ceil(length / self.sample_spacing)) + 1)
        along = np.linspace(0.0, 1.0, sample_count)
        centres = p0[None, :] + along[:, None] * delta[None, :]
        samples = centres[:, None, :] + self.corridor_offsets[None, :, None] * normal[None, None, :]
        if np.any(np.abs(samples) > self.boundary + 1e-12):
            return False, float("inf"), float("inf")
        gx = terrain_values(
            self.dz_dx,
            samples[..., 0],
            samples[..., 1],
            self.half_extent,
        )
        gy = terrain_values(
            self.dz_dy,
            samples[..., 0],
            samples[..., 1],
            self.half_extent,
        )
        slopes = np.arctan(np.hypot(gx, gy))
        maximum = float(np.max(slopes))
        mean = float(np.mean(slopes))
        return maximum <= self.max_slope_rad + 1e-12, maximum, mean

    def _edge_metrics(self, ix: int, iy: int, nx: int, ny: int) -> tuple[bool, float, float]:
        key = (ix, iy, nx, ny) if (ix, iy) <= (nx, ny) else (nx, ny, ix, iy)
        if key not in self._edge_cache:
            self._edge_cache[key] = self._segment_metrics(
                self._point(ix, iy), self._point(nx, ny)
            )
        return self._edge_cache[key]

    @staticmethod
    def _direction_angle(direction_index: int) -> float:
        dx, dy = RouteReconstructor.DIRECTIONS[direction_index]
        return math.atan2(float(dy), float(dx))

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        start_xy = np.asarray(start, dtype=np.float64)
        goal_xy = np.asarray(goal, dtype=np.float64)
        start_ix, start_iy = self._index(start_xy)
        goal_ix, goal_iy = self._index(goal_xy)
        initial_heading = math.radians(float(self.settings["start_heading_degrees"]))
        start_direction = min(
            range(len(self.DIRECTIONS)),
            key=lambda index: abs(wrapped_angle(self._direction_angle(index) - initial_heading)),
        )
        start_state = (start_ix, start_iy, start_direction)
        queue: list[tuple[float, float, tuple[int, int, int]]] = []
        heapq.heappush(queue, (float(np.linalg.norm(goal_xy - start_xy)), 0.0, start_state))
        best: dict[tuple[int, int, int], float] = {start_state: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        goal_state: tuple[int, int, int] | None = None
        expanded = 0

        while queue:
            _, cost, state = heapq.heappop(queue)
            if cost > best.get(state, float("inf")) + 1e-12:
                continue
            ix, iy, previous_direction = state
            expanded += 1
            if ix == goal_ix and iy == goal_iy:
                goal_state = state
                break
            for direction_index, (dx, dy) in enumerate(self.DIRECTIONS):
                nx = ix + int(dx)
                ny = iy + int(dy)
                if not (0 <= nx < len(self.coordinates) and 0 <= ny < len(self.coordinates)):
                    continue
                valid, _, mean_slope = self._edge_metrics(ix, iy, nx, ny)
                if not valid:
                    continue
                edge_length = self.grid_spacing * math.hypot(float(dx), float(dy))
                slope_fraction = mean_slope / max(self.max_slope_rad, 1e-12)
                turn_angle = abs(
                    wrapped_angle(
                        self._direction_angle(direction_index)
                        - self._direction_angle(previous_direction)
                    )
                )
                edge_cost = edge_length * (
                    1.0 + self.slope_cost_weight * slope_fraction * slope_fraction
                ) + self.turn_cost * turn_angle
                candidate = cost + edge_cost
                next_state = (nx, ny, direction_index)
                if candidate + 1e-12 >= best.get(next_state, float("inf")):
                    continue
                best[next_state] = candidate
                parent[next_state] = state
                heuristic = math.hypot(
                    float(goal_ix - nx) * self.grid_spacing,
                    float(goal_iy - ny) * self.grid_spacing,
                )
                heapq.heappush(queue, (candidate + heuristic, candidate, next_state))

        if goal_state is None:
            raise RuntimeError("No route satisfies the frozen 16-degree corridor constraint")
        states = [goal_state]
        while states[-1] != start_state:
            states.append(parent[states[-1]])
        states.reverse()
        raw = np.asarray([self._point(ix, iy) for ix, iy, _ in states], dtype=np.float64)
        simplified = self._simplify(raw)
        waypoints = self._resample(
            simplified, float(self.settings["output_waypoint_spacing_m"])
        )
        metrics = self.metrics(waypoints)
        metrics.update(
            {
                "expanded_heading_states": expanded,
                "cached_edges": len(self._edge_cache),
                "raw_grid_point_count": len(raw),
                "simplified_vertex_count": len(simplified),
                "waypoint_count": len(waypoints),
                "initial_grid_direction_degrees": math.degrees(
                    self._direction_angle(start_direction)
                ),
            }
        )
        return waypoints, metrics

    def _simplify(self, points: np.ndarray) -> np.ndarray:
        maximum_span = float(self.settings["line_of_sight_max_span_m"])
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            furthest = index + 1
            for candidate in range(index + 2, len(points)):
                if float(np.linalg.norm(points[candidate] - points[index])) > maximum_span:
                    break
                valid, _, _ = self._segment_metrics(points[index], points[candidate])
                if valid:
                    furthest = candidate
            result.append(points[furthest])
            index = furthest
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        total = float(cumulative[-1])
        distances = np.arange(0.0, total, spacing)
        if not len(distances) or distances[-1] < total - 1e-9:
            distances = np.append(distances, total)
        result = np.empty((len(distances), 2), dtype=np.float64)
        segment_index = 0
        for index, distance in enumerate(distances):
            while (
                segment_index < len(segments) - 1
                and cumulative[segment_index + 1] < distance
            ):
                segment_index += 1
            denominator = max(segments[segment_index], 1e-12)
            fraction = (distance - cumulative[segment_index]) / denominator
            result[index] = points[segment_index] + fraction * (
                points[segment_index + 1] - points[segment_index]
            )
        result[0] = points[0]
        result[-1] = points[-1]
        return result

    def metrics(self, points: np.ndarray) -> dict[str, Any]:
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        heights = terrain_values(
            self.heights, points[:, 0], points[:, 1], self.half_extent
        )
        changes = np.diff(heights)
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        turns = [
            abs(wrapped_angle(float(right - left)))
            for left, right in zip(headings[:-1], headings[1:])
        ]
        maximum_slope = 0.0
        for left, right in zip(points[:-1], points[1:]):
            valid, slope, _ = self._segment_metrics(left, right)
            if not valid:
                raise RuntimeError("Resampled route violates its own slope constraint")
            maximum_slope = max(maximum_slope, slope)
        return {
            "path_length_m": float(np.sum(lengths)),
            "cumulative_ascent_m": float(np.sum(np.maximum(changes, 0.0))),
            "cumulative_descent_m": float(np.sum(np.maximum(-changes, 0.0))),
            "maximum_corridor_slope_degrees": math.degrees(maximum_slope),
            "cumulative_turning_degrees": math.degrees(float(np.sum(turns))),
            "first_segment_heading_degrees": math.degrees(float(headings[0])),
        }


class Polyline:
    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.segments = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        if np.any(self.lengths <= 0.0):
            raise ValueError("Polyline contains a zero-length segment")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.total_length = float(self.cumulative[-1])

    def point_at(self, distance: float) -> np.ndarray:
        value = float(np.clip(distance, 0.0, self.total_length))
        index = min(
            int(np.searchsorted(self.cumulative, value, side="right") - 1),
            len(self.lengths) - 1,
        )
        fraction = (value - self.cumulative[index]) / self.lengths[index]
        return self.points[index] + fraction * self.segments[index]

    def project(
        self,
        position: np.ndarray,
        previous_distance: float,
        search_ahead_m: float,
    ) -> tuple[float, float]:
        start_index = max(
            0,
            int(np.searchsorted(self.cumulative, max(0.0, previous_distance - 1.0))) - 1,
        )
        maximum_distance = min(self.total_length, previous_distance + search_ahead_m)
        end_index = min(
            len(self.lengths) - 1,
            int(np.searchsorted(self.cumulative, maximum_distance, side="right")),
        )
        best_distance = previous_distance
        best_error = float("inf")
        for index in range(start_index, end_index + 1):
            segment = self.segments[index]
            fraction = float(
                np.clip(
                    np.dot(position - self.points[index], segment)
                    / (self.lengths[index] ** 2),
                    0.0,
                    1.0,
                )
            )
            projected = self.points[index] + fraction * segment
            error = float(np.linalg.norm(position - projected))
            distance = float(self.cumulative[index] + fraction * self.lengths[index])
            if distance + 1e-9 < previous_distance:
                continue
            if error < best_error:
                best_error = error
                best_distance = distance
        if not np.isfinite(best_error):
            projected = self.point_at(previous_distance)
            best_error = float(np.linalg.norm(position - projected))
        return best_distance, best_error


def set_initial_heading(
    env: Any,
    *,
    heading_degrees: float | None,
    heights: np.ndarray,
    half_extent: float,
) -> dict[str, Any]:
    qpos = np.asarray(env.unwrapped.init_qpos, dtype=np.float64).copy()
    original_yaw = quaternion_yaw_angle(qpos[3:7])
    if heading_degrees is None:
        return {
            "initial_heading_mode": "scene_default",
            "scene_init_yaw_degrees": math.degrees(original_yaw),
            "target_init_yaw_degrees": None,
        }
    spacing = 2.0 * half_extent / (heights.shape[0] - 1)
    dz_dy, dz_dx = np.gradient(heights, spacing, spacing)
    x, y = float(qpos[0]), float(qpos[1])
    gx = float(terrain_values(dz_dx, x, y, half_extent))
    gy = float(terrain_values(dz_dy, x, y, half_extent))
    heading = math.radians(float(heading_degrees))
    planar_forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    forward = np.asarray(
        [planar_forward[0], planar_forward[1], gx * planar_forward[0] + gy * planar_forward[1]],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    normal = np.asarray([-gx, -gy, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, normal)
    forward /= np.linalg.norm(forward)
    rotation = np.column_stack((forward, left, normal))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    qpos[3:7] = quaternion
    env.unwrapped.init_qpos[:] = qpos
    return {
        "initial_heading_mode": "terrain_tangent_override_before_seeded_reset",
        "scene_init_yaw_degrees": math.degrees(original_yaw),
        "target_init_yaw_degrees": float(heading_degrees),
        "target_init_quaternion_wxyz": quaternion.tolist(),
        "start_gradient_xy": [gx, gy],
    }


def verify_dynamic_goal_rewrite(env: Any, observation: np.ndarray) -> dict[str, Any]:
    position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
    original_goal = np.asarray(env.goal_xy, dtype=np.float64).copy()
    original_slow_radius = float(env.slow_radius)
    alternative_goal = position + np.asarray([-3.0, 2.0], dtype=np.float64)
    env.goal_xy = alternative_goal
    alternative = env._command_observation(observation)  # Intentional adapter-level probe.
    env.goal_xy = original_goal
    env.slow_radius = original_slow_radius
    restored = env._command_observation(alternative)
    command_slice = slice(113, 118)
    command_changed = bool(
        not np.allclose(
            np.asarray(alternative)[command_slice],
            np.asarray(restored)[command_slice],
            atol=1e-10,
            rtol=0.0,
        )
    )
    if restored.shape != observation.shape or not command_changed:
        raise RuntimeError("Dynamic goal observation rewrite probe failed")
    observation[:] = restored
    return {
        "verified": True,
        "observation_shape": list(restored.shape),
        "alternative_curve_command": np.asarray(alternative)[command_slice].tolist(),
        "restored_curve_command": np.asarray(restored)[command_slice].tolist(),
    }


def evaluate_episode(
    *,
    fixed_config: dict[str, Any],
    policy_config: dict[str, Any],
    model: PPO,
    scene: Path,
    heights: np.ndarray,
    route: Polyline,
    condition: dict[str, Any],
    seed: int,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    condition_config = copy.deepcopy(fixed_config)
    task = condition_config["task_adapter"]
    task["maximum_abs_curvature_per_m"] = float(
        condition["maximum_abs_curvature_per_m"]
    )
    task["yaw_gain_per_second"] = float(condition["yaw_gain_per_second"])
    task["additional_task_reward"] = 0.0
    evaluation = settings["evaluation"]
    horizon_steps = int(evaluation["horizon_steps"])
    speed = float(condition["cruise_speed_m_per_s"])
    env = make_task_env(
        condition_config,
        policy_config,
        xml_path=scene,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=horizon_steps,
        cruise_speed=speed,
        terminate_on_success=False,
    )
    if env.terminate_on_success:
        raise RuntimeError("Route evaluation requires terminate_on_success=False")
    approved = condition_config["approved_map"]
    start = np.asarray(approved["start_xy_m"], dtype=np.float64)
    final_goal = np.asarray(approved["goal_xy_m"], dtype=np.float64)
    half_extent = float(approved["map_half_extent_m"])
    initial_audit = set_initial_heading(
        env,
        heading_degrees=condition["initial_heading_degrees"],
        heights=heights,
        half_extent=half_extent,
    )
    route_mode = str(condition["route_mode"])
    lookahead = condition.get("lookahead_m")
    if route_mode == "waypoint_route":
        if lookahead is None or float(lookahead) <= 0.0:
            raise ValueError("Waypoint route requires positive lookahead_m")
        env.goal_xy = route.point_at(float(lookahead))
        env.slow_radius = env.arrival_radius
    elif route_mode == "direct_goal":
        env.goal_xy = final_goal.copy()
    else:
        raise ValueError(f"Unknown route_mode: {route_mode}")

    try:
        observation, _ = env.reset(seed=seed)
        if observation.shape != model.observation_space.shape:
            raise ValueError(
                f"observation mismatch: environment {observation.shape}, "
                f"model {model.observation_space.shape}"
            )
        reset_yaw = quaternion_yaw_angle(
            np.asarray(env.unwrapped.data.qpos[3:7], dtype=np.float64)
        )
        initial_audit["reset_actual_yaw_degrees"] = math.degrees(reset_yaw)
        initial_audit["reset_yaw_error_from_target_degrees"] = (
            wrapped_angle(reset_yaw - math.radians(float(condition["initial_heading_degrees"])))
            * 180.0
            / math.pi
            if condition["initial_heading_degrees"] is not None
            else None
        )
        dynamic_goal_audit = (
            verify_dynamic_goal_rewrite(env, observation)
            if route_mode == "waypoint_route"
            else {"verified": False, "reason": "direct_goal_does_not_rewrite_goal"}
        )

        hold_required = max(
            1,
            int(math.ceil(float(evaluation["hold_seconds"]) / float(env.unwrapped.dt))),
        )
        arrival_radius = float(evaluation["arrival_radius_m"])
        hold_radius = float(evaluation["hold_radius_m"])
        dwell = evaluation["stable_dwell"]
        slip_threshold = float(dwell["maximum_contact_tangential_speed_m_per_s"])
        trace_stride = int(evaluation["trace_stride_steps"])
        if trace_stride <= 0:
            raise ValueError("trace_stride_steps must be positive")

        progress = 0.0
        cross_track_squared_sum = 0.0
        maximum_cross_track = 0.0
        travelled = 0.0
        last_position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        minimum_final_distance = float(np.linalg.norm(final_goal - last_position))
        final_goal_entered = False
        spatial_hold_run = 0
        longest_spatial_hold = 0
        spatial_hold_completed = False
        stable_hold_run = 0
        longest_stable_hold = 0
        stable_dwell_completed = False
        success_step: int | None = None
        airborne_steps = 0
        slip_steps = 0
        first_airborne_step: int | None = None
        first_slip_step: int | None = None
        maximum_contact_speed = 0.0
        maximum_tilt = 0.0
        traces: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        completed_steps = 0
        started = time.perf_counter()

        for step_index in range(horizon_steps):
            position_before = np.asarray(
                env.unwrapped.data.qpos[:2], dtype=np.float64
            ).copy()
            if route_mode == "waypoint_route":
                progress, cross_track = route.project(
                    position_before,
                    progress,
                    search_ahead_m=max(15.0, 4.0 * float(lookahead)),
                )
                target_distance = min(route.total_length, progress + float(lookahead))
                env.goal_xy = route.point_at(target_distance)
                env.slow_radius = (
                    float(task["slow_radius_m"])
                    if target_distance >= route.total_length - 1e-9
                    else env.arrival_radius
                )
                observation = env._command_observation(observation)
            else:
                cross_track = float("nan")
                target_distance = route.total_length

            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            completed_steps = step_index + 1
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
            position = qpos[:2].copy()
            travelled += float(np.linalg.norm(position - last_position))
            last_position = position
            final_distance = float(np.linalg.norm(final_goal - position))
            minimum_final_distance = min(minimum_final_distance, final_distance)

            if route_mode == "waypoint_route":
                progress, cross_track = route.project(
                    position,
                    progress,
                    search_ahead_m=max(15.0, 4.0 * float(lookahead)),
                )
                cross_track_squared_sum += cross_track * cross_track
                maximum_cross_track = max(maximum_cross_track, cross_track)

            contact_mask = np.asarray(
                info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool
            )
            if contact_mask.shape != (4,):
                raise ValueError("Expected four foot-contact indicators")
            support_count = int(np.count_nonzero(contact_mask))
            airborne = support_count == 0
            if airborne:
                airborne_steps += 1
                if first_airborne_step is None:
                    first_airborne_step = completed_steps
            contact_speeds = np.asarray(
                info.get(
                    "proxygap_foot_contact_tangential_speeds_m_per_s_step",
                    np.zeros(4),
                ),
                dtype=np.float64,
            )
            active_speeds = contact_speeds[contact_mask]
            step_contact_speed = (
                float(np.max(active_speeds)) if active_speeds.size else 0.0
            )
            maximum_contact_speed = max(maximum_contact_speed, step_contact_speed)
            slip_violation = step_contact_speed > slip_threshold
            if slip_violation:
                slip_steps += 1
                if first_slip_step is None:
                    first_slip_step = completed_steps

            tilt = quaternion_tilt_angle(qpos[3:7])
            maximum_tilt = max(maximum_tilt, tilt)
            planar_speed = float(np.linalg.norm(qvel[:2]))
            root_angular_speed = float(np.linalg.norm(qvel[3:6]))
            if final_distance <= arrival_radius:
                final_goal_entered = True
            if final_goal_entered and final_distance <= hold_radius:
                spatial_hold_run += 1
            else:
                spatial_hold_run = 0
            longest_spatial_hold = max(longest_spatial_hold, spatial_hold_run)
            spatial_hold_completed = spatial_hold_completed or spatial_hold_run >= hold_required

            stable_step = bool(
                final_goal_entered
                and final_distance <= hold_radius
                and not terminated
                and not airborne
                and not slip_violation
                and support_count >= int(dwell["minimum_supporting_feet"])
                and planar_speed <= float(dwell["maximum_planar_speed_m_per_s"])
                and root_angular_speed
                <= float(dwell["maximum_root_angular_speed_rad_per_s"])
                and tilt <= math.radians(float(dwell["maximum_torso_tilt_degrees"]))
            )
            if stable_step:
                stable_hold_run += 1
            else:
                stable_hold_run = 0
            longest_stable_hold = max(longest_stable_hold, stable_hold_run)
            if not stable_dwell_completed and stable_hold_run >= hold_required:
                stable_dwell_completed = True
                success_step = completed_steps

            if (
                completed_steps == 1
                or completed_steps % trace_stride == 0
                or final_distance <= hold_radius + 1.0
                or terminated
                or truncated
            ):
                traces.append(
                    {
                        "condition_id": condition["condition_id"],
                        "evaluation_seed": seed,
                        "step": completed_steps,
                        "time_seconds": completed_steps * float(env.unwrapped.dt),
                        "x_m": float(position[0]),
                        "y_m": float(position[1]),
                        "yaw_degrees": math.degrees(quaternion_yaw_angle(qpos[3:7])),
                        "target_x_m": float(env.goal_xy[0]),
                        "target_y_m": float(env.goal_xy[1]),
                        "route_progress_m": progress,
                        "route_cross_track_m": cross_track,
                        "final_goal_distance_m": final_distance,
                        "support_count": support_count,
                        "airborne": int(airborne),
                        "contact_speed_exceeded": int(slip_violation),
                        "maximum_contact_speed_m_per_s": step_contact_speed,
                        "planar_speed_m_per_s": planar_speed,
                        "root_angular_speed_rad_per_s": root_angular_speed,
                        "torso_tilt_degrees": math.degrees(tilt),
                        "spatial_hold_run_steps": spatial_hold_run,
                        "stable_hold_run_steps": stable_hold_run,
                        "reward": float(reward),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                    }
                )

            if terminated or truncated:
                break
            if stable_dwell_completed and bool(
                evaluation["stop_harness_after_qualified_dwell"]
            ):
                break

        elapsed_wall = time.perf_counter() - started
        summary = env.episode_summary()
        fall = bool(summary.get("fall", False))
        whole_episode_safety = bool(
            not fall and airborne_steps == 0 and slip_steps == 0
        )
        qualified_completion = bool(
            spatial_hold_completed and stable_dwell_completed and whole_episode_safety
        )
        positive_work = float(
            np.sum(summary.get("actuator_positive_mechanical_work_j_by_actuator", []))
        )
        negative_work = float(
            np.sum(
                summary.get(
                    "actuator_negative_mechanical_work_abs_j_by_actuator", []
                )
            )
        )
        initial_distance = float(np.linalg.norm(final_goal - start))
        row = {
            "condition_id": condition["condition_id"],
            "route_mode": route_mode,
            "initial_heading_target_degrees": condition["initial_heading_degrees"],
            "reset_actual_yaw_degrees": initial_audit["reset_actual_yaw_degrees"],
            "cruise_speed_m_per_s": speed,
            "lookahead_m": lookahead,
            "maximum_abs_curvature_per_m": condition[
                "maximum_abs_curvature_per_m"
            ],
            "yaw_gain_per_second": condition["yaw_gain_per_second"],
            "evaluation_seed": seed,
            "completed_steps": completed_steps,
            "elapsed_simulation_seconds": completed_steps * float(env.unwrapped.dt),
            "elapsed_wall_seconds": elapsed_wall,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "fall": int(fall),
            "termination_category": summary.get("termination_category", "unknown"),
            "final_goal_entered": int(final_goal_entered),
            "spatial_hold_completed": int(spatial_hold_completed),
            "stable_dwell_completed": int(stable_dwell_completed),
            "qualified_completion": int(qualified_completion),
            "success_step": success_step,
            "success_time_seconds": (
                success_step * float(env.unwrapped.dt)
                if success_step is not None
                else None
            ),
            "final_distance_m": float(np.linalg.norm(final_goal - last_position)),
            "minimum_final_distance_m": minimum_final_distance,
            "net_final_goal_progress_m": initial_distance
            - float(np.linalg.norm(final_goal - last_position)),
            "travelled_planar_distance_m": travelled,
            "route_progress_m": progress,
            "route_progress_fraction": progress / route.total_length,
            "route_cross_track_rms_m": (
                math.sqrt(cross_track_squared_sum / max(completed_steps, 1))
                if route_mode == "waypoint_route"
                else None
            ),
            "route_cross_track_max_m": (
                maximum_cross_track if route_mode == "waypoint_route" else None
            ),
            "longest_spatial_hold_steps": longest_spatial_hold,
            "longest_stable_hold_steps": longest_stable_hold,
            "airborne_steps": airborne_steps,
            "airborne_fraction": airborne_steps / max(completed_steps, 1),
            "first_airborne_step": first_airborne_step,
            "contact_speed_exceedance_steps": slip_steps,
            "contact_speed_exceedance_fraction": slip_steps
            / max(completed_steps, 1),
            "first_contact_speed_exceedance_step": first_slip_step,
            "maximum_contact_speed_m_per_s": maximum_contact_speed,
            "maximum_torso_tilt_degrees": math.degrees(maximum_tilt),
            "positive_mechanical_work_j_diagnostic": positive_work,
            "negative_mechanical_work_abs_j_diagnostic": negative_work,
            "absolute_mechanical_work_j_diagnostic": positive_work + negative_work,
            "v2_relative_mission_energy_available": 0,
            "dynamic_goal_rewrite_verified": int(dynamic_goal_audit["verified"]),
        }
        audit = {
            "condition_id": condition["condition_id"],
            "evaluation_seed": seed,
            "initial_heading": initial_audit,
            "dynamic_goal_observation": dynamic_goal_audit,
        }
        return row, traces, audit
    finally:
        env.close()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identifiers = list(dict.fromkeys(str(row["condition_id"]) for row in rows))
    for identifier in identifiers:
        group = [row for row in rows if row["condition_id"] == identifier]
        result.append(
            {
                "condition_id": identifier,
                "episodes": len(group),
                "final_goal_entered_count": sum(int(row["final_goal_entered"]) for row in group),
                "spatial_hold_completed_count": sum(
                    int(row["spatial_hold_completed"]) for row in group
                ),
                "stable_dwell_completed_count": sum(
                    int(row["stable_dwell_completed"]) for row in group
                ),
                "qualified_completion_count": sum(
                    int(row["qualified_completion"]) for row in group
                ),
                "fall_count": sum(int(row["fall"]) for row in group),
                "mean_net_progress_m": float(
                    np.mean([float(row["net_final_goal_progress_m"]) for row in group])
                ),
                "mean_minimum_final_distance_m": float(
                    np.mean([float(row["minimum_final_distance_m"]) for row in group])
                ),
                "mean_route_progress_fraction": float(
                    np.mean([float(row["route_progress_fraction"]) for row in group])
                ),
                "mean_airborne_fraction": float(
                    np.mean([float(row["airborne_fraction"]) for row in group])
                ),
                "mean_contact_speed_exceedance_fraction": float(
                    np.mean(
                        [
                            float(row["contact_speed_exceedance_fraction"])
                            for row in group
                        ]
                    )
                ),
            }
        )
    return result


def render_report(
    phase: str,
    route_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    claim_boundary: str,
) -> str:
    lines = [
        f"# 固定地图16°候选路线探索性评估（{phase}）",
        "",
        "## 结论边界",
        "",
        claim_boundary,
        "",
        "路线是根据封存高度图重新构造的候选，不是对丢失原始 waypoints 的精确复现。",
        "已有机械功只作诊断，不是V2相对任务能耗，也没有进入奖励或路线成本。",
        "",
        "## 路线重建",
        "",
        f"- 长度：{route_metrics['path_length_m']:.3f} m",
        f"- 累计爬升/下降：{route_metrics['cumulative_ascent_m']:.3f}/{route_metrics['cumulative_descent_m']:.3f} m",
        f"- 最大走廊坡度：{route_metrics['maximum_corridor_slope_degrees']:.3f}°",
        f"- 首段航向：{route_metrics['first_segment_heading_degrees']:.3f}°",
        f"- 累计转向：{route_metrics['cumulative_turning_degrees']:.3f}°",
        "",
    ]
    if not rows:
        lines.extend(["本阶段只重建路线，未执行动力学 episode。", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## 条件汇总",
            "",
            "| 条件 | n | 进入终点 | 位置保持 | 稳定dwell | 安全合格 | 摔倒 | 平均净推进(m) | 平均腾空率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        lines.append(
            "| {condition_id} | {episodes} | {final_goal_entered_count} | "
            "{spatial_hold_completed_count} | {stable_dwell_completed_count} | "
            "{qualified_completion_count} | {fall_count} | {mean_net_progress_m:.3f} | "
            "{mean_airborne_fraction:.2%} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- `进入终点`只表示曾进入1.5 m范围。",
            "- `位置保持`表示进入后在2.0 m范围内连续40步，不代表站立稳定。",
            "- `稳定dwell`还要求支撑、低速度、低角速度、健康姿态和无接触速度超限。",
            "- `安全合格`进一步要求全轮无摔倒、无四足同时腾空且无接触速度超限。",
            "- 所有失败和超时均保留在 `episode_rows.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_config(config: dict[str, Any], fixed: dict[str, Any]) -> None:
    if config.get("status") != "exploratory_route_following_not_formal":
        raise ValueError("Route evaluation configuration is not exploratory")
    if fixed["task_adapter"]["additional_task_reward"] != 0.0:
        raise ValueError("Fixed-map route evaluation must not add a task reward")
    if bool(config["evaluation"]["terminate_environment_on_wrapper_success"]):
        raise ValueError("terminate_on_success must remain false")
    identifiers: list[str] = []
    for group_name in ("smoke_conditions", "paired_conditions"):
        for condition in config[group_name]:
            identifiers.append(str(condition["condition_id"]))
            if condition["route_mode"] == "waypoint_route" and condition["lookahead_m"] is None:
                raise ValueError("Waypoint route condition lacks lookahead")
    if len(identifiers) != len(set(identifiers)) + 2:
        # D0 and D1 intentionally repeat between smoke and paired phases.
        raise ValueError("Unexpected duplicate condition identifiers")
    route_conditions = config["paired_conditions"][2:]
    observed = {
        (
            float(item["cruise_speed_m_per_s"]),
            float(item["lookahead_m"]),
            float(item["maximum_abs_curvature_per_m"]),
        )
        for item in route_conditions
    }
    expected = {
        (speed, lookahead, curvature)
        for speed in (0.4, 0.5)
        for lookahead in (3.0, 5.0)
        for curvature in (0.12, 0.2)
    }
    if observed != expected:
        raise ValueError("Paired route conditions do not form the required 2x2x2 matrix")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_path = ROOT / config["fixed_map_config"]
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    validate_config(config, fixed)
    policy = config["policy"]
    model_path = ROOT / policy["model_path"]
    if sha256(model_path) != policy["model_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    policy_path = ROOT / policy["configuration"]
    policy_config = json.loads(policy_path.read_text(encoding="utf-8"))
    approved = fixed["approved_map"]
    heights_path = ROOT / approved["heights_path"]
    if sha256(heights_path) != approved["heights_sha256"]:
        raise ValueError("Height-array SHA-256 mismatch")
    if list(approved["fixed_friction"]) != [1.0, 0.5, 0.5]:
        raise ValueError("Frozen terrain friction changed")
    heights = np.load(heights_path, allow_pickle=False)
    reconstructor = RouteReconstructor(
        heights,
        half_extent=float(approved["map_half_extent_m"]),
        settings=config["route_reconstruction"],
    )
    waypoints, route_metrics = reconstructor.plan(
        np.asarray(approved["start_xy_m"], dtype=np.float64),
        np.asarray(approved["goal_xy_m"], dtype=np.float64),
    )
    route = Polyline(waypoints)

    base_output = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["output_root"]).resolve()
    )
    output_root = base_output / args.phase.replace("-", "_")
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    (output_root / "route").mkdir(parents=True)
    (output_root / "logs").mkdir(parents=True)
    (output_root / "task").mkdir(parents=True)
    (output_root / "frozen_config.json").write_bytes(config_path.read_bytes())
    frozen_script = output_root / "frozen_evaluation_script.py"
    frozen_script.write_bytes(Path(__file__).resolve().read_bytes())
    route_rows = [
        {
            "waypoint_index": index,
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "terrain_height_m": float(
                terrain_values(
                    heights,
                    float(point[0]),
                    float(point[1]),
                    float(approved["map_half_extent_m"]),
                )
            ),
        }
        for index, point in enumerate(waypoints)
    ]
    route_csv = output_root / "route" / "route_waypoints.csv"
    write_csv(route_csv, route_rows)
    route_record = {
        "settings": config["route_reconstruction"],
        "settings_source": str(config_path),
        "height_array_sha256": approved["heights_sha256"],
        "metrics": route_metrics,
        "waypoints_csv": str(route_csv),
        "waypoints_csv_sha256": sha256(route_csv),
        "claim_boundary": config["route_reconstruction"]["claim_boundary"],
    }
    route_json = output_root / "route" / "route_manifest.json"
    write_json(route_json, route_record)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    scene_metadata: list[dict[str, Any]] = []
    if args.phase != "route-only":
        scenes, scene_metadata = prepare_task_scenes(
            fixed, output_root / "task", [0.0]
        )
        model = PPO.load(model_path, device="cpu")
        conditions = (
            list(config["smoke_conditions"])
            if args.phase == "smoke"
            else list(config["paired_conditions"])
        )
        seeds = (
            [int(config["evaluation"]["smoke_seed"])]
            if args.phase in ("smoke", "screen")
            else [int(value) for value in config["evaluation"]["paired_seeds"]]
        )
        for condition in conditions:
            for seed in seeds:
                row, episode_trace, audit = evaluate_episode(
                    fixed_config=fixed,
                    policy_config=policy_config,
                    model=model,
                    scene=scenes[0],
                    heights=heights,
                    route=route,
                    condition=condition,
                    seed=seed,
                    settings=config,
                )
                rows.append(row)
                traces.extend(episode_trace)
                audits.append(audit)
                print(
                    json.dumps(
                        {
                            "condition": condition["condition_id"],
                            "seed": seed,
                            "progress_m": row["net_final_goal_progress_m"],
                            "route_progress_fraction": row["route_progress_fraction"],
                            "fall": row["fall"],
                            "airborne_fraction": row["airborne_fraction"],
                            "spatial_hold": row["spatial_hold_completed"],
                            "stable_dwell": row["stable_dwell_completed"],
                            "qualified": row["qualified_completion"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        write_csv(output_root / "episode_rows.csv", rows)
        write_csv(output_root / "step_traces.csv", traces)
        write_json(output_root / "api_audits.json", audits)

    aggregates = aggregate(rows) if rows else []
    write_json(output_root / "aggregate_summary.json", aggregates)
    report = render_report(
        args.phase,
        route_metrics,
        rows,
        aggregates,
        config["claim_boundary"],
    )
    (output_root / "ROUTE_EVALUATION_REPORT_CN.md").write_text(
        report, encoding="utf-8"
    )
    execution = {
        "schema_version": "proxygap-fixed-map-waypoint-route-v1",
        "status": "complete",
        "phase": args.phase,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "script_path": str(frozen_script),
        "script_sha256": sha256(frozen_script),
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "fixed_map_height_sha256": approved["heights_sha256"],
        "fixed_friction": approved["fixed_friction"],
        "condim": approved["condim"],
        "route_manifest": str(route_json),
        "route_manifest_sha256": sha256(route_json),
        "episode_count": len(rows),
        "all_failures_retained": True,
        "terminate_on_success": False,
        "v2_energy_used_in_reward_or_route_cost": False,
        "scene_metadata": scene_metadata,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output_root / "execution_record.json", execution)
    print(json.dumps(execution, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
