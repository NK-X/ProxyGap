"""Fixed-map start-to-goal task adapter for the contact-aware Ant policy.

The wrapper supplies only the existing local forward/yaw command interface to
the learned policy.  Global position is used by this deterministic high-level
controller and by task diagnostics; it is not appended to the policy
observation.  The wrapped locomotion reward is preserved unchanged.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from .metrics import quaternion_tilt_angle
from .planar_transition import quaternion_yaw_angle, wrapped_angle_difference


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FixedGoalTerrainWrapper(gym.Wrapper):
    """Guide a local locomotion policy towards one fixed global goal.

    No task reward is added.  Training continues to optimise the V22 command
    tracking, stability, contact and action terms while the deterministic
    route controller turns the existing local command towards the goal.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        heights_path: str | Path,
        expected_height_sha256: str,
        map_half_extent_m: float,
        start_xy_m: Sequence[float],
        goal_xy_m: Sequence[float],
        spawn_fraction: float,
        cruise_speed_m_per_s: float,
        maximum_abs_curvature_per_m: float,
        yaw_gain_per_second: float = 1.5,
        slow_radius_m: float = 5.0,
        arrival_radius_m: float = 1.5,
        hold_radius_m: float = 2.0,
        hold_seconds: float = 2.0,
        hold_speed_m_per_s: float = 0.05,
        terminate_on_success: bool = False,
        terrain_relative_healthy_clearance_m: tuple[float, float] = (0.18, 1.40),
        maximum_healthy_tilt_degrees: float = 80.0,
        unhealthy_grace_steps: int = 5,
        slip_speed_threshold_m_per_s: float = 0.20,
    ) -> None:
        super().__init__(env)
        self.heights_path = Path(heights_path).resolve()
        if not self.heights_path.is_file():
            raise FileNotFoundError(self.heights_path)
        observed_hash = file_sha256(self.heights_path)
        if observed_hash.lower() != str(expected_height_sha256).lower():
            raise ValueError("Frozen terrain height-array SHA-256 mismatch")
        self.height_sha256 = observed_hash
        self.heights = np.load(self.heights_path, allow_pickle=False)
        if self.heights.ndim != 2 or min(self.heights.shape) < 2:
            raise ValueError("Terrain heights must be a two-dimensional grid")
        self.map_half_extent_m = self._positive(map_half_extent_m, "map_half_extent_m")
        self.start_xy = self._xy(start_xy_m, "start_xy_m")
        self.goal_xy = self._xy(goal_xy_m, "goal_xy_m")
        if not 0.0 <= float(spawn_fraction) < 1.0:
            raise ValueError("spawn_fraction must lie in [0, 1)")
        self.spawn_fraction = float(spawn_fraction)
        self.cruise_speed = self._positive(cruise_speed_m_per_s, "cruise_speed_m_per_s")
        self.maximum_abs_curvature = self._positive(
            maximum_abs_curvature_per_m,
            "maximum_abs_curvature_per_m",
        )
        self.yaw_gain = self._positive(yaw_gain_per_second, "yaw_gain_per_second")
        self.slow_radius = self._positive(slow_radius_m, "slow_radius_m")
        self.arrival_radius = self._positive(arrival_radius_m, "arrival_radius_m")
        self.hold_radius = self._positive(hold_radius_m, "hold_radius_m")
        if self.hold_radius < self.arrival_radius:
            raise ValueError("hold_radius_m must be at least arrival_radius_m")
        self.hold_seconds = self._positive(hold_seconds, "hold_seconds")
        self.hold_speed = self._positive(hold_speed_m_per_s, "hold_speed_m_per_s")
        self.terminate_on_success = bool(terminate_on_success)
        low_clearance, high_clearance = (
            float(value) for value in terrain_relative_healthy_clearance_m
        )
        if not 0.0 < low_clearance < high_clearance:
            raise ValueError("Invalid terrain-relative healthy clearance range")
        self.healthy_clearance = (low_clearance, high_clearance)
        self.maximum_healthy_tilt = math.radians(
            self._positive(maximum_healthy_tilt_degrees, "maximum_healthy_tilt_degrees")
        )
        if unhealthy_grace_steps <= 0:
            raise ValueError("unhealthy_grace_steps must be positive")
        self.unhealthy_grace_steps = int(unhealthy_grace_steps)
        self.slip_speed_threshold = self._positive(
            slip_speed_threshold_m_per_s,
            "slip_speed_threshold_m_per_s",
        )
        dt = float(self.unwrapped.dt)
        self.required_hold_steps = max(1, int(math.ceil(self.hold_seconds / dt)))
        self._reset_metrics()

    @staticmethod
    def _positive(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{label} must be positive and finite")
        return result

    @staticmethod
    def _xy(value: Sequence[float], label: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain two finite values")
        return result

    def set_task_speed(self, cruise_speed_m_per_s: float) -> None:
        """Update the route-controller speed between frozen training stages."""
        self.cruise_speed = self._positive(
            cruise_speed_m_per_s,
            "cruise_speed_m_per_s",
        )

    def _reset_metrics(self) -> None:
        self._task_steps = 0
        self._initial_distance = float("nan")
        self._previous_distance = float("nan")
        self._minimum_distance = float("inf")
        self._final_distance = float("nan")
        self._cumulative_positive_progress = 0.0
        self._cumulative_regress = 0.0
        self._goal_entered = False
        self._goal_hold_run_steps = 0
        self._longest_goal_hold_run_steps = 0
        self._task_success = False
        self._success_step: int | None = None
        self._terrain_unhealthy_run_steps = 0
        self._terrain_unhealthy_terminated = False
        self._terrain_unhealthy_reason = "none"
        self._minimum_torso_clearance = float("inf")
        self._maximum_torso_clearance = float("-inf")
        self._maximum_torso_tilt = 0.0
        self._lateral_deviation_squared_sum = 0.0
        self._maximum_lateral_deviation = 0.0
        self._airborne_steps = 0
        self._first_airborne_step: int | None = None
        self._slip_violation_steps = 0
        self._first_slip_step: int | None = None
        self._maximum_contact_slip_speed = 0.0
        self._last_position = np.asarray([float("nan"), float("nan")])

    def _terrain_height(self, x: float, y: float) -> float:
        rows, cols = self.heights.shape
        extent = self.map_half_extent_m
        col_f = np.clip((x + extent) / (2.0 * extent) * (cols - 1), 0, cols - 1)
        row_f = np.clip((y + extent) / (2.0 * extent) * (rows - 1), 0, rows - 1)
        col0 = min(int(math.floor(col_f)), cols - 2)
        row0 = min(int(math.floor(row_f)), rows - 2)
        tx = float(col_f - col0)
        ty = float(row_f - row0)
        z00 = float(self.heights[row0, col0])
        z10 = float(self.heights[row0, col0 + 1])
        z01 = float(self.heights[row0 + 1, col0])
        z11 = float(self.heights[row0 + 1, col0 + 1])
        return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * (
            (1.0 - tx) * z01 + tx * z11
        )

    def _position(self) -> np.ndarray:
        return np.asarray(self.unwrapped.data.qpos[:2], dtype=np.float64).copy()

    def _distance_to_goal(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_xy - position))

    def _lateral_deviation(self, position: np.ndarray) -> float:
        direction = self.goal_xy - self.start_xy
        offset = position - self.start_xy
        return float(abs(direction[0] * offset[1] - direction[1] * offset[0]) / np.linalg.norm(direction))

    def _command_observation(self, observation: np.ndarray) -> np.ndarray:
        position = self._position()
        vector = self.goal_xy - position
        distance = float(np.linalg.norm(vector))
        yaw = quaternion_yaw_angle(np.asarray(self.unwrapped.data.qpos[3:7]))
        if distance <= self.arrival_radius:
            target_heading = yaw
            target_speed = self.hold_speed
            yaw_rate = 0.0
        else:
            target_heading = float(math.atan2(vector[1], vector[0]))
            heading_error = wrapped_angle_difference(target_heading, yaw)
            speed_scale = min(1.0, max(self.hold_speed / self.cruise_speed, distance / self.slow_radius))
            target_speed = self.cruise_speed * speed_scale
            maximum_yaw_rate = target_speed * self.maximum_abs_curvature
            yaw_rate = float(
                np.clip(
                    self.yaw_gain * heading_error,
                    -maximum_yaw_rate,
                    maximum_yaw_rate,
                )
            )
        return self.env.set_external_curve_command(
            observation,
            target_heading=target_heading,
            yaw_rate=yaw_rate,
            speed=target_speed,
            lateral_speed=0.0,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._reset_metrics()
        position = self._position()
        distance = self._distance_to_goal(position)
        self._initial_distance = distance
        self._previous_distance = distance
        self._minimum_distance = distance
        self._final_distance = distance
        self._last_position = position
        observation = self._command_observation(observation)
        live_info = dict(info)
        live_info.update(self._live_info())
        return observation, live_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._task_steps += 1
        position = self._position()
        distance = self._distance_to_goal(position)
        delta = self._previous_distance - distance
        self._cumulative_positive_progress += max(0.0, delta)
        self._cumulative_regress += max(0.0, -delta)
        self._previous_distance = distance
        self._minimum_distance = min(self._minimum_distance, distance)
        self._final_distance = distance
        self._last_position = position

        lateral_deviation = self._lateral_deviation(position)
        self._lateral_deviation_squared_sum += lateral_deviation**2
        self._maximum_lateral_deviation = max(
            self._maximum_lateral_deviation,
            lateral_deviation,
        )

        if distance <= self.arrival_radius:
            self._goal_entered = True
        if distance <= self.hold_radius:
            self._goal_hold_run_steps += 1
        else:
            self._goal_hold_run_steps = 0
        self._longest_goal_hold_run_steps = max(
            self._longest_goal_hold_run_steps,
            self._goal_hold_run_steps,
        )
        if (
            not self._task_success
            and self._goal_hold_run_steps >= self.required_hold_steps
        ):
            self._task_success = True
            self._success_step = self._task_steps

        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        terrain_height = self._terrain_height(float(qpos[0]), float(qpos[1]))
        clearance = float(qpos[2] - terrain_height)
        tilt = quaternion_tilt_angle(qpos[3:7])
        self._minimum_torso_clearance = min(self._minimum_torso_clearance, clearance)
        self._maximum_torso_clearance = max(self._maximum_torso_clearance, clearance)
        self._maximum_torso_tilt = max(self._maximum_torso_tilt, tilt)
        state_finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(self.unwrapped.data.qvel)))
        out_of_bounds = bool(np.any(np.abs(position) > self.map_half_extent_m))
        clearance_healthy = self.healthy_clearance[0] <= clearance <= self.healthy_clearance[1]
        tilt_healthy = bool(np.isfinite(tilt) and tilt <= self.maximum_healthy_tilt)
        healthy = state_finite and not out_of_bounds and clearance_healthy and tilt_healthy
        if healthy:
            self._terrain_unhealthy_run_steps = 0
        else:
            self._terrain_unhealthy_run_steps += 1
            if not state_finite:
                self._terrain_unhealthy_reason = "non_finite"
            elif out_of_bounds:
                self._terrain_unhealthy_reason = "out_of_bounds"
            elif not clearance_healthy:
                self._terrain_unhealthy_reason = "terrain_relative_torso_clearance"
            else:
                self._terrain_unhealthy_reason = "torso_tilt"
        if self._terrain_unhealthy_run_steps >= self.unhealthy_grace_steps:
            terminated = True
            self._terrain_unhealthy_terminated = True

        contact_mask = np.asarray(
            info.get("proxygap_foot_contact_mask_step", np.zeros(4)),
            dtype=bool,
        )
        if contact_mask.shape != (4,):
            raise ValueError(
                "proxygap_foot_contact_mask_step must contain four foot states"
            )
        airborne = bool(not np.any(contact_mask))
        if airborne:
            self._airborne_steps += 1
            if self._first_airborne_step is None:
                self._first_airborne_step = self._task_steps
        slip_speeds = np.asarray(
            info.get(
                "proxygap_foot_contact_tangential_speeds_m_per_s_step",
                np.zeros(4),
            ),
            dtype=np.float64,
        )
        if slip_speeds.shape == (4,):
            active_speeds = slip_speeds[contact_mask]
            step_max_slip = float(active_speeds.max()) if active_speeds.size else 0.0
            self._maximum_contact_slip_speed = max(
                self._maximum_contact_slip_speed,
                step_max_slip,
            )
            if step_max_slip > self.slip_speed_threshold:
                self._slip_violation_steps += 1
                if self._first_slip_step is None:
                    self._first_slip_step = self._task_steps

        if self.terminate_on_success and self._task_success:
            terminated = True
        info.update(self._live_info())
        if not (terminated or truncated):
            observation = self._command_observation(observation)
        if terminated or truncated:
            summary = self.episode_summary()
            info.update({f"proxygap_{key}": value for key, value in summary.items()})
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _live_info(self) -> dict[str, Any]:
        return {
            "proxygap_fixed_goal_distance_m": self._final_distance,
            "proxygap_fixed_goal_minimum_distance_m": self._minimum_distance,
            "proxygap_fixed_goal_entered": self._goal_entered,
            "proxygap_fixed_goal_hold_steps": self._goal_hold_run_steps,
            "proxygap_fixed_goal_success": self._task_success,
            "proxygap_fixed_goal_spawn_fraction": self.spawn_fraction,
            "proxygap_fixed_goal_cruise_speed_m_per_s": self.cruise_speed,
            "proxygap_terrain_relative_unhealthy_run_steps": self._terrain_unhealthy_run_steps,
        }

    def episode_summary(self) -> dict[str, Any]:
        summary = dict(self.env.episode_summary())
        elapsed = max(1, self._task_steps)
        inner_termination_category = summary.get("termination_category", "unknown")
        inner_fall = bool(summary.get("fall", False))
        if self._terrain_unhealthy_terminated:
            termination_category = f"terrain_relative_{self._terrain_unhealthy_reason}"
        elif self._task_success and self.terminate_on_success:
            termination_category = "goal_success"
        else:
            termination_category = str(inner_termination_category)
        qualified = bool(
            self._task_success
            and not self._terrain_unhealthy_terminated
            and self._airborne_steps == 0
            and self._slip_violation_steps == 0
        )
        summary.update(
            {
                "fixed_goal_map_height_sha256": self.height_sha256,
                "fixed_goal_spawn_fraction": self.spawn_fraction,
                "fixed_goal_start_xy_m": self.start_xy.tolist(),
                "fixed_goal_goal_xy_m": self.goal_xy.tolist(),
                "fixed_goal_initial_distance_m": self._initial_distance,
                "fixed_goal_final_distance_m": self._final_distance,
                "fixed_goal_minimum_distance_m": self._minimum_distance,
                "fixed_goal_net_progress_m": self._initial_distance - self._final_distance,
                "fixed_goal_cumulative_positive_progress_m": self._cumulative_positive_progress,
                "fixed_goal_cumulative_regress_m": self._cumulative_regress,
                "fixed_goal_entered": self._goal_entered,
                "fixed_goal_hold_required_steps": self.required_hold_steps,
                "fixed_goal_longest_hold_run_steps": self._longest_goal_hold_run_steps,
                "fixed_goal_longest_hold_run_seconds": self._longest_goal_hold_run_steps * float(self.unwrapped.dt),
                "fixed_goal_success": self._task_success,
                "fixed_goal_success_step": self._success_step,
                "fixed_goal_success_time_seconds": (
                    self._success_step * float(self.unwrapped.dt)
                    if self._success_step is not None
                    else None
                ),
                "fixed_goal_qualified_no_fall_no_airborne_no_slip": qualified,
                "fixed_goal_cruise_speed_m_per_s": self.cruise_speed,
                "fixed_goal_route_lateral_deviation_rms_m": math.sqrt(
                    self._lateral_deviation_squared_sum / elapsed
                ),
                "fixed_goal_route_lateral_deviation_max_m": self._maximum_lateral_deviation,
                "terrain_relative_minimum_torso_clearance_m": self._minimum_torso_clearance,
                "terrain_relative_maximum_torso_clearance_m": self._maximum_torso_clearance,
                "terrain_relative_maximum_torso_tilt_rad": self._maximum_torso_tilt,
                "terrain_relative_unhealthy_terminated": self._terrain_unhealthy_terminated,
                "terrain_relative_unhealthy_reason": self._terrain_unhealthy_reason,
                "task_airborne_step_count": self._airborne_steps,
                "task_airborne_step_fraction": self._airborne_steps / elapsed,
                "task_first_airborne_step": self._first_airborne_step,
                "task_first_airborne_time_seconds": (
                    self._first_airborne_step * float(self.unwrapped.dt)
                    if self._first_airborne_step is not None
                    else None
                ),
                "task_slip_speed_threshold_m_per_s": self.slip_speed_threshold,
                "task_slip_violation_step_count": self._slip_violation_steps,
                "task_slip_violation_step_fraction": self._slip_violation_steps / elapsed,
                "task_first_slip_step": self._first_slip_step,
                "task_first_slip_time_seconds": (
                    self._first_slip_step * float(self.unwrapped.dt)
                    if self._first_slip_step is not None
                    else None
                ),
                "task_maximum_contact_slip_speed_m_per_s": self._maximum_contact_slip_speed,
                "inner_absolute_z_termination_category": inner_termination_category,
                "inner_absolute_z_fall": inner_fall,
                "termination_category": termination_category,
                "fall": self._terrain_unhealthy_terminated,
                "unhealthy_termination": self._terrain_unhealthy_terminated,
            }
        )
        return summary
