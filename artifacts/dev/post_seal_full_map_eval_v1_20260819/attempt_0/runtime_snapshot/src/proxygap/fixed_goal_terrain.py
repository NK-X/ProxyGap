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
        yaw_deadband_degrees: float = 0.0,
        curvature_speed_reduction_gain: float = 0.0,
        minimum_turn_speed_fraction: float = 1.0,
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
        augment_local_terrain_observation: bool = False,
        terrain_frame_shaping_enabled: bool = False,
        terrain_preview_longitudinal_m: Sequence[float] = (0.5, 1.0, 1.5),
        terrain_preview_lateral_m: Sequence[float] = (-0.4, 0.0, 0.4),
        local_terrain_height_bound_m: float | None = None,
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
        self.augment_local_terrain_observation = bool(
            augment_local_terrain_observation
        )
        self.terrain_frame_shaping_enabled = bool(
            terrain_frame_shaping_enabled
        )
        self.local_terrain_height_bound = (
            None
            if local_terrain_height_bound_m is None
            else self._positive(
                local_terrain_height_bound_m,
                "local_terrain_height_bound_m",
            )
        )
        self.terrain_preview_longitudinal = self._preview_axis(
            terrain_preview_longitudinal_m,
            "terrain_preview_longitudinal_m",
            require_positive=True,
        )
        self.terrain_preview_lateral = self._preview_axis(
            terrain_preview_lateral_m,
            "terrain_preview_lateral_m",
            require_positive=False,
        )
        spacing = 2.0 * self.map_half_extent_m / (self.heights.shape[0] - 1)
        self._terrain_dz_dy, self._terrain_dz_dx = np.gradient(
            self.heights,
            spacing,
            spacing,
        )
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
        self.yaw_gain = self._non_negative(
            yaw_gain_per_second,
            "yaw_gain_per_second",
        )
        self.yaw_deadband = math.radians(
            self._non_negative(yaw_deadband_degrees, "yaw_deadband_degrees")
        )
        if self.yaw_deadband >= math.pi:
            raise ValueError("yaw_deadband_degrees must be below 180 degrees")
        self.curvature_speed_reduction_gain = self._non_negative(
            curvature_speed_reduction_gain,
            "curvature_speed_reduction_gain",
        )
        self.minimum_turn_speed_fraction = float(minimum_turn_speed_fraction)
        if not 0.0 < self.minimum_turn_speed_fraction <= 1.0:
            raise ValueError("minimum_turn_speed_fraction must lie in (0, 1]")
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
        if self.augment_local_terrain_observation:
            if not isinstance(self.observation_space, gym.spaces.Box):
                raise TypeError("local terrain preview requires a Box observation space")
            dtype = self.observation_space.dtype
            height_span = (
                self.local_terrain_height_bound
                if self.local_terrain_height_bound is not None
                else max(float(np.ptp(self.heights)), 1e-6)
            )
            preview_count = (
                len(self.terrain_preview_longitudinal)
                * len(self.terrain_preview_lateral)
            )
            terrain_low = np.concatenate(
                (
                    np.full(preview_count, -height_span, dtype=dtype),
                    np.full(3, -1.0, dtype=dtype),
                    np.asarray([-math.pi / 2.0], dtype=dtype),
                )
            )
            terrain_high = np.concatenate(
                (
                    np.full(preview_count, height_span, dtype=dtype),
                    np.full(3, 1.0, dtype=dtype),
                    np.asarray([math.pi / 2.0], dtype=dtype),
                )
            )
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    (np.asarray(self.observation_space.low, dtype=dtype), terrain_low)
                ),
                high=np.concatenate(
                    (np.asarray(self.observation_space.high, dtype=dtype), terrain_high)
                ),
                dtype=dtype,
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
    def _non_negative(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
        return result

    @staticmethod
    def _xy(value: Sequence[float], label: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain two finite values")
        return result

    @staticmethod
    def _preview_axis(
        value: Sequence[float],
        label: str,
        *,
        require_positive: bool,
    ) -> tuple[float, float, float]:
        result = tuple(float(item) for item in value)
        if len(result) != 3 or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain three finite values")
        if require_positive and any(item <= 0.0 for item in result):
            raise ValueError(f"{label} values must be positive")
        if any(left >= right for left, right in zip(result, result[1:])):
            raise ValueError(f"{label} values must be strictly increasing")
        return result

    def set_task_speed(self, cruise_speed_m_per_s: float) -> None:
        """Update the route-controller speed between frozen training stages."""
        self.cruise_speed = self._positive(
            cruise_speed_m_per_s,
            "cruise_speed_m_per_s",
        )

    def _reset_metrics(self) -> None:
        self._terrain_frame_context_ready = False
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
        return self._terrain_value(self.heights, x, y)

    def _terrain_normal(self, x: float, y: float) -> np.ndarray:
        gradient = np.asarray(
            [
                self._terrain_value(self._terrain_dz_dx, x, y),
                self._terrain_value(self._terrain_dz_dy, x, y),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(gradient)):
            raise ValueError("heightfield gradient is non-finite")
        normal = np.asarray([-gradient[0], -gradient[1], 1.0], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm <= 0.0:
            raise ValueError("heightfield normal is invalid")
        return normal / normal_norm

    def _terrain_value(self, values: np.ndarray, x: float, y: float) -> float:
        rows, cols = self.heights.shape
        extent = self.map_half_extent_m
        col_f = np.clip((x + extent) / (2.0 * extent) * (cols - 1), 0, cols - 1)
        row_f = np.clip((y + extent) / (2.0 * extent) * (rows - 1), 0, rows - 1)
        col0 = min(int(math.floor(col_f)), cols - 2)
        row0 = min(int(math.floor(row_f)), rows - 2)
        tx = float(col_f - col0)
        ty = float(row_f - row0)
        z00 = float(values[row0, col0])
        z10 = float(values[row0, col0 + 1])
        z01 = float(values[row0 + 1, col0])
        z11 = float(values[row0 + 1, col0 + 1])
        return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * (
            (1.0 - tx) * z01 + tx * z11
        )

    def _local_terrain_observation(
        self,
        position: np.ndarray,
        target_heading: float,
    ) -> np.ndarray:
        forward = np.asarray(
            [math.cos(target_heading), math.sin(target_heading)],
            dtype=np.float64,
        )
        left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        reference_height = self._terrain_height(float(position[0]), float(position[1]))
        relative_heights: list[float] = []
        for longitudinal in self.terrain_preview_longitudinal:
            for lateral in self.terrain_preview_lateral:
                sample = position + longitudinal * forward + lateral * left
                relative_heights.append(
                    self._terrain_height(float(sample[0]), float(sample[1]))
                    - reference_height
                )

        gradient = np.asarray(
            [
                self._terrain_value(
                    self._terrain_dz_dx,
                    float(position[0]),
                    float(position[1]),
                ),
                self._terrain_value(
                    self._terrain_dz_dy,
                    float(position[0]),
                    float(position[1]),
                ),
            ],
            dtype=np.float64,
        )
        normal_world = np.asarray([-gradient[0], -gradient[1], 1.0])
        normal_world /= np.linalg.norm(normal_world)
        normal_target_frame = np.asarray(
            [
                float(np.dot(normal_world[:2], forward)),
                float(np.dot(normal_world[:2], left)),
                float(normal_world[2]),
            ],
            dtype=np.float64,
        )
        signed_forward_slope = math.atan(float(np.dot(gradient, forward)))
        return np.concatenate(
            (
                np.asarray(relative_heights, dtype=np.float64),
                normal_target_frame,
                np.asarray([signed_forward_slope], dtype=np.float64),
            )
        )

    def _append_local_terrain_observation(
        self,
        observation: np.ndarray,
        position: np.ndarray,
        target_heading: float,
    ) -> np.ndarray:
        if not self.augment_local_terrain_observation:
            return observation
        terrain = self._local_terrain_observation(position, target_heading)
        return np.concatenate(
            (np.asarray(observation), terrain.astype(observation.dtype, copy=False))
        )

    def _position(self) -> np.ndarray:
        return np.asarray(self.unwrapped.data.qpos[:2], dtype=np.float64).copy()

    def _distance_to_goal(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_xy - position))

    def _update_goal_state(self, distance: float) -> None:
        """Advance the arrival-plus-hold state without accepting the hold annulus alone.

        ``hold_radius`` is intentionally larger than ``arrival_radius`` to
        provide hysteresis after a genuine arrival.  It must not independently
        establish success, otherwise an agent can stop in the annulus between
        the two radii and be recorded as having reached the goal.
        """
        if distance <= self.arrival_radius:
            self._goal_entered = True
        if self._goal_entered and distance <= self.hold_radius:
            self._goal_hold_run_steps += 1
        else:
            self._goal_hold_run_steps = 0
        self._longest_goal_hold_run_steps = max(
            self._longest_goal_hold_run_steps,
            self._goal_hold_run_steps,
        )
        if (
            not self._task_success
            and self._goal_entered
            and self._goal_hold_run_steps >= self.required_hold_steps
        ):
            self._task_success = True
            self._success_step = self._task_steps

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
            effective_heading_error = math.copysign(
                max(0.0, abs(heading_error) - self.yaw_deadband),
                heading_error,
            )
            yaw_rate = float(
                np.clip(
                    self.yaw_gain * effective_heading_error,
                    -maximum_yaw_rate,
                    maximum_yaw_rate,
                )
            )
            if maximum_yaw_rate > 0.0 and self.curvature_speed_reduction_gain > 0.0:
                turn_fraction = min(1.0, abs(yaw_rate) / maximum_yaw_rate)
                turn_speed_fraction = max(
                    self.minimum_turn_speed_fraction,
                    1.0 - self.curvature_speed_reduction_gain * turn_fraction,
                )
                target_speed *= turn_speed_fraction
                maximum_yaw_rate = target_speed * self.maximum_abs_curvature
                yaw_rate = float(
                    np.clip(yaw_rate, -maximum_yaw_rate, maximum_yaw_rate)
                )
        command_observation = self.env.set_external_curve_command(
            observation,
            target_heading=target_heading,
            yaw_rate=yaw_rate,
            speed=target_speed,
            lateral_speed=0.0,
        )
        if self.terrain_frame_shaping_enabled:
            self.env.set_terrain_shaping_context(
                height_sampler=self._terrain_height,
                normal_sampler=self._terrain_normal,
                target_heading=target_heading,
            )
            self._terrain_frame_context_ready = True
        return self._append_local_terrain_observation(
            command_observation,
            position,
            target_heading,
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

        self._update_goal_state(distance)

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
        elif self.augment_local_terrain_observation:
            vector = self.goal_xy - position
            target_heading = (
                float(math.atan2(vector[1], vector[0]))
                if float(np.linalg.norm(vector)) > self.arrival_radius
                else quaternion_yaw_angle(np.asarray(self.unwrapped.data.qpos[3:7]))
            )
            observation = self._append_local_terrain_observation(
                observation,
                position,
                target_heading,
            )
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
            "proxygap_fixed_goal_yaw_gain_per_second": self.yaw_gain,
            "proxygap_fixed_goal_yaw_deadband_rad": self.yaw_deadband,
            "proxygap_terrain_relative_unhealthy_run_steps": self._terrain_unhealthy_run_steps,
            "proxygap_local_terrain_observation_enabled": (
                self.augment_local_terrain_observation
            ),
            "proxygap_terrain_frame_shaping_enabled": (
                self.terrain_frame_shaping_enabled
            ),
            "proxygap_terrain_frame_context_valid": bool(
                self._terrain_frame_context_ready
            ),
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
                "fixed_goal_local_terrain_observation_enabled": (
                    self.augment_local_terrain_observation
                ),
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
