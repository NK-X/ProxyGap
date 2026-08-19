"""Curve-agnostic tangent-aligned gait training for Ant-v5.

The policy receives only local motion commands.  A curve generator supplies a
world-frame tangent velocity, signed yaw rate, and heading error; no global
path position or waypoint error participates in the reward.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from .ant_wrapper import (
    DEFAULT_FOOT_GEOM_NAMES,
    bounded_squared_signal_penalty,
    make_proxygap_ant_env,
)
from .planar_transition import (
    planar_velocity_tracking_value,
    quaternion_yaw_angle,
    wrapped_angle_difference,
)


CURVE_PROFILES = (
    "random",
    "straight",
    "constant_left",
    "constant_right",
    "s_curve",
    "external",
)


def pseudo_huber_penalty(signal: float, *, scale: float) -> float:
    """Return a smooth, non-saturating penalty with zero at the target."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pseudo-Huber scale must be positive and finite")
    if not np.isfinite(signal):
        return 0.0
    ratio = float(signal) / float(scale)
    return float(math.sqrt(1.0 + ratio * ratio) - 1.0)


class CurvedGaitCommandWrapper(gym.Wrapper):
    """Train locomotion under tangent velocity and yaw-rate commands.

    Observation append order is ``vx_command, vy_command, yaw_rate_command,
    sin_heading_error, cos_heading_error`` followed, when explicitly enabled,
    by the four binary foot-contact indicators.  The optional contact columns
    expose support state without prescribing a named gait or route geometry.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        profile: str = "random",
        command_frame: str = "world_tangent",
        observation_frame: str = "world",
        speed_min: float = 0.6,
        speed_max: float = 1.0,
        max_abs_lateral_speed: float = 0.0,
        lateral_speed_slew_rate: float = 0.40,
        fixed_lateral_speed: float = 0.0,
        max_abs_curvature: float = 0.15,
        curvature_slew_rate: float = 0.20,
        segment_steps_min: int = 80,
        segment_steps_max: int = 160,
        warmup_steps: int = 20,
        s_curve_period_steps: int = 240,
        planar_tracking_weight: float = 0.5,
        planar_tracking_scale: float = 0.5,
        planar_tracking_function: str = "pseudo_huber",
        cross_axis_velocity_weight: float = 0.05,
        cross_axis_velocity_scale: float = 0.5,
        heading_alignment_weight: float = 0.5,
        heading_alignment_scale: float = math.radians(5.0),
        heading_alignment_function: str = "pseudo_huber",
        yaw_rate_tracking_weight: float = 0.25,
        yaw_rate_tracking_scale: float = 0.20,
        yaw_rate_tracking_function: str = "pseudo_huber",
        heading_tolerance: float = math.radians(5.0),
        heading_termination_threshold: float = math.radians(20.0),
        heading_termination_consecutive_steps: int = 5,
        heading_termination_enabled: bool = True,
        augment_foot_contact_mask: bool = False,
    ) -> None:
        super().__init__(env)
        if profile not in CURVE_PROFILES:
            raise ValueError(f"unsupported curve profile: {profile}")
        self.profile = str(profile)
        if command_frame not in {"world_tangent", "body_tangent"}:
            raise ValueError(f"unsupported curve command frame: {command_frame}")
        self.command_frame = str(command_frame)
        if observation_frame not in {"world", "target_tangent"}:
            raise ValueError(f"unsupported curve observation frame: {observation_frame}")
        self.observation_frame = str(observation_frame)
        self.speed_min = self._positive(speed_min, "speed_min")
        self.speed_max = self._positive(speed_max, "speed_max")
        if self.speed_max < self.speed_min:
            raise ValueError("speed_max must be at least speed_min")
        self.max_abs_lateral_speed = self._non_negative(
            max_abs_lateral_speed,
            "max_abs_lateral_speed",
        )
        self.lateral_speed_slew_rate = self._positive(
            lateral_speed_slew_rate,
            "lateral_speed_slew_rate",
        )
        self.fixed_lateral_speed = float(fixed_lateral_speed)
        if not np.isfinite(self.fixed_lateral_speed):
            raise ValueError("fixed_lateral_speed must be finite")
        if abs(self.fixed_lateral_speed) > self.max_abs_lateral_speed + 1e-12:
            raise ValueError("fixed_lateral_speed exceeds max_abs_lateral_speed")
        self.max_abs_curvature = self._non_negative(
            max_abs_curvature,
            "max_abs_curvature",
        )
        self.curvature_slew_rate = self._positive(
            curvature_slew_rate,
            "curvature_slew_rate",
        )
        if segment_steps_min <= 0 or segment_steps_max < segment_steps_min:
            raise ValueError("curve segment interval is invalid")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if s_curve_period_steps <= 0:
            raise ValueError("s_curve_period_steps must be positive")
        self.segment_steps_min = int(segment_steps_min)
        self.segment_steps_max = int(segment_steps_max)
        self.warmup_steps = int(warmup_steps)
        self.s_curve_period_steps = int(s_curve_period_steps)
        self.planar_tracking_weight = self._non_negative(
            planar_tracking_weight,
            "planar_tracking_weight",
        )
        self.planar_tracking_scale = self._positive(
            planar_tracking_scale,
            "planar_tracking_scale",
        )
        if planar_tracking_function not in {"exponential", "pseudo_huber"}:
            raise ValueError("unsupported planar_tracking_function")
        self.planar_tracking_function = str(planar_tracking_function)
        self.cross_axis_velocity_weight = self._non_negative(
            cross_axis_velocity_weight,
            "cross_axis_velocity_weight",
        )
        self.cross_axis_velocity_scale = self._positive(
            cross_axis_velocity_scale,
            "cross_axis_velocity_scale",
        )
        self.heading_alignment_weight = self._non_negative(
            heading_alignment_weight,
            "heading_alignment_weight",
        )
        self.heading_alignment_scale = self._positive(
            heading_alignment_scale,
            "heading_alignment_scale",
        )
        if heading_alignment_function not in {"pseudo_huber", "bounded_squared"}:
            raise ValueError("unsupported heading_alignment_function")
        self.heading_alignment_function = str(heading_alignment_function)
        self.yaw_rate_tracking_weight = self._non_negative(
            yaw_rate_tracking_weight,
            "yaw_rate_tracking_weight",
        )
        self.yaw_rate_tracking_scale = self._positive(
            yaw_rate_tracking_scale,
            "yaw_rate_tracking_scale",
        )
        if yaw_rate_tracking_function not in {"pseudo_huber", "bounded_squared"}:
            raise ValueError("unsupported yaw_rate_tracking_function")
        self.yaw_rate_tracking_function = str(yaw_rate_tracking_function)
        self.heading_tolerance = self._positive(
            heading_tolerance,
            "heading_tolerance",
        )
        self.heading_termination_threshold = self._positive(
            heading_termination_threshold,
            "heading_termination_threshold",
        )
        if heading_termination_consecutive_steps <= 0:
            raise ValueError(
                "heading_termination_consecutive_steps must be positive"
            )
        self.heading_termination_consecutive_steps = int(
            heading_termination_consecutive_steps
        )
        self.heading_termination_enabled = bool(heading_termination_enabled)
        self.augment_foot_contact_mask = bool(augment_foot_contact_mask)
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("curved gait wrapper requires a Box observation space")
        dtype = self.observation_space.dtype
        speed_limit = max(1.0, self.speed_max)
        yaw_rate_limit = max(1.0, self.speed_max * self.max_abs_curvature)
        optional_contact_low = (
            np.zeros(4, dtype=dtype)
            if self.augment_foot_contact_mask
            else np.asarray([], dtype=dtype)
        )
        optional_contact_high = (
            np.ones(4, dtype=dtype)
            if self.augment_foot_contact_mask
            else np.asarray([], dtype=dtype)
        )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.asarray(self.observation_space.low, dtype=dtype),
                    np.asarray(
                        [-speed_limit, -speed_limit, -yaw_rate_limit, -1.0, -1.0],
                        dtype=dtype,
                    ),
                    optional_contact_low,
                )
            ),
            high=np.concatenate(
                (
                    np.asarray(self.observation_space.high, dtype=dtype),
                    np.asarray(
                        [speed_limit, speed_limit, yaw_rate_limit, 1.0, 1.0],
                        dtype=dtype,
                    ),
                    optional_contact_high,
                )
            ),
            dtype=dtype,
        )
        self._reset_state()

    @staticmethod
    def _positive(value: float, label: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{label} must be positive and finite")
        return number

    @staticmethod
    def _non_negative(value: float, label: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{label} must be finite and non-negative")
        return number

    def set_curriculum(
        self,
        max_abs_curvature: float,
        speed_min: float | None = None,
        speed_max: float | None = None,
        max_abs_lateral_speed: float | None = None,
    ) -> None:
        """Update only command difficulty between locked training stages."""
        self.max_abs_curvature = self._non_negative(
            max_abs_curvature,
            "max_abs_curvature",
        )
        if speed_min is not None:
            self.speed_min = self._positive(speed_min, "speed_min")
        if speed_max is not None:
            self.speed_max = self._positive(speed_max, "speed_max")
        if self.speed_max < self.speed_min:
            raise ValueError("speed_max must be at least speed_min")
        if max_abs_lateral_speed is not None:
            self.max_abs_lateral_speed = self._non_negative(
                max_abs_lateral_speed,
                "max_abs_lateral_speed",
            )

    def _reset_state(self) -> None:
        self._elapsed_steps = 0
        self._target_speed = self.speed_max
        self._target_lateral_speed = self.fixed_lateral_speed
        self._current_lateral_speed = self.fixed_lateral_speed
        self._desired_lateral_speed = self.fixed_lateral_speed
        self._target_heading = 0.0
        self._current_curvature = 0.0
        self._desired_curvature = 0.0
        self._segment_steps_remaining = self.warmup_steps
        self._command_xy = np.asarray([self._target_speed, 0.0], dtype=np.float64)
        self._yaw_rate_command = 0.0
        self._heading_error = 0.0
        self._heading_violation_run = 0
        self._heading_constraint_terminated = False
        self._external_yaw_rate_command = 0.0
        self._foot_contact_mask = np.zeros(4, dtype=np.float64)
        self._objective_return = 0.0
        self._outer_shaping_sum = 0.0
        self._tracking_reward_sum = 0.0
        self._cross_axis_reward_sum = 0.0
        self._heading_reward_sum = 0.0
        self._yaw_rate_reward_sum = 0.0
        self._tangent_speed_error_squared_sum = 0.0
        self._cross_axis_velocity_squared_sum = 0.0
        self._heading_error_squared_sum = 0.0
        self._heading_error_abs_max = 0.0
        self._heading_within_tolerance_steps = 0
        self._yaw_rate_error_squared_sum = 0.0
        self._curvature_abs_sum = 0.0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._last_base_observation = np.asarray(observation).copy()
        self._reset_state()
        self._update_foot_contact_mask(info)
        rng = self.unwrapped.np_random
        self._target_speed = float(rng.uniform(self.speed_min, self.speed_max))
        self._target_lateral_speed = self.fixed_lateral_speed
        self._current_lateral_speed = self.fixed_lateral_speed
        self._desired_lateral_speed = self.fixed_lateral_speed
        self._target_heading = self._torso_yaw()
        self._heading_error = 0.0
        if self.profile != "random":
            self._current_curvature = self._profile_curvature(0)
            self._desired_curvature = self._current_curvature
        self._refresh_command()
        live_info = dict(info)
        live_info.update(self._live_info())
        return self._augment_observation(observation), live_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        applied_curvature = float(self._current_curvature)
        applied_yaw_rate = (
            float(self._external_yaw_rate_command)
            if self.profile == "external"
            else float(self._target_speed * applied_curvature)
        )
        applied_heading = (
            float(self._target_heading)
            if self.profile == "external"
            else float(
                wrapped_angle_difference(
                    self._target_heading
                    + applied_yaw_rate * float(self.unwrapped.dt),
                    0.0,
                )
            )
        )
        tangent = np.asarray(
            [math.cos(applied_heading), math.sin(applied_heading)],
            dtype=np.float64,
        )
        cross_direction = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        applied_command = (
            self._target_speed * tangent
            + self._target_lateral_speed * cross_direction
        )

        observation, inner_reward, terminated, truncated, info = self.env.step(action)
        self._last_base_observation = np.asarray(observation).copy()
        info = dict(info)
        self._update_foot_contact_mask(info)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        velocity_xy = qvel[:2].copy()
        actual_yaw_rate = float(qvel[5])
        tangent_speed = float(np.dot(velocity_xy, tangent))
        cross_axis_velocity = float(np.dot(velocity_xy, cross_direction))
        cross_axis_velocity_error = (
            cross_axis_velocity - self._target_lateral_speed
        )
        heading_error = wrapped_angle_difference(self._torso_yaw(), applied_heading)
        yaw_rate_error = actual_yaw_rate - applied_yaw_rate

        tracking_value = planar_velocity_tracking_value(
            velocity_xy,
            applied_command,
            scale=self.planar_tracking_scale,
            function=self.planar_tracking_function,
        )
        tracking_reward = self.planar_tracking_weight * tracking_value
        cross_axis_penalty = bounded_squared_signal_penalty(
            cross_axis_velocity_error,
            scale=self.cross_axis_velocity_scale,
        )
        cross_axis_reward = -self.cross_axis_velocity_weight * cross_axis_penalty
        if self.heading_alignment_function == "bounded_squared":
            heading_penalty = bounded_squared_signal_penalty(
                heading_error,
                scale=self.heading_alignment_scale,
            )
        else:
            heading_penalty = pseudo_huber_penalty(
                heading_error,
                scale=self.heading_alignment_scale,
            )
        heading_reward = -self.heading_alignment_weight * heading_penalty
        if self.yaw_rate_tracking_function == "bounded_squared":
            yaw_rate_penalty = bounded_squared_signal_penalty(
                yaw_rate_error,
                scale=self.yaw_rate_tracking_scale,
            )
        else:
            yaw_rate_penalty = pseudo_huber_penalty(
                yaw_rate_error,
                scale=self.yaw_rate_tracking_scale,
            )
        yaw_rate_reward = -self.yaw_rate_tracking_weight * yaw_rate_penalty
        outer_shaping = (
            tracking_reward
            + cross_axis_reward
            + heading_reward
            + yaw_rate_reward
        )
        reward = float(inner_reward) + outer_shaping

        self._elapsed_steps += 1
        self._target_heading = applied_heading
        self._heading_error = heading_error
        self._objective_return += reward
        self._outer_shaping_sum += outer_shaping
        self._tracking_reward_sum += tracking_reward
        self._cross_axis_reward_sum += cross_axis_reward
        self._heading_reward_sum += heading_reward
        self._yaw_rate_reward_sum += yaw_rate_reward
        self._tangent_speed_error_squared_sum += (
            tangent_speed - self._target_speed
        ) ** 2
        self._cross_axis_velocity_squared_sum += cross_axis_velocity**2
        self._heading_error_squared_sum += heading_error**2
        self._heading_error_abs_max = max(
            self._heading_error_abs_max,
            abs(heading_error),
        )
        if abs(heading_error) <= self.heading_tolerance:
            self._heading_within_tolerance_steps += 1
        self._yaw_rate_error_squared_sum += yaw_rate_error**2
        self._curvature_abs_sum += abs(applied_curvature)

        if abs(heading_error) > self.heading_termination_threshold:
            self._heading_violation_run += 1
        else:
            self._heading_violation_run = 0
        if (
            self.heading_termination_enabled
            and
            self._heading_violation_run
            >= self.heading_termination_consecutive_steps
        ):
            terminated = True
            self._heading_constraint_terminated = True

        self._advance_curvature()
        self._refresh_command()
        info.update(
            {
                "reward_curve_velocity_tracking": float(tracking_reward),
                "reward_curve_cross_axis_shaping": float(cross_axis_reward),
                "reward_curve_heading_alignment": float(heading_reward),
                "reward_curve_yaw_rate_tracking": float(yaw_rate_reward),
                "reward_curve_outer_shaping": float(outer_shaping),
                "proxygap_curve_tangent_heading_step": applied_heading,
                "proxygap_curve_heading_error_step": heading_error,
                "proxygap_curve_curvature_step": applied_curvature,
                "proxygap_curve_yaw_rate_command_step": applied_yaw_rate,
                "proxygap_curve_actual_yaw_rate_step": actual_yaw_rate,
                "proxygap_curve_tangent_speed_step": tangent_speed,
                "proxygap_curve_cross_axis_velocity_step": cross_axis_velocity,
                "proxygap_heading_constraint_terminated": (
                    self._heading_constraint_terminated
                ),
            }
        )
        info.update(self._live_info())
        if terminated or truncated:
            summary = self.episode_summary()
            info.update({f"proxygap_{key}": value for key, value in summary.items()})
        return (
            self._augment_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def _profile_curvature(self, step: int) -> float:
        if self.profile == "straight":
            return 0.0
        if self.profile == "constant_left":
            return self.max_abs_curvature
        if self.profile == "constant_right":
            return -self.max_abs_curvature
        if self.profile == "s_curve":
            phase = 2.0 * math.pi * float(step) / self.s_curve_period_steps
            return float(self.max_abs_curvature * math.sin(phase))
        if self.profile == "external":
            return float(self._current_curvature)
        return self._desired_curvature

    def _sample_random_curvature(self) -> float:
        if self.max_abs_curvature <= 0:
            return 0.0
        rng = self.unwrapped.np_random
        if float(rng.random()) < 0.20:
            return 0.0
        return float(rng.uniform(-self.max_abs_curvature, self.max_abs_curvature))

    def _sample_random_lateral_speed(self) -> float:
        if self.max_abs_lateral_speed <= 0:
            return 0.0
        rng = self.unwrapped.np_random
        if float(rng.random()) < 0.20:
            return 0.0
        return float(
            rng.uniform(-self.max_abs_lateral_speed, self.max_abs_lateral_speed)
        )

    def _advance_curvature(self) -> None:
        if self.profile == "external":
            return
        if self.profile == "random":
            self._segment_steps_remaining -= 1
            if self._segment_steps_remaining <= 0:
                self._desired_curvature = self._sample_random_curvature()
                self._desired_lateral_speed = self._sample_random_lateral_speed()
                rng = self.unwrapped.np_random
                self._segment_steps_remaining = int(
                    rng.integers(
                        self.segment_steps_min,
                        self.segment_steps_max + 1,
                    )
                )
        else:
            self._desired_curvature = self._profile_curvature(self._elapsed_steps)
        max_delta = self.curvature_slew_rate * float(self.unwrapped.dt)
        delta = float(
            np.clip(
                self._desired_curvature - self._current_curvature,
                -max_delta,
                max_delta,
            )
        )
        self._current_curvature += delta
        max_lateral_delta = self.lateral_speed_slew_rate * float(self.unwrapped.dt)
        lateral_delta = float(
            np.clip(
                self._desired_lateral_speed - self._current_lateral_speed,
                -max_lateral_delta,
                max_lateral_delta,
            )
        )
        self._current_lateral_speed += lateral_delta

    def _refresh_command(self) -> None:
        self._target_lateral_speed = self._current_lateral_speed
        self._yaw_rate_command = (
            self._external_yaw_rate_command
            if self.profile == "external"
            else self._target_speed * self._current_curvature
        )
        if self.command_frame == "body_tangent":
            self._command_xy = np.asarray(
                [self._target_speed, self._target_lateral_speed],
                dtype=np.float64,
            )
        else:
            self._command_xy = self._target_speed * np.asarray(
                [math.cos(self._target_heading), math.sin(self._target_heading)],
                dtype=np.float64,
            )

    def set_external_curve_command(
        self,
        observation: np.ndarray,
        *,
        target_heading: float,
        yaw_rate: float,
        speed: float,
        lateral_speed: float = 0.0,
    ) -> np.ndarray:
        """Replace a local command for a high-level route controller.

        The high-level controller may use route position, but only the same
        five local curve-command values used in training are returned to the
        policy. Route coordinates never enter the learned observation.
        """
        if self.profile != "external":
            raise RuntimeError("external curve commands require profile='external'")
        values = np.asarray(observation)
        augmented_dimension = 122 if self.augment_foot_contact_mask else 118
        if values.shape == (augmented_dimension,):
            if not hasattr(self, "_last_base_observation"):
                raise RuntimeError("external command requires a reset observation")
            base_observation = np.asarray(self._last_base_observation)
        elif values.shape == (113,):
            base_observation = values
            self._last_base_observation = values.copy()
        else:
            raise ValueError(
                "external command requires a 113-value base observation or "
                f"a {augmented_dimension}-value augmented observation"
            )
        target_speed = self._positive(speed, "external speed")
        target_lateral_speed = float(lateral_speed)
        target_yaw_rate = float(yaw_rate)
        heading = float(target_heading)
        if (
            not np.isfinite(target_lateral_speed)
            or not np.isfinite(target_yaw_rate)
            or not np.isfinite(heading)
        ):
            raise ValueError("external lateral speed, heading and yaw rate must be finite")
        if abs(target_lateral_speed) > max(1.0, self.speed_max) + 1e-12:
            raise ValueError("external lateral speed exceeds the command limit")
        curvature = target_yaw_rate / target_speed
        if abs(curvature) > self.max_abs_curvature + 1e-12:
            raise ValueError("external command exceeds configured curvature limit")
        self._target_speed = target_speed
        self._target_lateral_speed = target_lateral_speed
        self._current_lateral_speed = target_lateral_speed
        self._desired_lateral_speed = target_lateral_speed
        self._target_heading = wrapped_angle_difference(heading, 0.0)
        self._current_curvature = curvature
        self._desired_curvature = curvature
        self._external_yaw_rate_command = target_yaw_rate
        self._heading_error = wrapped_angle_difference(
            self._torso_yaw(),
            self._target_heading,
        )
        self._refresh_command()
        return self._augment_observation(base_observation)

    def set_terrain_shaping_context(
        self,
        *,
        height_sampler: Any,
        normal_sampler: Any,
        target_heading: float,
    ) -> None:
        """Forward the fixed-map terrain frame to the inner reward wrapper."""

        setter = getattr(self.env, "set_terrain_shaping_context", None)
        if setter is None:
            raise RuntimeError("inner locomotion wrapper lacks terrain-frame support")
        setter(
            height_sampler=height_sampler,
            normal_sampler=normal_sampler,
            target_heading=target_heading,
        )

    def _torso_yaw(self) -> float:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return quaternion_yaw_angle(qpos[3:7])

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        values = self._observation_in_command_frame(observation)
        command = np.asarray(
            [
                self._command_xy[0],
                self._command_xy[1],
                self._yaw_rate_command,
                math.sin(self._heading_error),
                math.cos(self._heading_error),
            ],
            dtype=values.dtype,
        )
        if not self.augment_foot_contact_mask:
            return np.concatenate((values, command))
        return np.concatenate((values, command, self._foot_contact_mask)).astype(
            values.dtype,
            copy=False,
        )

    def _update_foot_contact_mask(self, info: dict[str, Any]) -> None:
        """Capture the contact state associated with the current observation."""
        if not self.augment_foot_contact_mask:
            return
        raw_mask = info.get("proxygap_foot_contact_mask_step")
        if raw_mask is None:
            raise KeyError("foot-contact observation requires contact diagnostics")
        mask = np.asarray(raw_mask, dtype=np.float64)
        if mask.shape != (4,):
            raise ValueError("foot-contact observation requires four foot indicators")
        self._foot_contact_mask = mask.copy()

    def _observation_in_command_frame(self, observation: np.ndarray) -> np.ndarray:
        """Express world-vector fields in the target-tangent frame.

        Ant-v5 contributes 105 values: qpos without root x/y (13), qvel
        (14), and external body force/torque vectors (78).  The preserved
        previous-action extension occupies the final eight values and is
        body-fixed, so it is copied without rotation.
        """
        values = np.asarray(observation)
        if self.observation_frame == "world":
            return values
        if values.shape != (113,):
            raise ValueError(
                "target-tangent canonicalisation requires the preserved "
                "113-dimensional Ant observation"
            )
        transformed = values.copy()
        angle = float(self._target_heading)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        # Left-multiply the root wxyz quaternion by yaw(-target_heading).
        half_cosine = math.cos(0.5 * angle)
        half_sine = math.sin(0.5 * angle)
        w, x, y, z = (float(value) for value in values[1:5])
        transformed[1:5] = (
            half_cosine * w + half_sine * z,
            half_cosine * x + half_sine * y,
            half_cosine * y - half_sine * x,
            half_cosine * z - half_sine * w,
        )
        # q and -q encode the same rotation. Wrapping the target yaw at pi can
        # otherwise flip every quaternion component and create a false policy
        # discontinuity once per revolution.
        if float(transformed[1]) < 0.0:
            transformed[1:5] *= -1.0

        def rotate_xy(array: np.ndarray, first: int, second: int) -> None:
            original_x = float(array[first])
            original_y = float(array[second])
            array[first] = cosine * original_x + sine * original_y
            array[second] = -sine * original_x + cosine * original_y

        # qvel root linear xyz then angular xyz.
        rotate_xy(transformed, 13, 14)
        rotate_xy(transformed, 16, 17)
        # cfrc_ext for 13 non-world bodies: torque xyz, force xyz.
        for offset in range(27, 105, 6):
            rotate_xy(transformed, offset, offset + 1)
            rotate_xy(transformed, offset + 3, offset + 4)
        return transformed

    def _live_info(self) -> dict[str, Any]:
        return {
            "proxygap_curve_profile": self.profile,
            "proxygap_curve_command_frame": self.command_frame,
            "proxygap_curve_observation_frame": self.observation_frame,
            "proxygap_curve_foot_contact_observation_enabled": (
                self.augment_foot_contact_mask
            ),
            "proxygap_curve_foot_contact_mask": self._foot_contact_mask.copy(),
            "proxygap_curve_command_xy": self._command_xy.copy(),
            "proxygap_curve_yaw_rate_command": self._yaw_rate_command,
            "proxygap_curve_target_heading": self._target_heading,
            "proxygap_curve_next_curvature": self._current_curvature,
            "proxygap_curve_max_abs_curvature": self.max_abs_curvature,
            "proxygap_condition_objective_return": self._objective_return,
            "proxygap_proxy_return": self._objective_return,
        }

    def episode_summary(self) -> dict[str, Any]:
        summary = dict(self.env.episode_summary())
        elapsed = max(1, self._elapsed_steps)
        inner_shaping = float(summary.get("reward_shaping_sum", 0.0))
        base_proxy = float(summary.get("base_proxy_return", 0.0))
        combined_shaping = inner_shaping + self._outer_shaping_sum
        summary.update(
            {
                "condition_objective_return": self._objective_return,
                "proxy_return": self._objective_return,
                "reward_shaping_sum": combined_shaping,
                "reward_curve_outer_shaping_sum": self._outer_shaping_sum,
                "reward_curve_velocity_tracking_sum": self._tracking_reward_sum,
                "reward_curve_cross_axis_shaping_sum": self._cross_axis_reward_sum,
                "reward_curve_heading_alignment_sum": self._heading_reward_sum,
                "reward_curve_yaw_rate_tracking_sum": self._yaw_rate_reward_sum,
                "curve_reward_reconciliation_error": (
                    self._objective_return - base_proxy - combined_shaping
                ),
                "curve_profile": self.profile,
                "curve_command_frame": self.command_frame,
                "curve_observation_frame": self.observation_frame,
                "curve_episode_steps": self._elapsed_steps,
                "curve_target_speed_m_per_s": self._target_speed,
                "curve_mean_abs_curvature_per_m": (
                    self._curvature_abs_sum / elapsed
                ),
                "curve_tangent_speed_rmse_m_per_s": math.sqrt(
                    self._tangent_speed_error_squared_sum / elapsed
                ),
                "curve_cross_axis_velocity_rms_m_per_s": math.sqrt(
                    self._cross_axis_velocity_squared_sum / elapsed
                ),
                "curve_heading_error_rms_rad": math.sqrt(
                    self._heading_error_squared_sum / elapsed
                ),
                "curve_heading_error_max_abs_rad": self._heading_error_abs_max,
                "curve_heading_within_tolerance_fraction": (
                    self._heading_within_tolerance_steps / elapsed
                ),
                "curve_yaw_rate_error_rmse_rad_per_s": math.sqrt(
                    self._yaw_rate_error_squared_sum / elapsed
                ),
                "curve_heading_constraint_terminated": (
                    self._heading_constraint_terminated
                ),
                "curve_max_abs_curvature_per_m": self.max_abs_curvature,
                "curve_heading_tolerance_rad": self.heading_tolerance,
                "curve_heading_termination_threshold_rad": (
                    self.heading_termination_threshold
                ),
                "curve_heading_termination_enabled": (
                    self.heading_termination_enabled
                ),
                "curve_uses_global_path_position_reward": False,
            }
        )
        return summary


def make_curved_gait_env(
    *,
    condition_id: str = "curved_gait",
    ctrl_cost_weight: float = 0.5,
    seed: int | None = None,
    render_mode: str | None = None,
    xml_file: str | Path | None = None,
    max_episode_steps: int | None = None,
    terminate_when_unhealthy: bool = True,
    orientation_shaping_weight: float = 0.1,
    orientation_shaping_scale: float = 1.0,
    orientation_shaping_function: str = "cosine",
    action_rate_shaping_weight: float = 0.2,
    vertical_velocity_shaping_weight: float = 0.05,
    vertical_velocity_shaping_scale: float = 1.014092584749083,
    roll_pitch_angular_velocity_shaping_weight: float = 0.05,
    roll_pitch_angular_velocity_shaping_scale: float = 1.9893176307304792,
    foot_landing_height_threshold: float = 0.03,
    foot_lateral_velocity_shaping_weight: float = 0.025,
    foot_lateral_velocity_shaping_scale: float = 1.0,
    foot_vertical_velocity_shaping_weight: float = 0.025,
    foot_vertical_velocity_shaping_scale: float = 1.0,
    airborne_shaping_weight: float = 0.0,
    foot_contact_gap_shaping_weight: float = 0.0,
    foot_contact_gap_grace_seconds: float = 0.5,
    foot_contact_gap_scale_seconds: float = 0.5,
    foot_geom_names: Sequence[str] = DEFAULT_FOOT_GEOM_NAMES,
    augment_previous_applied_action: bool = True,
    terrain_frame_shaping_enabled: bool = False,
    **curve_kwargs: Any,
) -> CurvedGaitCommandWrapper:
    """Create the preserved pre-pitch reward package plus curve commands."""
    base = make_proxygap_ant_env(
        ctrl_cost_weight=ctrl_cost_weight,
        condition_id=condition_id,
        seed=seed,
        render_mode=render_mode,
        xml_file=xml_file,
        max_episode_steps=max_episode_steps,
        terminate_when_unhealthy=terminate_when_unhealthy,
        orientation_shaping_weight=orientation_shaping_weight,
        orientation_shaping_scale=orientation_shaping_scale,
        orientation_shaping_function=orientation_shaping_function,
        lateral_drift_shaping_weight=0.0,
        replace_forward_reward_with_tracking=True,
        forward_velocity_target=1.0,
        forward_velocity_tracking_scale=0.5,
        forward_velocity_tracking_weight=0.0,
        action_rate_shaping_weight=action_rate_shaping_weight,
        vertical_velocity_shaping_weight=vertical_velocity_shaping_weight,
        vertical_velocity_shaping_scale=vertical_velocity_shaping_scale,
        roll_pitch_angular_velocity_shaping_weight=(
            roll_pitch_angular_velocity_shaping_weight
        ),
        roll_pitch_angular_velocity_shaping_scale=(
            roll_pitch_angular_velocity_shaping_scale
        ),
        foot_landing_height_threshold=foot_landing_height_threshold,
        foot_lateral_velocity_shaping_weight=(
            foot_lateral_velocity_shaping_weight
        ),
        foot_lateral_velocity_shaping_scale=foot_lateral_velocity_shaping_scale,
        foot_vertical_velocity_shaping_weight=(
            foot_vertical_velocity_shaping_weight
        ),
        foot_vertical_velocity_shaping_scale=foot_vertical_velocity_shaping_scale,
        airborne_shaping_weight=airborne_shaping_weight,
        foot_contact_gap_shaping_weight=foot_contact_gap_shaping_weight,
        foot_contact_gap_grace_seconds=foot_contact_gap_grace_seconds,
        foot_contact_gap_scale_seconds=foot_contact_gap_scale_seconds,
        pitch_balance_shaping_weight=0.0,
        foot_geom_names=tuple(foot_geom_names),
        augment_previous_applied_action=augment_previous_applied_action,
        terrain_frame_shaping_enabled=bool(terrain_frame_shaping_enabled),
    )
    return CurvedGaitCommandWrapper(base, **curve_kwargs)


def transfer_planar_policy_to_curved_gait(
    source_model: PPO,
    target_model: PPO,
) -> dict[str, Any]:
    """Copy a 115-column planar policy and zero-init three curve columns."""
    source_dimension = int(source_model.observation_space.shape[0])
    target_dimension = int(target_model.observation_space.shape[0])
    if target_dimension != source_dimension + 3:
        raise ValueError("curved gait policy must append exactly three columns")
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    for name, source_value in source_state.items():
        if name not in target_state:
            raise KeyError(f"Target policy is missing source parameter: {name}")
        target_value = target_state[name]
        if target_value.shape == source_value.shape:
            target_state[name] = source_value.detach().clone()
            copied.append(name)
            continue
        first_layer = name in {
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        }
        compatible = (
            first_layer
            and target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1] == source_value.shape[1] + 3
        )
        if not compatible:
            raise ValueError(
                f"Unsupported transfer shape for {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        expanded_value = target_value.detach().clone()
        expanded_value.zero_()
        expanded_value[:, : source_value.shape[1]] = source_value.detach()
        target_state[name] = expanded_value
        expanded.append(name)
    target_model.policy.load_state_dict(target_state, strict=True)
    return {
        "source_observation_dimension": source_dimension,
        "target_observation_dimension": target_dimension,
        "action_dimension": int(target_model.action_space.shape[0]),
        "copied_parameter_tensors": copied,
        "expanded_parameter_tensors": expanded,
        "new_curve_command_columns_initialised_to_zero": 3,
    }


def transfer_curved_policy_with_contact_observation(
    source_model: PPO,
    target_model: PPO,
) -> dict[str, Any]:
    """Copy a curved policy and zero-initialise four appended contact inputs."""
    source_dimension = int(source_model.observation_space.shape[0])
    target_dimension = int(target_model.observation_space.shape[0])
    appended_columns = 4
    if target_dimension != source_dimension + appended_columns:
        raise ValueError("contact-aware policy must append exactly four columns")
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    for name, source_value in source_state.items():
        if name not in target_state:
            raise KeyError(f"Target policy is missing source parameter: {name}")
        target_value = target_state[name]
        if target_value.shape == source_value.shape:
            target_state[name] = source_value.detach().clone()
            copied.append(name)
            continue
        first_layer = name in {
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        }
        compatible = (
            first_layer
            and target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1]
            == source_value.shape[1] + appended_columns
        )
        if not compatible:
            raise ValueError(
                f"Unsupported contact transfer shape for {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        expanded_value = target_value.detach().clone()
        expanded_value.zero_()
        expanded_value[:, : source_value.shape[1]] = source_value.detach()
        target_state[name] = expanded_value
        expanded.append(name)
    target_model.policy.load_state_dict(target_state, strict=True)
    return {
        "source_observation_dimension": source_dimension,
        "target_observation_dimension": target_dimension,
        "action_dimension": int(target_model.action_space.shape[0]),
        "copied_parameter_tensors": copied,
        "expanded_parameter_tensors": expanded,
        "new_foot_contact_columns_initialised_to_zero": appended_columns,
    }
