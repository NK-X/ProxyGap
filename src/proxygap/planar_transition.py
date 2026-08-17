"""Command-conditioned planar translation and stop-to-lateral transitions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from .ant_wrapper import (
    DEFAULT_FOOT_GEOM_NAMES,
    bounded_squared_signal_penalty,
    make_proxygap_ant_env,
)


# Ant-v5 actuator order is leg 4, leg 1, leg 2, leg 3, with one hip and
# ankle actuator per leg. Under a positive 90-degree rotation about world z,
# destination legs receive actions from the preceding leg in that cycle.
QUARTER_TURN_ACTION_PERMUTATION = np.asarray(
    [6, 7, 0, 1, 2, 3, 4, 5],
    dtype=np.int64,
)


def quarter_turn_action(action: np.ndarray) -> np.ndarray:
    """Rotate an eight-motor Ant action by positive 90 degrees about z."""
    values = np.asarray(action)
    if values.shape != (8,):
        raise ValueError("quarter-turn action mapping requires exactly 8 motors")
    return values[QUARTER_TURN_ACTION_PERMUTATION].copy()


def quaternion_yaw_angle(quaternion_wxyz: np.ndarray) -> float:
    """Return signed yaw in radians from a MuJoCo ``w, x, y, z`` quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.isfinite(norm):
        return float("nan")
    w, x, y, z = quaternion / norm
    sine = 2.0 * (w * z + x * y)
    cosine = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(sine, cosine))


def wrapped_angle_difference(angle: float, reference: float) -> float:
    """Return ``angle-reference`` wrapped to ``[-pi, pi)``."""
    if not np.isfinite(angle) or not np.isfinite(reference):
        return float("nan")
    return float((angle - reference + math.pi) % (2.0 * math.pi) - math.pi)


def planar_velocity_tracking_value(
    velocity_xy: np.ndarray,
    command_xy: np.ndarray,
    *,
    scale: float,
    function: str = "exponential",
) -> float:
    """Return a bounded reward for tracking a two-dimensional velocity command."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("planar velocity tracking scale must be positive and finite")
    velocity = np.asarray(velocity_xy, dtype=np.float64)
    command = np.asarray(command_xy, dtype=np.float64)
    if velocity.shape != (2,) or command.shape != (2,):
        raise ValueError("velocity and command must both have shape (2,)")
    if not np.isfinite(velocity).all() or not np.isfinite(command).all():
        return 0.0
    error = (velocity - command) / float(scale)
    squared_error = float(np.dot(error, error))
    if function == "exponential":
        return float(np.exp(-squared_error))
    if function == "pseudo_huber":
        # Maximum 1 at the target, with a useful gradient even when a stopped
        # policy is one full metre per second away from its commanded speed.
        return float(1.0 - math.sqrt(1.0 + squared_error))
    raise ValueError(f"Unsupported planar tracking function: {function}")


class PlanarCommandTransitionWrapper(gym.Wrapper):
    """Expose a planar command and train forward -> brake -> lateral motion.

    The wrapped ProxyGap environment removes Ant's original positive-x reward
    before this wrapper adds a two-dimensional command-tracking term. A sudden
    lateral request is governed through a zero-velocity braking phase. The
    torso's initial yaw remains the reference throughout the episode, so the
    learned transition is translation rather than body-yaw steering.
    """

    PHASE_FORWARD = "forward"
    PHASE_BRAKE = "brake"
    PHASE_LATERAL = "lateral"

    def __init__(
        self,
        env: gym.Env,
        *,
        initial_command_xy: Sequence[float] = (1.0, 0.0),
        lateral_command_xy: Sequence[float] = (0.0, 1.0),
        switch_step_min: int = 160,
        switch_step_max: int = 320,
        brake_min_steps: int = 5,
        brake_max_steps: int = 30,
        stop_speed_threshold: float = 0.15,
        stop_consecutive_steps: int = 3,
        planar_tracking_weight: float = 0.5,
        planar_tracking_scale: float = 0.5,
        planar_tracking_function: str = "exponential",
        cross_axis_velocity_weight: float = 0.025,
        cross_axis_velocity_scale: float = 1.0,
        yaw_shaping_weight: float = 0.1,
        yaw_shaping_scale: float = math.radians(15.0),
        yaw_shaping_function: str = "bounded_squared",
        brake_speed_weight: float = 0.25,
        brake_speed_scale: float = 0.5,
    ) -> None:
        super().__init__(env)
        self.initial_command_xy = self._command(initial_command_xy, "initial")
        self.lateral_command_xy = self._command(lateral_command_xy, "lateral")
        if not np.isclose(np.dot(self.initial_command_xy, self.lateral_command_xy), 0.0):
            raise ValueError("initial and lateral commands must be orthogonal")
        if switch_step_min <= 0 or switch_step_max < switch_step_min:
            raise ValueError("switch-step interval is invalid")
        if brake_min_steps < 0 or brake_max_steps < max(1, brake_min_steps):
            raise ValueError("brake-step interval is invalid")
        if stop_consecutive_steps <= 0:
            raise ValueError("stop_consecutive_steps must be positive")
        self.switch_step_min = int(switch_step_min)
        self.switch_step_max = int(switch_step_max)
        self.brake_min_steps = int(brake_min_steps)
        self.brake_max_steps = int(brake_max_steps)
        self.stop_consecutive_steps = int(stop_consecutive_steps)
        self.stop_speed_threshold = self._positive(
            stop_speed_threshold,
            "stop_speed_threshold",
        )
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
        self.yaw_shaping_weight = self._non_negative(
            yaw_shaping_weight,
            "yaw_shaping_weight",
        )
        self.yaw_shaping_scale = self._positive(
            yaw_shaping_scale,
            "yaw_shaping_scale",
        )
        if yaw_shaping_function not in {"bounded_squared", "pseudo_huber"}:
            raise ValueError("unsupported yaw_shaping_function")
        self.yaw_shaping_function = str(yaw_shaping_function)
        self.brake_speed_weight = self._non_negative(
            brake_speed_weight,
            "brake_speed_weight",
        )
        self.brake_speed_scale = self._positive(
            brake_speed_scale,
            "brake_speed_scale",
        )
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("planar command wrapper requires a Box observation space")
        dtype = self.observation_space.dtype
        command_limit = float(
            max(
                1.0,
                np.max(np.abs(self.initial_command_xy)),
                np.max(np.abs(self.lateral_command_xy)),
            )
        )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.asarray(self.observation_space.low, dtype=dtype),
                    np.full(2, -command_limit, dtype=dtype),
                )
            ),
            high=np.concatenate(
                (
                    np.asarray(self.observation_space.high, dtype=dtype),
                    np.full(2, command_limit, dtype=dtype),
                )
            ),
            dtype=dtype,
        )
        self._reset_state()

    @staticmethod
    def _command(values: Sequence[float], label: str) -> np.ndarray:
        command = np.asarray(values, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError(f"{label} command must contain two finite values")
        if float(np.linalg.norm(command)) <= 0:
            raise ValueError(f"{label} command must be non-zero")
        return command

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

    def _reset_state(self) -> None:
        self._phase = self.PHASE_FORWARD
        self._target_command = self.initial_command_xy.copy()
        self._switch_step = self.switch_step_min
        self._elapsed_steps = 0
        self._brake_steps = 0
        self._below_threshold_run = 0
        self._stop_achieved = False
        self._stop_transition_forced = False
        self._stop_speed = float("nan")
        self._switch_speed = float("nan")
        self._minimum_brake_speed = float("inf")
        self._initial_yaw = 0.0
        self._objective_return = 0.0
        self._outer_shaping_sum = 0.0
        self._tracking_reward_sum = 0.0
        self._tracking_error_squared_sum = 0.0
        self._cross_axis_reward_sum = 0.0
        self._cross_axis_penalty_sum = 0.0
        self._yaw_reward_sum = 0.0
        self._yaw_penalty_sum = 0.0
        self._brake_reward_sum = 0.0
        self._brake_speed_penalty_sum = 0.0
        self._phase_counts = {
            self.PHASE_FORWARD: 0,
            self.PHASE_BRAKE: 0,
            self.PHASE_LATERAL: 0,
        }
        self._phase_velocity_sums = {
            phase: np.zeros(2, dtype=np.float64) for phase in self._phase_counts
        }
        self._yaw_error_squared_sum = 0.0
        self._yaw_error_abs_max = 0.0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._reset_state()
        rng = self.unwrapped.np_random
        self._switch_step = int(
            rng.integers(self.switch_step_min, self.switch_step_max + 1)
        )
        self._initial_yaw = self._torso_yaw()
        info = dict(info)
        info.update(self._live_info(self.PHASE_FORWARD, self.initial_command_xy))
        return self._augment_observation(observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        applied_phase = self._phase
        applied_command = self._target_command.copy()
        observation, inner_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        velocity_xy = qvel[:2].copy()
        planar_speed = float(np.linalg.norm(velocity_xy))
        tracking_value = planar_velocity_tracking_value(
            velocity_xy,
            applied_command,
            scale=self.planar_tracking_scale,
            function=self.planar_tracking_function,
        )
        tracking_reward = self.planar_tracking_weight * tracking_value
        tracking_error_squared = float(
            np.sum(np.square((velocity_xy - applied_command) / self.planar_tracking_scale))
        )
        command_norm = float(np.linalg.norm(applied_command))
        if command_norm > 0:
            command_direction = applied_command / command_norm
            cross_direction = np.asarray(
                [-command_direction[1], command_direction[0]],
                dtype=np.float64,
            )
            cross_axis_velocity = float(np.dot(velocity_xy, cross_direction))
        else:
            cross_axis_velocity = 0.0
        cross_axis_penalty = bounded_squared_signal_penalty(
            cross_axis_velocity,
            scale=self.cross_axis_velocity_scale,
        )
        cross_axis_reward = -self.cross_axis_velocity_weight * cross_axis_penalty
        yaw = self._torso_yaw()
        yaw_error = wrapped_angle_difference(yaw, self._initial_yaw)
        if self.yaw_shaping_function == "pseudo_huber":
            yaw_penalty = float(
                math.sqrt(1.0 + (yaw_error / self.yaw_shaping_scale) ** 2) - 1.0
            )
        else:
            yaw_penalty = bounded_squared_signal_penalty(
                yaw_error,
                scale=self.yaw_shaping_scale,
            )
        yaw_reward = -self.yaw_shaping_weight * yaw_penalty
        brake_speed_penalty = (
            bounded_squared_signal_penalty(planar_speed, scale=self.brake_speed_scale)
            if applied_phase == self.PHASE_BRAKE
            else 0.0
        )
        brake_reward = -self.brake_speed_weight * brake_speed_penalty
        outer_shaping = tracking_reward + cross_axis_reward + yaw_reward + brake_reward
        reward = float(inner_reward) + outer_shaping

        self._objective_return += reward
        self._outer_shaping_sum += outer_shaping
        self._tracking_reward_sum += tracking_reward
        self._tracking_error_squared_sum += tracking_error_squared
        self._cross_axis_reward_sum += cross_axis_reward
        self._cross_axis_penalty_sum += cross_axis_penalty
        self._yaw_reward_sum += yaw_reward
        self._yaw_penalty_sum += yaw_penalty
        self._brake_reward_sum += brake_reward
        self._brake_speed_penalty_sum += brake_speed_penalty
        self._phase_counts[applied_phase] += 1
        self._phase_velocity_sums[applied_phase] += velocity_xy
        if np.isfinite(yaw_error):
            self._yaw_error_squared_sum += yaw_error * yaw_error
            self._yaw_error_abs_max = max(self._yaw_error_abs_max, abs(yaw_error))

        self._elapsed_steps += 1
        if applied_phase == self.PHASE_FORWARD and self._elapsed_steps >= self._switch_step:
            self._phase = self.PHASE_BRAKE
            self._target_command = np.zeros(2, dtype=np.float64)
            self._switch_speed = planar_speed
        elif applied_phase == self.PHASE_BRAKE:
            self._brake_steps += 1
            self._minimum_brake_speed = min(self._minimum_brake_speed, planar_speed)
            if planar_speed <= self.stop_speed_threshold:
                self._below_threshold_run += 1
            else:
                self._below_threshold_run = 0
            eligible = self._brake_steps >= self.brake_min_steps
            if eligible and self._below_threshold_run >= self.stop_consecutive_steps:
                self._stop_achieved = True
                self._stop_speed = planar_speed
                self._phase = self.PHASE_LATERAL
                self._target_command = self.lateral_command_xy.copy()
            elif self._brake_steps >= self.brake_max_steps:
                self._stop_transition_forced = True
                self._stop_speed = planar_speed
                self._phase = self.PHASE_LATERAL
                self._target_command = self.lateral_command_xy.copy()

        info["reward_planar_velocity_tracking"] = float(tracking_reward)
        info["reward_command_cross_axis_shaping"] = float(cross_axis_reward)
        info["reward_yaw_shaping"] = float(yaw_reward)
        info["reward_brake_speed_shaping"] = float(brake_reward)
        info["reward_outer_planar_shaping"] = float(outer_shaping)
        info["proxygap_planar_velocity_step"] = velocity_xy.copy()
        info["proxygap_planar_speed_step"] = planar_speed
        info["proxygap_torso_yaw_step"] = yaw
        info["proxygap_yaw_error_step"] = yaw_error
        info.update(self._live_info(applied_phase, applied_command))
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

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation)
        return np.concatenate(
            (values, self._target_command.astype(values.dtype)),
        )

    def _torso_yaw(self) -> float:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return quaternion_yaw_angle(qpos[3:7])

    def _phase_mean_velocity(self, phase: str) -> np.ndarray:
        count = self._phase_counts[phase]
        if count <= 0:
            return np.full(2, np.nan, dtype=np.float64)
        return self._phase_velocity_sums[phase] / count

    def _live_info(
        self,
        applied_phase: str,
        applied_command: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "proxygap_command_phase_step": applied_phase,
            "proxygap_command_xy_step": applied_command.copy(),
            "proxygap_next_command_phase": self._phase,
            "proxygap_next_command_xy": self._target_command.copy(),
            "proxygap_command_switch_step": self._switch_step,
            "proxygap_brake_steps": self._brake_steps,
            "proxygap_stop_achieved": self._stop_achieved,
            "proxygap_stop_transition_forced": self._stop_transition_forced,
            "proxygap_condition_objective_return": self._objective_return,
            "proxygap_proxy_return": self._objective_return,
        }

    def episode_summary(self) -> dict[str, Any]:
        summary = dict(self.env.episode_summary())
        inner_shaping = float(summary.get("reward_shaping_sum", 0.0))
        base_proxy = float(summary.get("base_proxy_return", 0.0))
        combined_shaping = inner_shaping + self._outer_shaping_sum
        forward_mean = self._phase_mean_velocity(self.PHASE_FORWARD)
        brake_mean = self._phase_mean_velocity(self.PHASE_BRAKE)
        lateral_mean = self._phase_mean_velocity(self.PHASE_LATERAL)
        elapsed = max(1, self._elapsed_steps)
        summary.update(
            {
                "condition_objective_return": self._objective_return,
                "proxy_return": self._objective_return,
                "reward_shaping_sum": combined_shaping,
                "reward_planar_outer_shaping_sum": self._outer_shaping_sum,
                "reward_planar_velocity_tracking_sum": self._tracking_reward_sum,
                "planar_tracking_error_squared_sum": self._tracking_error_squared_sum,
                "reward_command_cross_axis_shaping_sum": self._cross_axis_reward_sum,
                "command_cross_axis_penalty_sum": self._cross_axis_penalty_sum,
                "reward_yaw_shaping_sum": self._yaw_reward_sum,
                "yaw_penalty_sum": self._yaw_penalty_sum,
                "reward_brake_speed_shaping_sum": self._brake_reward_sum,
                "brake_speed_penalty_sum": self._brake_speed_penalty_sum,
                "planar_reward_reconciliation_error": (
                    self._objective_return - base_proxy - combined_shaping
                ),
                "command_switch_step": self._switch_step,
                "forward_phase_steps": self._phase_counts[self.PHASE_FORWARD],
                "brake_phase_steps": self._phase_counts[self.PHASE_BRAKE],
                "lateral_phase_steps": self._phase_counts[self.PHASE_LATERAL],
                "brake_duration_seconds": (
                    self._phase_counts[self.PHASE_BRAKE]
                    * float(self.unwrapped.dt)
                ),
                "stop_achieved": self._stop_achieved,
                "stop_transition_forced": self._stop_transition_forced,
                "switch_planar_speed": self._switch_speed,
                "stop_planar_speed": self._stop_speed,
                "minimum_brake_planar_speed": (
                    self._minimum_brake_speed
                    if np.isfinite(self._minimum_brake_speed)
                    else float("nan")
                ),
                "forward_phase_mean_vx": float(forward_mean[0]),
                "forward_phase_mean_vy": float(forward_mean[1]),
                "brake_phase_mean_vx": float(brake_mean[0]),
                "brake_phase_mean_vy": float(brake_mean[1]),
                "lateral_phase_mean_vx": float(lateral_mean[0]),
                "lateral_phase_mean_vy": float(lateral_mean[1]),
                "yaw_error_rms_rad": math.sqrt(
                    self._yaw_error_squared_sum / elapsed
                ),
                "yaw_error_max_abs_rad": self._yaw_error_abs_max,
                "planar_tracking_weight": self.planar_tracking_weight,
                "planar_tracking_scale": self.planar_tracking_scale,
                "planar_tracking_function": self.planar_tracking_function,
                "cross_axis_velocity_weight": self.cross_axis_velocity_weight,
                "cross_axis_velocity_scale": self.cross_axis_velocity_scale,
                "yaw_shaping_weight": self.yaw_shaping_weight,
                "yaw_shaping_scale": self.yaw_shaping_scale,
                "yaw_shaping_function": self.yaw_shaping_function,
                "brake_speed_weight": self.brake_speed_weight,
                "brake_speed_scale": self.brake_speed_scale,
                "stop_speed_threshold": self.stop_speed_threshold,
                "stop_consecutive_steps": self.stop_consecutive_steps,
                "initial_command_xy": self.initial_command_xy.tolist(),
                "lateral_command_xy": self.lateral_command_xy.tolist(),
                "quarter_turn_action_permutation": (
                    QUARTER_TURN_ACTION_PERMUTATION.tolist()
                ),
            }
        )
        return summary


def make_planar_transition_env(
    *,
    condition_id: str = "planar_transition",
    ctrl_cost_weight: float = 0.5,
    seed: int | None = None,
    render_mode: str | None = None,
    xml_file: str | Path | None = None,
    max_episode_steps: int | None = None,
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
    foot_geom_names: Sequence[str] = DEFAULT_FOOT_GEOM_NAMES,
    augment_previous_applied_action: bool = True,
    initial_command_xy: Sequence[float] = (1.0, 0.0),
    lateral_command_xy: Sequence[float] = (0.0, 1.0),
    switch_step_min: int = 160,
    switch_step_max: int = 320,
    brake_min_steps: int = 5,
    brake_max_steps: int = 30,
    stop_speed_threshold: float = 0.15,
    stop_consecutive_steps: int = 3,
    planar_tracking_weight: float = 0.5,
    planar_tracking_scale: float = 0.5,
    planar_tracking_function: str = "exponential",
    cross_axis_velocity_weight: float = 0.025,
    cross_axis_velocity_scale: float = 1.0,
    yaw_shaping_weight: float = 0.1,
    yaw_shaping_scale: float = math.radians(15.0),
    yaw_shaping_function: str = "bounded_squared",
    brake_speed_weight: float = 0.25,
    brake_speed_scale: float = 0.5,
) -> PlanarCommandTransitionWrapper:
    """Create the pre-pitch reward package plus planar command transition."""
    base = make_proxygap_ant_env(
        ctrl_cost_weight=ctrl_cost_weight,
        condition_id=condition_id,
        seed=seed,
        render_mode=render_mode,
        xml_file=xml_file,
        max_episode_steps=max_episode_steps,
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
        pitch_balance_shaping_weight=0.0,
        foot_geom_names=tuple(foot_geom_names),
        augment_previous_applied_action=augment_previous_applied_action,
    )
    return PlanarCommandTransitionWrapper(
        base,
        initial_command_xy=initial_command_xy,
        lateral_command_xy=lateral_command_xy,
        switch_step_min=switch_step_min,
        switch_step_max=switch_step_max,
        brake_min_steps=brake_min_steps,
        brake_max_steps=brake_max_steps,
        stop_speed_threshold=stop_speed_threshold,
        stop_consecutive_steps=stop_consecutive_steps,
        planar_tracking_weight=planar_tracking_weight,
        planar_tracking_scale=planar_tracking_scale,
        planar_tracking_function=planar_tracking_function,
        cross_axis_velocity_weight=cross_axis_velocity_weight,
        cross_axis_velocity_scale=cross_axis_velocity_scale,
        yaw_shaping_weight=yaw_shaping_weight,
        yaw_shaping_scale=yaw_shaping_scale,
        yaw_shaping_function=yaw_shaping_function,
        brake_speed_weight=brake_speed_weight,
        brake_speed_scale=brake_speed_scale,
    )


def transfer_pretrained_policy(
    source_model: PPO,
    target_model: PPO,
) -> dict[str, Any]:
    """Transfer a straight-walking policy into a command-augmented PPO model.

    Every equal-shaped tensor is copied exactly. The policy and value first
    layers retain all source observation columns, while the two new command
    columns are zero-initialised. Consequently the transferred target produces
    the same initial action as the source for any command value.
    """
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
            and target_value.shape[1] == source_value.shape[1] + 2
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
        "source_observation_dimension": int(source_model.observation_space.shape[0]),
        "target_observation_dimension": int(target_model.observation_space.shape[0]),
        "action_dimension": int(target_model.action_space.shape[0]),
        "copied_parameter_tensors": copied,
        "expanded_parameter_tensors": expanded,
        "new_command_columns_initialised_to_zero": True,
    }


def distill_quarter_turn_command_adapter(
    source_model: PPO,
    target_model: PPO,
    source_env: gym.Env,
    *,
    rollout_steps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    forward_weight: float = 2.0,
    brake_weight: float = 1.0,
    lateral_weight: float = 2.0,
) -> dict[str, Any]:
    """Warm-start command gating from the audited 90-degree motor symmetry.

    Straight observations retain the source action, brake observations target
    zero motor input, and lateral observations target the source action with
    leg pairs rotated by positive 90 degrees. PPO subsequently refines this
    kinematic prior against actual world-frame dynamics.
    """
    if rollout_steps <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("distillation sizes must be positive")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("distillation learning rate must be positive")
    rng = np.random.default_rng(seed)
    observation, _ = source_env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for step in range(int(rollout_steps)):
        observation_array = np.asarray(observation, dtype=np.float32)
        action, _ = source_model.predict(observation_array, deterministic=True)
        action_array = np.asarray(action, dtype=np.float32)
        observations.append(observation_array.copy())
        actions.append(action_array.copy())
        observation, _, terminated, truncated, _ = source_env.step(action_array)
        if terminated or truncated:
            observation, _ = source_env.reset(seed=seed + step + 1)
    base_observations = np.asarray(observations, dtype=np.float32)
    source_actions = np.asarray(actions, dtype=np.float32)
    if target_model.observation_space.shape[0] != base_observations.shape[1] + 2:
        raise ValueError("target model must append exactly two command values")
    forward_observations = np.concatenate(
        (
            base_observations,
            np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (rollout_steps, 1)),
        ),
        axis=1,
    )
    brake_observations = np.concatenate(
        (
            base_observations,
            np.zeros((rollout_steps, 2), dtype=np.float32),
        ),
        axis=1,
    )
    lateral_base = base_observations.copy()
    # The final eight base-observation values are the previous applied motor
    # action in this repository's audited pre-pitch wrapper.
    lateral_base[:, -8:] = lateral_base[:, -8:][:, QUARTER_TURN_ACTION_PERMUTATION]
    lateral_observations = np.concatenate(
        (
            lateral_base,
            np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (rollout_steps, 1)),
        ),
        axis=1,
    )
    inputs = np.concatenate(
        (forward_observations, brake_observations, lateral_observations),
        axis=0,
    )
    targets = np.concatenate(
        (
            source_actions,
            np.zeros_like(source_actions),
            source_actions[:, QUARTER_TURN_ACTION_PERMUTATION],
        ),
        axis=0,
    )
    sample_weights = np.concatenate(
        (
            np.full(rollout_steps, forward_weight, dtype=np.float32),
            np.full(rollout_steps, brake_weight, dtype=np.float32),
            np.full(rollout_steps, lateral_weight, dtype=np.float32),
        )
    )
    actor_parameters = list(target_model.policy.mlp_extractor.policy_net.parameters())
    actor_parameters.extend(target_model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=float(learning_rate))
    device = target_model.device
    target_model.policy.set_training_mode(True)
    final_loss = float("nan")
    for _ in range(int(epochs)):
        indices = rng.permutation(inputs.shape[0])
        for start in range(0, inputs.shape[0], int(batch_size)):
            selection = indices[start : start + int(batch_size)]
            observation_tensor = torch.as_tensor(inputs[selection], device=device)
            target_tensor = torch.as_tensor(targets[selection], device=device)
            weight_tensor = torch.as_tensor(sample_weights[selection], device=device)
            distribution = target_model.policy.get_distribution(observation_tensor)
            predicted = distribution.distribution.mean
            loss = torch.mean(weight_tensor[:, None] * (predicted - target_tensor) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    target_model.policy.set_training_mode(False)
    with torch.no_grad():
        probe_count = min(1024, rollout_steps)
        probe_tensor = torch.as_tensor(inputs[: 3 * probe_count], device=device)
        # The first contiguous slice above contains only forward samples; use
        # explicit slices so every command is audited independently.
        def prediction(values: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(values[:probe_count], device=device)
            return (
                target_model.policy.get_distribution(tensor)
                .distribution.mean.detach()
                .cpu()
                .numpy()
            )

        forward_prediction = prediction(forward_observations)
        brake_prediction = prediction(brake_observations)
        lateral_prediction = prediction(lateral_observations)
    return {
        "enabled": True,
        "rollout_steps": int(rollout_steps),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "final_batch_loss": final_loss,
        "forward_action_mae": float(
            np.mean(np.abs(forward_prediction - source_actions[:probe_count]))
        ),
        "brake_action_mae": float(np.mean(np.abs(brake_prediction))),
        "lateral_permuted_action_mae": float(
            np.mean(
                np.abs(
                    lateral_prediction
                    - source_actions[:probe_count, QUARTER_TURN_ACTION_PERMUTATION]
                )
            )
        ),
        "quarter_turn_action_permutation": QUARTER_TURN_ACTION_PERMUTATION.tolist(),
    }


def make_ppo_from_config(
    env: gym.Env,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> PPO:
    """Build the locked PPO architecture used for planar transition training."""
    policy_kwargs = dict(config["policy_kwargs"])
    activation_name = policy_kwargs.pop("activation_fn", "Tanh")
    if activation_name != "Tanh":
        raise ValueError("Only the audited Tanh activation is supported")
    from torch import nn

    policy_kwargs["activation_fn"] = nn.Tanh
    return PPO(
        str(config["policy"]),
        env,
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        learning_rate=float(config["learning_rate"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        normalize_advantage=bool(config["normalize_advantage"]),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        device=str(config.get("device", "cpu")),
        verbose=0,
    )
