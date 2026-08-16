"""Ant-v5 construction, reward decomposition and diagnostic logging."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, TextIO

import gymnasium as gym
import numpy as np

from .metrics import (
    DEFAULT_ACTION_SATURATION_THRESHOLD,
    DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
    EPSILON,
    EpisodeMetrics,
    quaternion_tilt_angle,
)


STEP_LOG_SCHEMA = [
    "episode_index",
    "step_index",
    "condition_id",
    "x_position",
    "y_position",
    "torso_height",
    "state_is_finite",
    "lateral_offset",
    "lateral_velocity",
    "torso_tilt_rad",
    "squared_action_step",
    "squared_action_change_step",
    "action_change_defined_step",
    "action_saturation_fraction_step",
    "proposed_action",
    "applied_action",
    "proposed_action_change_l2_step",
    "requested_action_change_l2_step",
    "applied_action_change_l2_step",
    "action_correction_l2_step",
    "action_slew_intervened_step",
    "action_constraint_enabled",
    "action_slew_l2_limit",
    "condition_objective_reward_step",
    "base_proxy_reward_step",
    "common_rescored_reward_step",
    "shaping_reward_step",
    "reward_forward_step",
    "reward_forward_tracking_step",
    "reward_forward_replacement_step",
    "forward_velocity_target",
    "forward_velocity_tracking_scale",
    "reward_ctrl_step",
    "reward_contact_step",
    "reward_survive_step",
    "reward_lateral_shaping_step",
    "lateral_shaping_signal",
    "lateral_velocity_target",
    "lateral_penalty_step",
    "reward_effort_shaping_step",
    "reward_orientation_shaping_step",
    "orientation_shaping_function",
    "orientation_penalty_step",
    "reward_action_rate_shaping_step",
    "action_rate_penalty_step",
    "root_vertical_velocity_step",
    "root_roll_pitch_angular_speed_step",
    "vertical_velocity_penalty_step",
    "roll_pitch_angular_velocity_penalty_step",
    "reward_vertical_velocity_shaping_step",
    "reward_roll_pitch_angular_velocity_shaping_step",
    "terminated",
    "truncated",
    "termination_category",
]


SUPPORTED_ORIENTATION_SHAPING_FUNCTIONS = ("tanh", "cosine")
SUPPORTED_LATERAL_SHAPING_SIGNALS = (
    "offset_tanh",
    "velocity_tanh_squared",
)


def project_action_l2_slew(
    proposed_action: np.ndarray,
    previous_applied_action: np.ndarray,
    *,
    limit: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, bool, float, float]:
    """Project a bounded proposed action onto an L2 slew ball.

    The function is deterministic and side-effect free so the control-layer
    constraint can be regression-tested independently of MuJoCo.
    """
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("action slew limit must be positive and finite")
    proposed = np.asarray(proposed_action, dtype=np.float64)
    previous = np.asarray(previous_applied_action, dtype=np.float64)
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if proposed.shape != previous.shape or proposed.shape != low.shape:
        raise ValueError("action, previous action and bounds must share one shape")
    if high.shape != low.shape:
        raise ValueError("action bounds must share one shape")
    if not np.isfinite(proposed).all() or not np.isfinite(previous).all():
        raise ValueError("action slew projection requires finite actions")
    bounded_proposed = np.clip(proposed, low, high)
    requested_delta = bounded_proposed - previous
    requested_norm = float(np.linalg.norm(requested_delta, ord=2))
    if requested_norm <= limit:
        applied = bounded_proposed
        intervened = False
    else:
        applied = previous + requested_delta * (limit / requested_norm)
        applied = np.clip(applied, low, high)
        intervened = True
    correction_norm = float(np.linalg.norm(bounded_proposed - applied, ord=2))
    return applied, intervened, requested_norm, correction_norm


def orientation_penalty_value(
    torso_tilt: float,
    *,
    function: str,
    scale: float = 1.0,
) -> float:
    """Return a bounded posture penalty before applying its reward weight.

    ``cosine`` is the pilot intervention: ``(1 - cos(theta)) / 2`` maps an
    upright torso to zero and a fully inverted torso to one. ``tanh`` remains
    available only so historical configurations retain their original meaning.
    """
    if scale <= 0:
        raise ValueError("orientation shaping scale must be positive")
    function_name = str(function).strip().lower()
    if function_name not in SUPPORTED_ORIENTATION_SHAPING_FUNCTIONS:
        raise ValueError(
            "orientation shaping function must be one of "
            f"{SUPPORTED_ORIENTATION_SHAPING_FUNCTIONS}"
        )
    if not np.isfinite(torso_tilt):
        return 1.0
    tilt = float(np.clip(torso_tilt, 0.0, np.pi))
    if function_name == "cosine":
        if not np.isclose(scale, 1.0):
            raise ValueError("cosine orientation shaping requires scale=1.0")
        return float((1.0 - np.cos(tilt)) / 2.0)
    return float(np.tanh(tilt / scale))


def lateral_penalty_value(
    *,
    lateral_offset: float,
    lateral_velocity: float,
    velocity_target: float,
    signal: str,
    scale: float,
) -> float:
    """Return a bounded lateral-control penalty before reward weighting.

    ``offset_tanh`` preserves the historical implementation. The corrected
    ``velocity_tanh_squared`` signal uses the Ant-v5 lateral velocity that is
    present in the policy observation, avoiding an unobserved absolute-position
    target.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("lateral shaping scale must be positive and finite")
    signal_name = str(signal).strip().lower()
    if signal_name not in SUPPORTED_LATERAL_SHAPING_SIGNALS:
        raise ValueError(
            "lateral shaping signal must be one of "
            f"{SUPPORTED_LATERAL_SHAPING_SIGNALS}"
        )
    if signal_name == "offset_tanh":
        if not np.isfinite(lateral_offset):
            return 1.0
        return float(np.tanh(abs(float(lateral_offset)) / scale))
    if not np.isfinite(lateral_velocity) or not np.isfinite(velocity_target):
        return 1.0
    scaled_error = (float(lateral_velocity) - float(velocity_target)) / scale
    return float(np.tanh(scaled_error * scaled_error))


def forward_velocity_tracking_value(
    forward_velocity: float,
    *,
    target: float,
    scale: float,
) -> float:
    """Return a bounded reward for tracking a commanded forward velocity."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("forward velocity tracking scale must be positive and finite")
    if not np.isfinite(forward_velocity) or not np.isfinite(target):
        return 0.0
    scaled_error = (float(forward_velocity) - float(target)) / float(scale)
    return float(np.exp(-(scaled_error * scaled_error)))


def normalised_action_rate_penalty(
    current_action: np.ndarray,
    previous_action: np.ndarray | None,
) -> float:
    """Return squared action change normalised to [0, 1] for [-1, 1] actions."""
    if previous_action is None:
        return 0.0
    current = np.asarray(current_action, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)
    if current.shape != previous.shape or current.size == 0:
        raise ValueError("current and previous actions must share one non-empty shape")
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        return 1.0
    squared_change = float(np.sum(np.square(current - previous)))
    return float(np.clip(squared_change / (4.0 * current.size), 0.0, 1.0))


def bounded_squared_signal_penalty(value: float, *, scale: float) -> float:
    """Map a signed physical signal to a finite penalty in [0, 1]."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bounded signal scale must be positive and finite")
    if not np.isfinite(value):
        return 1.0
    scaled = float(value) / float(scale)
    return float(np.tanh(scaled * scaled))


class ProxyGapAntWrapper(gym.Wrapper):
    """Record condition objectives and external locomotion diagnostics."""

    def __init__(
        self,
        env: gym.Env,
        condition_id: str,
        ctrl_cost_weight: float,
        forward_progress_shaping_weight: float = 0.0,
        lateral_drift_shaping_weight: float = 0.0,
        lateral_drift_shaping_scale: float = 1.0,
        lateral_shaping_signal: str = "offset_tanh",
        lateral_velocity_target: float = 0.0,
        effort_shaping_weight: float = 0.0,
        effort_shaping_scale: float = 1.0,
        orientation_shaping_weight: float = 0.0,
        orientation_shaping_scale: float = 1.0,
        orientation_shaping_function: str = "tanh",
        replace_forward_reward_with_tracking: bool = False,
        forward_velocity_target: float = 1.0,
        forward_velocity_tracking_scale: float = 0.5,
        action_rate_shaping_weight: float = 0.0,
        vertical_velocity_shaping_weight: float = 0.0,
        vertical_velocity_shaping_scale: float = 1.0,
        roll_pitch_angular_velocity_shaping_weight: float = 0.0,
        roll_pitch_angular_velocity_shaping_scale: float = 1.0,
        common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
        effort_distance_min: float = EPSILON,
        action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
        augment_previous_applied_action: bool = False,
        action_slew_l2_limit: float | None = None,
        step_log_path: str | Path | None = None,
    ) -> None:
        super().__init__(env)
        if (
            lateral_drift_shaping_scale <= 0
            or effort_shaping_scale <= 0
            or orientation_shaping_scale <= 0
        ):
            raise ValueError("shaping scales must be positive")
        self.condition_id = condition_id
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.forward_progress_shaping_weight = float(forward_progress_shaping_weight)
        self.lateral_drift_shaping_weight = float(lateral_drift_shaping_weight)
        self.lateral_drift_shaping_scale = float(lateral_drift_shaping_scale)
        self.lateral_shaping_signal = str(lateral_shaping_signal).strip().lower()
        self.lateral_velocity_target = float(lateral_velocity_target)
        lateral_penalty_value(
            lateral_offset=0.0,
            lateral_velocity=self.lateral_velocity_target,
            velocity_target=self.lateral_velocity_target,
            signal=self.lateral_shaping_signal,
            scale=self.lateral_drift_shaping_scale,
        )
        self.effort_shaping_weight = float(effort_shaping_weight)
        self.effort_shaping_scale = float(effort_shaping_scale)
        self.orientation_shaping_weight = float(orientation_shaping_weight)
        self.orientation_shaping_scale = float(orientation_shaping_scale)
        self.orientation_shaping_function = str(
            orientation_shaping_function
        ).strip().lower()
        orientation_penalty_value(
            0.0,
            function=self.orientation_shaping_function,
            scale=self.orientation_shaping_scale,
        )
        self.replace_forward_reward_with_tracking = bool(
            replace_forward_reward_with_tracking
        )
        self.forward_velocity_target = float(forward_velocity_target)
        self.forward_velocity_tracking_scale = float(
            forward_velocity_tracking_scale
        )
        forward_velocity_tracking_value(
            self.forward_velocity_target,
            target=self.forward_velocity_target,
            scale=self.forward_velocity_tracking_scale,
        )
        self.action_rate_shaping_weight = float(action_rate_shaping_weight)
        if self.action_rate_shaping_weight < 0 or not np.isfinite(
            self.action_rate_shaping_weight
        ):
            raise ValueError("action rate shaping weight must be finite and non-negative")
        self.vertical_velocity_shaping_weight = float(
            vertical_velocity_shaping_weight
        )
        self.vertical_velocity_shaping_scale = float(vertical_velocity_shaping_scale)
        self.roll_pitch_angular_velocity_shaping_weight = float(
            roll_pitch_angular_velocity_shaping_weight
        )
        self.roll_pitch_angular_velocity_shaping_scale = float(
            roll_pitch_angular_velocity_shaping_scale
        )
        for name, weight in (
            ("vertical_velocity_shaping_weight", self.vertical_velocity_shaping_weight),
            (
                "roll_pitch_angular_velocity_shaping_weight",
                self.roll_pitch_angular_velocity_shaping_weight,
            ),
        ):
            if weight < 0 or not np.isfinite(weight):
                raise ValueError(f"{name} must be finite and non-negative")
        bounded_squared_signal_penalty(
            0.0, scale=self.vertical_velocity_shaping_scale
        )
        bounded_squared_signal_penalty(
            0.0, scale=self.roll_pitch_angular_velocity_shaping_scale
        )
        self.common_rescore_ctrl_cost_weight = float(common_rescore_ctrl_cost_weight)
        self.effort_distance_min = float(effort_distance_min)
        self.action_saturation_threshold = float(action_saturation_threshold)
        self.augment_previous_applied_action = bool(
            augment_previous_applied_action
        )
        self.action_slew_l2_limit = (
            None if action_slew_l2_limit is None else float(action_slew_l2_limit)
        )
        if self.action_slew_l2_limit is not None:
            if self.action_slew_l2_limit <= 0 or not np.isfinite(
                self.action_slew_l2_limit
            ):
                raise ValueError("action_slew_l2_limit must be positive and finite")
            if not self.augment_previous_applied_action:
                raise ValueError(
                    "The previous applied action must be observable when the "
                    "action slew constraint is enabled"
                )
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("ProxyGap Ant requires a Box action space")
        self._action_low = np.asarray(self.action_space.low, dtype=np.float64)
        self._action_high = np.asarray(self.action_space.high, dtype=np.float64)
        if self.augment_previous_applied_action:
            if not isinstance(self.observation_space, gym.spaces.Box):
                raise TypeError("ProxyGap Ant requires a Box observation space")
            observation_dtype = self.observation_space.dtype
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    (
                        np.asarray(self.observation_space.low, dtype=observation_dtype),
                        self._action_low.astype(observation_dtype),
                    )
                ),
                high=np.concatenate(
                    (
                        np.asarray(self.observation_space.high, dtype=observation_dtype),
                        self._action_high.astype(observation_dtype),
                    )
                ),
                dtype=observation_dtype,
            )
        healthy_range = tuple(float(value) for value in self.unwrapped._healthy_z_range)
        action_dimension = int(np.prod(self.action_space.shape))
        self.metrics = EpisodeMetrics(
            condition_ctrl_cost_weight=self.ctrl_cost_weight,
            common_rescore_ctrl_cost_weight=self.common_rescore_ctrl_cost_weight,
            effort_distance_min=self.effort_distance_min,
            action_saturation_threshold=self.action_saturation_threshold,
            environment_dt=float(self.unwrapped.dt),
            action_dimension=action_dimension,
            healthy_z_range=(healthy_range[0], healthy_range[1]),
            evaluation_horizon_steps=int(self.env.spec.max_episode_steps or 1000),
        )
        self._episode_index = 0
        self._step_index = 0
        self._previous_applied_action = np.zeros(action_dimension, dtype=np.float64)
        self._previous_proposed_action: np.ndarray | None = None
        self._slew_intervention_count = 0
        self._cumulative_action_correction_l2 = 0.0
        self._max_action_correction_l2 = 0.0
        self._cumulative_proposed_squared_action_change = 0.0
        self._proposed_action_change_transition_count = 0
        self._reward_forward_tracking_sum = 0.0
        self._reward_forward_replacement_sum = 0.0
        self._reward_action_rate_shaping_sum = 0.0
        self._action_rate_penalty_sum = 0.0
        self._reward_vertical_velocity_shaping_sum = 0.0
        self._vertical_velocity_penalty_sum = 0.0
        self._reward_roll_pitch_angular_velocity_shaping_sum = 0.0
        self._roll_pitch_angular_velocity_penalty_sum = 0.0
        self._step_handle: TextIO | None = None
        self._step_writer: csv.DictWriter | None = None
        if step_log_path is not None:
            path = Path(step_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._step_handle = gzip.open(path, "wt", newline="", encoding="utf-8")
            self._step_writer = csv.DictWriter(self._step_handle, fieldnames=STEP_LOG_SCHEMA)
            self._step_writer.writeheader()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._previous_applied_action.fill(0.0)
        self._previous_proposed_action = None
        self._slew_intervention_count = 0
        self._cumulative_action_correction_l2 = 0.0
        self._max_action_correction_l2 = 0.0
        self._cumulative_proposed_squared_action_change = 0.0
        self._proposed_action_change_transition_count = 0
        self._reward_forward_tracking_sum = 0.0
        self._reward_forward_replacement_sum = 0.0
        self._reward_action_rate_shaping_sum = 0.0
        self._action_rate_penalty_sum = 0.0
        self._reward_vertical_velocity_shaping_sum = 0.0
        self._vertical_velocity_penalty_sum = 0.0
        self._reward_roll_pitch_angular_velocity_shaping_sum = 0.0
        self._roll_pitch_angular_velocity_penalty_sum = 0.0
        x_position, y_position = self._root_xy(info)
        self.metrics.reset(initial_x=x_position, initial_y=y_position)
        self._episode_index += 1
        self._step_index = 0
        info = dict(info)
        info.update(self._prefixed_summary())
        return self._augment_observation(observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        proposed_action = np.asarray(action, dtype=np.float64).reshape(
            self.action_space.shape
        )
        proposed_action = np.clip(
            proposed_action,
            self._action_low,
            self._action_high,
        )
        previous_applied_action = self._previous_applied_action.copy()
        if self.action_slew_l2_limit is None:
            applied_action = proposed_action.copy()
            requested_action_change_l2 = float(
                np.linalg.norm(proposed_action - previous_applied_action, ord=2)
            )
            action_slew_intervened = False
            action_correction_l2 = 0.0
        else:
            (
                applied_action,
                action_slew_intervened,
                requested_action_change_l2,
                action_correction_l2,
            ) = project_action_l2_slew(
                proposed_action,
                previous_applied_action,
                limit=self.action_slew_l2_limit,
                action_low=self._action_low,
                action_high=self._action_high,
            )
        applied_action_change_l2 = float(
            np.linalg.norm(applied_action - previous_applied_action, ord=2)
        )
        previous_proposed_action = (
            None
            if self._previous_proposed_action is None
            else self._previous_proposed_action.copy()
        )
        if previous_proposed_action is None:
            proposed_action_change_l2 = float("nan")
        else:
            proposed_delta = proposed_action - previous_proposed_action
            proposed_action_change_l2 = float(np.linalg.norm(proposed_delta, ord=2))
            self._cumulative_proposed_squared_action_change += float(
                np.dot(proposed_delta, proposed_delta)
            )
            self._proposed_action_change_transition_count += 1
        self._previous_proposed_action = proposed_action.copy()
        self._previous_applied_action = applied_action.copy()
        if action_slew_intervened:
            self._slew_intervention_count += 1
        self._cumulative_action_correction_l2 += action_correction_l2
        self._max_action_correction_l2 = max(
            self._max_action_correction_l2,
            action_correction_l2,
        )
        observation, base_reward, terminated, truncated, info = self.env.step(
            applied_action
        )
        info = dict(info)
        x_position, y_position = self._root_xy(info)
        lateral_offset = abs(y_position - self.metrics.initial_y)
        lateral_velocity = self._root_y_velocity(info)
        torso_tilt = self._torso_tilt()
        torso_height, state_is_finite = self._health_state()
        squared_action = float(np.sum(np.square(applied_action)))
        forward_velocity = self._root_x_velocity(info)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        root_vertical_velocity = float(qvel[2])
        root_roll_pitch_angular_speed = float(np.linalg.norm(qvel[3:5]))
        saturated_fraction = float(
            np.mean(np.abs(applied_action) >= self.action_saturation_threshold)
        )

        forward_shaping_reward = self.forward_progress_shaping_weight * float(
            info.get("reward_forward", 0.0)
        )
        forward_tracking_reward = forward_velocity_tracking_value(
            forward_velocity,
            target=self.forward_velocity_target,
            scale=self.forward_velocity_tracking_scale,
        )
        forward_replacement_reward = (
            forward_tracking_reward - float(info.get("reward_forward", 0.0))
            if self.replace_forward_reward_with_tracking
            else 0.0
        )
        lateral_penalty = lateral_penalty_value(
            lateral_offset=lateral_offset,
            lateral_velocity=lateral_velocity,
            velocity_target=self.lateral_velocity_target,
            signal=self.lateral_shaping_signal,
            scale=self.lateral_drift_shaping_scale,
        )
        lateral_shaping_reward = -self.lateral_drift_shaping_weight * lateral_penalty
        effort_shaping_reward = -self.effort_shaping_weight * float(
            np.tanh(squared_action / self.effort_shaping_scale)
        )
        orientation_penalty = orientation_penalty_value(
            torso_tilt,
            function=self.orientation_shaping_function,
            scale=self.orientation_shaping_scale,
        )
        orientation_shaping_reward = (
            -self.orientation_shaping_weight * orientation_penalty
        )
        action_rate_penalty = normalised_action_rate_penalty(
            proposed_action,
            previous_proposed_action,
        )
        action_rate_shaping_reward = (
            -self.action_rate_shaping_weight * action_rate_penalty
        )
        vertical_velocity_penalty = bounded_squared_signal_penalty(
            root_vertical_velocity,
            scale=self.vertical_velocity_shaping_scale,
        )
        vertical_velocity_shaping_reward = (
            -self.vertical_velocity_shaping_weight * vertical_velocity_penalty
        )
        roll_pitch_angular_velocity_penalty = bounded_squared_signal_penalty(
            root_roll_pitch_angular_speed,
            scale=self.roll_pitch_angular_velocity_shaping_scale,
        )
        roll_pitch_angular_velocity_shaping_reward = (
            -self.roll_pitch_angular_velocity_shaping_weight
            * roll_pitch_angular_velocity_penalty
        )
        shaping_reward = (
            forward_shaping_reward
            + forward_replacement_reward
            + lateral_shaping_reward
            + effort_shaping_reward
            + orientation_shaping_reward
            + action_rate_shaping_reward
            + vertical_velocity_shaping_reward
            + roll_pitch_angular_velocity_shaping_reward
        )
        self._reward_forward_tracking_sum += forward_tracking_reward
        self._reward_forward_replacement_sum += forward_replacement_reward
        self._reward_action_rate_shaping_sum += action_rate_shaping_reward
        self._action_rate_penalty_sum += action_rate_penalty
        self._reward_vertical_velocity_shaping_sum += vertical_velocity_shaping_reward
        self._vertical_velocity_penalty_sum += vertical_velocity_penalty
        self._reward_roll_pitch_angular_velocity_shaping_sum += (
            roll_pitch_angular_velocity_shaping_reward
        )
        self._roll_pitch_angular_velocity_penalty_sum += (
            roll_pitch_angular_velocity_penalty
        )
        observed_reward = float(base_reward) + shaping_reward
        common_rescored_reward = float(
            info.get("reward_forward", 0.0)
            + info.get("reward_survive", 0.0)
            + info.get("reward_contact", 0.0)
            - self.common_rescore_ctrl_cost_weight * squared_action
        )

        info["reward_base_proxy"] = float(base_reward)
        info["reward_shaping"] = float(shaping_reward)
        info["reward_forward_shaping"] = float(forward_shaping_reward)
        info["reward_forward_tracking"] = float(forward_tracking_reward)
        info["reward_forward_replacement"] = float(forward_replacement_reward)
        info["reward_lateral_shaping"] = float(lateral_shaping_reward)
        info["lateral_penalty"] = float(lateral_penalty)
        info["reward_effort_shaping"] = float(effort_shaping_reward)
        info["reward_orientation_shaping"] = float(orientation_shaping_reward)
        info["reward_action_rate_shaping"] = float(action_rate_shaping_reward)
        info["action_rate_penalty"] = float(action_rate_penalty)
        info["reward_vertical_velocity_shaping"] = float(
            vertical_velocity_shaping_reward
        )
        info["vertical_velocity_penalty"] = float(vertical_velocity_penalty)
        info["reward_roll_pitch_angular_velocity_shaping"] = float(
            roll_pitch_angular_velocity_shaping_reward
        )
        info["roll_pitch_angular_velocity_penalty"] = float(
            roll_pitch_angular_velocity_penalty
        )
        info["orientation_penalty"] = float(orientation_penalty)
        info["reward_common_rescored"] = common_rescored_reward
        self.metrics.update(
            action=applied_action,
            reward=observed_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            torso_tilt=torso_tilt,
            torso_height=torso_height,
            state_is_finite=state_is_finite,
        )
        summary: dict[str, Any] = (
            self.metrics.summary()
            if terminated or truncated
            else self.metrics.live_summary()
        )
        summary.update(self._constraint_summary())
        info.update({f"proxygap_{key}": value for key, value in summary.items()})
        info["proxygap_torso_tilt_step"] = torso_tilt
        info["proxygap_condition_id"] = self.condition_id
        info["proxygap_ctrl_cost_weight"] = self.ctrl_cost_weight
        info["proxygap_common_rescore_ctrl_cost_weight"] = (
            self.common_rescore_ctrl_cost_weight
        )
        info["proxygap_forward_progress_shaping_weight"] = (
            self.forward_progress_shaping_weight
        )
        info["proxygap_lateral_drift_shaping_weight"] = self.lateral_drift_shaping_weight
        info["proxygap_lateral_drift_shaping_scale"] = self.lateral_drift_shaping_scale
        info["proxygap_lateral_shaping_signal"] = self.lateral_shaping_signal
        info["proxygap_lateral_velocity_target"] = self.lateral_velocity_target
        info["proxygap_lateral_penalty_step"] = lateral_penalty
        info["proxygap_lateral_velocity_step"] = lateral_velocity
        info["proxygap_effort_shaping_weight"] = self.effort_shaping_weight
        info["proxygap_orientation_shaping_weight"] = self.orientation_shaping_weight
        info["proxygap_orientation_shaping_function"] = (
            self.orientation_shaping_function
        )
        info["proxygap_orientation_penalty_step"] = orientation_penalty
        info["proxygap_replace_forward_reward_with_tracking"] = (
            self.replace_forward_reward_with_tracking
        )
        info["proxygap_forward_velocity_target"] = self.forward_velocity_target
        info["proxygap_forward_velocity_tracking_scale"] = (
            self.forward_velocity_tracking_scale
        )
        info["proxygap_forward_velocity_step"] = forward_velocity
        info["proxygap_action_rate_shaping_weight"] = self.action_rate_shaping_weight
        info["proxygap_action_rate_penalty_step"] = action_rate_penalty
        info["proxygap_root_vertical_velocity_step"] = root_vertical_velocity
        info["proxygap_root_roll_pitch_angular_speed_step"] = (
            root_roll_pitch_angular_speed
        )
        info["proxygap_lateral_offset_step"] = lateral_offset
        info["proxygap_torso_height_step"] = torso_height
        info["proxygap_state_is_finite_step"] = state_is_finite
        info["proxygap_squared_action_step"] = squared_action
        info["proxygap_squared_action_change_step"] = (
            self.metrics.latest_squared_action_change
        )
        info["proxygap_action_change_defined_step"] = bool(
            self.metrics.action_change_transition_count > 0
        )
        info["proxygap_action_saturation_fraction_step"] = saturated_fraction
        info["proxygap_proposed_action"] = proposed_action.copy()
        info["proxygap_applied_action"] = applied_action.copy()
        info["proxygap_proposed_action_change_l2_step"] = (
            proposed_action_change_l2
        )
        info["proxygap_requested_action_change_l2_step"] = (
            requested_action_change_l2
        )
        info["proxygap_applied_action_change_l2_step"] = applied_action_change_l2
        info["proxygap_action_correction_l2_step"] = action_correction_l2
        info["proxygap_action_slew_intervened_step"] = action_slew_intervened
        self._step_index += 1
        self._write_step_record(
            x_position=x_position,
            y_position=y_position,
            torso_height=torso_height,
            state_is_finite=state_is_finite,
            lateral_offset=lateral_offset,
            lateral_velocity=lateral_velocity,
            lateral_penalty=lateral_penalty,
            torso_tilt=torso_tilt,
            squared_action=squared_action,
            squared_action_change=self.metrics.latest_squared_action_change,
            action_change_defined=bool(
                self.metrics.action_change_transition_count > 0
            ),
            saturated_fraction=saturated_fraction,
            proposed_action=proposed_action,
            applied_action=applied_action,
            proposed_action_change_l2=proposed_action_change_l2,
            requested_action_change_l2=requested_action_change_l2,
            applied_action_change_l2=applied_action_change_l2,
            action_correction_l2=action_correction_l2,
            action_slew_intervened=action_slew_intervened,
            observed_reward=observed_reward,
            base_reward=float(base_reward),
            common_rescored_reward=common_rescored_reward,
            shaping_reward=shaping_reward,
            orientation_penalty=orientation_penalty,
            forward_tracking_reward=forward_tracking_reward,
            forward_replacement_reward=forward_replacement_reward,
            action_rate_shaping_reward=action_rate_shaping_reward,
            action_rate_penalty=action_rate_penalty,
            root_vertical_velocity=root_vertical_velocity,
            root_roll_pitch_angular_speed=root_roll_pitch_angular_speed,
            vertical_velocity_penalty=vertical_velocity_penalty,
            roll_pitch_angular_velocity_penalty=(
                roll_pitch_angular_velocity_penalty
            ),
            vertical_velocity_shaping_reward=vertical_velocity_shaping_reward,
            roll_pitch_angular_velocity_shaping_reward=(
                roll_pitch_angular_velocity_shaping_reward
            ),
            info=info,
            terminated=terminated,
            truncated=truncated,
            termination_category=self.metrics.termination_category,
        )
        return (
            self._augment_observation(observation),
            observed_reward,
            terminated,
            truncated,
            info,
        )

    def episode_summary(self) -> dict[str, Any]:
        summary = self.metrics.summary()
        summary.update(self._constraint_summary())
        summary.update(
            {
                "condition_id": self.condition_id,
                "ctrl_cost_weight": self.ctrl_cost_weight,
                "forward_progress_shaping_weight": self.forward_progress_shaping_weight,
                "lateral_drift_shaping_weight": self.lateral_drift_shaping_weight,
                "lateral_drift_shaping_scale": self.lateral_drift_shaping_scale,
                "lateral_shaping_signal": self.lateral_shaping_signal,
                "lateral_velocity_target": self.lateral_velocity_target,
                "effort_shaping_weight": self.effort_shaping_weight,
                "effort_shaping_scale": self.effort_shaping_scale,
                "orientation_shaping_weight": self.orientation_shaping_weight,
                "orientation_shaping_scale": self.orientation_shaping_scale,
                "orientation_shaping_function": self.orientation_shaping_function,
                "replace_forward_reward_with_tracking": self.replace_forward_reward_with_tracking,
                "forward_velocity_target": self.forward_velocity_target,
                "forward_velocity_tracking_scale": self.forward_velocity_tracking_scale,
                "reward_forward_tracking_sum": self._reward_forward_tracking_sum,
                "reward_forward_replacement_sum": self._reward_forward_replacement_sum,
                "action_rate_shaping_weight": self.action_rate_shaping_weight,
                "reward_action_rate_shaping_sum": self._reward_action_rate_shaping_sum,
                "action_rate_penalty_sum": self._action_rate_penalty_sum,
                "vertical_velocity_shaping_weight": self.vertical_velocity_shaping_weight,
                "vertical_velocity_shaping_scale": self.vertical_velocity_shaping_scale,
                "reward_vertical_velocity_shaping_sum": self._reward_vertical_velocity_shaping_sum,
                "vertical_velocity_penalty_sum": self._vertical_velocity_penalty_sum,
                "roll_pitch_angular_velocity_shaping_weight": self.roll_pitch_angular_velocity_shaping_weight,
                "roll_pitch_angular_velocity_shaping_scale": self.roll_pitch_angular_velocity_shaping_scale,
                "reward_roll_pitch_angular_velocity_shaping_sum": self._reward_roll_pitch_angular_velocity_shaping_sum,
                "roll_pitch_angular_velocity_penalty_sum": self._roll_pitch_angular_velocity_penalty_sum,
            }
        )
        return summary

    def close(self) -> None:
        if self._step_handle is not None and not self._step_handle.closed:
            self._step_handle.flush()
            self._step_handle.close()
        super().close()

    def _prefixed_summary(self) -> dict[str, Any]:
        summary = self.metrics.summary()
        summary.update(self._constraint_summary())
        return {f"proxygap_{key}": value for key, value in summary.items()}

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation)
        if not self.augment_previous_applied_action:
            return observation_array
        return np.concatenate(
            (
                observation_array,
                self._previous_applied_action.astype(observation_array.dtype),
            )
        )

    def _constraint_summary(self) -> dict[str, Any]:
        episode_length = int(self.metrics.episode_length)
        proposed_mean_squared_change = (
            self._cumulative_proposed_squared_action_change
            / self._proposed_action_change_transition_count
            if self._proposed_action_change_transition_count > 0
            else float("nan")
        )
        return {
            "action_observation_augmented": self.augment_previous_applied_action,
            "action_constraint_enabled": self.action_slew_l2_limit is not None,
            "action_slew_l2_limit": (
                float(self.action_slew_l2_limit)
                if self.action_slew_l2_limit is not None
                else float("nan")
            ),
            "action_slew_intervention_count": int(self._slew_intervention_count),
            "action_slew_intervention_rate": (
                self._slew_intervention_count / episode_length
                if episode_length > 0
                else 0.0
            ),
            "cumulative_action_correction_l2": float(
                self._cumulative_action_correction_l2
            ),
            "mean_action_correction_l2": (
                self._cumulative_action_correction_l2 / episode_length
                if episode_length > 0
                else 0.0
            ),
            "max_action_correction_l2": float(self._max_action_correction_l2),
            "cumulative_proposed_squared_action_change": float(
                self._cumulative_proposed_squared_action_change
            ),
            "proposed_action_change_transition_count": int(
                self._proposed_action_change_transition_count
            ),
            "proposed_normalised_action_roughness": (
                proposed_mean_squared_change / (4.0 * self.metrics.action_dimension)
                if np.isfinite(proposed_mean_squared_change)
                else float("nan")
            ),
        }

    def _root_xy(self, info: dict[str, Any]) -> tuple[float, float]:
        if "x_position" in info and "y_position" in info:
            return float(info["x_position"]), float(info["y_position"])
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return float(qpos[0]), float(qpos[1])

    def _root_y_velocity(self, info: dict[str, Any]) -> float:
        if "y_velocity" in info:
            return float(info["y_velocity"])
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        return float(qvel[1])

    def _root_x_velocity(self, info: dict[str, Any]) -> float:
        if "x_velocity" in info:
            return float(info["x_velocity"])
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        return float(qvel[0])

    def _torso_tilt(self) -> float:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return quaternion_tilt_angle(qpos[3:7])

    def _health_state(self) -> tuple[float, bool]:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        return float(qpos[2]), bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())

    def _write_step_record(self, **values: Any) -> None:
        if self._step_writer is None:
            return
        info = values.pop("info")
        row = {
            "episode_index": self._episode_index,
            "step_index": self._step_index,
            "condition_id": self.condition_id,
            "x_position": values["x_position"],
            "y_position": values["y_position"],
            "torso_height": values["torso_height"],
            "state_is_finite": values["state_is_finite"],
            "lateral_offset": values["lateral_offset"],
            "lateral_velocity": values["lateral_velocity"],
            "torso_tilt_rad": values["torso_tilt"],
            "squared_action_step": values["squared_action"],
            "squared_action_change_step": values["squared_action_change"],
            "action_change_defined_step": values["action_change_defined"],
            "action_saturation_fraction_step": values["saturated_fraction"],
            "proposed_action": json.dumps(
                np.asarray(values["proposed_action"]).tolist(), separators=(",", ":")
            ),
            "applied_action": json.dumps(
                np.asarray(values["applied_action"]).tolist(), separators=(",", ":")
            ),
            "proposed_action_change_l2_step": values[
                "proposed_action_change_l2"
            ],
            "requested_action_change_l2_step": values[
                "requested_action_change_l2"
            ],
            "applied_action_change_l2_step": values[
                "applied_action_change_l2"
            ],
            "action_correction_l2_step": values["action_correction_l2"],
            "action_slew_intervened_step": values["action_slew_intervened"],
            "action_constraint_enabled": self.action_slew_l2_limit is not None,
            "action_slew_l2_limit": (
                self.action_slew_l2_limit
                if self.action_slew_l2_limit is not None
                else ""
            ),
            "condition_objective_reward_step": values["observed_reward"],
            "base_proxy_reward_step": values["base_reward"],
            "common_rescored_reward_step": values["common_rescored_reward"],
            "shaping_reward_step": values["shaping_reward"],
            "reward_forward_step": info.get("reward_forward", 0.0),
            "reward_forward_tracking_step": values["forward_tracking_reward"],
            "reward_forward_replacement_step": values["forward_replacement_reward"],
            "forward_velocity_target": self.forward_velocity_target,
            "forward_velocity_tracking_scale": self.forward_velocity_tracking_scale,
            "reward_ctrl_step": info.get("reward_ctrl", 0.0),
            "reward_contact_step": info.get("reward_contact", 0.0),
            "reward_survive_step": info.get("reward_survive", 0.0),
            "reward_lateral_shaping_step": info.get("reward_lateral_shaping", 0.0),
            "lateral_shaping_signal": self.lateral_shaping_signal,
            "lateral_velocity_target": self.lateral_velocity_target,
            "lateral_penalty_step": values["lateral_penalty"],
            "reward_effort_shaping_step": info.get("reward_effort_shaping", 0.0),
            "reward_orientation_shaping_step": info.get("reward_orientation_shaping", 0.0),
            "orientation_shaping_function": self.orientation_shaping_function,
            "orientation_penalty_step": values["orientation_penalty"],
            "reward_action_rate_shaping_step": values["action_rate_shaping_reward"],
            "action_rate_penalty_step": values["action_rate_penalty"],
            "root_vertical_velocity_step": values["root_vertical_velocity"],
            "root_roll_pitch_angular_speed_step": values[
                "root_roll_pitch_angular_speed"
            ],
            "vertical_velocity_penalty_step": values[
                "vertical_velocity_penalty"
            ],
            "roll_pitch_angular_velocity_penalty_step": values[
                "roll_pitch_angular_velocity_penalty"
            ],
            "reward_vertical_velocity_shaping_step": values[
                "vertical_velocity_shaping_reward"
            ],
            "reward_roll_pitch_angular_velocity_shaping_step": values[
                "roll_pitch_angular_velocity_shaping_reward"
            ],
            "terminated": values["terminated"],
            "truncated": values["truncated"],
            "termination_category": values["termination_category"],
        }
        self._step_writer.writerow(row)
        if self._step_index % 100 == 0 or values["terminated"] or values["truncated"]:
            assert self._step_handle is not None
            self._step_handle.flush()


def make_proxygap_ant_env(
    *,
    ctrl_cost_weight: float = 0.5,
    condition_id: str = "reference",
    seed: int | None = None,
    render_mode: str | None = None,
    xml_file: str | Path | None = None,
    max_episode_steps: int | None = None,
    forward_progress_shaping_weight: float = 0.0,
    lateral_drift_shaping_weight: float = 0.0,
    lateral_drift_shaping_scale: float = 1.0,
    lateral_shaping_signal: str = "offset_tanh",
    lateral_velocity_target: float = 0.0,
    effort_shaping_weight: float = 0.0,
    effort_shaping_scale: float = 1.0,
    orientation_shaping_weight: float = 0.0,
    orientation_shaping_scale: float = 1.0,
    orientation_shaping_function: str = "tanh",
    replace_forward_reward_with_tracking: bool = False,
    forward_velocity_target: float = 1.0,
    forward_velocity_tracking_scale: float = 0.5,
    action_rate_shaping_weight: float = 0.0,
    vertical_velocity_shaping_weight: float = 0.0,
    vertical_velocity_shaping_scale: float = 1.0,
    roll_pitch_angular_velocity_shaping_weight: float = 0.0,
    roll_pitch_angular_velocity_shaping_scale: float = 1.0,
    common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
    effort_distance_min: float = EPSILON,
    action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
    augment_previous_applied_action: bool = False,
    action_slew_l2_limit: float | None = None,
    step_log_path: str | Path | None = None,
) -> ProxyGapAntWrapper:
    """Create Ant-v5 with separately logged objective and diagnostic terms."""
    kwargs: dict[str, Any] = {
        "ctrl_cost_weight": float(ctrl_cost_weight),
        "render_mode": render_mode,
    }
    if xml_file is not None:
        kwargs["xml_file"] = str(Path(xml_file).resolve())
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = gym.make("Ant-v5", **kwargs)
    wrapped = ProxyGapAntWrapper(
        env=env,
        condition_id=condition_id,
        ctrl_cost_weight=float(ctrl_cost_weight),
        forward_progress_shaping_weight=float(forward_progress_shaping_weight),
        lateral_drift_shaping_weight=float(lateral_drift_shaping_weight),
        lateral_drift_shaping_scale=float(lateral_drift_shaping_scale),
        lateral_shaping_signal=str(lateral_shaping_signal),
        lateral_velocity_target=float(lateral_velocity_target),
        effort_shaping_weight=float(effort_shaping_weight),
        effort_shaping_scale=float(effort_shaping_scale),
        orientation_shaping_weight=float(orientation_shaping_weight),
        orientation_shaping_scale=float(orientation_shaping_scale),
        orientation_shaping_function=str(orientation_shaping_function),
        replace_forward_reward_with_tracking=bool(
            replace_forward_reward_with_tracking
        ),
        forward_velocity_target=float(forward_velocity_target),
        forward_velocity_tracking_scale=float(forward_velocity_tracking_scale),
        action_rate_shaping_weight=float(action_rate_shaping_weight),
        vertical_velocity_shaping_weight=float(vertical_velocity_shaping_weight),
        vertical_velocity_shaping_scale=float(vertical_velocity_shaping_scale),
        roll_pitch_angular_velocity_shaping_weight=float(
            roll_pitch_angular_velocity_shaping_weight
        ),
        roll_pitch_angular_velocity_shaping_scale=float(
            roll_pitch_angular_velocity_shaping_scale
        ),
        common_rescore_ctrl_cost_weight=float(common_rescore_ctrl_cost_weight),
        effort_distance_min=float(effort_distance_min),
        action_saturation_threshold=float(action_saturation_threshold),
        augment_previous_applied_action=bool(augment_previous_applied_action),
        action_slew_l2_limit=action_slew_l2_limit,
        step_log_path=step_log_path,
    )
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped
