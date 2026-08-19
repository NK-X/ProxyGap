"""Ant-v5 construction, reward decomposition and diagnostic logging."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, TextIO

import gymnasium as gym
import mujoco
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
    "torso_pitch_rad",
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
    "forward_velocity_tracking_weight",
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
    "foot_landing_height_threshold",
    "foot_landing_active_count_step",
    "foot_landing_mask_step",
    "foot_contact_point_heights_step",
    "foot_contact_mask_step",
    "foot_contact_counts_step",
    "foot_normal_forces_n_step",
    "foot_tangential_forces_n_step",
    "foot_contact_tangential_speeds_m_per_s_step",
    "foot_contact_slip_distance_m_step",
    "airborne_penalty_step",
    "reward_airborne_shaping_step",
    "foot_contact_gap_penalty_step",
    "foot_contact_gap_penalties_step",
    "reward_foot_contact_gap_shaping_step",
    "foot_lateral_velocities_step",
    "foot_vertical_velocities_step",
    "foot_lateral_velocity_penalty_step",
    "foot_vertical_velocity_penalty_step",
    "foot_lateral_velocity_penalties_step",
    "foot_vertical_velocity_penalties_step",
    "reward_foot_lateral_velocity_shaping_step",
    "reward_foot_vertical_velocity_shaping_step",
    "foot_landing_transition_mask_step",
    "pitch_balance_shaping_weight",
    "pitch_balance_event_active_step",
    "pitch_balance_event_started_step",
    "pitch_balance_event_completed_step",
    "pitch_balance_event_landed_count_step",
    "pitch_balance_event_positive_steps_step",
    "pitch_balance_event_negative_steps_step",
    "pitch_balance_event_neutral_steps_step",
    "pitch_balance_event_score_step",
    "reward_pitch_balance_shaping_step",
    "actuator_joint_torques_n_m_step",
    "actuator_joint_velocities_rad_per_s_step",
    "actuator_mechanical_powers_w_step",
    "terminated",
    "truncated",
    "termination_category",
]


SUPPORTED_ORIENTATION_SHAPING_FUNCTIONS = ("tanh", "cosine")
SUPPORTED_LATERAL_SHAPING_SIGNALS = (
    "offset_tanh",
    "velocity_tanh_squared",
)

DEFAULT_FOOT_GEOM_NAMES = (
    "left_ankle_geom",
    "right_ankle_geom",
    "third_ankle_geom",
    "fourth_ankle_geom",
)


def validated_terrain_normal(
    normal_world: np.ndarray,
    *,
    unit_tolerance: float = 1e-6,
) -> np.ndarray:
    """Return a copied, upward unit terrain normal or fail closed.

    Terrain-frame shaping is deliberately stricter than generic vector
    normalisation: the heightfield adapter is responsible for supplying an
    already normalised vector.  Rejecting malformed context avoids silently
    changing the magnitude of velocity projections or accepting a downward
    frame.
    """

    normal = np.asarray(normal_world, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("terrain normal must contain three finite values")
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or abs(norm - 1.0) > float(unit_tolerance):
        raise ValueError("terrain normal must already be unit length")
    if float(normal[2]) <= 0.0:
        raise ValueError("terrain normal must point into the upper hemisphere")
    return normal.copy()


def target_tangent_frame(
    normal_world: np.ndarray,
    target_heading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct target-forward, target-left and normal terrain axes.

    The requested horizontal heading is orthogonally projected onto the local
    tangent plane.  ``left = normal x forward`` preserves the conventional
    world ``+y`` left axis on a flat surface with a ``+x`` target heading.
    """

    normal = validated_terrain_normal(normal_world)
    heading = float(target_heading)
    if not np.isfinite(heading):
        raise ValueError("target heading must be finite")
    horizontal_forward = np.asarray(
        [math.cos(heading), math.sin(heading), 0.0],
        dtype=np.float64,
    )
    tangent_forward = horizontal_forward - float(
        np.dot(horizontal_forward, normal)
    ) * normal
    forward_norm = float(np.linalg.norm(tangent_forward))
    if not np.isfinite(forward_norm) or forward_norm <= EPSILON:
        raise ValueError("target heading cannot define a terrain tangent")
    tangent_forward /= forward_norm
    tangent_left = np.cross(normal, tangent_forward)
    left_norm = float(np.linalg.norm(tangent_left))
    if not np.isfinite(left_norm) or left_norm <= EPSILON:
        raise ValueError("terrain normal and target tangent are degenerate")
    tangent_left /= left_norm
    return tangent_forward, tangent_left, normal


def project_velocity_onto_axis(
    velocity_world: np.ndarray,
    axis_world: np.ndarray,
) -> float:
    """Project a finite world-frame velocity onto a validated unit axis."""

    velocity = np.asarray(velocity_world, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("world velocity must contain three finite values")
    axis = np.asarray(axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("projection axis must contain three finite values")
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or abs(axis_norm - 1.0) > 1e-6:
        raise ValueError("projection axis must already be unit length")
    return float(np.dot(velocity, axis))


def angular_speed_perpendicular_to_normal(
    angular_velocity_world: np.ndarray,
    normal_world: np.ndarray,
) -> float:
    """Return angular speed excluding rotation about the terrain normal."""

    angular_velocity = np.asarray(angular_velocity_world, dtype=np.float64)
    if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
        raise ValueError("world angular velocity must contain three finite values")
    normal = validated_terrain_normal(normal_world)
    perpendicular = angular_velocity - float(
        np.dot(angular_velocity, normal)
    ) * normal
    return float(np.linalg.norm(perpendicular))


def quaternion_tilt_relative_to_normal(
    quaternion_wxyz: np.ndarray,
    normal_world: np.ndarray,
) -> float:
    """Return full torso-up misalignment from a local terrain normal."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("torso quaternion must contain four finite values")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm < EPSILON or not np.isfinite(quaternion_norm):
        return float("nan")
    w, x, y, z = quaternion / quaternion_norm
    torso_up = np.asarray(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )
    normal = validated_terrain_normal(normal_world)
    alignment = float(np.clip(np.dot(torso_up, normal), -1.0, 1.0))
    return float(math.acos(alignment))


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


def quaternion_pitch_angle(quaternion_wxyz: np.ndarray) -> float:
    """Return signed torso pitch in radians from a MuJoCo ``w, x, y, z`` quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < EPSILON or not np.isfinite(norm):
        return float("nan")
    w, x, y, z = quaternion / norm
    sine_pitch = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(sine_pitch, -1.0, 1.0)))


class PitchBalanceEventTracker:
    """Track pitch-sign time from the first through fourth distinct foot landing."""

    def __init__(self, foot_count: int = 4) -> None:
        if foot_count <= 0:
            raise ValueError("foot_count must be positive")
        self.foot_count = int(foot_count)
        self.reset()

    def reset(self, initial_grounded: np.ndarray | None = None) -> None:
        if initial_grounded is None:
            grounded = np.zeros(self.foot_count, dtype=bool)
        else:
            grounded = np.asarray(initial_grounded, dtype=bool)
            if grounded.shape != (self.foot_count,):
                raise ValueError("initial_grounded must contain one flag per foot")
        self.previous_grounded = grounded.copy()
        self.active = False
        self.landed = np.zeros(self.foot_count, dtype=bool)
        self.positive_steps = 0
        self.negative_steps = 0
        self.neutral_steps = 0
        self.completed_event_count = 0
        self.balance_score_sum = 0.0
        self.active_positive_step_sum = 0
        self.active_negative_step_sum = 0
        self.active_neutral_step_sum = 0

    def update(
        self,
        grounded: np.ndarray,
        signed_pitch: float,
    ) -> dict[str, Any]:
        current_grounded = np.asarray(grounded, dtype=bool)
        if current_grounded.shape != (self.foot_count,):
            raise ValueError("grounded must contain one flag per foot")
        landing_transitions = current_grounded & ~self.previous_grounded
        self.previous_grounded = current_grounded.copy()

        started = False
        if not self.active and bool(np.any(landing_transitions)):
            self.active = True
            self.landed.fill(False)
            self.positive_steps = 0
            self.negative_steps = 0
            self.neutral_steps = 0
            started = True
        if self.active:
            self.landed |= landing_transitions
            if np.isfinite(signed_pitch) and signed_pitch > 0.0:
                self.positive_steps += 1
                self.active_positive_step_sum += 1
            elif np.isfinite(signed_pitch) and signed_pitch < 0.0:
                self.negative_steps += 1
                self.active_negative_step_sum += 1
            else:
                self.neutral_steps += 1
                self.active_neutral_step_sum += 1

        completed = bool(self.active and np.all(self.landed))
        landed_count = int(np.sum(self.landed)) if self.active else 0
        positive_steps = int(self.positive_steps) if self.active else 0
        negative_steps = int(self.negative_steps) if self.active else 0
        neutral_steps = int(self.neutral_steps) if self.active else 0
        score = 0.0
        if completed:
            signed_steps = positive_steps + negative_steps
            if signed_steps > 0:
                score = 1.0 - abs(positive_steps - negative_steps) / signed_steps
            self.completed_event_count += 1
            self.balance_score_sum += score
            self.active = False
            self.landed.fill(False)
            self.positive_steps = 0
            self.negative_steps = 0
            self.neutral_steps = 0

        return {
            "landing_transitions": landing_transitions.copy(),
            "started": started,
            "completed": completed,
            "active": self.active,
            "landed_count": landed_count,
            "positive_steps": positive_steps,
            "negative_steps": negative_steps,
            "neutral_steps": neutral_steps,
            "score": float(score),
        }


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
        forward_velocity_tracking_weight: float = 1.0,
        action_rate_shaping_weight: float = 0.0,
        vertical_velocity_shaping_weight: float = 0.0,
        vertical_velocity_shaping_scale: float = 1.0,
        roll_pitch_angular_velocity_shaping_weight: float = 0.0,
        roll_pitch_angular_velocity_shaping_scale: float = 1.0,
        foot_landing_height_threshold: float = 0.03,
        foot_lateral_velocity_shaping_weight: float = 0.0,
        foot_lateral_velocity_shaping_scale: float = 1.0,
        foot_vertical_velocity_shaping_weight: float = 0.0,
        foot_vertical_velocity_shaping_scale: float = 1.0,
        airborne_shaping_weight: float = 0.0,
        foot_contact_gap_shaping_weight: float = 0.0,
        foot_contact_gap_grace_seconds: float = 0.5,
        foot_contact_gap_scale_seconds: float = 0.5,
        pitch_balance_shaping_weight: float = 0.0,
        foot_geom_names: tuple[str, ...] = DEFAULT_FOOT_GEOM_NAMES,
        common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
        effort_distance_min: float = EPSILON,
        action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
        augment_previous_applied_action: bool = False,
        action_slew_l2_limit: float | None = None,
        terrain_frame_shaping_enabled: bool = False,
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
        self.forward_velocity_tracking_weight = float(
            forward_velocity_tracking_weight
        )
        if self.forward_velocity_tracking_weight < 0 or not np.isfinite(
            self.forward_velocity_tracking_weight
        ):
            raise ValueError(
                "forward velocity tracking weight must be finite and non-negative"
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
        self.foot_landing_height_threshold = float(foot_landing_height_threshold)
        self.foot_lateral_velocity_shaping_weight = float(
            foot_lateral_velocity_shaping_weight
        )
        self.foot_lateral_velocity_shaping_scale = float(
            foot_lateral_velocity_shaping_scale
        )
        self.foot_vertical_velocity_shaping_weight = float(
            foot_vertical_velocity_shaping_weight
        )
        self.foot_vertical_velocity_shaping_scale = float(
            foot_vertical_velocity_shaping_scale
        )
        self.airborne_shaping_weight = float(airborne_shaping_weight)
        self.foot_contact_gap_shaping_weight = float(
            foot_contact_gap_shaping_weight
        )
        self.foot_contact_gap_grace_seconds = float(
            foot_contact_gap_grace_seconds
        )
        self.foot_contact_gap_scale_seconds = float(
            foot_contact_gap_scale_seconds
        )
        self.pitch_balance_shaping_weight = float(pitch_balance_shaping_weight)
        if (
            not np.isfinite(self.foot_landing_height_threshold)
            or self.foot_landing_height_threshold < 0
        ):
            raise ValueError("foot landing height threshold must be finite and non-negative")
        for name, weight in (
            (
                "foot_lateral_velocity_shaping_weight",
                self.foot_lateral_velocity_shaping_weight,
            ),
            (
                "foot_vertical_velocity_shaping_weight",
                self.foot_vertical_velocity_shaping_weight,
            ),
            ("airborne_shaping_weight", self.airborne_shaping_weight),
            (
                "foot_contact_gap_shaping_weight",
                self.foot_contact_gap_shaping_weight,
            ),
            ("pitch_balance_shaping_weight", self.pitch_balance_shaping_weight),
        ):
            if weight < 0 or not np.isfinite(weight):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.foot_contact_gap_grace_seconds)
            or self.foot_contact_gap_grace_seconds < 0
            or not np.isfinite(self.foot_contact_gap_scale_seconds)
            or self.foot_contact_gap_scale_seconds <= 0
        ):
            raise ValueError(
                "foot contact gap grace must be non-negative and scale positive"
            )
        bounded_squared_signal_penalty(
            0.0, scale=self.foot_lateral_velocity_shaping_scale
        )
        bounded_squared_signal_penalty(
            0.0, scale=self.foot_vertical_velocity_shaping_scale
        )
        self.foot_geom_names = tuple(str(name) for name in foot_geom_names)
        if len(self.foot_geom_names) != 4 or len(set(self.foot_geom_names)) != 4:
            raise ValueError("exactly four distinct foot geometry names are required")
        geom_ids = tuple(
            int(
                mujoco.mj_name2id(
                    self.unwrapped.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
            )
            for name in self.foot_geom_names
        )
        missing = [
            name
            for name, geom_id in zip(self.foot_geom_names, geom_ids)
            if geom_id < 0
        ]
        foot_shaping_enabled = (
            self.foot_lateral_velocity_shaping_weight > 0
            or self.foot_vertical_velocity_shaping_weight > 0
            or self.airborne_shaping_weight > 0
            or self.foot_contact_gap_shaping_weight > 0
            or self.pitch_balance_shaping_weight > 0
        )
        if missing and foot_shaping_enabled:
            raise ValueError(f"foot geometries not found in MuJoCo model: {missing}")
        self._foot_geom_ids = () if missing else geom_ids
        if self._foot_geom_ids and foot_shaping_enabled:
            non_capsules = [
                name
                for name, geom_id in zip(self.foot_geom_names, self._foot_geom_ids)
                if int(self.unwrapped.model.geom_type[geom_id])
                != int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            ]
            if non_capsules:
                raise ValueError(
                    "foot landing shaping currently requires capsule geometries: "
                    f"{non_capsules}"
                )
        actuator_joint_ids = np.asarray(
            self.unwrapped.model.actuator_trnid[:, 0], dtype=np.int64
        )
        if np.any(actuator_joint_ids < 0):
            raise ValueError("ProxyGap Ant diagnostics require joint actuators")
        self._actuator_joint_dof_addresses = np.asarray(
            self.unwrapped.model.jnt_dofadr[actuator_joint_ids], dtype=np.int64
        )
        if len(set(self._actuator_joint_dof_addresses.tolist())) != len(
            self._actuator_joint_dof_addresses
        ):
            raise ValueError("ProxyGap Ant diagnostics require one actuator per joint")
        self.actuator_joint_names = tuple(
            str(
                mujoco.mj_id2name(
                    self.unwrapped.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    int(joint_id),
                )
            )
            for joint_id in actuator_joint_ids
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
        self.terrain_frame_shaping_enabled = bool(
            terrain_frame_shaping_enabled
        )
        self._terrain_height_sampler: Callable[[float, float], float] | None = None
        self._terrain_normal_sampler: (
            Callable[[float, float], np.ndarray] | None
        ) = None
        self._terrain_target_heading: float | None = None
        self._terrain_frame_context_valid = False
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
        self._reward_foot_lateral_velocity_shaping_sum = 0.0
        self._foot_lateral_velocity_penalty_sum = 0.0
        self._reward_foot_vertical_velocity_shaping_sum = 0.0
        self._foot_vertical_velocity_penalty_sum = 0.0
        self._reward_airborne_shaping_sum = 0.0
        self._reward_foot_contact_gap_shaping_sum = 0.0
        self._foot_contact_gap_penalty_sum = 0.0
        self._foot_contact_gap_penalty_sum_by_foot = np.zeros(
            4, dtype=np.float64
        )
        self._foot_landing_active_count_sum = 0
        self._foot_landing_active_count_by_foot = np.zeros(4, dtype=np.int64)
        self._foot_lateral_velocity_penalty_sum_by_foot = np.zeros(
            4, dtype=np.float64
        )
        self._foot_vertical_velocity_penalty_sum_by_foot = np.zeros(
            4, dtype=np.float64
        )
        self._foot_contact_step_count_by_foot = np.zeros(4, dtype=np.int64)
        self._foot_contact_transition_count_by_foot = np.zeros(4, dtype=np.int64)
        self._current_foot_no_contact_run_steps = np.zeros(4, dtype=np.int64)
        self._longest_foot_no_contact_run_steps = np.zeros(4, dtype=np.int64)
        self._previous_foot_contact_mask = np.zeros(4, dtype=bool)
        self._support_count_step_counts = np.zeros(5, dtype=np.int64)
        self._support_mask_step_counts = np.zeros(16, dtype=np.int64)
        self._foot_normal_force_time_integral_by_foot = np.zeros(
            4, dtype=np.float64
        )
        self._foot_tangential_force_time_integral_by_foot = np.zeros(
            4, dtype=np.float64
        )
        self._foot_contact_slip_distance_by_foot = np.zeros(4, dtype=np.float64)
        self._foot_contact_slip_speed_max_by_foot = np.zeros(4, dtype=np.float64)
        self._airborne_step_count = 0
        self._current_airborne_run_steps = 0
        self._longest_airborne_run_steps = 0
        actuator_count = len(self.actuator_joint_names)
        self._actuator_abs_torque_time_integral = np.zeros(
            actuator_count, dtype=np.float64
        )
        self._actuator_positive_mechanical_work = np.zeros(
            actuator_count, dtype=np.float64
        )
        self._actuator_negative_mechanical_work_abs = np.zeros(
            actuator_count, dtype=np.float64
        )
        self._pitch_balance_tracker = PitchBalanceEventTracker(foot_count=4)
        self._reward_pitch_balance_shaping_sum = 0.0
        self._step_handle: TextIO | None = None
        self._step_writer: csv.DictWriter | None = None
        if step_log_path is not None:
            path = Path(step_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._step_handle = gzip.open(path, "wt", newline="", encoding="utf-8")
            self._step_writer = csv.DictWriter(self._step_handle, fieldnames=STEP_LOG_SCHEMA)
            self._step_writer.writeheader()

    def set_terrain_shaping_context(
        self,
        *,
        height_sampler: Callable[[float, float], float],
        normal_sampler: Callable[[float, float], np.ndarray],
        target_heading: float,
    ) -> None:
        """Install the heightfield/target frame used by the *next* step.

        The fixed-goal adapter owns the frozen terrain and therefore supplies
        the samplers.  Validation is performed immediately at the current root
        position and again at every sampled root/foot position.  An enabled
        wrapper refuses to step without a valid context.
        """

        self._terrain_frame_context_valid = False
        if not self.terrain_frame_shaping_enabled:
            raise RuntimeError(
                "terrain shaping context cannot be installed when the feature is disabled"
            )
        if not callable(height_sampler) or not callable(normal_sampler):
            raise TypeError("terrain height and normal samplers must be callable")
        heading = float(target_heading)
        if not np.isfinite(heading):
            raise ValueError("terrain shaping target heading must be finite")
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        if qpos.shape[0] < 2 or not np.all(np.isfinite(qpos[:2])):
            raise ValueError("root position is unavailable for terrain context")
        x, y = float(qpos[0]), float(qpos[1])
        height = float(height_sampler(x, y))
        if not np.isfinite(height):
            raise ValueError("terrain height sampler returned a non-finite value")
        normal = validated_terrain_normal(normal_sampler(x, y))
        target_tangent_frame(normal, heading)
        self._terrain_height_sampler = height_sampler
        self._terrain_normal_sampler = normal_sampler
        self._terrain_target_heading = heading
        self._terrain_frame_context_valid = True

    def _terrain_frame_at(
        self,
        x: float,
        y: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        if not self.terrain_frame_shaping_enabled:
            raise RuntimeError("terrain frame requested while shaping is disabled")
        if (
            not self._terrain_frame_context_valid
            or self._terrain_height_sampler is None
            or self._terrain_normal_sampler is None
            or self._terrain_target_heading is None
        ):
            raise RuntimeError("terrain-frame shaping has no valid next-step context")
        height = float(self._terrain_height_sampler(float(x), float(y)))
        if not np.isfinite(height):
            self._terrain_frame_context_valid = False
            raise ValueError("terrain height sampler returned a non-finite value")
        normal = validated_terrain_normal(
            self._terrain_normal_sampler(float(x), float(y))
        )
        _, tangent_left, normal = target_tangent_frame(
            normal,
            self._terrain_target_heading,
        )
        return height, tangent_left, normal, np.asarray(
            [
                math.cos(self._terrain_target_heading),
                math.sin(self._terrain_target_heading),
                0.0,
            ],
            dtype=np.float64,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._terrain_height_sampler = None
        self._terrain_normal_sampler = None
        self._terrain_target_heading = None
        self._terrain_frame_context_valid = False
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
        self._reward_foot_lateral_velocity_shaping_sum = 0.0
        self._foot_lateral_velocity_penalty_sum = 0.0
        self._reward_foot_vertical_velocity_shaping_sum = 0.0
        self._foot_vertical_velocity_penalty_sum = 0.0
        self._reward_airborne_shaping_sum = 0.0
        self._reward_foot_contact_gap_shaping_sum = 0.0
        self._foot_contact_gap_penalty_sum = 0.0
        self._foot_contact_gap_penalty_sum_by_foot.fill(0.0)
        self._foot_landing_active_count_sum = 0
        self._foot_landing_active_count_by_foot.fill(0)
        self._foot_lateral_velocity_penalty_sum_by_foot.fill(0.0)
        self._foot_vertical_velocity_penalty_sum_by_foot.fill(0.0)
        self._foot_contact_step_count_by_foot.fill(0)
        self._foot_contact_transition_count_by_foot.fill(0)
        self._current_foot_no_contact_run_steps.fill(0)
        self._longest_foot_no_contact_run_steps.fill(0)
        self._support_count_step_counts.fill(0)
        self._support_mask_step_counts.fill(0)
        self._foot_normal_force_time_integral_by_foot.fill(0.0)
        self._foot_tangential_force_time_integral_by_foot.fill(0.0)
        self._foot_contact_slip_distance_by_foot.fill(0.0)
        self._foot_contact_slip_speed_max_by_foot.fill(0.0)
        self._airborne_step_count = 0
        self._current_airborne_run_steps = 0
        self._longest_airborne_run_steps = 0
        self._actuator_abs_torque_time_integral.fill(0.0)
        self._actuator_positive_mechanical_work.fill(0.0)
        self._actuator_negative_mechanical_work_abs.fill(0.0)
        self._reward_pitch_balance_shaping_sum = 0.0
        initial_grounded = (
            np.zeros(4, dtype=bool)
            if self.terrain_frame_shaping_enabled
            else self._foot_landing_kinematics()[3]
        )
        self._pitch_balance_tracker.reset(initial_grounded=initial_grounded)
        initial_foot_contact_mask = self._foot_contact_diagnostics()[0]
        self._previous_foot_contact_mask = initial_foot_contact_mask.copy()
        x_position, y_position = self._root_xy(info)
        self.metrics.reset(initial_x=x_position, initial_y=y_position)
        self._episode_index += 1
        self._step_index = 0
        info = dict(info)
        info["proxygap_foot_contact_mask_step"] = (
            initial_foot_contact_mask.copy()
        )
        info["proxygap_terrain_frame_shaping_enabled"] = bool(
            self.terrain_frame_shaping_enabled
        )
        info["proxygap_terrain_frame_context_valid"] = False
        info["proxygap_terrain_frame_shaping_applied_step"] = False
        info.update(self._prefixed_summary())
        return self._augment_observation(observation), info

    def _foot_landing_kinematics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return distal-foot heights, lateral/vertical speeds and landing mask.

        Ant's ankle capsules are authored from the ankle joint to the distal
        contact sphere. MuJoCo compiles that distal endpoint onto the negative
        local-z axis of each named geometry. By default, height and velocity
        retain the frozen world-z/world-y definition. When the optional
        terrain frame is enabled, height is clearance above the heightfield at
        the distal sphere's XY position and the two velocity components are
        projected onto local terrain-normal and target-left tangent axes.
        """
        foot_count = len(self.foot_geom_names)
        if not self._foot_geom_ids:
            unavailable = np.full(foot_count, np.nan, dtype=np.float64)
            return (
                unavailable.copy(),
                unavailable.copy(),
                unavailable.copy(),
                np.zeros(foot_count, dtype=bool),
            )

        model = self.unwrapped.model
        data = self.unwrapped.data
        qvel = np.asarray(data.qvel, dtype=np.float64)
        heights = np.empty(foot_count, dtype=np.float64)
        lateral_velocities = np.empty(foot_count, dtype=np.float64)
        vertical_velocities = np.empty(foot_count, dtype=np.float64)
        for index, geom_id in enumerate(self._foot_geom_ids):
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            distal_center = (
                np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
                - rotation[:, 2] * float(model.geom_size[geom_id, 1])
            )
            sphere_radius = float(model.geom_size[geom_id, 0])
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                distal_center,
                int(model.geom_bodyid[geom_id]),
            )
            velocity = jacobian_position @ qvel
            if self.terrain_frame_shaping_enabled:
                terrain_height, tangent_left, normal, _ = self._terrain_frame_at(
                    float(distal_center[0]),
                    float(distal_center[1]),
                )
                heights[index] = float(
                    distal_center[2] - sphere_radius - terrain_height
                )
                lateral_velocities[index] = project_velocity_onto_axis(
                    velocity,
                    tangent_left,
                )
                vertical_velocities[index] = project_velocity_onto_axis(
                    velocity,
                    normal,
                )
            else:
                heights[index] = float(distal_center[2] - sphere_radius)
                lateral_velocities[index] = float(velocity[1])
                vertical_velocities[index] = float(velocity[2])
        landing_mask = heights <= self.foot_landing_height_threshold
        return heights, lateral_velocities, vertical_velocities, landing_mask

    def _foot_contact_diagnostics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Measure actual world-ground contacts for each distal foot geometry.

        MuJoCo contact forces are sampled after each control step. Their force
        time integrals support within-simulation participation comparisons, but
        are not a substitute for high-rate force-plate measurements.
        """
        foot_count = len(self.foot_geom_names)
        contact_mask = np.zeros(foot_count, dtype=bool)
        contact_counts = np.zeros(foot_count, dtype=np.int64)
        normal_forces = np.zeros(foot_count, dtype=np.float64)
        tangential_forces = np.zeros(foot_count, dtype=np.float64)
        tangential_speeds = np.zeros(foot_count, dtype=np.float64)
        if not self._foot_geom_ids:
            return (
                contact_mask,
                contact_counts,
                normal_forces,
                tangential_forces,
                tangential_speeds,
            )

        model = self.unwrapped.model
        data = self.unwrapped.data
        qvel = np.asarray(data.qvel, dtype=np.float64)
        foot_lookup = {
            int(geom_id): index for index, geom_id in enumerate(self._foot_geom_ids)
        }
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if geom_1 in foot_lookup:
                foot_geom, other_geom = geom_1, geom_2
            elif geom_2 in foot_lookup:
                foot_geom, other_geom = geom_2, geom_1
            else:
                continue
            if int(model.geom_bodyid[other_geom]) != 0:
                continue
            foot_index = foot_lookup[foot_geom]
            contact_mask[foot_index] = True
            contact_counts[foot_index] += 1
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            normal_forces[foot_index] += max(0.0, float(contact_force[0]))
            tangential_forces[foot_index] += float(
                np.linalg.norm(contact_force[1:3])
            )

            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                np.asarray(contact.pos, dtype=np.float64),
                int(model.geom_bodyid[foot_geom]),
            )
            contact_velocity = jacobian_position @ qvel
            contact_normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
            contact_normal /= max(float(np.linalg.norm(contact_normal)), EPSILON)
            tangent_velocity = contact_velocity - float(
                np.dot(contact_velocity, contact_normal)
            ) * contact_normal
            tangential_speeds[foot_index] = max(
                tangential_speeds[foot_index],
                float(np.linalg.norm(tangent_velocity)),
            )
        return (
            contact_mask,
            contact_counts,
            normal_forces,
            tangential_forces,
            tangential_speeds,
        )

    def _actuator_diagnostics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return joint torque, joint speed and signed mechanical power."""
        data = self.unwrapped.data
        joint_torques = np.asarray(
            data.qfrc_actuator[self._actuator_joint_dof_addresses],
            dtype=np.float64,
        ).copy()
        joint_velocities = np.asarray(
            data.qvel[self._actuator_joint_dof_addresses],
            dtype=np.float64,
        ).copy()
        return joint_torques, joint_velocities, joint_torques * joint_velocities

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terrain_frame_shaping_enabled and not (
            self._terrain_frame_context_valid
            and self._terrain_height_sampler is not None
            and self._terrain_normal_sampler is not None
            and self._terrain_target_heading is not None
        ):
            raise RuntimeError(
                "enabled terrain-frame shaping requires a valid next-step context"
            )
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
        torso_pitch = self._torso_pitch()
        torso_height, state_is_finite = self._health_state()
        squared_action = float(np.sum(np.square(applied_action)))
        forward_velocity = self._root_x_velocity(info)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        if self.terrain_frame_shaping_enabled:
            qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
            _, _, root_terrain_normal, _ = self._terrain_frame_at(
                float(qpos[0]),
                float(qpos[1]),
            )
            root_vertical_velocity = project_velocity_onto_axis(
                qvel[:3],
                root_terrain_normal,
            )
            if np.allclose(
                root_terrain_normal,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                atol=1e-12,
                rtol=0.0,
            ):
                # Preserve the frozen formula bit-for-bit on canonical flat
                # terrain; the slope branch below performs a world projection.
                root_roll_pitch_angular_speed = float(np.linalg.norm(qvel[3:5]))
            else:
                torso_body_id = int(
                    mujoco.mj_name2id(
                        self.unwrapped.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "torso",
                    )
                )
                if torso_body_id < 0:
                    raise RuntimeError("terrain-frame shaping requires torso body")
                torso_rotation = np.asarray(
                    self.unwrapped.data.xmat[torso_body_id],
                    dtype=np.float64,
                ).reshape(3, 3)
                angular_velocity_world = torso_rotation @ qvel[3:6]
                root_roll_pitch_angular_speed = (
                    angular_speed_perpendicular_to_normal(
                        angular_velocity_world,
                        root_terrain_normal,
                    )
                )
        else:
            root_terrain_normal = np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            )
            root_vertical_velocity = float(qvel[2])
            root_roll_pitch_angular_speed = float(np.linalg.norm(qvel[3:5]))
        saturated_fraction = float(
            np.mean(np.abs(applied_action) >= self.action_saturation_threshold)
        )

        forward_shaping_reward = self.forward_progress_shaping_weight * float(
            info.get("reward_forward", 0.0)
        )
        forward_tracking_reward = (
            self.forward_velocity_tracking_weight
            * forward_velocity_tracking_value(
                forward_velocity,
                target=self.forward_velocity_target,
                scale=self.forward_velocity_tracking_scale,
            )
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
        (
            foot_contact_point_heights,
            foot_lateral_velocities,
            foot_vertical_velocities,
            foot_landing_mask,
        ) = self._foot_landing_kinematics()
        (
            foot_contact_mask,
            foot_contact_counts,
            foot_normal_forces,
            foot_tangential_forces,
            foot_contact_tangential_speeds,
        ) = self._foot_contact_diagnostics()
        environment_dt = float(self.unwrapped.dt)
        next_foot_no_contact_run_steps = np.where(
            foot_contact_mask,
            0,
            self._current_foot_no_contact_run_steps + 1,
        )
        foot_contact_gap_excess_seconds = np.maximum(
            next_foot_no_contact_run_steps * environment_dt
            - self.foot_contact_gap_grace_seconds,
            0.0,
        )
        foot_contact_gap_penalties = np.square(
            np.tanh(
                foot_contact_gap_excess_seconds
                / self.foot_contact_gap_scale_seconds
            )
        )
        foot_contact_gap_penalty = float(np.mean(foot_contact_gap_penalties))
        foot_contact_gap_shaping_reward = (
            -self.foot_contact_gap_shaping_weight
            * foot_contact_gap_penalty
        )
        foot_contact_slip_distance = (
            foot_contact_tangential_speeds * environment_dt
        )
        (
            actuator_joint_torques,
            actuator_joint_velocities,
            actuator_mechanical_powers,
        ) = self._actuator_diagnostics()
        foot_lateral_velocity_penalties = np.asarray(
            [
                bounded_squared_signal_penalty(
                    velocity,
                    scale=self.foot_lateral_velocity_shaping_scale,
                )
                if active
                else 0.0
                for velocity, active in zip(
                    foot_lateral_velocities,
                    foot_landing_mask,
                )
            ],
            dtype=np.float64,
        )
        foot_vertical_velocity_penalties = np.asarray(
            [
                bounded_squared_signal_penalty(
                    velocity,
                    scale=self.foot_vertical_velocity_shaping_scale,
                )
                if active
                else 0.0
                for velocity, active in zip(
                    foot_vertical_velocities,
                    foot_landing_mask,
                )
            ],
            dtype=np.float64,
        )
        foot_lateral_velocity_penalty = float(
            np.sum(foot_lateral_velocity_penalties)
        )
        foot_vertical_velocity_penalty = float(
            np.sum(foot_vertical_velocity_penalties)
        )
        foot_lateral_velocity_shaping_reward = (
            -self.foot_lateral_velocity_shaping_weight
            * foot_lateral_velocity_penalty
        )
        foot_vertical_velocity_shaping_reward = (
            -self.foot_vertical_velocity_shaping_weight
            * foot_vertical_velocity_penalty
        )
        airborne_penalty = float(not np.any(foot_contact_mask))
        airborne_shaping_reward = (
            -self.airborne_shaping_weight * airborne_penalty
        )
        pitch_balance_event = self._pitch_balance_tracker.update(
            foot_landing_mask,
            torso_pitch,
        )
        pitch_balance_shaping_reward = (
            self.pitch_balance_shaping_weight
            * float(pitch_balance_event["score"])
            if bool(pitch_balance_event["completed"])
            else 0.0
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
            + foot_lateral_velocity_shaping_reward
            + foot_vertical_velocity_shaping_reward
            + airborne_shaping_reward
            + foot_contact_gap_shaping_reward
            + pitch_balance_shaping_reward
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
        self._reward_foot_lateral_velocity_shaping_sum += (
            foot_lateral_velocity_shaping_reward
        )
        self._foot_lateral_velocity_penalty_sum += foot_lateral_velocity_penalty
        self._reward_foot_vertical_velocity_shaping_sum += (
            foot_vertical_velocity_shaping_reward
        )
        self._foot_vertical_velocity_penalty_sum += foot_vertical_velocity_penalty
        self._reward_airborne_shaping_sum += airborne_shaping_reward
        self._reward_foot_contact_gap_shaping_sum += (
            foot_contact_gap_shaping_reward
        )
        self._foot_contact_gap_penalty_sum += foot_contact_gap_penalty
        self._foot_contact_gap_penalty_sum_by_foot += foot_contact_gap_penalties
        self._reward_pitch_balance_shaping_sum += pitch_balance_shaping_reward
        self._foot_landing_active_count_sum += int(np.sum(foot_landing_mask))
        self._foot_landing_active_count_by_foot += foot_landing_mask.astype(np.int64)
        self._foot_lateral_velocity_penalty_sum_by_foot += (
            foot_lateral_velocity_penalties
        )
        self._foot_vertical_velocity_penalty_sum_by_foot += (
            foot_vertical_velocity_penalties
        )
        self._foot_contact_step_count_by_foot += foot_contact_mask.astype(np.int64)
        contact_transitions = np.logical_and(
            foot_contact_mask,
            np.logical_not(self._previous_foot_contact_mask),
        )
        self._foot_contact_transition_count_by_foot += contact_transitions.astype(
            np.int64
        )
        self._current_foot_no_contact_run_steps = (
            next_foot_no_contact_run_steps
        )
        self._longest_foot_no_contact_run_steps = np.maximum(
            self._longest_foot_no_contact_run_steps,
            self._current_foot_no_contact_run_steps,
        )
        support_count = int(np.sum(foot_contact_mask))
        support_mask_index = int(
            sum(
                (1 << index) if active else 0
                for index, active in enumerate(foot_contact_mask)
            )
        )
        self._support_count_step_counts[support_count] += 1
        self._support_mask_step_counts[support_mask_index] += 1
        self._previous_foot_contact_mask = foot_contact_mask.copy()
        self._foot_normal_force_time_integral_by_foot += (
            foot_normal_forces * environment_dt
        )
        self._foot_tangential_force_time_integral_by_foot += (
            foot_tangential_forces * environment_dt
        )
        self._foot_contact_slip_distance_by_foot += foot_contact_slip_distance
        self._foot_contact_slip_speed_max_by_foot = np.maximum(
            self._foot_contact_slip_speed_max_by_foot,
            foot_contact_tangential_speeds,
        )
        if np.any(foot_contact_mask):
            self._current_airborne_run_steps = 0
        else:
            self._airborne_step_count += 1
            self._current_airborne_run_steps += 1
            self._longest_airborne_run_steps = max(
                self._longest_airborne_run_steps,
                self._current_airborne_run_steps,
            )
        self._actuator_abs_torque_time_integral += (
            np.abs(actuator_joint_torques) * environment_dt
        )
        self._actuator_positive_mechanical_work += (
            np.maximum(actuator_mechanical_powers, 0.0) * environment_dt
        )
        self._actuator_negative_mechanical_work_abs += (
            np.maximum(-actuator_mechanical_powers, 0.0) * environment_dt
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
        info["reward_foot_lateral_velocity_shaping"] = float(
            foot_lateral_velocity_shaping_reward
        )
        info["foot_lateral_velocity_penalty"] = foot_lateral_velocity_penalty
        info["reward_foot_vertical_velocity_shaping"] = float(
            foot_vertical_velocity_shaping_reward
        )
        info["foot_vertical_velocity_penalty"] = foot_vertical_velocity_penalty
        info["airborne_penalty"] = airborne_penalty
        info["reward_airborne_shaping"] = float(airborne_shaping_reward)
        info["foot_contact_gap_penalty"] = foot_contact_gap_penalty
        info["reward_foot_contact_gap_shaping"] = float(
            foot_contact_gap_shaping_reward
        )
        info["reward_pitch_balance_shaping"] = float(
            pitch_balance_shaping_reward
        )
        info["pitch_balance_event_score"] = float(pitch_balance_event["score"])
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
        info["proxygap_torso_pitch_step"] = torso_pitch
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
        info["proxygap_forward_velocity_tracking_weight"] = (
            self.forward_velocity_tracking_weight
        )
        info["proxygap_forward_velocity_step"] = forward_velocity
        info["proxygap_action_rate_shaping_weight"] = self.action_rate_shaping_weight
        info["proxygap_action_rate_penalty_step"] = action_rate_penalty
        info["proxygap_root_vertical_velocity_step"] = root_vertical_velocity
        info["proxygap_root_roll_pitch_angular_speed_step"] = (
            root_roll_pitch_angular_speed
        )
        info["proxygap_terrain_frame_shaping_enabled"] = bool(
            self.terrain_frame_shaping_enabled
        )
        info["proxygap_terrain_frame_context_valid"] = bool(
            self._terrain_frame_context_valid
            if self.terrain_frame_shaping_enabled
            else False
        )
        info["proxygap_terrain_frame_shaping_applied_step"] = bool(
            self.terrain_frame_shaping_enabled
            and self._terrain_frame_context_valid
        )
        info["proxygap_terrain_frame_normal_world_step"] = (
            root_terrain_normal.copy()
        )
        info["proxygap_terrain_frame_target_heading_rad_step"] = (
            float(self._terrain_target_heading)
            if self.terrain_frame_shaping_enabled
            and self._terrain_target_heading is not None
            else float("nan")
        )
        info["proxygap_foot_landing_height_threshold"] = (
            self.foot_landing_height_threshold
        )
        info["proxygap_foot_landing_active_count_step"] = int(
            np.sum(foot_landing_mask)
        )
        info["proxygap_foot_landing_mask_step"] = foot_landing_mask.copy()
        info["proxygap_foot_landing_transition_mask_step"] = pitch_balance_event[
            "landing_transitions"
        ].copy()
        info["proxygap_pitch_balance_event_active_step"] = bool(
            pitch_balance_event["active"]
        )
        info["proxygap_pitch_balance_event_started_step"] = bool(
            pitch_balance_event["started"]
        )
        info["proxygap_pitch_balance_event_completed_step"] = bool(
            pitch_balance_event["completed"]
        )
        info["proxygap_pitch_balance_event_landed_count_step"] = int(
            pitch_balance_event["landed_count"]
        )
        info["proxygap_pitch_balance_event_positive_steps_step"] = int(
            pitch_balance_event["positive_steps"]
        )
        info["proxygap_pitch_balance_event_negative_steps_step"] = int(
            pitch_balance_event["negative_steps"]
        )
        info["proxygap_pitch_balance_event_neutral_steps_step"] = int(
            pitch_balance_event["neutral_steps"]
        )
        info["proxygap_pitch_balance_event_score_step"] = float(
            pitch_balance_event["score"]
        )
        info["proxygap_foot_contact_point_heights_step"] = (
            foot_contact_point_heights.copy()
        )
        info["proxygap_foot_contact_mask_step"] = foot_contact_mask.copy()
        info["proxygap_foot_contact_counts_step"] = foot_contact_counts.copy()
        info["proxygap_foot_normal_forces_n_step"] = foot_normal_forces.copy()
        info["proxygap_foot_tangential_forces_n_step"] = (
            foot_tangential_forces.copy()
        )
        info["proxygap_foot_contact_tangential_speeds_m_per_s_step"] = (
            foot_contact_tangential_speeds.copy()
        )
        info["proxygap_foot_contact_slip_distance_m_step"] = (
            foot_contact_slip_distance.copy()
        )
        info["proxygap_foot_lateral_velocities_step"] = (
            foot_lateral_velocities.copy()
        )
        info["proxygap_foot_vertical_velocities_step"] = (
            foot_vertical_velocities.copy()
        )
        info["proxygap_foot_lateral_velocity_penalties_step"] = (
            foot_lateral_velocity_penalties.copy()
        )
        info["proxygap_foot_vertical_velocity_penalties_step"] = (
            foot_vertical_velocity_penalties.copy()
        )
        info["proxygap_actuator_joint_torques_n_m_step"] = (
            actuator_joint_torques.copy()
        )
        info["proxygap_actuator_joint_velocities_rad_per_s_step"] = (
            actuator_joint_velocities.copy()
        )
        info["proxygap_actuator_mechanical_powers_w_step"] = (
            actuator_mechanical_powers.copy()
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
            torso_pitch=torso_pitch,
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
            foot_contact_point_heights=foot_contact_point_heights,
            foot_contact_mask=foot_contact_mask,
            foot_contact_counts=foot_contact_counts,
            foot_normal_forces=foot_normal_forces,
            foot_tangential_forces=foot_tangential_forces,
            foot_contact_tangential_speeds=foot_contact_tangential_speeds,
            foot_contact_slip_distance=foot_contact_slip_distance,
            airborne_penalty=airborne_penalty,
            airborne_shaping_reward=airborne_shaping_reward,
            foot_contact_gap_penalty=foot_contact_gap_penalty,
            foot_contact_gap_penalties=foot_contact_gap_penalties,
            foot_contact_gap_shaping_reward=foot_contact_gap_shaping_reward,
            foot_lateral_velocities=foot_lateral_velocities,
            foot_vertical_velocities=foot_vertical_velocities,
            foot_landing_mask=foot_landing_mask,
            foot_lateral_velocity_penalty=foot_lateral_velocity_penalty,
            foot_vertical_velocity_penalty=foot_vertical_velocity_penalty,
            foot_lateral_velocity_penalties=foot_lateral_velocity_penalties,
            foot_vertical_velocity_penalties=foot_vertical_velocity_penalties,
            foot_lateral_velocity_shaping_reward=(
                foot_lateral_velocity_shaping_reward
            ),
            foot_vertical_velocity_shaping_reward=(
                foot_vertical_velocity_shaping_reward
            ),
            pitch_balance_event=pitch_balance_event,
            pitch_balance_shaping_reward=pitch_balance_shaping_reward,
            actuator_joint_torques=actuator_joint_torques,
            actuator_joint_velocities=actuator_joint_velocities,
            actuator_mechanical_powers=actuator_mechanical_powers,
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
        episode_length = int(summary["episode_length"])
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
                "forward_velocity_tracking_weight": self.forward_velocity_tracking_weight,
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
                "foot_geom_names": list(self.foot_geom_names),
                "foot_landing_height_threshold": self.foot_landing_height_threshold,
                "foot_landing_active_count_sum": self._foot_landing_active_count_sum,
                "foot_landing_active_count_by_foot": self._foot_landing_active_count_by_foot.tolist(),
                "foot_contact_step_count_by_foot": self._foot_contact_step_count_by_foot.tolist(),
                "foot_contact_transition_count_by_foot": self._foot_contact_transition_count_by_foot.tolist(),
                "foot_contact_duty_fraction_by_foot": (
                    self._foot_contact_step_count_by_foot / episode_length
                    if episode_length > 0
                    else np.zeros(4, dtype=np.float64)
                ).tolist(),
                "longest_foot_no_contact_run_steps_by_foot": self._longest_foot_no_contact_run_steps.tolist(),
                "longest_foot_no_contact_run_seconds_by_foot": (
                    self._longest_foot_no_contact_run_steps
                    * self.metrics.environment_dt
                ).tolist(),
                "support_count_step_counts_0_to_4": self._support_count_step_counts.tolist(),
                "support_count_step_fractions_0_to_4": (
                    self._support_count_step_counts / episode_length
                    if episode_length > 0
                    else np.zeros(5, dtype=np.float64)
                ).tolist(),
                "support_mask_step_counts_0_to_15": self._support_mask_step_counts.tolist(),
                "foot_sampled_normal_force_time_integral_n_s_by_foot": self._foot_normal_force_time_integral_by_foot.tolist(),
                "foot_sampled_tangential_force_time_integral_n_s_by_foot": self._foot_tangential_force_time_integral_by_foot.tolist(),
                "foot_contact_slip_distance_m_by_foot": self._foot_contact_slip_distance_by_foot.tolist(),
                "foot_contact_slip_speed_max_m_per_s_by_foot": self._foot_contact_slip_speed_max_by_foot.tolist(),
                "airborne_step_count": int(self._airborne_step_count),
                "airborne_step_fraction": (
                    self._airborne_step_count / episode_length
                    if episode_length > 0
                    else 0.0
                ),
                "longest_airborne_run_steps": int(self._longest_airborne_run_steps),
                "longest_airborne_run_seconds": self._longest_airborne_run_steps * self.metrics.environment_dt,
                "airborne_shaping_weight": self.airborne_shaping_weight,
                "reward_airborne_shaping_sum": self._reward_airborne_shaping_sum,
                "foot_contact_gap_shaping_weight": self.foot_contact_gap_shaping_weight,
                "foot_contact_gap_grace_seconds": self.foot_contact_gap_grace_seconds,
                "foot_contact_gap_scale_seconds": self.foot_contact_gap_scale_seconds,
                "reward_foot_contact_gap_shaping_sum": self._reward_foot_contact_gap_shaping_sum,
                "foot_contact_gap_penalty_sum": self._foot_contact_gap_penalty_sum,
                "foot_contact_gap_penalty_sum_by_foot": self._foot_contact_gap_penalty_sum_by_foot.tolist(),
                "actuator_joint_names": list(self.actuator_joint_names),
                "actuator_abs_torque_time_integral_n_m_s_by_actuator": self._actuator_abs_torque_time_integral.tolist(),
                "actuator_positive_mechanical_work_j_by_actuator": self._actuator_positive_mechanical_work.tolist(),
                "actuator_negative_mechanical_work_abs_j_by_actuator": self._actuator_negative_mechanical_work_abs.tolist(),
                "actuator_abs_mechanical_work_j_by_actuator": (
                    self._actuator_positive_mechanical_work
                    + self._actuator_negative_mechanical_work_abs
                ).tolist(),
                "foot_lateral_velocity_shaping_weight": self.foot_lateral_velocity_shaping_weight,
                "foot_lateral_velocity_shaping_scale": self.foot_lateral_velocity_shaping_scale,
                "reward_foot_lateral_velocity_shaping_sum": self._reward_foot_lateral_velocity_shaping_sum,
                "foot_lateral_velocity_penalty_sum": self._foot_lateral_velocity_penalty_sum,
                "foot_lateral_velocity_penalty_sum_by_foot": self._foot_lateral_velocity_penalty_sum_by_foot.tolist(),
                "foot_vertical_velocity_shaping_weight": self.foot_vertical_velocity_shaping_weight,
                "foot_vertical_velocity_shaping_scale": self.foot_vertical_velocity_shaping_scale,
                "reward_foot_vertical_velocity_shaping_sum": self._reward_foot_vertical_velocity_shaping_sum,
                "foot_vertical_velocity_penalty_sum": self._foot_vertical_velocity_penalty_sum,
                "foot_vertical_velocity_penalty_sum_by_foot": self._foot_vertical_velocity_penalty_sum_by_foot.tolist(),
                "pitch_balance_shaping_weight": self.pitch_balance_shaping_weight,
                "reward_pitch_balance_shaping_sum": self._reward_pitch_balance_shaping_sum,
                "pitch_balance_event_completed_count": self._pitch_balance_tracker.completed_event_count,
                "pitch_balance_event_score_sum": self._pitch_balance_tracker.balance_score_sum,
                "pitch_balance_event_score_mean": (
                    self._pitch_balance_tracker.balance_score_sum
                    / self._pitch_balance_tracker.completed_event_count
                    if self._pitch_balance_tracker.completed_event_count > 0
                    else float("nan")
                ),
                "pitch_balance_positive_time_seconds": self._pitch_balance_tracker.active_positive_step_sum * self.metrics.environment_dt,
                "pitch_balance_negative_time_seconds": self._pitch_balance_tracker.active_negative_step_sum * self.metrics.environment_dt,
                "pitch_balance_neutral_time_seconds": self._pitch_balance_tracker.active_neutral_step_sum * self.metrics.environment_dt,
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
        if self.terrain_frame_shaping_enabled:
            _, _, normal, _ = self._terrain_frame_at(
                float(qpos[0]),
                float(qpos[1]),
            )
            return quaternion_tilt_relative_to_normal(qpos[3:7], normal)
        return quaternion_tilt_angle(qpos[3:7])

    def _torso_pitch(self) -> float:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return quaternion_pitch_angle(qpos[3:7])

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
            "torso_pitch_rad": values["torso_pitch"],
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
            "forward_velocity_tracking_weight": self.forward_velocity_tracking_weight,
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
            "foot_landing_height_threshold": self.foot_landing_height_threshold,
            "foot_landing_active_count_step": int(
                np.sum(values["foot_landing_mask"])
            ),
            "foot_landing_mask_step": json.dumps(
                np.asarray(values["foot_landing_mask"], dtype=bool).tolist(),
                separators=(",", ":"),
            ),
            "foot_contact_point_heights_step": json.dumps(
                np.asarray(values["foot_contact_point_heights"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_contact_mask_step": json.dumps(
                np.asarray(values["foot_contact_mask"], dtype=bool).tolist(),
                separators=(",", ":"),
            ),
            "foot_contact_counts_step": json.dumps(
                np.asarray(values["foot_contact_counts"], dtype=np.int64).tolist(),
                separators=(",", ":"),
            ),
            "foot_normal_forces_n_step": json.dumps(
                np.asarray(values["foot_normal_forces"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_tangential_forces_n_step": json.dumps(
                np.asarray(values["foot_tangential_forces"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_contact_tangential_speeds_m_per_s_step": json.dumps(
                np.asarray(values["foot_contact_tangential_speeds"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_contact_slip_distance_m_step": json.dumps(
                np.asarray(values["foot_contact_slip_distance"]).tolist(),
                separators=(",", ":"),
            ),
            "airborne_penalty_step": values["airborne_penalty"],
            "reward_airborne_shaping_step": values[
                "airborne_shaping_reward"
            ],
            "foot_contact_gap_penalty_step": values[
                "foot_contact_gap_penalty"
            ],
            "foot_contact_gap_penalties_step": json.dumps(
                np.asarray(values["foot_contact_gap_penalties"]).tolist(),
                separators=(",", ":"),
            ),
            "reward_foot_contact_gap_shaping_step": values[
                "foot_contact_gap_shaping_reward"
            ],
            "foot_lateral_velocities_step": json.dumps(
                np.asarray(values["foot_lateral_velocities"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_vertical_velocities_step": json.dumps(
                np.asarray(values["foot_vertical_velocities"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_lateral_velocity_penalty_step": values[
                "foot_lateral_velocity_penalty"
            ],
            "foot_vertical_velocity_penalty_step": values[
                "foot_vertical_velocity_penalty"
            ],
            "foot_lateral_velocity_penalties_step": json.dumps(
                np.asarray(values["foot_lateral_velocity_penalties"]).tolist(),
                separators=(",", ":"),
            ),
            "foot_vertical_velocity_penalties_step": json.dumps(
                np.asarray(values["foot_vertical_velocity_penalties"]).tolist(),
                separators=(",", ":"),
            ),
            "reward_foot_lateral_velocity_shaping_step": values[
                "foot_lateral_velocity_shaping_reward"
            ],
            "reward_foot_vertical_velocity_shaping_step": values[
                "foot_vertical_velocity_shaping_reward"
            ],
            "foot_landing_transition_mask_step": json.dumps(
                np.asarray(
                    values["pitch_balance_event"]["landing_transitions"],
                    dtype=bool,
                ).tolist(),
                separators=(",", ":"),
            ),
            "pitch_balance_shaping_weight": self.pitch_balance_shaping_weight,
            "pitch_balance_event_active_step": values["pitch_balance_event"]["active"],
            "pitch_balance_event_started_step": values["pitch_balance_event"]["started"],
            "pitch_balance_event_completed_step": values["pitch_balance_event"]["completed"],
            "pitch_balance_event_landed_count_step": values["pitch_balance_event"]["landed_count"],
            "pitch_balance_event_positive_steps_step": values["pitch_balance_event"]["positive_steps"],
            "pitch_balance_event_negative_steps_step": values["pitch_balance_event"]["negative_steps"],
            "pitch_balance_event_neutral_steps_step": values["pitch_balance_event"]["neutral_steps"],
            "pitch_balance_event_score_step": values["pitch_balance_event"]["score"],
            "reward_pitch_balance_shaping_step": values["pitch_balance_shaping_reward"],
            "actuator_joint_torques_n_m_step": json.dumps(
                np.asarray(values["actuator_joint_torques"]).tolist(),
                separators=(",", ":"),
            ),
            "actuator_joint_velocities_rad_per_s_step": json.dumps(
                np.asarray(values["actuator_joint_velocities"]).tolist(),
                separators=(",", ":"),
            ),
            "actuator_mechanical_powers_w_step": json.dumps(
                np.asarray(values["actuator_mechanical_powers"]).tolist(),
                separators=(",", ":"),
            ),
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
    terminate_when_unhealthy: bool = True,
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
    forward_velocity_tracking_weight: float = 1.0,
    action_rate_shaping_weight: float = 0.0,
    vertical_velocity_shaping_weight: float = 0.0,
    vertical_velocity_shaping_scale: float = 1.0,
    roll_pitch_angular_velocity_shaping_weight: float = 0.0,
    roll_pitch_angular_velocity_shaping_scale: float = 1.0,
    foot_landing_height_threshold: float = 0.03,
    foot_lateral_velocity_shaping_weight: float = 0.0,
    foot_lateral_velocity_shaping_scale: float = 1.0,
    foot_vertical_velocity_shaping_weight: float = 0.0,
    foot_vertical_velocity_shaping_scale: float = 1.0,
    airborne_shaping_weight: float = 0.0,
    foot_contact_gap_shaping_weight: float = 0.0,
    foot_contact_gap_grace_seconds: float = 0.5,
    foot_contact_gap_scale_seconds: float = 0.5,
    pitch_balance_shaping_weight: float = 0.0,
    foot_geom_names: tuple[str, ...] = DEFAULT_FOOT_GEOM_NAMES,
    common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
    effort_distance_min: float = EPSILON,
    action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
    augment_previous_applied_action: bool = False,
    action_slew_l2_limit: float | None = None,
    terrain_frame_shaping_enabled: bool = False,
    step_log_path: str | Path | None = None,
) -> ProxyGapAntWrapper:
    """Create Ant-v5 with separately logged objective and diagnostic terms."""
    kwargs: dict[str, Any] = {
        "ctrl_cost_weight": float(ctrl_cost_weight),
        "render_mode": render_mode,
        "terminate_when_unhealthy": bool(terminate_when_unhealthy),
    }
    temporary_xml_path: Path | None = None
    if xml_file is not None:
        resolved_xml_path = Path(xml_file).resolve()
        if not resolved_xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML does not exist: {resolved_xml_path}")
        try:
            str(resolved_xml_path).encode("ascii")
            mujoco_xml_path = resolved_xml_path
        except UnicodeEncodeError:
            # MuJoCo's Windows path loader can reject otherwise valid Unicode
            # paths. Ant's public render XML is self-contained, so an ASCII-
            # named temporary copy is sufficient and is removed after loading.
            with tempfile.NamedTemporaryFile(
                prefix="proxygap_mujoco_",
                suffix=".xml",
                delete=False,
            ) as temporary_xml:
                temporary_xml.write(resolved_xml_path.read_bytes())
                temporary_xml_path = Path(temporary_xml.name)
            mujoco_xml_path = temporary_xml_path
        kwargs["xml_file"] = str(mujoco_xml_path)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    try:
        env = gym.make("Ant-v5", **kwargs)
    finally:
        if temporary_xml_path is not None:
            temporary_xml_path.unlink(missing_ok=True)
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
        forward_velocity_tracking_weight=float(forward_velocity_tracking_weight),
        action_rate_shaping_weight=float(action_rate_shaping_weight),
        vertical_velocity_shaping_weight=float(vertical_velocity_shaping_weight),
        vertical_velocity_shaping_scale=float(vertical_velocity_shaping_scale),
        roll_pitch_angular_velocity_shaping_weight=float(
            roll_pitch_angular_velocity_shaping_weight
        ),
        roll_pitch_angular_velocity_shaping_scale=float(
            roll_pitch_angular_velocity_shaping_scale
        ),
        foot_landing_height_threshold=float(foot_landing_height_threshold),
        foot_lateral_velocity_shaping_weight=float(
            foot_lateral_velocity_shaping_weight
        ),
        foot_lateral_velocity_shaping_scale=float(
            foot_lateral_velocity_shaping_scale
        ),
        foot_vertical_velocity_shaping_weight=float(
            foot_vertical_velocity_shaping_weight
        ),
        foot_vertical_velocity_shaping_scale=float(
            foot_vertical_velocity_shaping_scale
        ),
        airborne_shaping_weight=float(airborne_shaping_weight),
        foot_contact_gap_shaping_weight=float(
            foot_contact_gap_shaping_weight
        ),
        foot_contact_gap_grace_seconds=float(foot_contact_gap_grace_seconds),
        foot_contact_gap_scale_seconds=float(foot_contact_gap_scale_seconds),
        pitch_balance_shaping_weight=float(pitch_balance_shaping_weight),
        foot_geom_names=tuple(foot_geom_names),
        common_rescore_ctrl_cost_weight=float(common_rescore_ctrl_cost_weight),
        effort_distance_min=float(effort_distance_min),
        action_saturation_threshold=float(action_saturation_threshold),
        augment_previous_applied_action=bool(augment_previous_applied_action),
        action_slew_l2_limit=action_slew_l2_limit,
        terrain_frame_shaping_enabled=bool(terrain_frame_shaping_enabled),
        step_log_path=step_log_path,
    )
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped
