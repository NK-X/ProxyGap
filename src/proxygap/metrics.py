"""Disaggregated diagnostics for the ProxyGap Ant-v5 study.

Reward returns, locomotion diagnostics and safety indicators deliberately remain
separate.  No aggregate ``true_performance`` scalar is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


EPSILON = 1e-8
DEFAULT_COMMON_RESCORE_CTRL_WEIGHT = 0.5
DEFAULT_ACTION_SATURATION_THRESHOLD = 0.95
DEFAULT_ENVIRONMENT_DT = 0.05
DEFAULT_ACTION_DIMENSION = 8
DEFAULT_EVALUATION_HORIZON_STEPS = 1000
DEFAULT_INVERSION_TILT_THRESHOLD_RAD = math.pi / 2.0
DEFAULT_SUSTAINED_INVERSION_SECONDS = 1.0
DEFAULT_INTENT_FORWARD_VELOCITY_MIN = 0.8
DEFAULT_INTENT_FORWARD_VELOCITY_MAX = 1.2
DEFAULT_INTENT_TILT_RMS_MAX_RAD = math.radians(15.0)
DEFAULT_INTENT_DIRECTION_ERROR_MAX_RAD = math.radians(5.0)
DEFAULT_INTENT_PATH_EFFICIENCY_MIN = 0.90
DEFAULT_INTENT_NORMALISED_ACTION_ROUGHNESS_MAX = 0.04
DEFAULT_INTENT_ACTION_SATURATION_RATE_MAX = 0.01


CSV_SCHEMA = [
    "episode",
    "checkpoint_fraction",
    "target_timesteps",
    "actual_model_timesteps",
    "condition_id",
    "reward_id",
    "constraint_id",
    "ctrl_cost_weight",
    "common_rescore_ctrl_cost_weight",
    "forward_progress_shaping_weight",
    "lateral_drift_shaping_weight",
    "lateral_drift_shaping_scale",
    "lateral_shaping_signal",
    "lateral_velocity_target",
    "effort_shaping_weight",
    "effort_shaping_scale",
    "orientation_shaping_weight",
    "orientation_shaping_scale",
    "orientation_shaping_function",
    "replace_forward_reward_with_tracking",
    "forward_velocity_target",
    "forward_velocity_tracking_scale",
    "reward_forward_tracking_sum",
    "reward_forward_replacement_sum",
    "action_rate_shaping_weight",
    "reward_action_rate_shaping_sum",
    "action_rate_penalty_sum",
    "vertical_velocity_shaping_weight",
    "vertical_velocity_shaping_scale",
    "reward_vertical_velocity_shaping_sum",
    "vertical_velocity_penalty_sum",
    "roll_pitch_angular_velocity_shaping_weight",
    "roll_pitch_angular_velocity_shaping_scale",
    "reward_roll_pitch_angular_velocity_shaping_sum",
    "roll_pitch_angular_velocity_penalty_sum",
    "training_seed",
    "seed",
    "condition_objective_return",
    "common_rescored_return",
    "proxy_return",
    "base_proxy_return",
    "reward_shaping_sum",
    "reward_forward_shaping_sum",
    "reward_lateral_shaping_sum",
    "reward_effort_shaping_sum",
    "reward_orientation_shaping_sum",
    "orientation_penalty_sum",
    "reward_forward_sum",
    "reward_ctrl_sum",
    "reward_contact_sum",
    "reward_survive_sum",
    "base_reward_reconciliation_error",
    "ctrl_cost_reconciliation_error",
    "net_forward_progress",
    "net_forward_progress_per_step",
    "environment_dt",
    "episode_duration_seconds",
    "mean_forward_velocity",
    "evaluation_horizon_steps",
    "fixed_horizon_duration_seconds",
    "fixed_horizon_mean_forward_velocity",
    "net_displacement_direction_error_rad",
    "net_displacement_direction_error_degrees",
    "cumulative_squared_action",
    "mean_squared_action_per_step",
    "action_saturation_rate",
    "cumulative_squared_action_change",
    "mean_squared_action_change_per_transition",
    "normalised_action_roughness",
    "action_change_transition_count",
    "effort_distance_min",
    "effort_per_distance_defined",
    "cumulative_squared_action_per_unit_distance",
    "control_effort",
    "control_effort_per_unit_distance",
    "unhealthy_termination",
    "termination_category",
    "low_z_termination",
    "high_z_termination",
    "non_finite_termination",
    "other_unhealthy_termination",
    "time_limit_truncation",
    "fall",
    "lateral_drift_final_abs",
    "lateral_drift_mean_abs",
    "lateral_drift_max_abs",
    "cumulative_lateral_path",
    "cumulative_planar_path",
    "forward_path_efficiency",
    "torso_tilt_mean",
    "torso_tilt_std",
    "torso_tilt_rms",
    "torso_tilt_p95",
    "torso_tilt_max",
    "inversion_tilt_threshold_rad",
    "inverted_step_fraction",
    "longest_inverted_run_steps",
    "longest_inverted_run_seconds",
    "sustained_inversion_seconds",
    "sustained_inversion",
    "full_horizon_completed",
    "intent_compliant",
    "intent_failure_reasons",
    "action_observation_augmented",
    "action_constraint_enabled",
    "action_slew_l2_limit",
    "action_slew_intervention_count",
    "action_slew_intervention_rate",
    "cumulative_action_correction_l2",
    "mean_action_correction_l2",
    "max_action_correction_l2",
    "cumulative_proposed_squared_action_change",
    "proposed_action_change_transition_count",
    "proposed_normalised_action_roughness",
    "episode_length",
    "terminated",
    "truncated",
]


def safe_divide(numerator: float, denominator: float) -> float:
    """Return NaN when a ratio would be numerically meaningless."""
    if abs(denominator) < EPSILON:
        return float("nan")
    return float(numerator / denominator)


def common_rescored_return(
    *,
    reward_forward_sum: float,
    reward_survive_sum: float,
    reward_contact_sum: float,
    cumulative_squared_action: float,
    ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
) -> float:
    """Score recorded components under one fixed Ant reward definition.

    Gymnasium logs contact reward as a signed (normally non-positive) term.
    The action penalty is reconstructed from the recorded squared action so that
    policies trained under different control-cost weights share one score scale.
    """
    if ctrl_cost_weight < 0:
        raise ValueError("ctrl_cost_weight must be non-negative")
    return float(
        reward_forward_sum
        + reward_survive_sum
        + reward_contact_sum
        - ctrl_cost_weight * cumulative_squared_action
    )


def classify_termination(
    *,
    terminated: bool,
    torso_height: float,
    state_is_finite: bool,
    healthy_z_range: tuple[float, float],
) -> str:
    """Separate low-z, high-z and non-finite unhealthy terminations."""
    if not terminated:
        return "none"
    if not state_is_finite:
        return "non_finite_state"
    low, high = healthy_z_range
    if torso_height < low:
        return "low_z_collapse"
    if torso_height > high:
        return "high_z_excursion"
    return "other_unhealthy"


def quaternion_tilt_angle(quaternion_wxyz: np.ndarray) -> float:
    """Return torso tilt from world vertical in radians.

    Gymnasium/MuJoCo stores the Ant root orientation in ``w, x, y, z`` order.
    The local z-axis alignment is clipped to ``[-1, 1]`` before ``acos``.
    """
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < EPSILON:
        return float("nan")
    w, x, y, z = q / norm
    _ = w, z
    vertical_alignment = 1.0 - 2.0 * (x * x + y * y)
    vertical_alignment = float(np.clip(vertical_alignment, -1.0, 1.0))
    return float(math.acos(vertical_alignment))


@dataclass
class EpisodeMetrics:
    """Mutable episode-level accumulator with explicit metric parameters."""

    condition_ctrl_cost_weight: float = 0.5
    common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT
    effort_distance_min: float = EPSILON
    action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD
    environment_dt: float = DEFAULT_ENVIRONMENT_DT
    action_dimension: int = DEFAULT_ACTION_DIMENSION
    evaluation_horizon_steps: int = DEFAULT_EVALUATION_HORIZON_STEPS
    inversion_tilt_threshold_rad: float = DEFAULT_INVERSION_TILT_THRESHOLD_RAD
    sustained_inversion_seconds: float = DEFAULT_SUSTAINED_INVERSION_SECONDS
    intent_forward_velocity_min: float = DEFAULT_INTENT_FORWARD_VELOCITY_MIN
    intent_forward_velocity_max: float = DEFAULT_INTENT_FORWARD_VELOCITY_MAX
    intent_tilt_rms_max_rad: float = DEFAULT_INTENT_TILT_RMS_MAX_RAD
    intent_direction_error_max_rad: float = DEFAULT_INTENT_DIRECTION_ERROR_MAX_RAD
    intent_path_efficiency_min: float = DEFAULT_INTENT_PATH_EFFICIENCY_MIN
    intent_normalised_action_roughness_max: float = (
        DEFAULT_INTENT_NORMALISED_ACTION_ROUGHNESS_MAX
    )
    intent_action_saturation_rate_max: float = (
        DEFAULT_INTENT_ACTION_SATURATION_RATE_MAX
    )
    healthy_z_range: tuple[float, float] = (0.2, 1.0)
    initial_x: float = 0.0
    initial_y: float = 0.0
    condition_objective_return: float = 0.0
    base_proxy_return: float = 0.0
    reward_shaping_sum: float = 0.0
    reward_forward_shaping_sum: float = 0.0
    reward_lateral_shaping_sum: float = 0.0
    reward_effort_shaping_sum: float = 0.0
    reward_orientation_shaping_sum: float = 0.0
    orientation_penalty_sum: float = 0.0
    reward_forward_sum: float = 0.0
    reward_ctrl_sum: float = 0.0
    reward_contact_sum: float = 0.0
    reward_survive_sum: float = 0.0
    cumulative_squared_action: float = 0.0
    action_component_count: int = 0
    saturated_action_component_count: int = 0
    cumulative_squared_action_change: float = 0.0
    action_change_transition_count: int = 0
    latest_squared_action_change: float = float("nan")
    previous_action: np.ndarray | None = field(default=None, repr=False)
    episode_length: int = 0
    terminated: bool = False
    truncated: bool = False
    termination_category: str = "none"
    latest_x: float = 0.0
    latest_y: float = 0.0
    cumulative_lateral_path: float = 0.0
    cumulative_planar_path: float = 0.0
    lateral_abs_history: list[float] = field(default_factory=list)
    torso_tilt_history: list[float] = field(default_factory=list)
    inverted_step_count: int = 0
    current_inverted_run_steps: int = 0
    longest_inverted_run_steps: int = 0

    def __post_init__(self) -> None:
        if self.condition_ctrl_cost_weight < 0:
            raise ValueError("condition_ctrl_cost_weight must be non-negative")
        if self.common_rescore_ctrl_cost_weight < 0:
            raise ValueError("common_rescore_ctrl_cost_weight must be non-negative")
        if self.effort_distance_min <= 0:
            raise ValueError("effort_distance_min must be positive")
        if not 0 < self.action_saturation_threshold <= 1:
            raise ValueError("action_saturation_threshold must be in (0, 1]")
        if self.environment_dt <= 0:
            raise ValueError("environment_dt must be positive")
        if self.action_dimension <= 0:
            raise ValueError("action_dimension must be positive")
        if self.evaluation_horizon_steps <= 0:
            raise ValueError("evaluation_horizon_steps must be positive")
        if not 0 < self.inversion_tilt_threshold_rad <= math.pi:
            raise ValueError("inversion_tilt_threshold_rad must be in (0, pi]")
        if self.sustained_inversion_seconds <= 0:
            raise ValueError("sustained_inversion_seconds must be positive")
        if not 0 <= self.intent_forward_velocity_min < self.intent_forward_velocity_max:
            raise ValueError("intent forward-velocity interval is invalid")
        if not 0 < self.intent_tilt_rms_max_rad <= math.pi:
            raise ValueError("intent_tilt_rms_max_rad must be in (0, pi]")
        if not 0 < self.intent_direction_error_max_rad <= math.pi:
            raise ValueError("intent_direction_error_max_rad must be in (0, pi]")
        if not 0 < self.intent_path_efficiency_min <= 1:
            raise ValueError("intent_path_efficiency_min must be in (0, 1]")
        if not 0 <= self.intent_normalised_action_roughness_max <= 1:
            raise ValueError(
                "intent_normalised_action_roughness_max must be in [0, 1]"
            )
        if not 0 <= self.intent_action_saturation_rate_max <= 1:
            raise ValueError("intent_action_saturation_rate_max must be in [0, 1]")

    def reset(self, *, initial_x: float, initial_y: float) -> None:
        self.initial_x = float(initial_x)
        self.initial_y = float(initial_y)
        self.condition_objective_return = 0.0
        self.base_proxy_return = 0.0
        self.reward_shaping_sum = 0.0
        self.reward_forward_shaping_sum = 0.0
        self.reward_lateral_shaping_sum = 0.0
        self.reward_effort_shaping_sum = 0.0
        self.reward_orientation_shaping_sum = 0.0
        self.orientation_penalty_sum = 0.0
        self.reward_forward_sum = 0.0
        self.reward_ctrl_sum = 0.0
        self.reward_contact_sum = 0.0
        self.reward_survive_sum = 0.0
        self.cumulative_squared_action = 0.0
        self.action_component_count = 0
        self.saturated_action_component_count = 0
        self.cumulative_squared_action_change = 0.0
        self.action_change_transition_count = 0
        self.latest_squared_action_change = float("nan")
        self.previous_action = None
        self.episode_length = 0
        self.terminated = False
        self.truncated = False
        self.termination_category = "none"
        self.latest_x = float(initial_x)
        self.latest_y = float(initial_y)
        self.cumulative_lateral_path = 0.0
        self.cumulative_planar_path = 0.0
        self.lateral_abs_history.clear()
        self.torso_tilt_history.clear()
        self.inverted_step_count = 0
        self.current_inverted_run_steps = 0
        self.longest_inverted_run_steps = 0

    def update(
        self,
        *,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        torso_tilt: float,
        torso_height: float = float("nan"),
        state_is_finite: bool = True,
    ) -> None:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.size != self.action_dimension:
            raise ValueError(
                f"Expected {self.action_dimension} action components, "
                f"received {action_array.size}"
            )
        self.condition_objective_return += float(reward)
        self.base_proxy_return += float(info.get("reward_base_proxy", reward))
        self.reward_shaping_sum += float(info.get("reward_shaping", 0.0))
        self.reward_forward_shaping_sum += float(
            info.get("reward_forward_shaping", 0.0)
        )
        self.reward_lateral_shaping_sum += float(
            info.get("reward_lateral_shaping", 0.0)
        )
        self.reward_effort_shaping_sum += float(
            info.get("reward_effort_shaping", 0.0)
        )
        self.reward_orientation_shaping_sum += float(
            info.get("reward_orientation_shaping", 0.0)
        )
        self.orientation_penalty_sum += float(info.get("orientation_penalty", 0.0))
        self.reward_forward_sum += float(info.get("reward_forward", 0.0))
        self.reward_ctrl_sum += float(info.get("reward_ctrl", 0.0))
        self.reward_contact_sum += float(info.get("reward_contact", 0.0))
        self.reward_survive_sum += float(info.get("reward_survive", 0.0))
        self.cumulative_squared_action += float(np.sum(np.square(action_array)))
        self.action_component_count += int(action_array.size)
        self.saturated_action_component_count += int(
            np.count_nonzero(np.abs(action_array) >= self.action_saturation_threshold)
        )
        if self.previous_action is None:
            self.latest_squared_action_change = float("nan")
        else:
            squared_change = float(
                np.sum(np.square(action_array - self.previous_action))
            )
            self.latest_squared_action_change = squared_change
            self.cumulative_squared_action_change += squared_change
            self.action_change_transition_count += 1
        self.previous_action = action_array.copy()
        self.episode_length += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        previous_x = self.latest_x
        previous_y = self.latest_y
        self.latest_x = float(info.get("x_position", self.latest_x))
        self.latest_y = float(info.get("y_position", self.latest_y))
        self.cumulative_lateral_path += abs(self.latest_y - previous_y)
        self.cumulative_planar_path += math.hypot(
            self.latest_x - previous_x,
            self.latest_y - previous_y,
        )
        self.lateral_abs_history.append(abs(self.latest_y - self.initial_y))
        if math.isfinite(torso_tilt):
            self.torso_tilt_history.append(float(torso_tilt))
            if torso_tilt >= self.inversion_tilt_threshold_rad:
                self.inverted_step_count += 1
                self.current_inverted_run_steps += 1
                self.longest_inverted_run_steps = max(
                    self.longest_inverted_run_steps,
                    self.current_inverted_run_steps,
                )
            else:
                self.current_inverted_run_steps = 0
        if terminated:
            self.termination_category = classify_termination(
                terminated=True,
                torso_height=float(torso_height),
                state_is_finite=bool(state_is_finite),
                healthy_z_range=self.healthy_z_range,
            )

    def summary(self) -> dict[str, float | int | bool | str]:
        net_forward_progress = self.latest_x - self.initial_x
        episode_duration_seconds = self.episode_length * self.environment_dt
        fixed_horizon_duration_seconds = (
            self.evaluation_horizon_steps * self.environment_dt
        )
        effort_defined = net_forward_progress > self.effort_distance_min
        effort_per_distance = (
            self.cumulative_squared_action / net_forward_progress
            if effort_defined
            else float("nan")
        )
        lateral_mean = (
            float(np.mean(self.lateral_abs_history))
            if self.lateral_abs_history
            else 0.0
        )
        lateral_max = max(self.lateral_abs_history, default=0.0)
        tilt = np.asarray(self.torso_tilt_history, dtype=np.float64)
        tilt_mean = float(np.mean(tilt)) if tilt.size else float("nan")
        tilt_std = float(np.std(tilt)) if tilt.size else float("nan")
        tilt_rms = float(np.sqrt(np.mean(np.square(tilt)))) if tilt.size else float("nan")
        tilt_p95 = float(np.percentile(tilt, 95)) if tilt.size else float("nan")
        tilt_max = float(np.max(tilt)) if tilt.size else float("nan")
        inverted_step_fraction = safe_divide(
            self.inverted_step_count,
            self.episode_length,
        )
        longest_inverted_run_seconds = (
            self.longest_inverted_run_steps * self.environment_dt
        )
        common_return = common_rescored_return(
            reward_forward_sum=self.reward_forward_sum,
            reward_survive_sum=self.reward_survive_sum,
            reward_contact_sum=self.reward_contact_sum,
            cumulative_squared_action=self.cumulative_squared_action,
            ctrl_cost_weight=self.common_rescore_ctrl_cost_weight,
        )
        reconstructed_base = common_rescored_return(
            reward_forward_sum=self.reward_forward_sum,
            reward_survive_sum=self.reward_survive_sum,
            reward_contact_sum=self.reward_contact_sum,
            cumulative_squared_action=self.cumulative_squared_action,
            ctrl_cost_weight=self.condition_ctrl_cost_weight,
        )
        low_z = self.termination_category == "low_z_collapse"
        high_z = self.termination_category == "high_z_excursion"
        non_finite = self.termination_category == "non_finite_state"
        other_unhealthy = self.termination_category == "other_unhealthy"
        action_saturation_rate = safe_divide(
            self.saturated_action_component_count,
            self.action_component_count,
        )
        mean_squared_action_change = safe_divide(
            self.cumulative_squared_action_change,
            self.action_change_transition_count,
        )
        # Each Ant action component lies in [-1, 1], so its largest squared
        # change is 4. Dividing by 4 * action_dimension maps roughness to [0, 1].
        normalised_action_roughness = safe_divide(
            mean_squared_action_change,
            4.0 * self.action_dimension,
        )
        forward_path_efficiency = safe_divide(
            net_forward_progress,
            self.cumulative_planar_path,
        )
        fixed_horizon_velocity = safe_divide(
            net_forward_progress,
            fixed_horizon_duration_seconds,
        )
        net_lateral_displacement = abs(self.latest_y - self.initial_y)
        direction_error = math.atan2(
            net_lateral_displacement,
            max(net_forward_progress, EPSILON),
        )
        sustained_inversion = bool(
            longest_inverted_run_seconds >= self.sustained_inversion_seconds
        )
        full_horizon_completed = bool(
            self.episode_length == self.evaluation_horizon_steps
            and self.truncated
            and not self.terminated
        )
        compliance_checks = {
            "full_horizon": full_horizon_completed,
            "forward_velocity": bool(
                math.isfinite(fixed_horizon_velocity)
                and self.intent_forward_velocity_min
                <= fixed_horizon_velocity
                <= self.intent_forward_velocity_max
            ),
            "no_sustained_inversion": not sustained_inversion,
            "tilt_rms": bool(
                math.isfinite(tilt_rms)
                and tilt_rms <= self.intent_tilt_rms_max_rad
            ),
            "direction_error": bool(
                math.isfinite(direction_error)
                and direction_error <= self.intent_direction_error_max_rad
            ),
            "path_efficiency": bool(
                math.isfinite(forward_path_efficiency)
                and forward_path_efficiency >= self.intent_path_efficiency_min
            ),
            "action_roughness": bool(
                math.isfinite(normalised_action_roughness)
                and normalised_action_roughness
                <= self.intent_normalised_action_roughness_max
            ),
            "action_saturation": bool(
                math.isfinite(action_saturation_rate)
                and action_saturation_rate
                <= self.intent_action_saturation_rate_max
            ),
        }
        intent_failure_reasons = ";".join(
            name for name, passed in compliance_checks.items() if not passed
        )
        return {
            "condition_objective_return": float(self.condition_objective_return),
            "common_rescore_ctrl_cost_weight": float(self.common_rescore_ctrl_cost_weight),
            "common_rescored_return": common_return,
            # Legacy aliases remain readable for formal v1 but are not v2 report labels.
            "proxy_return": float(self.condition_objective_return),
            "base_proxy_return": float(self.base_proxy_return),
            "reward_shaping_sum": float(self.reward_shaping_sum),
            "reward_forward_shaping_sum": float(self.reward_forward_shaping_sum),
            "reward_lateral_shaping_sum": float(self.reward_lateral_shaping_sum),
            "reward_effort_shaping_sum": float(self.reward_effort_shaping_sum),
            "reward_orientation_shaping_sum": float(self.reward_orientation_shaping_sum),
            "orientation_penalty_sum": float(self.orientation_penalty_sum),
            "reward_forward_sum": float(self.reward_forward_sum),
            "reward_ctrl_sum": float(self.reward_ctrl_sum),
            "reward_contact_sum": float(self.reward_contact_sum),
            "reward_survive_sum": float(self.reward_survive_sum),
            "base_reward_reconciliation_error": float(
                self.base_proxy_return - reconstructed_base
            ),
            "ctrl_cost_reconciliation_error": float(
                self.reward_ctrl_sum
                + self.condition_ctrl_cost_weight * self.cumulative_squared_action
            ),
            "net_forward_progress": float(net_forward_progress),
            "net_forward_progress_per_step": safe_divide(
                net_forward_progress,
                self.episode_length,
            ),
            "environment_dt": float(self.environment_dt),
            "episode_duration_seconds": float(episode_duration_seconds),
            "mean_forward_velocity": safe_divide(
                net_forward_progress,
                episode_duration_seconds,
            ),
            "evaluation_horizon_steps": int(self.evaluation_horizon_steps),
            "fixed_horizon_duration_seconds": float(
                fixed_horizon_duration_seconds
            ),
            "fixed_horizon_mean_forward_velocity": safe_divide(
                net_forward_progress,
                fixed_horizon_duration_seconds,
            ),
            "net_displacement_direction_error_rad": float(direction_error),
            "net_displacement_direction_error_degrees": float(
                math.degrees(direction_error)
            ),
            "cumulative_squared_action": float(self.cumulative_squared_action),
            "mean_squared_action_per_step": safe_divide(
                self.cumulative_squared_action,
                self.episode_length,
            ),
            "action_saturation_rate": action_saturation_rate,
            "cumulative_squared_action_change": float(
                self.cumulative_squared_action_change
            ),
            "mean_squared_action_change_per_transition": mean_squared_action_change,
            "normalised_action_roughness": normalised_action_roughness,
            "action_change_transition_count": int(
                self.action_change_transition_count
            ),
            "effort_distance_min": float(self.effort_distance_min),
            "effort_per_distance_defined": bool(effort_defined),
            "cumulative_squared_action_per_unit_distance": float(effort_per_distance),
            "control_effort": float(self.cumulative_squared_action),
            "control_effort_per_unit_distance": float(effort_per_distance),
            "unhealthy_termination": bool(self.terminated),
            "termination_category": self.termination_category,
            "low_z_termination": bool(low_z),
            "high_z_termination": bool(high_z),
            "non_finite_termination": bool(non_finite),
            "other_unhealthy_termination": bool(other_unhealthy),
            "time_limit_truncation": bool(self.truncated),
            "fall": bool(self.terminated),
            "lateral_drift_final_abs": float(abs(self.latest_y - self.initial_y)),
            "lateral_drift_mean_abs": lateral_mean,
            "lateral_drift_max_abs": float(lateral_max),
            "cumulative_lateral_path": float(self.cumulative_lateral_path),
            "cumulative_planar_path": float(self.cumulative_planar_path),
            "forward_path_efficiency": float(forward_path_efficiency),
            "torso_tilt_mean": tilt_mean,
            "torso_tilt_std": tilt_std,
            "torso_tilt_rms": tilt_rms,
            "torso_tilt_p95": tilt_p95,
            "torso_tilt_max": tilt_max,
            "inversion_tilt_threshold_rad": float(
                self.inversion_tilt_threshold_rad
            ),
            "inverted_step_fraction": inverted_step_fraction,
            "longest_inverted_run_steps": int(self.longest_inverted_run_steps),
            "longest_inverted_run_seconds": float(
                longest_inverted_run_seconds
            ),
            "sustained_inversion_seconds": float(
                self.sustained_inversion_seconds
            ),
            "sustained_inversion": sustained_inversion,
            "full_horizon_completed": full_horizon_completed,
            "intent_compliant": bool(all(compliance_checks.values())),
            "intent_failure_reasons": intent_failure_reasons,
            "episode_length": int(self.episode_length),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }

    def live_summary(self) -> dict[str, float | int | bool | str]:
        """Return inexpensive per-step values without history-wide reductions.

        Full distributional diagnostics such as tilt percentiles are calculated
        by :meth:`summary` at episode boundaries or explicit evaluation time.
        """
        net_forward_progress = self.latest_x - self.initial_x
        episode_duration_seconds = self.episode_length * self.environment_dt
        fixed_horizon_duration_seconds = (
            self.evaluation_horizon_steps * self.environment_dt
        )
        effort_defined = net_forward_progress > self.effort_distance_min
        effort_per_distance = (
            self.cumulative_squared_action / net_forward_progress
            if effort_defined
            else float("nan")
        )
        return {
            "condition_objective_return": float(self.condition_objective_return),
            "proxy_return": float(self.condition_objective_return),
            "base_proxy_return": float(self.base_proxy_return),
            "reward_shaping_sum": float(self.reward_shaping_sum),
            "orientation_penalty_sum": float(self.orientation_penalty_sum),
            "net_forward_progress": float(net_forward_progress),
            "environment_dt": float(self.environment_dt),
            "episode_duration_seconds": float(episode_duration_seconds),
            "mean_forward_velocity": safe_divide(
                net_forward_progress,
                episode_duration_seconds,
            ),
            "evaluation_horizon_steps": int(self.evaluation_horizon_steps),
            "fixed_horizon_duration_seconds": float(
                fixed_horizon_duration_seconds
            ),
            "fixed_horizon_mean_forward_velocity": safe_divide(
                net_forward_progress,
                fixed_horizon_duration_seconds,
            ),
            "cumulative_squared_action": float(self.cumulative_squared_action),
            "cumulative_squared_action_change": float(
                self.cumulative_squared_action_change
            ),
            "mean_squared_action_change_per_transition": safe_divide(
                self.cumulative_squared_action_change,
                self.action_change_transition_count,
            ),
            "normalised_action_roughness": safe_divide(
                safe_divide(
                    self.cumulative_squared_action_change,
                    self.action_change_transition_count,
                ),
                4.0 * self.action_dimension,
            ),
            "control_effort": float(self.cumulative_squared_action),
            "control_effort_per_unit_distance": float(effort_per_distance),
            "unhealthy_termination": bool(self.terminated),
            "termination_category": self.termination_category,
            "episode_length": int(self.episode_length),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }
