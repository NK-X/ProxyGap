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


CSV_SCHEMA = [
    "episode",
    "checkpoint_fraction",
    "target_timesteps",
    "actual_model_timesteps",
    "condition_id",
    "ctrl_cost_weight",
    "common_rescore_ctrl_cost_weight",
    "forward_progress_shaping_weight",
    "lateral_drift_shaping_weight",
    "effort_shaping_weight",
    "effort_shaping_scale",
    "stability_shaping_weight",
    "stability_shaping_scale",
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
    "reward_stability_shaping_sum",
    "reward_forward_sum",
    "reward_ctrl_sum",
    "reward_contact_sum",
    "reward_survive_sum",
    "base_reward_reconciliation_error",
    "ctrl_cost_reconciliation_error",
    "net_forward_progress",
    "net_forward_progress_per_step",
    "cumulative_squared_action",
    "mean_squared_action_per_step",
    "action_saturation_rate",
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
    "torso_tilt_mean",
    "torso_tilt_std",
    "torso_tilt_rms",
    "torso_tilt_p95",
    "torso_tilt_max",
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
    healthy_z_range: tuple[float, float] = (0.2, 1.0)
    initial_x: float = 0.0
    initial_y: float = 0.0
    condition_objective_return: float = 0.0
    base_proxy_return: float = 0.0
    reward_shaping_sum: float = 0.0
    reward_forward_shaping_sum: float = 0.0
    reward_lateral_shaping_sum: float = 0.0
    reward_effort_shaping_sum: float = 0.0
    reward_stability_shaping_sum: float = 0.0
    reward_forward_sum: float = 0.0
    reward_ctrl_sum: float = 0.0
    reward_contact_sum: float = 0.0
    reward_survive_sum: float = 0.0
    cumulative_squared_action: float = 0.0
    action_component_count: int = 0
    saturated_action_component_count: int = 0
    episode_length: int = 0
    terminated: bool = False
    truncated: bool = False
    termination_category: str = "none"
    latest_x: float = 0.0
    latest_y: float = 0.0
    cumulative_lateral_path: float = 0.0
    lateral_abs_history: list[float] = field(default_factory=list)
    torso_tilt_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.condition_ctrl_cost_weight < 0:
            raise ValueError("condition_ctrl_cost_weight must be non-negative")
        if self.common_rescore_ctrl_cost_weight < 0:
            raise ValueError("common_rescore_ctrl_cost_weight must be non-negative")
        if self.effort_distance_min <= 0:
            raise ValueError("effort_distance_min must be positive")
        if not 0 < self.action_saturation_threshold <= 1:
            raise ValueError("action_saturation_threshold must be in (0, 1]")

    def reset(self, *, initial_x: float, initial_y: float) -> None:
        self.initial_x = float(initial_x)
        self.initial_y = float(initial_y)
        self.condition_objective_return = 0.0
        self.base_proxy_return = 0.0
        self.reward_shaping_sum = 0.0
        self.reward_forward_shaping_sum = 0.0
        self.reward_lateral_shaping_sum = 0.0
        self.reward_effort_shaping_sum = 0.0
        self.reward_stability_shaping_sum = 0.0
        self.reward_forward_sum = 0.0
        self.reward_ctrl_sum = 0.0
        self.reward_contact_sum = 0.0
        self.reward_survive_sum = 0.0
        self.cumulative_squared_action = 0.0
        self.action_component_count = 0
        self.saturated_action_component_count = 0
        self.episode_length = 0
        self.terminated = False
        self.truncated = False
        self.termination_category = "none"
        self.latest_x = float(initial_x)
        self.latest_y = float(initial_y)
        self.cumulative_lateral_path = 0.0
        self.lateral_abs_history.clear()
        self.torso_tilt_history.clear()

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
        self.reward_stability_shaping_sum += float(
            info.get("reward_stability_shaping", 0.0)
        )
        self.reward_forward_sum += float(info.get("reward_forward", 0.0))
        self.reward_ctrl_sum += float(info.get("reward_ctrl", 0.0))
        self.reward_contact_sum += float(info.get("reward_contact", 0.0))
        self.reward_survive_sum += float(info.get("reward_survive", 0.0))
        self.cumulative_squared_action += float(np.sum(np.square(action_array)))
        self.action_component_count += int(action_array.size)
        self.saturated_action_component_count += int(
            np.count_nonzero(np.abs(action_array) >= self.action_saturation_threshold)
        )
        self.episode_length += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        previous_y = self.latest_y
        self.latest_x = float(info.get("x_position", self.latest_x))
        self.latest_y = float(info.get("y_position", self.latest_y))
        self.cumulative_lateral_path += abs(self.latest_y - previous_y)
        self.lateral_abs_history.append(abs(self.latest_y - self.initial_y))
        if math.isfinite(torso_tilt):
            self.torso_tilt_history.append(float(torso_tilt))
        if terminated:
            self.termination_category = classify_termination(
                terminated=True,
                torso_height=float(torso_height),
                state_is_finite=bool(state_is_finite),
                healthy_z_range=self.healthy_z_range,
            )

    def summary(self) -> dict[str, float | int | bool | str]:
        net_forward_progress = self.latest_x - self.initial_x
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
            "reward_stability_shaping_sum": float(self.reward_stability_shaping_sum),
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
            "cumulative_squared_action": float(self.cumulative_squared_action),
            "mean_squared_action_per_step": safe_divide(
                self.cumulative_squared_action,
                self.episode_length,
            ),
            "action_saturation_rate": action_saturation_rate,
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
            "torso_tilt_mean": tilt_mean,
            "torso_tilt_std": tilt_std,
            "torso_tilt_rms": tilt_rms,
            "torso_tilt_p95": tilt_p95,
            "torso_tilt_max": tilt_max,
            "episode_length": int(self.episode_length),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }
