"""Ant-v5 construction, reward decomposition and diagnostic logging."""

from __future__ import annotations

import csv
import gzip
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
    "torso_tilt_rad",
    "squared_action_step",
    "action_saturation_fraction_step",
    "condition_objective_reward_step",
    "base_proxy_reward_step",
    "common_rescored_reward_step",
    "shaping_reward_step",
    "reward_forward_step",
    "reward_ctrl_step",
    "reward_contact_step",
    "reward_survive_step",
    "reward_effort_shaping_step",
    "reward_stability_shaping_step",
    "terminated",
    "truncated",
    "termination_category",
]


class ProxyGapAntWrapper(gym.Wrapper):
    """Record condition objectives and external locomotion diagnostics."""

    def __init__(
        self,
        env: gym.Env,
        condition_id: str,
        ctrl_cost_weight: float,
        forward_progress_shaping_weight: float = 0.0,
        lateral_drift_shaping_weight: float = 0.0,
        effort_shaping_weight: float = 0.0,
        effort_shaping_scale: float = 1.0,
        stability_shaping_weight: float = 0.0,
        stability_shaping_scale: float = 1.0,
        common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
        effort_distance_min: float = EPSILON,
        action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
        step_log_path: str | Path | None = None,
    ) -> None:
        super().__init__(env)
        if effort_shaping_scale <= 0 or stability_shaping_scale <= 0:
            raise ValueError("shaping scales must be positive")
        self.condition_id = condition_id
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.forward_progress_shaping_weight = float(forward_progress_shaping_weight)
        self.lateral_drift_shaping_weight = float(lateral_drift_shaping_weight)
        self.effort_shaping_weight = float(effort_shaping_weight)
        self.effort_shaping_scale = float(effort_shaping_scale)
        self.stability_shaping_weight = float(stability_shaping_weight)
        self.stability_shaping_scale = float(stability_shaping_scale)
        self.common_rescore_ctrl_cost_weight = float(common_rescore_ctrl_cost_weight)
        self.effort_distance_min = float(effort_distance_min)
        self.action_saturation_threshold = float(action_saturation_threshold)
        healthy_range = tuple(float(value) for value in self.unwrapped._healthy_z_range)
        self.metrics = EpisodeMetrics(
            condition_ctrl_cost_weight=self.ctrl_cost_weight,
            common_rescore_ctrl_cost_weight=self.common_rescore_ctrl_cost_weight,
            effort_distance_min=self.effort_distance_min,
            action_saturation_threshold=self.action_saturation_threshold,
            healthy_z_range=(healthy_range[0], healthy_range[1]),
        )
        self._episode_index = 0
        self._step_index = 0
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
        x_position, y_position = self._root_xy(info)
        self.metrics.reset(initial_x=x_position, initial_y=y_position)
        self._episode_index += 1
        self._step_index = 0
        info = dict(info)
        info.update(self._prefixed_summary())
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float64)
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        x_position, y_position = self._root_xy(info)
        lateral_offset = abs(y_position - self.metrics.initial_y)
        torso_tilt = self._torso_tilt()
        torso_height, state_is_finite = self._health_state()
        squared_action = float(np.sum(np.square(action_array)))
        saturated_fraction = float(
            np.mean(np.abs(action_array) >= self.action_saturation_threshold)
        )

        forward_shaping_reward = self.forward_progress_shaping_weight * float(
            info.get("reward_forward", 0.0)
        )
        lateral_shaping_reward = -self.lateral_drift_shaping_weight * lateral_offset
        effort_shaping_reward = -self.effort_shaping_weight * float(
            np.tanh(squared_action / self.effort_shaping_scale)
        )
        stability_shaping_reward = -self.stability_shaping_weight * float(
            np.tanh(torso_tilt / self.stability_shaping_scale)
        )
        shaping_reward = (
            forward_shaping_reward
            + lateral_shaping_reward
            + effort_shaping_reward
            + stability_shaping_reward
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
        info["reward_lateral_shaping"] = float(lateral_shaping_reward)
        info["reward_effort_shaping"] = float(effort_shaping_reward)
        info["reward_stability_shaping"] = float(stability_shaping_reward)
        info["reward_common_rescored"] = common_rescored_reward
        self.metrics.update(
            action=action_array,
            reward=observed_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            torso_tilt=torso_tilt,
            torso_height=torso_height,
            state_is_finite=state_is_finite,
        )
        summary = self.metrics.summary()
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
        info["proxygap_effort_shaping_weight"] = self.effort_shaping_weight
        info["proxygap_stability_shaping_weight"] = self.stability_shaping_weight
        info["proxygap_lateral_offset_step"] = lateral_offset
        info["proxygap_torso_height_step"] = torso_height
        info["proxygap_state_is_finite_step"] = state_is_finite
        info["proxygap_squared_action_step"] = squared_action
        info["proxygap_action_saturation_fraction_step"] = saturated_fraction
        self._step_index += 1
        self._write_step_record(
            x_position=x_position,
            y_position=y_position,
            torso_height=torso_height,
            state_is_finite=state_is_finite,
            lateral_offset=lateral_offset,
            torso_tilt=torso_tilt,
            squared_action=squared_action,
            saturated_fraction=saturated_fraction,
            observed_reward=observed_reward,
            base_reward=float(base_reward),
            common_rescored_reward=common_rescored_reward,
            shaping_reward=shaping_reward,
            info=info,
            terminated=terminated,
            truncated=truncated,
            termination_category=str(summary["termination_category"]),
        )
        return observation, observed_reward, terminated, truncated, info

    def episode_summary(self) -> dict[str, Any]:
        summary = self.metrics.summary()
        summary.update(
            {
                "condition_id": self.condition_id,
                "ctrl_cost_weight": self.ctrl_cost_weight,
                "forward_progress_shaping_weight": self.forward_progress_shaping_weight,
                "lateral_drift_shaping_weight": self.lateral_drift_shaping_weight,
                "effort_shaping_weight": self.effort_shaping_weight,
                "effort_shaping_scale": self.effort_shaping_scale,
                "stability_shaping_weight": self.stability_shaping_weight,
                "stability_shaping_scale": self.stability_shaping_scale,
            }
        )
        return summary

    def close(self) -> None:
        if self._step_handle is not None and not self._step_handle.closed:
            self._step_handle.flush()
            self._step_handle.close()
        super().close()

    def _prefixed_summary(self) -> dict[str, Any]:
        return {f"proxygap_{key}": value for key, value in self.metrics.summary().items()}

    def _root_xy(self, info: dict[str, Any]) -> tuple[float, float]:
        if "x_position" in info and "y_position" in info:
            return float(info["x_position"]), float(info["y_position"])
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        return float(qpos[0]), float(qpos[1])

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
            "torso_tilt_rad": values["torso_tilt"],
            "squared_action_step": values["squared_action"],
            "action_saturation_fraction_step": values["saturated_fraction"],
            "condition_objective_reward_step": values["observed_reward"],
            "base_proxy_reward_step": values["base_reward"],
            "common_rescored_reward_step": values["common_rescored_reward"],
            "shaping_reward_step": values["shaping_reward"],
            "reward_forward_step": info.get("reward_forward", 0.0),
            "reward_ctrl_step": info.get("reward_ctrl", 0.0),
            "reward_contact_step": info.get("reward_contact", 0.0),
            "reward_survive_step": info.get("reward_survive", 0.0),
            "reward_effort_shaping_step": info.get("reward_effort_shaping", 0.0),
            "reward_stability_shaping_step": info.get("reward_stability_shaping", 0.0),
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
    max_episode_steps: int | None = None,
    forward_progress_shaping_weight: float = 0.0,
    lateral_drift_shaping_weight: float = 0.0,
    effort_shaping_weight: float = 0.0,
    effort_shaping_scale: float = 1.0,
    stability_shaping_weight: float = 0.0,
    stability_shaping_scale: float = 1.0,
    common_rescore_ctrl_cost_weight: float = DEFAULT_COMMON_RESCORE_CTRL_WEIGHT,
    effort_distance_min: float = EPSILON,
    action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
    step_log_path: str | Path | None = None,
) -> ProxyGapAntWrapper:
    """Create Ant-v5 with separately logged objective and diagnostic terms."""
    kwargs: dict[str, Any] = {
        "ctrl_cost_weight": float(ctrl_cost_weight),
        "render_mode": render_mode,
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = gym.make("Ant-v5", **kwargs)
    wrapped = ProxyGapAntWrapper(
        env=env,
        condition_id=condition_id,
        ctrl_cost_weight=float(ctrl_cost_weight),
        forward_progress_shaping_weight=float(forward_progress_shaping_weight),
        lateral_drift_shaping_weight=float(lateral_drift_shaping_weight),
        effort_shaping_weight=float(effort_shaping_weight),
        effort_shaping_scale=float(effort_shaping_scale),
        stability_shaping_weight=float(stability_shaping_weight),
        stability_shaping_scale=float(stability_shaping_scale),
        common_rescore_ctrl_cost_weight=float(common_rescore_ctrl_cost_weight),
        effort_distance_min=float(effort_distance_min),
        action_saturation_threshold=float(action_saturation_threshold),
        step_log_path=step_log_path,
    )
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped
