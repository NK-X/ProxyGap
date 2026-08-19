"""Deterministic paired command schedule for turn-balance continuation.

This module deliberately changes only the externally supplied local curve
command.  The wrapped locomotion reward, 122-value locomotion/contact/command
observation prefix, 13-value local-terrain preview, action space, robot,
contact contract, and energy measurements remain owned by the existing
wrappers.

The schedule is rank based rather than randomly sampled so that a completed
16-episode run is exactly auditable.  In the balanced condition, each turning
worker sees eight left and eight right episodes at one fixed magnitude; the
paired worker starts in the opposite phase.  The matched control uses this
same wrapper with zero curvature on every worker and episode.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fixed_goal_terrain import FixedGoalTerrainWrapper
from .planar_transition import quaternion_yaw_angle, wrapped_angle_difference


CONTROL_CONDITION_ID = "C0_STRAIGHT_CONTINUE"
BALANCED_CONDITION_ID = "C1_BALANCED_TURN"
CONDITION_IDS = (CONTROL_CONDITION_ID, BALANCED_CONDITION_ID)
PARALLEL_ENVIRONMENTS = 8
EPISODES_PER_WORKER = 16
EPISODE_STEPS = 512

_WORKER_MAGNITUDE: tuple[float, ...] = (
    0.0,
    0.0,
    0.10,
    0.10,
    0.20,
    0.20,
    0.35,
    0.35,
)


def scheduled_curvature_per_m(
    condition_id: str,
    worker_rank: int,
    episode_index: int,
) -> float:
    """Return the predeclared fixed curvature for one worker episode."""

    if condition_id not in CONDITION_IDS:
        raise ValueError(f"unsupported paired turn condition: {condition_id}")
    rank = int(worker_rank)
    episode = int(episode_index)
    if rank != worker_rank or not 0 <= rank < PARALLEL_ENVIRONMENTS:
        raise ValueError("worker_rank must be an integer in [0, 7]")
    if episode != episode_index or episode < 0:
        raise ValueError("episode_index must be a non-negative integer")
    if condition_id == CONTROL_CONDITION_ID:
        return 0.0
    magnitude = _WORKER_MAGNITUDE[rank]
    if magnitude == 0.0:
        return 0.0
    # The two workers assigned to one magnitude have opposite phase in every
    # episode; each individual worker flips sign on every reset.
    positive = (episode + rank) % 2 == 0
    return float(magnitude if positive else -magnitude)


def curvature_label(curvature_per_m: float) -> str:
    """Use stable string labels instead of floating-point dictionary keys."""

    value = float(curvature_per_m)
    if not np.isfinite(value):
        raise ValueError("curvature must be finite")
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "straight_000"
    side = "left" if value > 0.0 else "right"
    magnitude = int(round(abs(value) * 100.0))
    if not math.isclose(abs(value), magnitude / 100.0, abs_tol=1e-12):
        raise ValueError("curvature is outside the predeclared two-decimal grid")
    return f"{side}_{magnitude:03d}"


def expected_worker_exposure_steps(
    condition_id: str,
    worker_rank: int,
    *,
    episodes_per_worker: int = EPISODES_PER_WORKER,
    episode_steps: int = EPISODE_STEPS,
) -> dict[str, int]:
    """Return exact executed-step counts required from one completed worker."""

    if int(episodes_per_worker) != episodes_per_worker or episodes_per_worker <= 0:
        raise ValueError("episodes_per_worker must be a positive integer")
    if int(episode_steps) != episode_steps or episode_steps <= 0:
        raise ValueError("episode_steps must be a positive integer")
    counts: dict[str, int] = {}
    for episode in range(int(episodes_per_worker)):
        label = curvature_label(
            scheduled_curvature_per_m(condition_id, worker_rank, episode)
        )
        counts[label] = counts.get(label, 0) + int(episode_steps)
    return dict(sorted(counts.items()))


def expected_condition_exposure_steps(
    condition_id: str,
    *,
    parallel_environments: int = PARALLEL_ENVIRONMENTS,
    episodes_per_worker: int = EPISODES_PER_WORKER,
    episode_steps: int = EPISODE_STEPS,
) -> dict[str, int]:
    """Return exact aggregate step counts for one completed condition."""

    if int(parallel_environments) != PARALLEL_ENVIRONMENTS:
        raise ValueError("the frozen paired schedule requires exactly eight workers")
    total: dict[str, int] = {}
    for rank in range(PARALLEL_ENVIRONMENTS):
        for label, steps in expected_worker_exposure_steps(
            condition_id,
            rank,
            episodes_per_worker=episodes_per_worker,
            episode_steps=episode_steps,
        ).items():
            total[label] = total.get(label, 0) + steps
    return dict(sorted(total.items()))


class PairedTurnBalanceTerrainWrapper(FixedGoalTerrainWrapper):
    """Supply one fixed, auditable local turn command per training episode.

    ``FixedGoalTerrainWrapper`` is subclassed only to retain its already-tested
    terrain-relative health audit and 13D local preview implementation.  Its
    goal-seeking command calculation is replaced here.  Global position is
    used to sample the local terrain, exactly as before, but no global map
    coordinate or worker/episode identifier is appended to the policy input.
    """

    def __init__(
        self,
        env: Any,
        *,
        condition_id: str,
        worker_rank: int,
        expected_episode_steps: int = EPISODE_STEPS,
        fixed_speed_m_per_s: float = 0.55,
        fail_closed_training_contract: bool = True,
        **kwargs: Any,
    ) -> None:
        if condition_id not in CONDITION_IDS:
            raise ValueError(f"unsupported paired turn condition: {condition_id}")
        if int(worker_rank) != worker_rank or not 0 <= int(worker_rank) < 8:
            raise ValueError("worker_rank must be an integer in [0, 7]")
        if int(expected_episode_steps) != expected_episode_steps or expected_episode_steps <= 0:
            raise ValueError("expected_episode_steps must be a positive integer")
        speed = float(fixed_speed_m_per_s)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("fixed_speed_m_per_s must be positive and finite")

        self.turn_balance_condition_id = str(condition_id)
        self.turn_balance_worker_rank = int(worker_rank)
        self.turn_balance_expected_episode_steps = int(expected_episode_steps)
        self.turn_balance_fixed_speed_m_per_s = speed
        self.turn_balance_fail_closed = bool(fail_closed_training_contract)
        self._turn_balance_episode_index = -1
        self._turn_balance_episode_steps = 0
        self._turn_balance_episode_completed = True
        self._turn_balance_completed_episodes = 0
        self._turn_balance_heading_origin: float | None = None
        self._turn_balance_active_curvature = 0.0
        self._turn_balance_exposure_steps: dict[str, int] = {}
        self._turn_balance_first_reset_seed: int | None = None
        super().__init__(env, **kwargs)
        if not self.augment_local_terrain_observation:
            raise ValueError("paired turn continuation requires the existing 13D preview")
        if tuple(self.observation_space.shape or ()) != (135,):
            raise ValueError("paired turn continuation requires the frozen 135D observation")
        if tuple(self.action_space.shape or ()) != (8,):
            raise ValueError("paired turn continuation requires the frozen 8D action")

    @property
    def active_curvature_per_m(self) -> float:
        return float(self._turn_balance_active_curvature)

    @property
    def active_yaw_rate_rad_per_s(self) -> float:
        return float(
            self.turn_balance_fixed_speed_m_per_s
            * self._turn_balance_active_curvature
        )

    def _scheduled_target_heading(self, command_step_index: int) -> float:
        if self._turn_balance_heading_origin is None:
            raise RuntimeError("paired turn heading origin is not initialised")
        heading = self._turn_balance_heading_origin + (
            self.active_yaw_rate_rad_per_s
            * int(command_step_index)
            * float(self.unwrapped.dt)
        )
        return wrapped_angle_difference(float(heading), 0.0)

    def _initial_target_heading(self) -> float:
        position = self._position()
        vector = self.goal_xy - position
        if float(np.linalg.norm(vector)) > self.arrival_radius:
            return float(math.atan2(float(vector[1]), float(vector[0])))
        return quaternion_yaw_angle(np.asarray(self.unwrapped.data.qpos[3:7]))

    def _set_scheduled_command(
        self,
        observation: np.ndarray,
        *,
        command_step_index: int,
    ) -> np.ndarray:
        position = self._position()
        target_heading = self._scheduled_target_heading(command_step_index)
        command_observation = self.env.set_external_curve_command(
            observation,
            target_heading=target_heading,
            yaw_rate=self.active_yaw_rate_rad_per_s,
            speed=self.turn_balance_fixed_speed_m_per_s,
            lateral_speed=0.0,
        )
        if self.terrain_frame_shaping_enabled:
            # This protocol freezes terrain-frame reward shaping off.  Keeping
            # the branch correct prevents silent semantic drift if reused.
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

    def _command_observation(self, observation: np.ndarray) -> np.ndarray:
        if self._turn_balance_heading_origin is None:
            self._turn_balance_heading_origin = self._initial_target_heading()
        # FixedGoalTerrainWrapper increments _task_steps before requesting the
        # next observation, so this index is exactly the command that the next
        # action/reward transition will consume.
        return self._set_scheduled_command(
            observation,
            command_step_index=int(self._task_steps),
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self._turn_balance_episode_index >= 0 and not self._turn_balance_episode_completed:
            raise RuntimeError(
                "paired turn episode reset before the exact fixed horizon completed"
            )
        if self._turn_balance_episode_index < 0:
            requested_seed = kwargs.get("seed")
            self._turn_balance_first_reset_seed = (
                None if requested_seed is None else int(requested_seed)
            )
        self._turn_balance_episode_index += 1
        self._turn_balance_episode_steps = 0
        self._turn_balance_episode_completed = False
        self._turn_balance_heading_origin = None
        self._turn_balance_active_curvature = scheduled_curvature_per_m(
            self.turn_balance_condition_id,
            self.turn_balance_worker_rank,
            self._turn_balance_episode_index,
        )
        observation, info = super().reset(**kwargs)
        info = dict(info)
        info.update(self._turn_balance_live_info())
        return observation, info

    def _terminal_command_observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation)
        if values.shape != (135,):
            raise RuntimeError("terminal observation changed from the frozen 135D contract")
        # The parent terminal branch appends a goal-frame preview but does not
        # install the next external command.  Replace both here so TimeLimit's
        # terminal_observation is a self-consistent bootstrap state.
        return self._set_scheduled_command(
            values[:-13],
            command_step_index=int(self._task_steps),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._turn_balance_episode_completed:
            raise RuntimeError("step called after a completed paired turn episode")
        if self._turn_balance_episode_steps >= self.turn_balance_expected_episode_steps:
            raise RuntimeError("paired turn episode exceeded its exact fixed horizon")
        expected_heading = self._scheduled_target_heading(
            self._turn_balance_episode_steps
        )
        expected_yaw_rate = self.active_yaw_rate_rad_per_s
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)

        observed_yaw_rate = float(info["proxygap_curve_yaw_rate_command_step"])
        observed_heading = float(info["proxygap_curve_tangent_heading_step"])
        if not math.isclose(observed_yaw_rate, expected_yaw_rate, abs_tol=1e-12):
            raise RuntimeError("reward transition consumed the wrong yaw-rate command")
        if not math.isclose(
            wrapped_angle_difference(observed_heading, expected_heading),
            0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("reward transition consumed the wrong heading command")

        self._turn_balance_episode_steps += 1
        label = curvature_label(self.active_curvature_per_m)
        self._turn_balance_exposure_steps[label] = (
            self._turn_balance_exposure_steps.get(label, 0) + 1
        )

        values = np.asarray(observation)
        state_finite = bool(
            np.all(np.isfinite(values))
            and np.isfinite(float(reward))
            and np.all(np.isfinite(np.asarray(self.unwrapped.data.qpos)))
            and np.all(np.isfinite(np.asarray(self.unwrapped.data.qvel)))
        )
        if self.turn_balance_fail_closed and not state_finite:
            raise RuntimeError("non-finite paired turn training transition")

        expected_end = (
            self._turn_balance_episode_steps
            == self.turn_balance_expected_episode_steps
        )
        if self.turn_balance_fail_closed:
            if terminated:
                raise RuntimeError("paired turn training episode terminated early")
            if truncated != expected_end:
                raise RuntimeError(
                    "TimeLimit truncation did not occur at the exact fixed horizon"
                )
        if terminated or truncated:
            observation = self._terminal_command_observation(observation)
        if expected_end:
            self._turn_balance_episode_completed = True
            self._turn_balance_completed_episodes += 1

        info.update(self._turn_balance_live_info())
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _turn_balance_live_info(self) -> dict[str, Any]:
        return {
            "proxygap_turn_balance_condition_id": self.turn_balance_condition_id,
            "proxygap_turn_balance_worker_rank": self.turn_balance_worker_rank,
            "proxygap_turn_balance_episode_index": self._turn_balance_episode_index,
            "proxygap_turn_balance_episode_steps": self._turn_balance_episode_steps,
            "proxygap_turn_balance_curvature_per_m": self.active_curvature_per_m,
            "proxygap_turn_balance_yaw_rate_rad_per_s": self.active_yaw_rate_rad_per_s,
        }

    def turn_balance_state(self) -> dict[str, Any]:
        """Return integer exposure evidence for the parent training runner."""

        return {
            "condition_id": self.turn_balance_condition_id,
            "worker_rank": self.turn_balance_worker_rank,
            "active_episode_index": self._turn_balance_episode_index,
            "active_episode_steps": self._turn_balance_episode_steps,
            "active_episode_completed": self._turn_balance_episode_completed,
            "completed_episode_count": self._turn_balance_completed_episodes,
            "first_reset_seed": self._turn_balance_first_reset_seed,
            "executed_exposure_steps": dict(
                sorted(self._turn_balance_exposure_steps.items())
            ),
            "expected_exposure_steps_after_full_budget": (
                expected_worker_exposure_steps(
                    self.turn_balance_condition_id,
                    self.turn_balance_worker_rank,
                )
            ),
            "observation_dimension": int(self.observation_space.shape[0]),
            "action_dimension": int(self.action_space.shape[0]),
            "global_coordinates_appended_to_policy_observation": False,
            "energy_used_as_reward": False,
        }
