from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_fixed_standard_pair0_turn_balance_continuation as runner
from proxygap.paired_turn_balance import (
    BALANCED_CONDITION_ID,
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    PairedTurnBalanceTerrainWrapper,
    expected_condition_exposure_steps,
    expected_worker_exposure_steps,
    scheduled_curvature_per_m,
)
from proxygap.fixed_goal_terrain import FixedGoalTerrainWrapper


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v1_20260819.json"
)


class DeterministicExternalCommandEnv(gym.Env[np.ndarray, np.ndarray]):
    """Small deterministic stand-in for command/reward timing tests."""

    metadata: dict[str, Any] = {}

    def __init__(self, horizon: int = 512) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(122,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )
        self.dt = 0.05
        self.horizon = int(horizon)
        self.data = SimpleNamespace(
            qpos=np.asarray(
                [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0] + [0.0] * 8,
                dtype=np.float64,
            ),
            qvel=np.zeros(14, dtype=np.float64),
        )
        self._steps = 0
        self._command = {
            "target_heading": 0.0,
            "yaw_rate": 0.0,
            "speed": 0.55,
        }
        self.applied_commands: list[dict[str, float]] = []

    def _observation(self) -> np.ndarray:
        values = np.zeros(122, dtype=np.float32)
        heading = float(self._command["target_heading"])
        speed = float(self._command["speed"])
        values[113] = speed * math.cos(heading)
        values[114] = speed * math.sin(heading)
        values[115] = float(self._command["yaw_rate"])
        values[116] = math.sin(-heading)
        values[117] = math.cos(-heading)
        values[118:122] = 1.0
        return values

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._steps = 0
        self._command = {
            "target_heading": 0.0,
            "yaw_rate": 0.0,
            "speed": 0.55,
        }
        return self._observation(), {}

    def set_external_curve_command(
        self,
        observation: np.ndarray,
        *,
        target_heading: float,
        yaw_rate: float,
        speed: float,
        lateral_speed: float = 0.0,
    ) -> np.ndarray:
        del observation
        assert lateral_speed == 0.0
        self._command = {
            "target_heading": float(target_heading),
            "yaw_rate": float(yaw_rate),
            "speed": float(speed),
        }
        return self._observation()

    def step(self, action: np.ndarray):
        del action
        applied = dict(self._command)
        self.applied_commands.append(applied)
        self._steps += 1
        truncated = self._steps == self.horizon
        reward = applied["target_heading"] + 10.0 * applied["yaw_rate"]
        info = {
            "proxygap_curve_yaw_rate_command_step": applied["yaw_rate"],
            "proxygap_curve_tangent_heading_step": applied["target_heading"],
            "proxygap_foot_contact_mask_step": np.ones(4, dtype=np.int8),
            "proxygap_foot_contact_tangential_speeds_m_per_s_step": np.zeros(4),
        }
        return self._observation(), reward, False, truncated, info

    def episode_summary(self) -> dict[str, Any]:
        zeros = [0.0] * 8
        return {
            "termination_category": "time_limit" if self._steps == self.horizon else "running",
            "fall": False,
            "inner_absolute_z_fall": False,
            "cumulative_squared_action": 0.0,
            "actuator_abs_torque_time_integral_n_m_s_by_actuator": zeros,
            "actuator_positive_mechanical_work_j_by_actuator": zeros,
            "actuator_abs_mechanical_work_j_by_actuator": zeros,
        }


def make_dummy_wrapper(
    tmp_path: Path,
    *,
    condition_id: str,
    worker_rank: int,
    inner_horizon: int = 512,
) -> PairedTurnBalanceTerrainWrapper:
    heights = np.zeros((5, 5), dtype=np.float64)
    path = tmp_path / f"flat_{condition_id}_{worker_rank}.npy"
    np.save(path, heights, allow_pickle=False)
    return PairedTurnBalanceTerrainWrapper(
        DeterministicExternalCommandEnv(horizon=inner_horizon),
        condition_id=condition_id,
        worker_rank=worker_rank,
        heights_path=path,
        expected_height_sha256=runner.sha256(path),
        map_half_extent_m=20.0,
        start_xy_m=(0.0, 0.0),
        goal_xy_m=(6.0, 0.0),
        spawn_fraction=0.0,
        cruise_speed_m_per_s=0.55,
        maximum_abs_curvature_per_m=0.35,
        yaw_gain_per_second=1.5,
        slow_radius_m=5.0,
        arrival_radius_m=1.5,
        hold_radius_m=2.0,
        hold_seconds=2.0,
        hold_speed_m_per_s=0.05,
        terminate_on_success=False,
        terrain_relative_healthy_clearance_m=(0.18, 1.4),
        maximum_healthy_tilt_degrees=80.0,
        unhealthy_grace_steps=5,
        slip_speed_threshold_m_per_s=0.2,
        augment_local_terrain_observation=True,
        terrain_frame_shaping_enabled=False,
        terrain_preview_longitudinal_m=(0.5, 1.0, 1.5),
        terrain_preview_lateral_m=(-0.4, 0.0, 0.4),
        local_terrain_height_bound_m=1.0,
    )


def test_schedule_is_exactly_counterbalanced_and_digest_is_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert runner.canonical_sha256(runner.schedule_rows()) == (
        config["command_schedule"][
            "expected_worker_episode_and_exposure_table_canonical_sha256"
        ]
    )
    assert expected_condition_exposure_steps(CONTROL_CONDITION_ID) == {
        "straight_000": 65536
    }
    assert expected_condition_exposure_steps(BALANCED_CONDITION_ID) == {
        "left_010": 8192,
        "left_020": 8192,
        "left_035": 8192,
        "right_010": 8192,
        "right_020": 8192,
        "right_035": 8192,
        "straight_000": 16384,
    }
    for first, second in ((2, 3), (4, 5), (6, 7)):
        assert [
            scheduled_curvature_per_m(BALANCED_CONDITION_ID, first, episode)
            for episode in range(16)
        ] == [
            -scheduled_curvature_per_m(BALANCED_CONDITION_ID, second, episode)
            for episode in range(16)
        ]


def test_first_observation_and_reward_consume_the_preinstalled_command(
    tmp_path: Path,
) -> None:
    env = make_dummy_wrapper(
        tmp_path,
        condition_id=BALANCED_CONDITION_ID,
        worker_rank=2,
    )
    observation, info = env.reset(seed=100)
    assert observation.shape == (135,)
    assert observation.dtype == np.float32
    assert observation[115] == np.float32(0.055)
    assert info["proxygap_turn_balance_curvature_per_m"] == 0.1

    next_observation, reward, terminated, truncated, step_info = env.step(
        np.zeros(8, dtype=np.float32)
    )
    assert not terminated and not truncated
    assert reward == 0.55
    assert math.isclose(
        step_info["proxygap_curve_yaw_rate_command_step"], 0.055, abs_tol=1e-12
    )
    assert step_info["proxygap_curve_tangent_heading_step"] == 0.0
    expected_next_heading = 0.055 * 0.05
    assert np.isclose(next_observation[116], math.sin(-expected_next_heading))
    assert math.isclose(
        env.env.applied_commands[0]["yaw_rate"], 0.055, abs_tol=1e-12
    )
    env.close()


def test_c0_wrapper_preserves_fixed_goal_initial_and_first_step_semantics(
    tmp_path: Path,
) -> None:
    heights = np.zeros((5, 5), dtype=np.float64)
    path = tmp_path / "c0_parity_flat.npy"
    np.save(path, heights, allow_pickle=False)
    common = {
        "heights_path": path,
        "expected_height_sha256": runner.sha256(path),
        "map_half_extent_m": 20.0,
        "start_xy_m": (0.0, 0.0),
        "goal_xy_m": (6.0, 0.0),
        "spawn_fraction": 0.0,
        "cruise_speed_m_per_s": 0.55,
        "maximum_abs_curvature_per_m": 0.35,
        "yaw_gain_per_second": 1.5,
        "slow_radius_m": 5.0,
        "arrival_radius_m": 1.5,
        "hold_radius_m": 2.0,
        "hold_seconds": 2.0,
        "hold_speed_m_per_s": 0.05,
        "terminate_on_success": False,
        "terrain_relative_healthy_clearance_m": (0.18, 1.4),
        "maximum_healthy_tilt_degrees": 80.0,
        "unhealthy_grace_steps": 5,
        "slip_speed_threshold_m_per_s": 0.2,
        "augment_local_terrain_observation": True,
        "terrain_frame_shaping_enabled": False,
        "terrain_preview_longitudinal_m": (0.5, 1.0, 1.5),
        "terrain_preview_lateral_m": (-0.4, 0.0, 0.4),
        "local_terrain_height_bound_m": 1.0,
    }
    baseline = FixedGoalTerrainWrapper(DeterministicExternalCommandEnv(), **common)
    control = PairedTurnBalanceTerrainWrapper(
        DeterministicExternalCommandEnv(),
        condition_id=CONTROL_CONDITION_ID,
        worker_rank=0,
        **common,
    )
    try:
        baseline_observation, _ = baseline.reset(seed=105)
        control_observation, _ = control.reset(seed=105)
        assert np.array_equal(baseline_observation, control_observation)
        action = np.zeros(8, dtype=np.float32)
        baseline_next, baseline_reward, baseline_term, baseline_trunc, _ = (
            baseline.step(action)
        )
        control_next, control_reward, control_term, control_trunc, _ = control.step(
            action
        )
        assert np.array_equal(baseline_next, control_next)
        assert baseline_reward == control_reward
        assert baseline_term == control_term is False
        assert baseline_trunc == control_trunc is False
    finally:
        baseline.close()
        control.close()


def test_terminal_bootstrap_observation_is_advanced_one_command_step(
    tmp_path: Path,
) -> None:
    env = make_dummy_wrapper(
        tmp_path,
        condition_id=BALANCED_CONDITION_ID,
        worker_rank=2,
    )
    env.reset(seed=101)
    observation = None
    for step in range(512):
        observation, _, terminated, truncated, _ = env.step(
            np.zeros(8, dtype=np.float32)
        )
        assert not terminated
        assert truncated is (step == 511)
    assert observation is not None
    expected_bootstrap_heading = 0.055 * 512 * 0.05
    assert np.isclose(observation[116], math.sin(-expected_bootstrap_heading))
    assert math.isclose(
        env.env.applied_commands[-1]["target_heading"],
        0.055 * 511 * 0.05,
        abs_tol=1e-12,
    )
    state = env.turn_balance_state()
    assert state["completed_episode_count"] == 1
    assert state["first_reset_seed"] == 101
    assert state["executed_exposure_steps"] == {"left_010": 512}
    env.close()


def test_vecenv_auto_reset_counts_only_executed_transitions_and_flips_phase(
    tmp_path: Path,
) -> None:
    vector = DummyVecEnv(
        [
            lambda: make_dummy_wrapper(
                tmp_path,
                condition_id=BALANCED_CONDITION_ID,
                worker_rank=2,
            )
        ]
    )
    try:
        assert vector.seed(63806) == [63806]
        observation = vector.reset()
        assert observation[0, 115] == np.float32(0.055)
        first_terminal = None
        for transition in range(16 * 512):
            observation, _, done, infos = vector.step(
                np.zeros((1, 8), dtype=np.float32)
            )
            if transition == 511:
                assert bool(done[0])
                first_terminal = infos[0]["terminal_observation"].copy()
                assert observation[0, 115] == np.float32(-0.055)
        assert first_terminal is not None
        assert np.isclose(
            first_terminal[116], math.sin(-(0.055 * 512 * 0.05))
        )
        state = vector.env_method("turn_balance_state")[0]
        assert state["completed_episode_count"] == 16
        assert state["active_episode_index"] == 16
        assert state["active_episode_steps"] == 0
        assert state["first_reset_seed"] == 63806
        assert state["executed_exposure_steps"] == expected_worker_exposure_steps(
            BALANCED_CONDITION_ID, 2
        )
    finally:
        vector.close()


def test_early_reset_and_early_termination_fail_closed(tmp_path: Path) -> None:
    env = make_dummy_wrapper(
        tmp_path,
        condition_id=BALANCED_CONDITION_ID,
        worker_rank=2,
    )
    env.reset(seed=102)
    try:
        env.reset(seed=103)
    except RuntimeError as error:
        assert "before the exact fixed horizon" in str(error)
    else:
        raise AssertionError("early reset did not fail closed")
    env.close()

    early = make_dummy_wrapper(
        tmp_path,
        condition_id=BALANCED_CONDITION_ID,
        worker_rank=2,
        inner_horizon=2,
    )
    early.reset(seed=104)
    early.step(np.zeros(8, dtype=np.float32))
    try:
        early.step(np.zeros(8, dtype=np.float32))
    except RuntimeError as error:
        assert "TimeLimit truncation" in str(error)
    else:
        raise AssertionError("early truncation did not fail closed")
    early.close()


def test_real_checkpoint_loader_preserves_policy_optimizer_action_and_counter() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / config["source"]["checkpoint"]
    source = PPO.load(source_path, device="cpu")
    result = runner.loader_preflight(config, source_path, source)
    assert all(result["checks"].values())
    assert all(result["setup_learn_without_rollout_or_gradient_checks"].values())
    assert result["optimizer_state_entries"] == 13
    assert result["torch_num_threads"] == 2
    assert result["effective_first_reset_seeds"] == list(range(63806, 63814))
    assert result["learn_called"] is False
    assert result["gradient_update_performed"] is False


def test_two_branch_loads_are_independent_and_identical_to_one_source() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / config["source"]["checkpoint"]
    source = PPO.load(source_path, device="cpu")

    def vector() -> DummyVecEnv:
        return DummyVecEnv(
            [
                lambda: runner.StaticContractEnv(
                    source.observation_space, source.action_space
                )
                for _ in range(8)
            ]
        )

    control_vec = vector()
    balanced_vec = vector()
    try:
        control = runner.load_continuation_model(source_path, control_vec, config)
        balanced = runner.load_continuation_model(source_path, balanced_vec, config)
        assert control is not balanced
        assert runner._nested_equal(
            control.policy.state_dict(), balanced.policy.state_dict()
        )
        assert runner._nested_equal(
            control.policy.optimizer.state_dict(),
            balanced.policy.optimizer.state_dict(),
        )
        assert runner._nested_equal(
            source.policy.state_dict(), control.policy.state_dict()
        )
        assert control.num_timesteps == balanced.num_timesteps == 2_727_936
        assert control.seed == balanced.seed == 63806
    finally:
        control_vec.close()
        balanced_vec.close()


def test_configuration_is_validate_preflight_only_and_runner_has_no_training_call() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["execution"]["formal_training_authorised"] is False
    assert config["energy_boundary"]["status"] == (
        "measurement_only_not_reward_or_gate"
    )
    assert config["energy_boundary"]["reward_weight"] == 0.0
    assert config["invariants"]["global_map_input_added"] is False
    assert config["training"]["training_seed_count"] == 1
    assert config["training"]["multi_seed_training_robustness_claim_permitted"] is False
    assert config["post_result_video_archive_contract"]["predeclared_seed"] == 96131
    source = (
        ROOT
        / "scripts"
        / "run_fixed_standard_pair0_turn_balance_continuation.py"
    ).read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert "fixed_map" not in config["execution"]["permitted_modes"]
    assert tuple(row["condition_id"] for row in config["conditions"]) == CONDITION_IDS
