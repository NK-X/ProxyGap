from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap import (  # noqa: E402
    make_planar_transition_env,
    make_proxygap_ant_env,
    planar_velocity_tracking_value,
    quarter_turn_action,
    quaternion_yaw_angle,
    transfer_pretrained_policy,
    wrapped_angle_difference,
)
from proxygap.planar_transition import make_ppo_from_config  # noqa: E402
from run_planar_translation_transition import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "planar_translation_transition_v1_20260818.json"


def test_quarter_turn_action_mapping_cycles_four_leg_pairs() -> None:
    action = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(
        quarter_turn_action(action),
        np.asarray([6, 7, 0, 1, 2, 3, 4, 5]),
    )
    rotated = action.copy()
    for _ in range(4):
        rotated = quarter_turn_action(rotated)
    np.testing.assert_array_equal(rotated, action)


def test_yaw_helpers_are_signed_and_wrap_across_pi() -> None:
    angle = math.radians(30.0)
    quaternion = np.asarray([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
    assert math.isclose(quaternion_yaw_angle(quaternion), angle)
    assert math.isclose(
        wrapped_angle_difference(math.radians(-179), math.radians(179)),
        math.radians(2),
    )


def test_planar_tracking_uses_both_velocity_components() -> None:
    assert math.isclose(
        planar_velocity_tracking_value(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            scale=0.5,
        ),
        1.0,
    )
    assert planar_velocity_tracking_value(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        scale=0.5,
    ) < 0.001
    stopped_pseudo_huber = planar_velocity_tracking_value(
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        scale=0.5,
        function="pseudo_huber",
    )
    assert stopped_pseudo_huber < -1.0


def test_command_observation_and_brake_state_machine_reconcile_reward() -> None:
    env = make_planar_transition_env(
        switch_step_min=2,
        switch_step_max=2,
        brake_min_steps=1,
        brake_max_steps=3,
        stop_consecutive_steps=1,
    )
    observation, _ = env.reset(seed=818)
    assert observation.shape == (115,)
    np.testing.assert_allclose(observation[-2:], [1.0, 0.0])
    phases = []
    for _ in range(8):
        observation, _, _, _, info = env.step(np.zeros(8))
        phases.append(info["proxygap_command_phase_step"])
    assert phases[:2] == ["forward", "forward"]
    assert "brake" in phases
    assert phases[-1] == "lateral"
    np.testing.assert_allclose(observation[-2:], [0.0, 1.0])
    summary = env.episode_summary()
    assert abs(summary["planar_reward_reconciliation_error"]) < 1e-9
    assert summary["brake_phase_steps"] <= 3
    env.close()


def test_transfer_expands_only_command_columns_and_preserves_initial_action() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_env = make_proxygap_ant_env(augment_previous_applied_action=True)
    target_env = make_planar_transition_env()
    source_model = make_ppo_from_config(source_env, config["ppo"], seed=1)
    target_model = make_ppo_from_config(target_env, config["ppo"], seed=2)
    manifest = transfer_pretrained_policy(source_model, target_model)
    source_observation, _ = source_env.reset(seed=3)
    target_observation, _ = target_env.reset(seed=3)
    np.testing.assert_allclose(target_observation[:-2], source_observation)
    source_action, _ = source_model.predict(source_observation, deterministic=True)
    target_action, _ = target_model.predict(target_observation, deterministic=True)
    np.testing.assert_allclose(target_action, source_action, atol=1e-7, rtol=0.0)
    assert manifest["source_observation_dimension"] == 113
    assert manifest["target_observation_dimension"] == 115
    assert len(manifest["expanded_parameter_tensors"]) == 2
    source_env.close()
    target_env.close()


def test_planar_transition_config_is_frozen_and_uses_pre_pitch_model() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["base_policy"]["condition_id"] == "F1__FOOT_LANDING"
    assert config["base_policy"]["pitch_balance_reward_enabled"] is False
    assert config["commands"]["target_observation_dimension"] == 115
    assert config["reward"]["yaw_reference"] == "episode initial torso yaw"
