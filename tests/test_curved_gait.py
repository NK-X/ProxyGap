from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap import (  # noqa: E402
    make_curved_gait_env,
    make_planar_transition_env,
    pseudo_huber_penalty,
    transfer_planar_policy_to_curved_gait,
)
from proxygap.planar_transition import (  # noqa: E402
    make_ppo_from_config,
    quaternion_yaw_angle,
)
from run_curved_gait_training import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "curved_gait_tangent_v1_20260818.json"
BODY_FRAME_CONFIG = (
    ROOT / "configs" / "curved_gait_tangent_v2_body_frame_20260818.json"
)
BOUNDED_CURRICULUM_CONFIG = (
    ROOT
    / "configs"
    / "curved_gait_tangent_v3_bounded_curriculum_20260818.json"
)
CANONICAL_FRAME_CONFIG = (
    ROOT / "configs" / "curved_gait_tangent_v4_canonical_frame_20260818.json"
)
LONG_HORIZON_CONFIG = (
    ROOT / "configs" / "curved_gait_tangent_v5_long_horizon_20260818.json"
)


def test_pseudo_huber_penalty_is_zero_at_target_and_non_saturating() -> None:
    assert pseudo_huber_penalty(0.0, scale=0.2) == 0.0
    assert pseudo_huber_penalty(0.4, scale=0.2) > 1.0
    assert math.isclose(
        pseudo_huber_penalty(-0.4, scale=0.2),
        pseudo_huber_penalty(0.4, scale=0.2),
    )


def test_curve_observation_contains_only_local_motion_command() -> None:
    env = make_curved_gait_env(
        profile="constant_left",
        speed_min=1.0,
        speed_max=1.0,
        max_abs_curvature=0.2,
        heading_termination_consecutive_steps=20,
    )
    observation, info = env.reset(seed=818)
    assert observation.shape == (118,)
    target_heading = float(info["proxygap_curve_target_heading"])
    np.testing.assert_allclose(
        observation[-5:-3],
        [math.cos(target_heading), math.sin(target_heading)],
        atol=1e-12,
    )
    assert math.isclose(observation[-3], 0.2, abs_tol=1e-12)
    np.testing.assert_allclose(observation[-2:], [0.0, 1.0], atol=1e-12)
    assert info["proxygap_curve_profile"] == "constant_left"
    observation, _, _, _, step_info = env.step(np.zeros(8))
    assert math.isclose(
        step_info["proxygap_curve_tangent_heading_step"],
        target_heading + 0.01,
        abs_tol=1e-12,
    )
    assert "path_position" not in " ".join(step_info)
    summary = env.episode_summary()
    assert summary["curve_uses_global_path_position_reward"] is False
    assert abs(summary["curve_reward_reconciliation_error"]) < 1e-9
    env.close()


def test_heading_constraint_terminates_sustained_misalignment() -> None:
    env = make_curved_gait_env(
        profile="straight",
        speed_min=1.0,
        speed_max=1.0,
        max_abs_curvature=0.0,
        heading_termination_threshold=math.radians(5.0),
        heading_termination_consecutive_steps=1,
    )
    env.reset(seed=3)
    qpos = np.asarray(env.unwrapped.data.qpos).copy()
    qvel = np.asarray(env.unwrapped.data.qvel).copy()
    yaw = math.radians(30.0)
    qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    env.unwrapped.set_state(qpos, qvel)
    _, _, terminated, _, info = env.step(np.zeros(8))
    assert terminated
    assert info["proxygap_heading_constraint_terminated"] is True
    env.close()


def test_heading_constraint_can_be_disabled_during_early_curriculum() -> None:
    env = make_curved_gait_env(
        profile="straight",
        speed_min=1.0,
        speed_max=1.0,
        max_abs_curvature=0.0,
        heading_alignment_function="bounded_squared",
        yaw_rate_tracking_function="bounded_squared",
        heading_termination_threshold=math.radians(5.0),
        heading_termination_consecutive_steps=1,
        heading_termination_enabled=False,
    )
    env.reset(seed=3)
    qpos = np.asarray(env.unwrapped.data.qpos).copy()
    qvel = np.asarray(env.unwrapped.data.qvel).copy()
    yaw = math.radians(30.0)
    qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    env.unwrapped.set_state(qpos, qvel)
    _, _, terminated, _, info = env.step(np.zeros(8))
    assert not terminated
    assert info["proxygap_heading_constraint_terminated"] is False
    assert env.episode_summary()["curve_heading_termination_enabled"] is False
    env.close()


def test_body_frame_command_does_not_turn_into_a_lateral_translation_request() -> None:
    env = make_curved_gait_env(
        profile="constant_left",
        command_frame="body_tangent",
        speed_min=0.8,
        speed_max=0.8,
        max_abs_curvature=0.2,
        heading_termination_consecutive_steps=20,
    )
    observation, _ = env.reset(seed=9)
    np.testing.assert_allclose(observation[-5:-3], [0.8, 0.0], atol=1e-12)
    observation, _, _, _, info = env.step(np.zeros(8))
    np.testing.assert_allclose(observation[-5:-3], [0.8, 0.0], atol=1e-12)
    assert info["proxygap_curve_target_heading"] != 0.0
    assert info["proxygap_curve_command_frame"] == "body_tangent"
    env.close()


def test_target_tangent_frame_canonicalises_initial_yaw_and_world_vectors() -> None:
    env = make_curved_gait_env(
        profile="constant_left",
        command_frame="body_tangent",
        observation_frame="target_tangent",
        speed_min=0.8,
        speed_max=0.8,
        max_abs_curvature=0.2,
        heading_termination_enabled=False,
    )
    observation, info = env.reset(seed=9)
    assert abs(quaternion_yaw_angle(observation[1:5])) < 1e-12
    assert info["proxygap_curve_observation_frame"] == "target_tangent"
    observation, _, _, _, info = env.step(np.zeros(8))
    canonical_yaw = quaternion_yaw_angle(observation[1:5])
    assert math.isclose(
        canonical_yaw,
        info["proxygap_curve_heading_error_step"],
        abs_tol=1e-12,
    )
    assert env.episode_summary()["curve_observation_frame"] == "target_tangent"
    env.close()


def test_external_route_controller_exposes_only_local_curve_command() -> None:
    env = make_curved_gait_env(
        profile="external",
        command_frame="body_tangent",
        observation_frame="target_tangent",
        speed_min=0.8,
        speed_max=0.8,
        max_abs_curvature=0.35,
        heading_termination_enabled=False,
    )
    observation, _ = env.reset(seed=14)
    observation = env.set_external_curve_command(
        observation,
        target_heading=0.5,
        yaw_rate=0.1,
        speed=0.8,
        lateral_speed=-0.25,
    )
    np.testing.assert_allclose(observation[-5:-3], [0.8, -0.25], atol=1e-12)
    assert math.isclose(observation[-3], 0.1, abs_tol=1e-12)
    observation, _, _, _, info = env.step(np.zeros(8))
    assert math.isclose(
        info["proxygap_curve_tangent_heading_step"],
        0.5,
        abs_tol=1e-12,
    )
    observation = env.set_external_curve_command(
        observation,
        target_heading=0.55,
        yaw_rate=0.1,
        speed=0.8,
    )
    observation, _, _, _, info = env.step(np.zeros(8))
    assert math.isclose(
        info["proxygap_curve_tangent_heading_step"],
        0.55,
        abs_tol=1e-12,
    )
    assert "path" not in " ".join(info).lower()
    env.close()


def test_tangent_frame_quaternion_is_continuous_across_pi_wrap() -> None:
    env = make_curved_gait_env(
        profile="external",
        command_frame="body_tangent",
        observation_frame="target_tangent",
        speed_min=0.8,
        speed_max=0.8,
        max_abs_curvature=0.35,
        heading_termination_enabled=False,
    )
    observation, _ = env.reset(seed=19)
    canonical_quaternions = []
    for yaw in (math.pi - 1e-4, -math.pi + 1e-4):
        qpos = np.asarray(env.unwrapped.data.qpos).copy()
        qvel = np.asarray(env.unwrapped.data.qvel).copy()
        qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        env.unwrapped.set_state(qpos, qvel)
        raw = env.env._augment_observation(env.unwrapped._get_obs())
        observation = env.set_external_curve_command(
            raw,
            target_heading=yaw,
            yaw_rate=0.1,
            speed=0.8,
        )
        canonical_quaternions.append(observation[1:5].copy())
    np.testing.assert_allclose(
        canonical_quaternions[0],
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        canonical_quaternions[1],
        canonical_quaternions[0],
        atol=1e-12,
    )
    env.close()


def test_planar_policy_transfer_adds_only_three_zero_columns() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ppo = dict(config["ppo"])
    ppo["device"] = "cpu"
    ppo["batch_size"] = 64
    source_env = make_planar_transition_env(
        initial_command_xy=(1.0, 0.0),
        switch_step_min=200,
        switch_step_max=200,
    )
    target_env = make_curved_gait_env(
        profile="straight",
        speed_min=1.0,
        speed_max=1.0,
        max_abs_curvature=0.0,
    )
    source_model = make_ppo_from_config(source_env, ppo, seed=1)
    target_model = make_ppo_from_config(target_env, ppo, seed=2)
    manifest = transfer_planar_policy_to_curved_gait(source_model, target_model)
    source_observation, _ = source_env.reset(seed=5)
    target_observation, _ = target_env.reset(seed=5)
    np.testing.assert_allclose(target_observation[:-5], source_observation[:-2])
    source_action, _ = source_model.predict(
        target_observation[:-3],
        deterministic=True,
    )
    target_action, _ = target_model.predict(target_observation, deterministic=True)
    np.testing.assert_allclose(target_action, source_action, atol=1e-7, rtol=0.0)
    assert manifest["source_observation_dimension"] == 115
    assert manifest["target_observation_dimension"] == 118
    assert manifest["new_curve_command_columns_initialised_to_zero"] == 3
    source_env.close()
    target_env.close()


def test_curved_gait_config_preserves_gait_and_excludes_path_reward() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["base_policy"]["pitch_balance_reward_enabled"] is False
    assert config["commands"]["global_path_position_reward_enabled"] is False
    assert config["preserved_pre_pitch_reward"]["pitch_balance_shaping_weight"] == 0.0
    assert config["curriculum"][-1]["max_abs_curvature_per_m"] == 0.5


def test_body_frame_revision_is_frozen_and_keeps_path_outside_policy() -> None:
    config = json.loads(BODY_FRAME_CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["commands"]["command_frame"] == "body_tangent"
    assert config["commands"]["observation_append_order"][:2] == [
        "v_forward_command",
        "v_lateral_command",
    ]
    assert config["commands"]["global_path_position_reward_enabled"] is False


def test_bounded_curriculum_separates_training_from_strict_evaluation() -> None:
    config = json.loads(BOUNDED_CURRICULUM_CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["reward"]["heading_alignment_function"] == "bounded_squared"
    assert config["reward"]["yaw_rate_tracking_function"] == "bounded_squared"
    assert config["reward"]["training_heading_termination_enabled"] is False
    assert config["reward"]["evaluation_heading_termination_enabled"] is True
    assert config["execution"]["parallel_environments_per_seed"] == 8


def test_canonical_frame_config_keeps_path_out_and_rotates_only_local_state() -> None:
    config = json.loads(CANONICAL_FRAME_CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["commands"]["observation_frame"] == "target_tangent"
    assert config["commands"]["global_path_position_in_observation"] is False
    assert config["commands"]["global_path_position_reward_enabled"] is False
    assert "root_quaternion" in config["commands"]["canonicalised_world_fields"]


def test_long_horizon_continuation_keeps_route_outside_policy() -> None:
    config = json.loads(LONG_HORIZON_CONFIG.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=False)
    assert config["transfer"]["initialisation"] == "continue_same_observation_policy"
    assert config["base_policy"]["observation_dimension"] == 118
    assert config["evaluation_max_episode_steps"] == 2000
    assert config["commands"]["segment_steps_interval"] == [600, 1000]
    assert config["commands"]["global_path_position_in_observation"] is False
