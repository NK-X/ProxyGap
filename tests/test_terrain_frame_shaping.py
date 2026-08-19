from __future__ import annotations

import math
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import mujoco


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.ant_wrapper import (  # noqa: E402
    angular_speed_perpendicular_to_normal,
    make_proxygap_ant_env,
    project_velocity_onto_axis,
    quaternion_tilt_relative_to_normal,
    target_tangent_frame,
    validated_terrain_normal,
)
from run_fixed_goal_terrain_training import make_task_env  # noqa: E402


def test_terrain_normal_validation_fails_closed() -> None:
    np.testing.assert_array_equal(
        validated_terrain_normal(np.asarray([0.0, 0.0, 1.0])),
        np.asarray([0.0, 0.0, 1.0]),
    )
    for invalid in (
        np.asarray([0.0, 0.0, 2.0]),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([np.nan, 0.0, 1.0]),
        np.asarray([0.0, 1.0]),
    ):
        with pytest.raises(ValueError):
            validated_terrain_normal(invalid)


def test_analytic_x_slope_projections_are_target_tangent_correct() -> None:
    slope = 0.5
    normal = np.asarray([-slope, 0.0, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    forward, left, observed_normal = target_tangent_frame(normal, 0.0)
    expected_forward = np.asarray([1.0, 0.0, slope], dtype=np.float64)
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(forward, expected_forward, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(left, [0.0, 1.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed_normal, normal, atol=0.0, rtol=0.0)

    uphill_velocity = 2.0 * expected_forward
    assert project_velocity_onto_axis(uphill_velocity, normal) == pytest.approx(0.0)
    assert project_velocity_onto_axis(uphill_velocity, left) == pytest.approx(0.0)
    assert project_velocity_onto_axis(np.asarray([0.0, 3.0, 0.0]), left) == pytest.approx(3.0)

    angular_velocity = 2.0 * normal + 3.0 * left
    assert angular_speed_perpendicular_to_normal(
        angular_velocity,
        normal,
    ) == pytest.approx(3.0)


def test_full_normal_orientation_uses_both_slope_components() -> None:
    normal = np.asarray([-0.3, 0.4, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    observed = quaternion_tilt_relative_to_normal(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        normal,
    )
    assert observed == pytest.approx(math.acos(float(normal[2])))


def test_enabled_wrapper_refuses_to_step_without_context() -> None:
    env = make_proxygap_ant_env(
        condition_id="terrain_frame_missing_context",
        terrain_frame_shaping_enabled=True,
    )
    try:
        env.reset(seed=1501)
        with pytest.raises(RuntimeError, match="valid next-step context"):
            env.step(np.zeros(env.action_space.shape, dtype=np.float64))
    finally:
        env.close()


def test_free_joint_angular_velocity_is_rotated_to_world_before_projection() -> None:
    env = make_proxygap_ant_env(condition_id="free_joint_velocity_frame")
    try:
        env.reset(seed=1503)
        model = env.unwrapped.model
        data = env.unwrapped.data
        torso_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )
        object_velocity_world = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            torso_id,
            object_velocity_world,
            0,
        )
        torso_rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
        free_joint_angular_world = torso_rotation @ np.asarray(data.qvel[3:6])
        np.testing.assert_allclose(
            free_joint_angular_world,
            object_velocity_world[:3],
            atol=1e-12,
            rtol=0.0,
        )
    finally:
        env.close()


def test_canonical_flat_context_has_exact_reward_and_dynamics_parity() -> None:
    common = {
        "orientation_shaping_weight": 0.1,
        "orientation_shaping_function": "cosine",
        "vertical_velocity_shaping_weight": 0.05,
        "roll_pitch_angular_velocity_shaping_weight": 0.05,
        "foot_lateral_velocity_shaping_weight": 0.025,
        "foot_vertical_velocity_shaping_weight": 0.025,
    }
    control = make_proxygap_ant_env(
        condition_id="terrain_frame_flat_control",
        **common,
    )
    intervention = make_proxygap_ant_env(
        condition_id="terrain_frame_flat_intervention",
        terrain_frame_shaping_enabled=True,
        **common,
    )
    try:
        control_observation, _ = control.reset(seed=1502)
        intervention_observation, reset_info = intervention.reset(seed=1502)
        np.testing.assert_allclose(
            control_observation,
            intervention_observation,
            atol=0.0,
            rtol=0.0,
        )
        assert reset_info["proxygap_terrain_frame_shaping_enabled"] is True
        assert reset_info["proxygap_terrain_frame_context_valid"] is False
        intervention.set_terrain_shaping_context(
            height_sampler=lambda _x, _y: 0.0,
            normal_sampler=lambda _x, _y: np.asarray([0.0, 0.0, 1.0]),
            target_heading=0.0,
        )
        action = np.asarray([0.25, -0.1, 0.2, -0.3, 0.15, -0.2, 0.1, -0.05])
        control_result = control.step(action)
        intervention_result = intervention.step(action)
        np.testing.assert_allclose(
            control_result[0],
            intervention_result[0],
            atol=0.0,
            rtol=0.0,
        )
        assert control_result[1] == intervention_result[1]
        assert control_result[2:4] == intervention_result[2:4]
        control_info = control_result[4]
        intervention_info = intervention_result[4]
        for key in (
            "reward_shaping",
            "orientation_penalty",
            "vertical_velocity_penalty",
            "roll_pitch_angular_velocity_penalty",
            "foot_lateral_velocity_penalty",
            "foot_vertical_velocity_penalty",
            "proxygap_root_vertical_velocity_step",
            "proxygap_root_roll_pitch_angular_speed_step",
        ):
            assert control_info[key] == intervention_info[key], key
        for key in (
            "proxygap_foot_contact_point_heights_step",
            "proxygap_foot_lateral_velocities_step",
            "proxygap_foot_vertical_velocities_step",
        ):
            np.testing.assert_allclose(
                control_info[key],
                intervention_info[key],
                atol=0.0,
                rtol=0.0,
            )
        assert intervention_info["proxygap_terrain_frame_context_valid"] is True
        assert intervention_info["proxygap_terrain_frame_shaping_applied_step"] is True
    finally:
        control.close()
        intervention.close()


def test_fixed_goal_reset_installs_context_before_first_policy_step() -> None:
    config = json.loads(
        (ROOT / "configs" / "fixed_quad_terrain_v2_local_preview_pilot_v1_20260819.json")
        .read_text(encoding="utf-8")
    )
    config["task_adapter"]["terrain_frame_shaping_enabled"] = True
    reward_config = json.loads(
        (ROOT / config["base_policy"]["configuration"]).read_text(encoding="utf-8")
    )
    scene = ROOT / config["approved_map"]["xml_path"]
    env = make_task_env(
        config,
        reward_config,
        xml_path=scene,
        seed=1504,
        spawn_fraction=0.0,
        max_episode_steps=10,
        cruise_speed=0.55,
        terminate_on_success=False,
    )
    try:
        observation, reset_info = env.reset(seed=1504)
        assert observation.shape == (135,)
        assert reset_info["proxygap_terrain_frame_shaping_enabled"] is True
        assert reset_info["proxygap_terrain_frame_context_valid"] is True
        _, _, _, _, step_info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float64)
        )
        assert step_info["proxygap_terrain_frame_shaping_applied_step"] is True
        assert step_info["proxygap_terrain_frame_context_valid"] is True
    finally:
        env.close()
