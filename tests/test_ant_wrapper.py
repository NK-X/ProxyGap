from __future__ import annotations

import math
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from proxygap import (
    bounded_squared_signal_penalty,
    CSV_SCHEMA,
    DEFAULT_PPO_CONFIG,
    EpisodeMetrics,
    common_rescored_return,
    forward_velocity_tracking_value,
    lateral_penalty_value,
    make_proxygap_ant_env,
    normalised_action_rate_penalty,
    orientation_penalty_value,
    project_action_l2_slew,
    protocol_freeze_status,
    quaternion_tilt_angle,
)
from proxygap.experiment import (
    checkpoint_targets,
    evaluate_model,
    resolve_ppo_config,
    select_representative_evaluation_seed,
    summarise_evaluation,
)


def test_reference_ctrl_cost_weight_is_applied() -> None:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.5, condition_id="reference", seed=1)
    assert env.unwrapped._ctrl_cost_weight == 0.5
    env.close()


def test_large_render_floor_xml_preserves_default_dynamics() -> None:
    render_xml = Path(__file__).resolve().parents[1] / "assets" / "ant_render_large_floor.xml"
    default_env = make_proxygap_ant_env(condition_id="default_xml")
    render_env = make_proxygap_ant_env(condition_id="render_xml", xml_file=render_xml)
    default_observation, _ = default_env.reset(seed=816)
    render_observation, _ = render_env.reset(seed=816)
    np.testing.assert_allclose(default_observation, render_observation, atol=0.0, rtol=0.0)
    rng = np.random.default_rng(816)
    for _ in range(25):
        action = rng.uniform(-1.0, 1.0, size=default_env.action_space.shape)
        default_step = default_env.step(action)
        render_step = render_env.step(action)
        np.testing.assert_allclose(default_step[0], render_step[0], atol=1e-12, rtol=0.0)
        assert math.isclose(default_step[1], render_step[1], abs_tol=1e-12)
        assert default_step[2:4] == render_step[2:4]
    default_env.close()
    render_env.close()


def test_reduced_ctrl_cost_weight_is_the_only_reward_coefficient_changed_here() -> None:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.25, condition_id="candidate", seed=1)
    assert env.unwrapped._ctrl_cost_weight == 0.25
    assert env.unwrapped._forward_reward_weight == 1
    assert env.unwrapped._healthy_reward == 1.0
    env.close()


def test_wrapper_reset_and_step_expose_proxygap_metrics() -> None:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.5, condition_id="reference", seed=2)
    observation, info = env.reset(seed=2)
    assert observation.shape == env.observation_space.shape
    assert "proxygap_proxy_return" in info

    action = env.action_space.sample()
    _, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_forward" in info
    assert "reward_ctrl" in info
    assert "proxygap_net_forward_progress" in info
    assert "proxygap_control_effort_per_unit_distance" in info
    assert "proxygap_torso_tilt_step" in info
    env.close()


def test_action_slew_projection_is_identity_below_bound() -> None:
    proposed = np.full(8, 0.1)
    previous = np.zeros(8)
    applied, intervened, requested_norm, correction_norm = project_action_l2_slew(
        proposed,
        previous,
        limit=1.4,
        action_low=-np.ones(8),
        action_high=np.ones(8),
    )
    assert np.allclose(applied, proposed)
    assert intervened is False
    assert math.isclose(requested_norm, math.sqrt(0.08))
    assert correction_norm == 0.0


def test_action_slew_projection_hits_bound_above_limit() -> None:
    proposed = np.ones(8)
    previous = -np.ones(8)
    applied, intervened, requested_norm, correction_norm = project_action_l2_slew(
        proposed,
        previous,
        limit=1.4,
        action_low=-np.ones(8),
        action_high=np.ones(8),
    )
    assert intervened is True
    assert math.isclose(requested_norm, 2.0 * math.sqrt(8.0))
    assert math.isclose(np.linalg.norm(applied - previous), 1.4)
    assert correction_norm > 0.0
    assert np.all(applied >= -1.0)
    assert np.all(applied <= 1.0)


def test_guardrail_requires_previous_action_observation() -> None:
    with np.testing.assert_raises(ValueError):
        make_proxygap_ant_env(
            condition_id="invalid_guardrail",
            action_slew_l2_limit=1.4,
            augment_previous_applied_action=False,
        )


def test_augmented_comparator_and_guardrail_share_113_observations() -> None:
    comparator = make_proxygap_ant_env(
        condition_id="augmented_comparator",
        augment_previous_applied_action=True,
    )
    constrained = make_proxygap_ant_env(
        condition_id="guardrail",
        augment_previous_applied_action=True,
        action_slew_l2_limit=1.4,
    )
    for env in (comparator, constrained):
        observation, _ = env.reset(seed=123)
        assert observation.shape == (113,)
        assert np.allclose(observation[-8:], 0.0)
    comparator_action = np.full(8, 0.25)
    next_observation, _, _, _, info = comparator.step(comparator_action)
    assert np.allclose(next_observation[-8:], comparator_action)
    assert info["proxygap_action_constraint_enabled"] is False
    comparator.close()
    constrained.close()


def test_guardrail_logs_proposed_applied_and_intervention_metrics() -> None:
    env = make_proxygap_ant_env(
        condition_id="guardrail",
        augment_previous_applied_action=True,
        action_slew_l2_limit=0.1,
    )
    env.reset(seed=124)
    observation, _, _, _, info = env.step(np.ones(8))
    assert info["proxygap_action_slew_intervened_step"] is True
    assert math.isclose(info["proxygap_applied_action_change_l2_step"], 0.1)
    assert info["proxygap_requested_action_change_l2_step"] > 0.1
    assert np.allclose(observation[-8:], info["proxygap_applied_action"])
    summary = env.episode_summary()
    assert summary["action_slew_intervention_count"] == 1
    assert summary["action_slew_intervention_rate"] == 1.0
    assert summary["action_constraint_enabled"] is True
    env.close()


def test_intent_compliance_uses_all_frozen_dimensions() -> None:
    metrics = EpisodeMetrics(
        environment_dt=0.05,
        action_dimension=8,
        evaluation_horizon_steps=1000,
    )
    metrics.reset(initial_x=0.0, initial_y=0.0)
    for step in range(1, 1001):
        metrics.update(
            action=np.zeros(8),
            reward=0.0,
            terminated=False,
            truncated=step == 1000,
            info={"x_position": step * 0.05, "y_position": 0.0},
            torso_tilt=0.0,
            torso_height=0.5,
        )
    summary = metrics.summary()
    assert math.isclose(summary["fixed_horizon_mean_forward_velocity"], 1.0)
    assert summary["net_displacement_direction_error_degrees"] == 0.0
    assert summary["intent_compliant"] is True
    assert summary["intent_failure_reasons"] == ""


def test_episode_summary_contains_expected_schema_metrics() -> None:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.5, condition_id="reference", seed=3)
    env.reset(seed=3)
    for _ in range(3):
        env.step(env.action_space.sample())
    summary = env.episode_summary()
    expected = {
        "proxy_return",
        "reward_forward_sum",
        "reward_ctrl_sum",
        "reward_contact_sum",
        "reward_survive_sum",
        "net_forward_progress",
        "environment_dt",
        "episode_duration_seconds",
        "mean_forward_velocity",
        "control_effort",
        "control_effort_per_unit_distance",
        "condition_objective_return",
        "common_rescored_return",
        "cumulative_squared_action",
        "mean_squared_action_per_step",
        "mean_squared_action_change_per_transition",
        "normalised_action_roughness",
        "action_change_transition_count",
        "action_saturation_rate",
        "unhealthy_termination",
        "termination_category",
        "low_z_termination",
        "high_z_termination",
        "lateral_drift_max_abs",
        "cumulative_planar_path",
        "forward_path_efficiency",
        "torso_tilt_rms",
        "torso_tilt_p95",
        "fall",
        "lateral_drift_final_abs",
        "lateral_drift_mean_abs",
        "torso_tilt_mean",
        "torso_tilt_std",
        "episode_length",
        "terminated",
        "truncated",
    }
    assert expected.issubset(summary)
    assert summary["episode_length"] == 3
    assert math.isclose(summary["environment_dt"], env.unwrapped.dt)
    assert math.isclose(
        summary["episode_duration_seconds"],
        3 * env.unwrapped.dt,
    )
    env.close()


def test_velocity_and_normalised_action_roughness_have_explicit_units() -> None:
    metrics = EpisodeMetrics(environment_dt=0.05, action_dimension=8)
    metrics.reset(initial_x=0.0, initial_y=0.0)
    metrics.update(
        action=-np.ones(8),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"x_position": 0.5, "y_position": 0.0},
        torso_tilt=0.0,
    )
    metrics.update(
        action=np.ones(8),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"x_position": 1.0, "y_position": 0.0},
        torso_tilt=0.0,
    )
    summary = metrics.summary()
    assert math.isclose(summary["episode_duration_seconds"], 0.1)
    assert math.isclose(summary["mean_forward_velocity"], 10.0)
    assert math.isclose(summary["normalised_action_roughness"], 1.0)


def test_quaternion_tilt_angle_for_identity_orientation() -> None:
    assert quaternion_tilt_angle(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0


def test_normalised_cosine_orientation_penalty_has_geometric_anchors() -> None:
    assert orientation_penalty_value(0.0, function="cosine") == 0.0
    assert math.isclose(
        orientation_penalty_value(math.pi / 2.0, function="cosine"),
        0.5,
    )
    assert math.isclose(
        orientation_penalty_value(math.pi, function="cosine"),
        1.0,
    )


def test_forward_velocity_tracking_has_bounded_command_anchors() -> None:
    assert forward_velocity_tracking_value(1.0, target=1.0, scale=0.5) == 1.0
    assert math.isclose(
        forward_velocity_tracking_value(1.5, target=1.0, scale=0.5),
        math.exp(-1.0),
    )
    assert 0.0 <= forward_velocity_tracking_value(3.0, target=1.0, scale=0.5) < 1.0


def test_normalised_action_rate_penalty_has_fixed_action_range() -> None:
    assert normalised_action_rate_penalty(np.ones(8), None) == 0.0
    assert normalised_action_rate_penalty(np.zeros(8), np.zeros(8)) == 0.0
    assert math.isclose(normalised_action_rate_penalty(np.ones(8), -np.ones(8)), 1.0)


def test_bounded_squared_signal_penalty_has_finite_symmetric_anchors() -> None:
    assert bounded_squared_signal_penalty(0.0, scale=2.0) == 0.0
    assert math.isclose(
        bounded_squared_signal_penalty(2.0, scale=2.0),
        math.tanh(1.0),
    )
    assert math.isclose(
        bounded_squared_signal_penalty(-2.0, scale=2.0),
        math.tanh(1.0),
    )
    assert bounded_squared_signal_penalty(float("inf"), scale=2.0) == 1.0
    with np.testing.assert_raises(ValueError):
        bounded_squared_signal_penalty(0.0, scale=0.0)


def test_body_dynamics_shaping_is_bounded_logged_and_reconciled() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="body_dynamics_test",
        seed=29,
        vertical_velocity_shaping_weight=0.05,
        vertical_velocity_shaping_scale=1.014092584749083,
        roll_pitch_angular_velocity_shaping_weight=0.05,
        roll_pitch_angular_velocity_shaping_scale=1.9893176307304792,
    )
    env.reset(seed=29)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    expected_vertical = bounded_squared_signal_penalty(
        info["proxygap_root_vertical_velocity_step"],
        scale=1.014092584749083,
    )
    expected_angular = bounded_squared_signal_penalty(
        info["proxygap_root_roll_pitch_angular_speed_step"],
        scale=1.9893176307304792,
    )
    assert math.isclose(info["vertical_velocity_penalty"], expected_vertical)
    assert math.isclose(
        info["roll_pitch_angular_velocity_penalty"], expected_angular
    )
    assert -0.05 <= info["reward_vertical_velocity_shaping"] <= 0.0
    assert -0.05 <= info["reward_roll_pitch_angular_velocity_shaping"] <= 0.0
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    summary = env.episode_summary()
    assert math.isclose(
        summary["reward_vertical_velocity_shaping_sum"],
        info["reward_vertical_velocity_shaping"],
    )
    assert math.isclose(
        summary["reward_roll_pitch_angular_velocity_shaping_sum"],
        info["reward_roll_pitch_angular_velocity_shaping"],
    )
    env.close()


def test_tracking_replaces_only_forward_term_and_rate_penalty_is_separate() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="tracking_rate_test",
        seed=19,
        replace_forward_reward_with_tracking=True,
        forward_velocity_target=1.0,
        forward_velocity_tracking_scale=0.5,
        action_rate_shaping_weight=0.4,
        augment_previous_applied_action=True,
    )
    env.reset(seed=19)
    _, reward_a, _, _, info_a = env.step(np.zeros(8))
    _, reward_b, _, _, info_b = env.step(np.ones(8))
    expected_a = (
        info_a["reward_base_proxy"]
        - info_a["reward_forward"]
        + info_a["reward_forward_tracking"]
    )
    assert math.isclose(reward_a, expected_a)
    assert math.isclose(info_b["action_rate_penalty"], 0.25)
    assert math.isclose(info_b["reward_action_rate_shaping"], -0.1)
    expected_b = (
        info_b["reward_base_proxy"]
        - info_b["reward_forward"]
        + info_b["reward_forward_tracking"]
        - 0.1
    )
    assert math.isclose(reward_b, expected_b)
    summary = env.episode_summary()
    assert summary["replace_forward_reward_with_tracking"] is True
    assert summary["action_rate_penalty_sum"] > 0.0
    env.close()


def test_forward_tracking_weight_reduces_torso_x_reward_influence() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="weighted_tracking_test",
        seed=119,
        replace_forward_reward_with_tracking=True,
        forward_velocity_target=1.0,
        forward_velocity_tracking_scale=0.5,
        forward_velocity_tracking_weight=0.5,
    )
    env.reset(seed=119)
    _, reward, _, _, info = env.step(np.zeros(8))
    expected_tracking = 0.5 * forward_velocity_tracking_value(
        info["proxygap_forward_velocity_step"],
        target=1.0,
        scale=0.5,
    )
    assert math.isclose(info["reward_forward_tracking"], expected_tracking)
    assert math.isclose(
        reward,
        info["reward_base_proxy"]
        - info["reward_forward"]
        + expected_tracking,
    )
    assert env.episode_summary()["forward_velocity_tracking_weight"] == 0.5
    env.close()


def test_four_foot_landing_velocity_shaping_is_height_gated_and_reconciled() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="foot_landing_test",
        seed=120,
        foot_landing_height_threshold=0.03,
        foot_lateral_velocity_shaping_weight=0.025,
        foot_lateral_velocity_shaping_scale=1.0,
        foot_vertical_velocity_shaping_weight=0.025,
        foot_vertical_velocity_shaping_scale=1.0,
    )
    env.reset(seed=120)
    heights = np.asarray([0.01, 0.04, 0.02, 0.05])
    lateral = np.asarray([1.0, 100.0, -0.5, 100.0])
    vertical = np.asarray([-2.0, 100.0, 0.25, 100.0])
    landing_mask = heights <= 0.03
    env._foot_landing_kinematics = lambda: (
        heights,
        lateral,
        vertical,
        landing_mask,
    )
    _, reward, _, _, info = env.step(np.zeros(8))
    expected_lateral_penalty = bounded_squared_signal_penalty(
        1.0, scale=1.0
    ) + bounded_squared_signal_penalty(-0.5, scale=1.0)
    expected_vertical_penalty = bounded_squared_signal_penalty(
        -2.0, scale=1.0
    ) + bounded_squared_signal_penalty(0.25, scale=1.0)
    assert info["proxygap_foot_landing_active_count_step"] == 2
    assert np.array_equal(
        info["proxygap_foot_landing_mask_step"],
        np.asarray([True, False, True, False]),
    )
    assert np.array_equal(info["proxygap_foot_contact_point_heights_step"], heights)
    assert math.isclose(
        info["foot_lateral_velocity_penalty"], expected_lateral_penalty
    )
    assert math.isclose(
        info["foot_vertical_velocity_penalty"], expected_vertical_penalty
    )
    expected_foot_reward = -0.025 * (
        expected_lateral_penalty + expected_vertical_penalty
    )
    assert math.isclose(
        info["reward_foot_lateral_velocity_shaping"]
        + info["reward_foot_vertical_velocity_shaping"],
        expected_foot_reward,
    )
    assert math.isclose(reward, info["reward_base_proxy"] + expected_foot_reward)
    summary = env.episode_summary()
    assert summary["foot_landing_active_count_sum"] == 2
    assert summary["foot_landing_active_count_by_foot"] == [1, 0, 1, 0]
    assert math.isclose(
        summary["reward_foot_lateral_velocity_shaping_sum"]
        + summary["reward_foot_vertical_velocity_shaping_sum"],
        expected_foot_reward,
    )
    env.close()


def test_contact_force_slip_and_actuator_diagnostics_are_separate_from_reward() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="contact_diagnostics",
        seed=121,
    )
    env.reset(seed=121)
    reward = 0.0
    info: dict[str, object] = {}
    for _ in range(25):
        _, reward, terminated, truncated, info = env.step(np.zeros(8))
        if terminated or truncated:
            break
    for name in (
        "proxygap_foot_contact_mask_step",
        "proxygap_foot_contact_counts_step",
        "proxygap_foot_normal_forces_n_step",
        "proxygap_foot_tangential_forces_n_step",
        "proxygap_foot_contact_tangential_speeds_m_per_s_step",
        "proxygap_foot_contact_slip_distance_m_step",
    ):
        assert np.asarray(info[name]).shape == (4,)
    for name in (
        "proxygap_actuator_joint_torques_n_m_step",
        "proxygap_actuator_joint_velocities_rad_per_s_step",
        "proxygap_actuator_mechanical_powers_w_step",
    ):
        assert np.asarray(info[name]).shape == (8,)
    assert math.isclose(reward, float(info["reward_base_proxy"]))
    summary = env.episode_summary()
    assert summary["actuator_joint_names"] == [
        "hip_4",
        "ankle_4",
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
    ]
    assert len(summary["foot_contact_duty_fraction_by_foot"]) == 4
    assert len(summary["foot_contact_transition_count_by_foot"]) == 4
    assert len(summary["longest_foot_no_contact_run_steps_by_foot"]) == 4
    assert len(summary["support_count_step_counts_0_to_4"]) == 5
    assert len(summary["support_mask_step_counts_0_to_15"]) == 16
    assert sum(summary["support_count_step_counts_0_to_4"]) == summary[
        "episode_length"
    ]
    assert all(
        0.0 <= value <= 1.0
        for value in summary["foot_contact_duty_fraction_by_foot"]
    )
    assert all(
        value >= 0.0
        for value in summary[
            "foot_sampled_normal_force_time_integral_n_s_by_foot"
        ]
    )
    assert all(
        value >= 0.0
        for value in summary["actuator_abs_mechanical_work_j_by_actuator"]
    )
    assert 0.0 <= summary["airborne_step_fraction"] <= 1.0
    env.close()


def test_airborne_shaping_uses_actual_contact_mask_and_is_separately_logged() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="airborne_shaping",
        seed=122,
        airborne_shaping_weight=0.2,
    )
    env.reset(seed=122)
    zeros = np.zeros(4, dtype=np.float64)
    env._foot_contact_diagnostics = lambda: (
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=np.int64),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["airborne_penalty"] == 1.0
    assert math.isclose(info["reward_airborne_shaping"], -0.2)
    assert math.isclose(
        reward,
        info["reward_base_proxy"] + info["reward_shaping"],
    )
    summary = env.episode_summary()
    assert summary["airborne_step_count"] == 1
    assert summary["airborne_shaping_weight"] == 0.2
    assert math.isclose(summary["reward_airborne_shaping_sum"], -0.2)
    env.close()


def test_negative_airborne_shaping_weight_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        make_proxygap_ant_env(
            condition_id="invalid_airborne_shaping",
            airborne_shaping_weight=-0.1,
        )


def test_contact_gap_shaping_penalises_each_foot_after_its_grace_period() -> None:
    env = make_proxygap_ant_env(
        condition_id="contact_gap_shaping",
        seed=123,
        foot_contact_gap_shaping_weight=0.2,
        foot_contact_gap_grace_seconds=0.0,
        foot_contact_gap_scale_seconds=0.05,
    )
    env.reset(seed=123)
    zeros = np.zeros(4, dtype=np.float64)
    env._foot_contact_diagnostics = lambda: (
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=np.int64),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    expected_penalty = math.tanh(1.0) ** 2
    assert math.isclose(info["foot_contact_gap_penalty"], expected_penalty)
    assert math.isclose(
        info["reward_foot_contact_gap_shaping"],
        -0.2 * expected_penalty,
    )
    assert math.isclose(
        reward,
        info["reward_base_proxy"] + info["reward_shaping"],
    )
    summary = env.episode_summary()
    assert summary["foot_contact_gap_grace_seconds"] == 0.0
    assert summary["foot_contact_gap_scale_seconds"] == 0.05
    assert all(
        math.isclose(value, expected_penalty)
        for value in summary["foot_contact_gap_penalty_sum_by_foot"]
    )
    env.close()


def test_cosine_orientation_shaping_is_added_without_changing_base_reward() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="cosine_test",
        seed=9,
        orientation_shaping_weight=0.4,
        orientation_shaping_function="cosine",
    )
    env.reset(seed=9)
    env._torso_tilt = lambda: math.pi / 2.0
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert math.isclose(info["proxygap_orientation_penalty_step"], 0.5)
    assert math.isclose(info["reward_orientation_shaping"], -0.2)
    assert math.isclose(info["orientation_penalty"], 0.5)
    assert math.isclose(reward, info["reward_base_proxy"] - 0.2)
    assert info["proxygap_orientation_shaping_function"] == "cosine"
    env.close()


def test_invalid_orientation_shaping_configuration_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        orientation_penalty_value(0.0, function="unknown")
    with np.testing.assert_raises(ValueError):
        orientation_penalty_value(0.0, function="cosine", scale=2.0)


def test_control_effort_per_distance_is_nan_when_distance_is_zero() -> None:
    metrics = EpisodeMetrics()
    metrics.reset(initial_x=0.0, initial_y=0.0)
    metrics.update(
        action=np.ones(8),
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"x_position": 0.0, "y_position": 0.0},
        torso_tilt=0.1,
    )
    assert math.isnan(metrics.summary()["control_effort_per_unit_distance"])


def test_fixed_horizon_velocity_penalises_early_episode_end() -> None:
    metrics = EpisodeMetrics(
        environment_dt=0.05,
        action_dimension=8,
        evaluation_horizon_steps=1000,
    )
    metrics.reset(initial_x=0.0, initial_y=0.0)
    metrics.update(
        action=np.zeros(8),
        reward=0.0,
        terminated=True,
        truncated=False,
        info={"x_position": 1.0, "y_position": 0.0},
        torso_tilt=0.0,
        torso_height=1.1,
    )
    summary = metrics.summary()
    assert math.isclose(summary["mean_forward_velocity"], 20.0)
    assert math.isclose(summary["fixed_horizon_mean_forward_velocity"], 0.02)
    assert summary["full_horizon_completed"] is False


def test_sustained_inversion_uses_duration_not_single_frame() -> None:
    metrics = EpisodeMetrics(
        environment_dt=0.05,
        action_dimension=8,
        evaluation_horizon_steps=1000,
        sustained_inversion_seconds=1.0,
    )
    metrics.reset(initial_x=0.0, initial_y=0.0)
    for step in range(20):
        metrics.update(
            action=np.zeros(8),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"x_position": float(step), "y_position": 0.0},
            torso_tilt=math.pi / 2.0,
        )
    summary = metrics.summary()
    assert math.isclose(summary["inverted_step_fraction"], 1.0)
    assert summary["longest_inverted_run_steps"] == 20
    assert math.isclose(summary["longest_inverted_run_seconds"], 1.0)
    assert summary["sustained_inversion"] is True


def test_forward_path_efficiency_distinguishes_straight_and_lateral_motion() -> None:
    straight = EpisodeMetrics()
    straight.reset(initial_x=0.0, initial_y=0.0)
    straight.update(
        action=np.zeros(8),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"x_position": 1.0, "y_position": 0.0},
        torso_tilt=0.0,
    )
    diagonal = EpisodeMetrics()
    diagonal.reset(initial_x=0.0, initial_y=0.0)
    diagonal.update(
        action=np.zeros(8),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"x_position": 1.0, "y_position": 1.0},
        torso_tilt=0.0,
    )
    assert math.isclose(straight.summary()["forward_path_efficiency"], 1.0)
    assert diagonal.summary()["forward_path_efficiency"] < 1.0


def test_live_summary_avoids_history_distribution_metrics() -> None:
    metrics = EpisodeMetrics()
    metrics.reset(initial_x=0.0, initial_y=0.0)
    metrics.update(
        action=np.zeros(8),
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"x_position": 0.1, "y_position": 0.0},
        torso_tilt=0.2,
    )
    live = metrics.live_summary()
    assert live["net_forward_progress"] == 0.1
    assert "torso_tilt_p95" not in live
    assert "forward_path_efficiency" not in live


def test_csv_schema_has_required_diagnostic_columns() -> None:
    required = {
        "target_timesteps",
        "training_seed",
        "proxy_return",
        "base_proxy_return",
        "reward_shaping_sum",
        "reward_forward_shaping_sum",
        "reward_lateral_shaping_sum",
        "net_forward_progress",
        "control_effort_per_unit_distance",
        "condition_objective_return",
        "common_rescored_return",
        "cumulative_squared_action",
        "mean_squared_action_per_step",
        "unhealthy_termination",
        "termination_category",
        "low_z_termination",
        "high_z_termination",
        "torso_tilt_rms",
        "torso_tilt_p95",
        "fall",
        "lateral_drift_final_abs",
        "torso_tilt_std",
        "episode_length",
        "fixed_horizon_mean_forward_velocity",
        "inverted_step_fraction",
        "longest_inverted_run_seconds",
        "sustained_inversion",
        "orientation_shaping_function",
        "orientation_penalty_sum",
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
    }
    assert required.issubset(set(CSV_SCHEMA))


def test_checkpoint_targets_are_25_50_75_100_percent() -> None:
    assert checkpoint_targets(1000) == [
        (0.25, 250),
        (0.5, 500),
        (0.75, 750),
        (1.0, 1000),
    ]


def test_checkpoint_targets_accept_formal_six_checkpoint_schedule() -> None:
    expected_timesteps = [50_000, 100_000, 150_000, 200_000, 250_000, 300_000]
    targets = checkpoint_targets(300_000, expected_timesteps)
    assert [target for _, target in targets] == expected_timesteps
    assert targets[-1] == (1.0, 300_000)


def test_checkpoint_targets_reject_schedule_without_final_target() -> None:
    with np.testing.assert_raises(ValueError):
        checkpoint_targets(300_000, [50_000, 100_000])


def test_shaping_reward_is_separate_from_base_proxy_reward() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.25,
        condition_id="shaped",
        seed=4,
        lateral_drift_shaping_weight=0.1,
    )
    env.reset(seed=4)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    assert info["reward_shaping"] <= 0.0
    summary = env.episode_summary()
    assert math.isclose(summary["proxy_return"], reward)
    assert math.isclose(summary["base_proxy_return"], info["reward_base_proxy"])
    assert math.isclose(summary["reward_shaping_sum"], info["reward_shaping"])
    env.close()


def test_forward_progress_shaping_uses_logged_forward_reward() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.0625,
        condition_id="forward_shaped",
        seed=5,
        forward_progress_shaping_weight=0.5,
    )
    env.reset(seed=5)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert math.isclose(info["reward_forward_shaping"], 0.5 * info["reward_forward"])
    assert math.isclose(info["reward_lateral_shaping"], 0.0)
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    env.close()


def test_summary_accepts_boolean_values_round_tripped_through_csv() -> None:
    row = {
        "condition_id": "reference",
        "ctrl_cost_weight": "0.5",
        "lateral_drift_shaping_weight": "0.0",
        "checkpoint_fraction": "1.0",
        "target_timesteps": "300000",
        "proxy_return": "1.0",
        "base_proxy_return": "1.0",
        "reward_shaping_sum": "0.0",
        "reward_forward_shaping_sum": "0.0",
        "reward_lateral_shaping_sum": "0.0",
        "net_forward_progress": "0.5",
        "control_effort_per_unit_distance": "2.0",
        "fall": "False",
        "lateral_drift_final_abs": "0.1",
        "lateral_drift_mean_abs": "0.05",
        "torso_tilt_mean": "0.2",
        "torso_tilt_std": "0.01",
        "episode_length": "1000",
    }
    summary = summarise_evaluation([row])
    assert summary[0]["fall_mean"] == 0.0


def test_reset_with_same_seed_is_reproducible() -> None:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.5, condition_id="reference")
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    assert np.allclose(obs_a, obs_b)
    env.close()


def test_common_rescoring_is_independent_of_training_condition_weight() -> None:
    summaries = []
    action = np.ones(8)
    for condition_weight in (0.5, 0.0625):
        metrics = EpisodeMetrics(condition_ctrl_cost_weight=condition_weight)
        metrics.reset(initial_x=0.0, initial_y=0.0)
        base_reward = 2.0 + 1.0 - 0.1 - condition_weight * 8.0
        metrics.update(
            action=action,
            reward=base_reward,
            terminated=False,
            truncated=False,
            info={
                "reward_base_proxy": base_reward,
                "reward_forward": 2.0,
                "reward_survive": 1.0,
                "reward_contact": -0.1,
                "reward_ctrl": -condition_weight * 8.0,
                "x_position": 0.2,
                "y_position": 0.0,
            },
            torso_tilt=0.1,
            torso_height=0.5,
        )
        summaries.append(metrics.summary())
    assert summaries[0]["condition_objective_return"] != summaries[1]["condition_objective_return"]
    assert summaries[0]["common_rescored_return"] == summaries[1]["common_rescored_return"]
    assert summaries[0]["common_rescored_return"] == common_rescored_return(
        reward_forward_sum=2.0,
        reward_survive_sum=1.0,
        reward_contact_sum=-0.1,
        cumulative_squared_action=8.0,
        ctrl_cost_weight=0.5,
    )


def test_effort_ratio_uses_positive_locked_distance_threshold() -> None:
    metrics = EpisodeMetrics(effort_distance_min=0.1)
    metrics.reset(initial_x=0.0, initial_y=0.0)
    metrics.update(
        action=np.ones(8),
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"x_position": 0.05, "y_position": 0.0},
        torso_tilt=0.2,
        torso_height=0.5,
    )
    summary = metrics.summary()
    assert summary["effort_per_distance_defined"] is False
    assert math.isnan(summary["cumulative_squared_action_per_unit_distance"])
    assert summary["mean_squared_action_per_step"] == 8.0
    assert summary["action_saturation_rate"] == 1.0


def test_action_change_excludes_first_action_and_measures_transitions() -> None:
    metrics = EpisodeMetrics()
    metrics.reset(initial_x=0.0, initial_y=0.0)
    for action in (np.zeros(8), np.zeros(8), np.ones(8)):
        metrics.update(
            action=action,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"x_position": 0.0, "y_position": 0.0},
            torso_tilt=0.0,
        )
    summary = metrics.summary()
    assert summary["action_change_transition_count"] == 2
    assert summary["cumulative_squared_action_change"] == 8.0
    assert summary["mean_squared_action_change_per_transition"] == 4.0


def test_bounded_lateral_and_orientation_terms_are_separate_from_control_cost() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.25,
        condition_id="bounded_combined",
        seed=6,
        lateral_drift_shaping_weight=0.2,
        lateral_drift_shaping_scale=1.0,
        orientation_shaping_weight=0.3,
        orientation_shaping_scale=0.5,
    )
    env.reset(seed=6)
    _, reward, _, _, info = env.step(np.ones(env.action_space.shape) * 0.5)
    assert env.unwrapped._ctrl_cost_weight == 0.25
    assert -0.2 <= info["reward_lateral_shaping"] <= 0.0
    assert -0.3 <= info["reward_orientation_shaping"] <= 0.0
    assert info["reward_effort_shaping"] == 0.0
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    env.close()


def test_velocity_lateral_penalty_is_bounded_symmetric_and_zero_at_command() -> None:
    at_target = lateral_penalty_value(
        lateral_offset=9.0,
        lateral_velocity=0.0,
        velocity_target=0.0,
        signal="velocity_tanh_squared",
        scale=1.0,
    )
    positive = lateral_penalty_value(
        lateral_offset=0.0,
        lateral_velocity=1.0,
        velocity_target=0.0,
        signal="velocity_tanh_squared",
        scale=1.0,
    )
    negative = lateral_penalty_value(
        lateral_offset=0.0,
        lateral_velocity=-1.0,
        velocity_target=0.0,
        signal="velocity_tanh_squared",
        scale=1.0,
    )
    assert at_target == 0.0
    assert 0.0 < positive < 1.0
    assert math.isclose(positive, negative)


def test_velocity_lateral_reward_uses_observed_y_velocity_and_is_logged(
    tmp_path: Path,
) -> None:
    step_log = tmp_path / "velocity_lateral.csv.gz"
    env = make_proxygap_ant_env(
        condition_id="velocity_lateral",
        lateral_drift_shaping_weight=0.1,
        lateral_shaping_signal="velocity_tanh_squared",
        lateral_velocity_target=0.0,
        step_log_path=step_log,
    )
    env.reset(seed=816)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    expected_penalty = math.tanh(float(info["y_velocity"]) ** 2)
    assert math.isclose(info["lateral_penalty"], expected_penalty)
    assert math.isclose(info["reward_lateral_shaping"], -0.1 * expected_penalty)
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    env.close()
    with gzip.open(step_log, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["lateral_shaping_signal"] == "velocity_tanh_squared"
    assert math.isclose(float(rows[0]["lateral_velocity"]), float(info["y_velocity"]))
    assert math.isclose(float(rows[0]["lateral_penalty_step"]), expected_penalty)


def _force_termination_height(height: float, velocity: float) -> dict[str, object]:
    env = make_proxygap_ant_env(ctrl_cost_weight=0.5, condition_id="termination_test")
    env.reset(seed=7)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    qpos[2] = height
    qvel[2] = velocity
    env.unwrapped.set_state(qpos, qvel)
    _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
    summary = env.episode_summary()
    env.close()
    assert terminated is True
    return summary


def test_low_z_termination_is_classified_as_collapse() -> None:
    summary = _force_termination_height(-1.0, -10.0)
    assert summary["termination_category"] == "low_z_collapse"
    assert summary["low_z_termination"] is True
    assert summary["high_z_termination"] is False


def test_high_z_termination_is_classified_as_excursion() -> None:
    summary = _force_termination_height(1.2, 2.0)
    assert summary["termination_category"] == "high_z_excursion"
    assert summary["high_z_termination"] is True
    assert summary["low_z_termination"] is False


def test_gzip_step_log_contains_reconstructable_fields(tmp_path) -> None:
    path = tmp_path / "episode.csv.gz"
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="logged",
        step_log_path=path,
    )
    env.reset(seed=8)
    env.step(np.zeros(env.action_space.shape))
    env.close()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["common_rescored_reward_step"] != ""
    assert rows[0]["action_change_defined_step"] == "False"
    assert rows[0]["torso_height"] != ""
    assert rows[0]["termination_category"] == "none"
    assert len(json.loads(rows[0]["foot_contact_mask_step"])) == 4
    assert len(json.loads(rows[0]["foot_normal_forces_n_step"])) == 4
    assert len(json.loads(rows[0]["actuator_joint_torques_n_m_step"])) == 8


def test_prospective_ppo_config_requires_all_resolved_parameters() -> None:
    resolved = resolve_ppo_config(DEFAULT_PPO_CONFIG, require_complete=True)
    assert resolved["gamma"] == 0.99
    incomplete = dict(DEFAULT_PPO_CONFIG)
    incomplete.pop("gamma")
    with np.testing.assert_raises(ValueError):
        resolve_ppo_config(incomplete, require_complete=True)


def test_representative_video_rule_is_deterministic() -> None:
    rows = [
        {"condition_id": "c", "training_seed": 1, "seed": 10, "target_timesteps": 300, "net_forward_progress": 1.0},
        {"condition_id": "c", "training_seed": 1, "seed": 11, "target_timesteps": 300, "net_forward_progress": 3.0},
        {"condition_id": "c", "training_seed": 2, "seed": 10, "target_timesteps": 300, "net_forward_progress": 4.0},
        {"condition_id": "c", "training_seed": 2, "seed": 11, "target_timesteps": 300, "net_forward_progress": 6.0},
        {"condition_id": "c", "training_seed": 3, "seed": 10, "target_timesteps": 300, "net_forward_progress": 8.0},
        {"condition_id": "c", "training_seed": 3, "seed": 11, "target_timesteps": 300, "net_forward_progress": 10.0},
    ]
    selected = select_representative_evaluation_seed(rows, final_target_timesteps=300)
    assert selected["training_seed"] == 2
    assert selected["evaluation_seed"] == 10


def test_representative_video_rule_handles_two_seed_float_tie() -> None:
    rows = [
        {
            "condition_id": "c",
            "training_seed": 41101,
            "seed": 51101,
            "target_timesteps": 300000,
            "net_forward_progress": 15.89582734186817,
        },
        {
            "condition_id": "c",
            "training_seed": 41102,
            "seed": 51101,
            "target_timesteps": 300000,
            "net_forward_progress": 4.28982734186817,
        },
    ]
    selected = select_representative_evaluation_seed(
        rows,
        final_target_timesteps=300000,
    )
    assert selected["training_seed"] == 41101
    assert selected["evaluation_seed"] == 51101


def test_revision_gate_config_reports_only_explicit_unresolved_decisions() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "prospective_v2_revision_gate_20260810.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (
        config["proposal"]["distribution"]
        == "external_controlling_source_not_distributed"
    )
    status = protocol_freeze_status(config)
    codes = {item["code"] for item in status["blockers"]}
    assert status["status"] == "blocked"
    assert codes == {
        "ANALYSIS_ROUTE_UNDECIDED",
        "ATTRIBUTION_SCOPE_UNDECIDED",
        "EFFORT_DISTANCE_MIN_UNLOCKED",
        "INTERVENTION_PARAMETER_UNLOCKED",
    }
