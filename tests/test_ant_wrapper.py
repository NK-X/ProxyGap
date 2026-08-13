from __future__ import annotations

import math
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from proxygap import (
    CSV_SCHEMA,
    DEFAULT_PPO_CONFIG,
    EpisodeMetrics,
    common_rescored_return,
    make_proxygap_ant_env,
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
        "control_effort",
        "control_effort_per_unit_distance",
        "condition_objective_return",
        "common_rescored_return",
        "cumulative_squared_action",
        "mean_squared_action_per_step",
        "action_saturation_rate",
        "unhealthy_termination",
        "termination_category",
        "low_z_termination",
        "high_z_termination",
        "lateral_drift_max_abs",
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
    env.close()


def test_quaternion_tilt_angle_for_identity_orientation() -> None:
    assert quaternion_tilt_angle(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0


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


def test_bounded_effort_and_orientation_terms_are_separate() -> None:
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.0625,
        condition_id="bounded_combined",
        seed=6,
        effort_shaping_weight=0.2,
        effort_shaping_scale=2.0,
        stability_shaping_weight=0.3,
        stability_shaping_scale=0.5,
    )
    env.reset(seed=6)
    _, reward, _, _, info = env.step(np.ones(env.action_space.shape) * 0.5)
    assert -0.2 <= info["reward_effort_shaping"] <= 0.0
    assert -0.3 <= info["reward_stability_shaping"] <= 0.0
    assert math.isclose(reward, info["reward_base_proxy"] + info["reward_shaping"])
    env.close()


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
    assert rows[0]["torso_height"] != ""
    assert rows[0]["termination_category"] == "none"


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


def test_revision_gate_config_reports_only_explicit_unresolved_decisions() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "prospective_v2_revision_gate_20260810.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = protocol_freeze_status(config)
    codes = {item["code"] for item in status["blockers"]}
    assert status["status"] == "blocked"
    assert codes == {
        "ANALYSIS_ROUTE_UNDECIDED",
        "ATTRIBUTION_SCOPE_UNDECIDED",
        "EFFORT_DISTANCE_MIN_UNLOCKED",
        "INTERVENTION_PARAMETER_UNLOCKED",
    }
