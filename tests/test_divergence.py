from __future__ import annotations

from proxygap.divergence import (
    choose_minimal_departure_candidate,
    screen_pairwise_fixed_proxy,
    screen_divergence_candidates,
)


def rows_for_policy(weight: float, seed: int, early_reward: float, late_reward: float):
    rows = []
    for target, reward, drift, tilt in [
        (50, early_reward, 0.2, 0.1),
        (100, early_reward, 0.2, 0.1),
        (250, late_reward, 0.8, 0.3),
        (300, late_reward, 0.8, 0.3),
    ]:
        for episode in range(2):
            rows.append(
                {
                    "episode": episode,
                    "ctrl_cost_weight": weight,
                    "training_seed": seed,
                    "target_timesteps": target,
                    "condition_objective_return": reward,
                    "forward_path_efficiency": 0.9 if target < 200 else 0.6,
                    "lateral_drift_mean_abs": drift,
                    "torso_tilt_rms": tilt,
                    "unhealthy_termination": 0.0,
                    "action_saturation_rate": 0.1,
                }
            )
    return rows


def test_candidate_requires_reward_gain_and_consistent_external_harm() -> None:
    rows = rows_for_policy(0.25, 11, 100.0, 120.0)
    rows += rows_for_policy(0.25, 12, 90.0, 130.0)
    result = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
    )[0]
    assert result.candidate_for_confirmation is True
    assert result.consistently_worsened_diagnostics == (
        "lateral_drift_mean_abs",
        "negative_forward_path_efficiency",
        "torso_tilt_rms",
    )


def test_candidate_rejected_when_reward_does_not_increase() -> None:
    rows = rows_for_policy(0.25, 11, 120.0, 100.0)
    rows += rows_for_policy(0.25, 12, 130.0, 90.0)
    result = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
    )[0]
    assert result.candidate_for_confirmation is False


def test_returns_are_never_compared_across_weights() -> None:
    rows = rows_for_policy(0.5, 11, 100.0, 110.0)
    rows += rows_for_policy(0.25, 11, 1000.0, 900.0)
    result = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
        minimum_consistent_seeds=1,
    )
    by_weight = {item.ctrl_cost_weight: item for item in result}
    assert by_weight[0.5].reward_gain_seed_count == 1
    assert by_weight[0.25].reward_gain_seed_count == 0


def test_missing_checkpoint_is_rejected() -> None:
    rows = rows_for_policy(0.25, 11, 100.0, 120.0)
    rows = [row for row in rows if row["target_timesteps"] != 300]
    try:
        screen_divergence_candidates(
            rows,
            early_checkpoints=[50, 100],
            late_checkpoints=[250, 300],
            minimum_consistent_seeds=1,
        )
    except ValueError as error:
        assert "Missing checkpoint" in str(error)
    else:
        raise AssertionError("Missing checkpoint should fail screening")


def test_multiple_candidates_choose_smallest_departure_from_default() -> None:
    rows = []
    for weight in [0.375, 0.25, 0.125]:
        rows += rows_for_policy(weight, 11, 100.0, 120.0)
        rows += rows_for_policy(weight, 12, 100.0, 120.0)
    screens = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
    )
    assert choose_minimal_departure_candidate(
        screens,
        eligible_reduced_weights=[0.375, 0.25, 0.125],
    ) == 0.375


def test_no_qualifying_candidate_returns_stop_signal() -> None:
    rows = rows_for_policy(0.25, 11, 120.0, 100.0)
    rows += rows_for_policy(0.25, 12, 120.0, 100.0)
    screens = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
    )
    assert choose_minimal_departure_candidate(
        screens,
        eligible_reduced_weights=[0.25],
    ) is None


def test_csv_boolean_strings_are_converted_explicitly() -> None:
    rows = rows_for_policy(0.25, 11, 100.0, 120.0)
    rows += rows_for_policy(0.25, 12, 100.0, 120.0)
    for row in rows:
        row["unhealthy_termination"] = "False"
    result = screen_divergence_candidates(
        rows,
        early_checkpoints=[50, 100],
        late_checkpoints=[250, 300],
    )[0]
    assert result.candidate_for_confirmation is True


def pairwise_rows(weight: float, seed: int, *, proxy_components: tuple[float, float, float, float], path_efficiency: float, tilt: float):
    forward, survive, contact, effort = proxy_components
    return [
        {
            "ctrl_cost_weight": weight,
            "training_seed": seed,
            "target_timesteps": 100,
            "reward_forward_sum": forward,
            "reward_survive_sum": survive,
            "reward_contact_sum": contact,
            "cumulative_squared_action": effort,
            "forward_path_efficiency": path_efficiency,
            "lateral_drift_mean_abs": 0.2,
            "torso_tilt_rms": tilt,
            "unhealthy_termination": False,
            "action_saturation_rate": 0.1,
        }
    ]


def test_pairwise_screen_uses_candidate_formula_for_both_policies() -> None:
    rows = []
    for seed in [11, 12]:
        rows += pairwise_rows(0.5, seed, proxy_components=(10, 10, 0, 10), path_efficiency=0.9, tilt=0.1)
        rows += pairwise_rows(0.25, seed, proxy_components=(15, 10, 0, 10), path_efficiency=0.6, tilt=0.3)
    screen = screen_pairwise_fixed_proxy(rows, checkpoint=100)[0]
    assert screen.candidate_for_confirmation is True
    assert screen.positive_proxy_seed_count == 2
    assert screen.consistently_worsened_diagnostics == (
        "negative_forward_path_efficiency",
        "torso_tilt_rms",
    )


def test_pairwise_screen_rejects_unpaired_training_seeds() -> None:
    rows = pairwise_rows(0.5, 11, proxy_components=(10, 10, 0, 10), path_efficiency=0.9, tilt=0.1)
    rows += pairwise_rows(0.25, 12, proxy_components=(15, 10, 0, 10), path_efficiency=0.6, tilt=0.3)
    try:
        screen_pairwise_fixed_proxy(rows, checkpoint=100)
    except ValueError as error:
        assert "identical training seeds" in str(error)
    else:
        raise AssertionError("Unpaired seed sets must be rejected")
