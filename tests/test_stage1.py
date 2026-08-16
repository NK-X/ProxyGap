from __future__ import annotations

from proxygap.stage1 import screen_stage1_endpoint, validate_stage1_config


def row(
    weight: float,
    training_seed: int,
    evaluation_seed: int,
    *,
    forward: float,
    effort: float,
    progress: float = 5.0,
    path_efficiency: float = 0.8,
    saturation: float = 0.01,
    roughness: float = 0.01,
) -> dict:
    return {
        "ctrl_cost_weight": weight,
        "training_seed": training_seed,
        "target_timesteps": 300000,
        "seed": evaluation_seed,
        "reward_forward_sum": forward,
        "reward_survive_sum": 10.0,
        "reward_contact_sum": 0.0,
        "cumulative_squared_action": effort,
        "net_forward_progress": progress,
        "forward_path_efficiency": path_efficiency,
        "unhealthy_termination": False,
        "lateral_drift_mean_abs": 0.2,
        "torso_tilt_rms": 0.1,
        "action_saturation_rate": saturation,
        "normalised_action_roughness": roughness,
    }


def paired_rows(*, tiny_harm: bool = False, incomplete_command: bool = False):
    rows = []
    for training_seed in [11, 12]:
        for evaluation_seed in [101, 102]:
            rows.append(
                row(0.5, training_seed, evaluation_seed, forward=10, effort=10)
            )
            path = 0.799 if tiny_harm else 0.65
            saturation = 0.05
            roughness = 0.015 if incomplete_command else 0.05
            rows.append(
                row(
                    0.25,
                    training_seed,
                    evaluation_seed,
                    forward=14,
                    effort=10,
                    path_efficiency=path,
                    saturation=saturation,
                    roughness=roughness,
                )
            )
    return rows


def test_stage1_uses_fixed_candidate_formula_and_practical_margins() -> None:
    screen = screen_stage1_endpoint(paired_rows(), checkpoint=300000)[0]
    assert screen.positive_proxy_seed_count == 2
    assert screen.consistently_harmed_domains == (
        "locomotion_effectiveness",
        "command_quality",
    )
    assert screen.consistently_harmed_metrics_by_domain == {
        "locomotion_effectiveness": ("forward_path_efficiency",),
        "command_quality": (
            "action_saturation_rate",
            "normalised_action_roughness",
        ),
    }
    assert screen.strong_development_candidate is True


def test_tiny_numeric_change_is_not_called_harm() -> None:
    screen = screen_stage1_endpoint(
        paired_rows(tiny_harm=True, incomplete_command=True),
        checkpoint=300000,
    )[0]
    assert screen.consistently_harmed_domains == ()
    assert screen.strong_development_candidate is False


def test_command_quality_requires_both_prespecified_indicators() -> None:
    screen = screen_stage1_endpoint(
        paired_rows(incomplete_command=True),
        checkpoint=300000,
    )[0]
    assert "locomotion_effectiveness" in screen.consistently_harmed_domains
    assert "command_quality" not in screen.consistently_harmed_domains


def test_unpaired_evaluation_seeds_are_rejected() -> None:
    rows = paired_rows()
    rows[-1]["seed"] = 999
    try:
        screen_stage1_endpoint(rows, checkpoint=300000)
    except ValueError as error:
        assert "unpaired evaluation seeds" in str(error)
    else:
        raise AssertionError("Unpaired evaluation episodes must be rejected")


def test_different_harmed_metrics_do_not_fake_cross_seed_consistency() -> None:
    rows = paired_rows(tiny_harm=True, incomplete_command=True)
    for item in rows:
        if item["ctrl_cost_weight"] != 0.25:
            continue
        if item["training_seed"] == 11:
            item["net_forward_progress"] = 3.5
            item["forward_path_efficiency"] = 0.799
        else:
            item["net_forward_progress"] = 5.0
            item["forward_path_efficiency"] = 0.65
    screen = screen_stage1_endpoint(rows, checkpoint=300000)[0]
    assert "locomotion_effectiveness" not in screen.consistently_harmed_domains
    assert "locomotion_effectiveness" not in (
        screen.consistently_harmed_metrics_by_domain
    )


def test_stage1_config_rejects_shaping_and_seed_leakage() -> None:
    config = {
        "environment": "Ant-v5",
        "algorithm": "PPO",
        "stage_scope": "stage_1_detection_only",
        "reward": {"lateral_drift_shaping_weight": 1.0},
        "development": {
            "ordered_analysis_grid": [0.5, 0.25, 0.125],
            "new_dense_weights": [0.25],
            "training_seeds": [11],
            "checkpoint_timesteps": [50000, 100000, 150000, 200000, 250000, 300000],
        },
        "formal_confirmation": {"training_seeds": [11]},
    }
    errors = validate_stage1_config(config)
    assert "All shaping weights must be zero in stage one" in errors
    assert "Development and formal training seeds must be disjoint" in errors


def valid_bidirectional_config() -> dict:
    return {
        "environment": "Ant-v5",
        "algorithm": "PPO",
        "stage_scope": "stage_1_detection_only",
        "reward": {
            "forward_progress_shaping_weight": 0.0,
            "lateral_drift_shaping_weight": 0.0,
            "effort_shaping_weight": 0.0,
            "orientation_shaping_weight": 0.0,
        },
        "development": {
            "directionality": "bidirectional",
            "ordered_analysis_grid": [0.75, 0.625, 0.5, 0.25, 0.125],
            "new_upper_weights": [0.625, 0.75],
            "training_seeds": [11, 12],
            "checkpoint_timesteps": [
                50000,
                100000,
                150000,
                200000,
                250000,
                300000,
            ],
        },
        "development_screen": {
            "proxy_relative_noninferiority_margin": 0.05,
        },
        "formal_confirmation": {"training_seeds": [21, 22]},
    }


def test_bidirectional_config_accepts_weights_on_both_sides() -> None:
    assert validate_stage1_config(valid_bidirectional_config()) == []


def test_bidirectional_config_rejects_a_one_sided_grid() -> None:
    config = valid_bidirectional_config()
    config["development"]["ordered_analysis_grid"] = [0.5, 0.25, 0.125]
    config["development"]["new_upper_weights"] = []
    errors = validate_stage1_config(config)
    assert "A bidirectional grid requires weights below and above 0.5" in errors


def test_proxy_noninferiority_is_distinct_from_strict_gain() -> None:
    rows = paired_rows()
    for item in rows:
        if item["ctrl_cost_weight"] == 0.25:
            # Under R_0.25 this makes the candidate 0.5 points worse than the
            # reference, which is inside a 5% margin around reference R=17.5.
            item["reward_forward_sum"] = 9.5
    screen = screen_stage1_endpoint(
        rows,
        checkpoint=300000,
        proxy_relative_noninferiority_margin=0.05,
    )[0]
    assert screen.positive_proxy_seed_count == 0
    assert screen.noninferior_proxy_seed_count == 2
    assert screen.strong_development_candidate is False
    assert screen.noninferior_development_candidate is True


def test_stage1_config_rejects_excessive_proxy_margin() -> None:
    config = valid_bidirectional_config()
    config["development_screen"]["proxy_relative_noninferiority_margin"] = 0.25
    errors = validate_stage1_config(config)
    assert "proxy_relative_noninferiority_margin must lie in [0, 0.20]" in errors
