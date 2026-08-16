from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyse_stage1_budget_extension.py"
)
SPEC = importlib.util.spec_from_file_location("budget_extension_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def episode_row(
    weight: float,
    seed: int,
    checkpoint: int,
    *,
    forward: float,
    effort: float,
    drift: float,
    progress: float,
) -> dict:
    return {
        "ctrl_cost_weight": weight,
        "training_seed": seed,
        "target_timesteps": checkpoint,
        "reward_forward_sum": forward,
        "reward_survive_sum": 100.0,
        "reward_contact_sum": -1.0,
        "cumulative_squared_action": effort,
        "net_forward_progress": progress,
        "mean_forward_velocity": progress / 50.0,
        "forward_path_efficiency": 0.8,
        "unhealthy_termination": 0.0,
        "episode_length": 1000.0,
        "episode_duration_seconds": 50.0,
        "lateral_drift_mean_abs": drift,
        "lateral_drift_final_abs": drift,
        "cumulative_lateral_path": drift * 2.0,
        "cumulative_planar_path": 10.0,
        "torso_tilt_rms": 0.1,
        "action_saturation_rate": 0.01,
        "normalised_action_roughness": 0.02,
    }


def test_expected_actual_timesteps_respects_rollout_alignment() -> None:
    assert MODULE.expected_actual_timesteps(
        301056, [500000, 750000, 1000000], 2048
    ) == {500000: 501760, 750000: 751616, 1000000: 1001472}


def test_reward_reconciliation_tolerance_matches_existing_metric_contract() -> None:
    assert MODULE.REWARD_RECONCILIATION_TOLERANCE == 1e-3


def test_matched_proxy_uses_candidate_weight_for_both_policies() -> None:
    rows = []
    for seed in [41101, 41102]:
        rows.append(
            episode_row(
                0.5,
                seed,
                1000000,
                forward=20.0,
                effort=20.0,
                drift=0.2,
                progress=5.0,
            )
        )
        rows.append(
            episode_row(
                0.21875,
                seed,
                1000000,
                forward=22.0,
                effort=20.0,
                drift=0.8,
                progress=5.0,
            )
        )
    contrasts = MODULE.paired_candidate_contrasts(
        pd.DataFrame(rows), candidate_weights=[0.21875]
    )
    assert list(contrasts["delta_matched_proxy_return"]) == pytest.approx([2.0, 2.0])
    assert list(contrasts["delta_lateral_drift_mean_abs"]) == pytest.approx(
        [0.6, 0.6]
    )


def test_reference_competence_requires_each_training_seed_to_pass_both() -> None:
    rows = pd.DataFrame(
        [
            {
                "ctrl_cost_weight": 0.5,
                "training_seed": 41101,
                "target_timesteps": 1000000,
                "unhealthy_termination": 0.0,
                "mean_forward_velocity": 0.2,
                "net_forward_progress": 10.0,
            },
            {
                "ctrl_cost_weight": 0.5,
                "training_seed": 41102,
                "target_timesteps": 1000000,
                "unhealthy_termination": 0.3,
                "mean_forward_velocity": 0.2,
                "net_forward_progress": 10.0,
            },
        ]
    )
    competence = MODULE.reference_competence(rows)
    assert list(competence["joint_competence_gate_pass"]) == [True, False]
