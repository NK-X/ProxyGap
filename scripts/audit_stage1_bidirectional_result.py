"""Audit the bidirectional development result without changing its primary gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_root", required=True, type=Path)
    parser.add_argument("--reference_weight", type=float, default=0.5)
    parser.add_argument("--candidate_weight", type=float, default=0.21875)
    parser.add_argument("--endpoint", type=int, default=300_000)
    return parser.parse_args()


def policy_summary(rows: pd.DataFrame, weight: float) -> pd.DataFrame:
    selected = rows[np.isclose(rows["ctrl_cost_weight"].astype(float), weight)].copy()
    if selected.empty:
        raise ValueError(f"No endpoint rows found for weight {weight}")
    selected["matched_proxy_return"] = (
        selected["reward_forward_sum"]
        + selected["reward_survive_sum"]
        + selected["reward_contact_sum"]
        - weight * selected["cumulative_squared_action"]
    )
    fields = [
        "matched_proxy_return",
        "net_forward_progress",
        "mean_forward_velocity",
        "forward_path_efficiency",
        "unhealthy_termination",
        "episode_length",
        "episode_duration_seconds",
        "lateral_drift_mean_abs",
        "lateral_drift_final_abs",
        "cumulative_lateral_path",
        "cumulative_planar_path",
        "torso_tilt_rms",
        "cumulative_squared_action",
        "action_saturation_rate",
        "normalised_action_roughness",
    ]
    summary = selected.groupby("training_seed", as_index=False)[fields].mean()
    summary["ctrl_cost_weight"] = weight
    summary["final_lateral_per_abs_forward"] = summary[
        "lateral_drift_final_abs"
    ] / summary["net_forward_progress"].abs().clip(lower=1e-8)
    summary["lateral_path_fraction"] = summary[
        "cumulative_lateral_path"
    ] / summary["cumulative_planar_path"].clip(lower=1e-8)
    return summary


def paired_audit(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if set(reference["training_seed"]) != set(candidate["training_seed"]):
        raise ValueError("Reference and candidate training seeds are not paired")
    merged = candidate.merge(
        reference,
        on="training_seed",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    difference_fields = [
        "matched_proxy_return",
        "net_forward_progress",
        "mean_forward_velocity",
        "forward_path_efficiency",
        "unhealthy_termination",
        "episode_length",
        "episode_duration_seconds",
        "lateral_drift_mean_abs",
        "lateral_drift_final_abs",
        "final_lateral_per_abs_forward",
        "lateral_path_fraction",
        "torso_tilt_rms",
        "cumulative_squared_action",
        "action_saturation_rate",
        "normalised_action_roughness",
    ]
    for field in difference_fields:
        merged[f"delta_{field}"] = (
            merged[f"{field}_candidate"] - merged[f"{field}_reference"]
        )
    return merged


def screen_for_weight(result: dict[str, Any], weight: float) -> dict[str, Any]:
    for screen in result["endpoint_screens"]:
        if np.isclose(float(screen["candidate_weight"]), weight):
            return screen
    raise ValueError(f"No endpoint screen found for weight {weight}")


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    metrics = pd.read_csv(root / "combined_episode_metrics.csv")
    result = json.loads((root / "stage1_development_result.json").read_text())
    endpoint_rows = metrics[metrics["target_timesteps"].astype(int) == args.endpoint]

    reference = policy_summary(endpoint_rows, args.reference_weight)
    candidate = policy_summary(endpoint_rows, args.candidate_weight)
    paired = paired_audit(reference, candidate)
    reference.to_csv(root / "reference_competence_audit.csv", index=False)
    paired.to_csv(root / "candidate_alternative_explanation_audit.csv", index=False)

    candidate_screen = screen_for_weight(result, args.candidate_weight)
    diagnostic_sensitivity = pd.read_csv(root / "margin_sensitivity.csv")
    diagnostic_sensitivity = diagnostic_sensitivity[
        np.isclose(
            diagnostic_sensitivity["candidate_weight"].astype(float),
            args.candidate_weight,
        )
    ]
    proxy_sensitivity = pd.read_csv(root / "proxy_margin_sensitivity.csv")
    proxy_sensitivity = proxy_sensitivity[
        np.isclose(
            proxy_sensitivity["candidate_weight"].astype(float),
            args.candidate_weight,
        )
    ]
    persistence = pd.read_csv(root / "late_checkpoint_persistence.csv")
    persistence_row = persistence[
        np.isclose(persistence["candidate_weight"].astype(float), args.candidate_weight)
    ].iloc[0]

    gym_threshold = gym.spec("Ant-v5").reward_threshold
    reference_majority_unhealthy = bool(
        (reference["unhealthy_termination"] > 0.5).all()
    )
    candidate_longer_both_seeds = bool(
        (paired["delta_episode_duration_seconds"] > 0).all()
    )
    absolute_lateral_harm_both_seeds = bool(
        (paired["delta_lateral_drift_mean_abs"] >= 0.5).all()
    )
    normalised_final_lateral_harm_both_seeds = bool(
        (paired["delta_final_lateral_per_abs_forward"] > 0).all()
    )
    lateral_path_fraction_harm_both_seeds = bool(
        (paired["delta_lateral_path_fraction"] > 0).all()
    )
    path_efficiency_harm_both_seeds = bool(
        (paired["delta_forward_path_efficiency"] <= -0.10).all()
    )

    audit = {
        "status": "development_candidate_nominated_formal_protocol_blocked",
        "endpoint": args.endpoint,
        "reference_weight": args.reference_weight,
        "candidate_weight": args.candidate_weight,
        "predeclared_primary_screen": {
            "strict_proxy_gain_in_both_seeds": (
                int(candidate_screen["positive_proxy_seed_count"])
                == len(candidate_screen["paired_training_seeds"])
            ),
            "noninferior_proxy_in_both_seeds": (
                int(candidate_screen["noninferior_proxy_seed_count"])
                == len(candidate_screen["paired_training_seeds"])
            ),
            "consistently_harmed_domains": candidate_screen[
                "consistently_harmed_domains"
            ],
            "strong_development_candidate": candidate_screen[
                "strong_development_candidate"
            ],
            "noninferior_development_candidate": candidate_screen[
                "noninferior_development_candidate"
            ],
            "late_checkpoint_count": int(
                persistence_row["late_checkpoints_strong_count"]
            ),
            "late_window_persistent": bool(persistence_row["late_window_persistent"]),
        },
        "sensitivity": {
            "diagnostic_margin_pass_by_scale": {
                str(float(row.margin_scale)): bool(row.primary_gate_candidate)
                for row in diagnostic_sensitivity.itertuples(index=False)
            },
            "proxy_margin_pass_by_relative_margin": {
                str(float(row.proxy_relative_noninferiority_margin)): bool(
                    row.noninferior_development_candidate
                )
                for row in proxy_sensitivity.itertuples(index=False)
            },
        },
        "reference_competence_audit": {
            "registered_ant_v5_reward_threshold": gym_threshold,
            "threshold_role": (
                "Context only; it was not frozen as the project's competence gate."
            ),
            "majority_unhealthy_in_each_development_seed": (
                reference_majority_unhealthy
            ),
            "policy_rows": reference.to_dict("records"),
        },
        "alternative_explanation_audit": {
            "candidate_episode_duration_higher_in_both_seeds": (
                candidate_longer_both_seeds
            ),
            "absolute_lateral_margin_crossed_in_both_seeds": (
                absolute_lateral_harm_both_seeds
            ),
            "final_lateral_per_abs_forward_worse_in_both_seeds": (
                normalised_final_lateral_harm_both_seeds
            ),
            "lateral_path_fraction_worse_in_both_seeds": (
                lateral_path_fraction_harm_both_seeds
            ),
            "path_efficiency_margin_crossed_in_both_seeds": (
                path_efficiency_harm_both_seeds
            ),
            "interpretation": (
                "The predeclared absolute-drift screen is positive, but the candidate "
                "also runs longer and the two normalised lateral checks do not worsen "
                "in both seeds. A corridor-following interpretation must therefore be "
                "frozen before formal confirmation; otherwise exposure time and travel "
                "distance remain plausible alternative explanations."
            ),
        },
        "adjudication": {
            "development_nomination": "retain",
            "formal_launch": "blocked",
            "blockers": [
                "reference_minimum_competence_rule_not_frozen",
                "reference_policy_is_majority_unhealthy_in_both_development_seeds",
                "absolute_lateral_drift_construct_requires_corridor_intent_decision",
                "300k_budget_is_below_the_one_million_step_PPO_MuJoCo_benchmark_scale",
                "formal_condition_matrix_and_seed_gate_not_yet_frozen",
            ],
            "claim_boundary": (
                "The result is a two-seed development nomination. It is not held-out "
                "confirmation of high reward and low overall performance."
            ),
        },
    }
    (root / "stage1_bidirectional_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
