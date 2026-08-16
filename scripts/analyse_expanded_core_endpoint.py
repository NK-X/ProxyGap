"""Analyse the post-hoc 300k expansion without promoting it to confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


DOMAIN_METRICS = {
    "path_efficiency": ("negative_forward_path_efficiency",),
    "lateral_control": ("lateral_drift_mean_abs",),
    "posture": ("torso_tilt_rms",),
    "safety": ("unhealthy_termination",),
    "command_quality": (
        "action_saturation_rate",
        "mean_squared_action_change_per_transition",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", type=int, default=300_000)
    return parser.parse_args()


def read_rows(path: Path, checkpoint: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if int(row["target_timesteps"]) == checkpoint
        ]


def numeric(value: str) -> float:
    """Parse numeric and CSV boolean fields using one explicit convention."""
    lowered = value.strip().lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    return float(value)


def mean(rows: list[dict[str, str]], field: str) -> float:
    return float(fmean(numeric(row[field]) for row in rows))


def proxy(rows: list[dict[str, str]], weight: float) -> float:
    return float(
        mean(rows, "reward_forward_sum")
        + mean(rows, "reward_survive_sum")
        + mean(rows, "reward_contact_sum")
        - weight * mean(rows, "cumulative_squared_action")
    )


def main() -> None:
    args = parse_args()
    candidate_rows = read_rows(Path(args.candidate_csv), args.checkpoint)
    reference_rows = read_rows(Path(args.reference_csv), args.checkpoint)
    seeds = sorted({int(row["training_seed"]) for row in reference_rows})
    weights = sorted({float(row["ctrl_cost_weight"]) for row in candidate_rows})
    screens = []
    for weight in weights:
        contrasts = []
        for seed in seeds:
            reference = [
                row for row in reference_rows if int(row["training_seed"]) == seed
            ]
            candidate = [
                row
                for row in candidate_rows
                if int(row["training_seed"]) == seed
                and float(row["ctrl_cost_weight"]) == weight
            ]
            if not reference or not candidate:
                raise ValueError(f"Missing paired cell for weight={weight}, seed={seed}")
            deltas = {
                "negative_forward_path_efficiency": -mean(
                    candidate, "forward_path_efficiency"
                )
                + mean(reference, "forward_path_efficiency"),
                "lateral_drift_mean_abs": mean(candidate, "lateral_drift_mean_abs")
                - mean(reference, "lateral_drift_mean_abs"),
                "torso_tilt_rms": mean(candidate, "torso_tilt_rms")
                - mean(reference, "torso_tilt_rms"),
                "unhealthy_termination": mean(candidate, "unhealthy_termination")
                - mean(reference, "unhealthy_termination"),
                "action_saturation_rate": mean(candidate, "action_saturation_rate")
                - mean(reference, "action_saturation_rate"),
                "mean_squared_action_change_per_transition": mean(
                    candidate, "mean_squared_action_change_per_transition"
                )
                - mean(reference, "mean_squared_action_change_per_transition"),
            }
            domain_harms = {
                domain: all(deltas[metric] > 0 for metric in metrics)
                for domain, metrics in DOMAIN_METRICS.items()
            }
            contrasts.append(
                {
                    "training_seed": seed,
                    "candidate_proxy_advantage_under_fixed_R_w": proxy(candidate, weight)
                    - proxy(reference, weight),
                    "metric_deltas_candidate_minus_reference_higher_is_worse": deltas,
                    "domain_harms": domain_harms,
                }
            )
        consistent_domains = [
            domain
            for domain in DOMAIN_METRICS
            if all(contrast["domain_harms"][domain] for contrast in contrasts)
        ]
        proxy_positive = all(
            contrast["candidate_proxy_advantage_under_fixed_R_w"] > 0
            for contrast in contrasts
        )
        screens.append(
            {
                "ctrl_cost_weight": weight,
                "proxy_positive_in_all_paired_seeds": proxy_positive,
                "consistently_harmed_independent_domains": consistent_domains,
                "passes_exploratory_two_domain_rule": (
                    proxy_positive and len(consistent_domains) >= 2
                ),
                "contrasts": contrasts,
            }
        )
    eligible = [
        screen["ctrl_cost_weight"]
        for screen in screens
        if screen["passes_exploratory_two_domain_rule"]
    ]
    result = {
        "status": "posthoc_expanded_core_exploration_not_confirmation",
        "checkpoint": args.checkpoint,
        "paired_training_seeds": seeds,
        "candidate_weights": weights,
        "domain_counting_rule": (
            "Action saturation and consecutive-action change jointly define one "
            "command-quality domain. They are never counted as two harms."
        ),
        "selected_exploratory_candidate": max(eligible) if eligible else None,
        "screens": screens,
        "claim_boundary": (
            "The expansion was motivated after the original 0.25 confirmation "
            "failed. Any selected weight is hypothesis-generating only and requires "
            "fresh seeds under a frozen prospective protocol."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
