"""Post-hoc development re-evaluation of command smoothness.

This script does not retrain policies and never overwrites the original 300k
confirmation data.  The new action-change diagnostic was proposed after that
confirmation failed, so every output is explicitly exploratory and cannot be
reported as preregistered confirmation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    evaluate_model,
    summarise_evaluation,
    write_standard_outputs,
)


CONDITIONS = {
    "reference": 0.5,
    "ctrl_0p25": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--evaluation_seed_base", type=int, default=51_101)
    parser.add_argument("--max_episode_steps", type=int, default=1_000)
    return parser.parse_args()


def mean(rows: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    evaluation_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    seed_dirs = sorted((source_root / "runs").glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {source_root / 'runs'}")

    for seed_dir in seed_dirs:
        training_seed = int(seed_dir.name.removeprefix("seed_"))
        for condition_id, weight in CONDITIONS.items():
            model_path = (
                seed_dir
                / condition_id
                / "models"
                / condition_id
                / f"checkpoint_{args.checkpoint:06d}.zip"
            )
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = PPO.load(model_path, device="cpu")
            rows, elapsed = evaluate_model(
                model,
                condition_id=condition_id,
                ctrl_cost_weight=weight,
                checkpoint_fraction=1.0,
                seed=args.evaluation_seed_base,
                episodes=args.episodes,
                target_timesteps=args.checkpoint,
                actual_model_timesteps=args.checkpoint,
                training_seed=training_seed,
                max_episode_steps=args.max_episode_steps,
                step_log_dir=output_root / "logs" / "evaluation_steps",
            )
            evaluation_rows.extend(rows)
            runtime_rows.append(
                {
                    "condition_id": condition_id,
                    "ctrl_cost_weight": weight,
                    "training_seed": training_seed,
                    "checkpoint": args.checkpoint,
                    "episodes": args.episodes,
                    "evaluation_elapsed_sec": elapsed,
                    "source_model": str(model_path),
                }
            )

    summary_rows = summarise_evaluation(evaluation_rows)
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summary_rows,
    )

    by_cell: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in evaluation_rows:
        key = (int(row["training_seed"]), str(row["condition_id"]))
        by_cell.setdefault(key, []).append(row)

    contrasts = []
    domain_names = [
        "path_efficiency",
        "lateral_control",
        "posture",
        "safety",
        "command_quality",
    ]
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.removeprefix("seed_"))
        reference = by_cell[(seed, "reference")]
        candidate = by_cell[(seed, "ctrl_0p25")]
        candidate_proxy = (
            mean(candidate, "reward_forward_sum")
            + mean(candidate, "reward_survive_sum")
            + mean(candidate, "reward_contact_sum")
            - 0.25 * mean(candidate, "cumulative_squared_action")
        )
        reference_proxy = (
            mean(reference, "reward_forward_sum")
            + mean(reference, "reward_survive_sum")
            + mean(reference, "reward_contact_sum")
            - 0.25 * mean(reference, "cumulative_squared_action")
        )
        metric_deltas = {
            "negative_forward_path_efficiency": -mean(
                candidate, "forward_path_efficiency"
            )
            - (-mean(reference, "forward_path_efficiency")),
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
            "path_efficiency": metric_deltas["negative_forward_path_efficiency"] > 0,
            "lateral_control": metric_deltas["lateral_drift_mean_abs"] > 0,
            "posture": metric_deltas["torso_tilt_rms"] > 0,
            "safety": metric_deltas["unhealthy_termination"] > 0,
            # These are two indicators of one command-quality construct.
            "command_quality": (
                metric_deltas["action_saturation_rate"] > 0
                and metric_deltas["mean_squared_action_change_per_transition"] > 0
            ),
        }
        contrasts.append(
            {
                "training_seed": seed,
                "candidate_proxy_advantage_under_R_0p25": candidate_proxy
                - reference_proxy,
                "metric_deltas_candidate_minus_reference_higher_is_worse": metric_deltas,
                "domain_harms": domain_harms,
            }
        )

    consistent_domains = [
        name
        for name in domain_names
        if all(item["domain_harms"][name] for item in contrasts)
    ]
    proxy_positive_all_seeds = all(
        item["candidate_proxy_advantage_under_R_0p25"] > 0 for item in contrasts
    )
    result = {
        "status": "posthoc_development_measurement_not_confirmation",
        "source_run": str(source_root),
        "checkpoint": args.checkpoint,
        "training_seeds": [
            int(path.name.removeprefix("seed_")) for path in seed_dirs
        ],
        "evaluation_seed_base": args.evaluation_seed_base,
        "episodes_per_policy": args.episodes,
        "diagnostic_definition": (
            "Mean sum of squared differences between consecutive eight-dimensional "
            "actions; the first action of each episode is excluded."
        ),
        "proxy_positive_in_all_seeds": proxy_positive_all_seeds,
        "consistently_harmed_domains": consistent_domains,
        "passes_original_two_domain_rule": (
            proxy_positive_all_seeds and len(consistent_domains) >= 2
        ),
        "contrasts": contrasts,
        "interpretation_boundary": (
            "Action saturation and action change are correlated indicators of one "
            "command-quality domain, not two independent harms. This post-hoc result "
            "may motivate a fresh held-out protocol but cannot rescue the failed "
            "preregistered 300k confirmation."
        ),
    }
    result_path = output_root / "posthoc_action_smoothness_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    run_config = {
        "status": "posthoc_development_reanalysis",
        "source_root": str(source_root),
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "evaluation_seed_base": args.evaluation_seed_base,
        "models_retrained": False,
        "original_outputs_overwritten": False,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
