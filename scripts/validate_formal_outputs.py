"""Validate the grain and invariants of one completed formal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / "artifacts" / "formal" / config["config_id"]
    eval_path = output_root / "logs" / "evaluation_metrics.csv"
    runtime_path = output_root / "logs" / "training_runtime.csv"
    evaluation = pd.read_csv(eval_path)
    runtime = pd.read_csv(runtime_path)

    conditions = [item["condition_id"] for item in config["conditions"]]
    seeds = [int(value) for value in config["training_seeds"]]
    targets = [int(value) for value in config["checkpoint_timesteps"]]
    episodes = int(config["eval_episodes_per_checkpoint"])
    expected_eval_rows = len(conditions) * len(seeds) * len(targets) * episodes
    expected_runtime_rows = len(conditions) * len(seeds) * len(targets)
    expected_eval_seeds = list(
        range(int(config["evaluation_seed_base"]), int(config["evaluation_seed_base"]) + episodes)
    )

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    add("completed_marker", (output_root / "completed.json").exists(), str(output_root / "completed.json"))
    add("evaluation_row_count", len(evaluation) == expected_eval_rows, {"actual": len(evaluation), "expected": expected_eval_rows})
    add("runtime_row_count", len(runtime) == expected_runtime_rows, {"actual": len(runtime), "expected": expected_runtime_rows})

    key = ["condition_id", "training_seed", "target_timesteps", "episode"]
    duplicate_rows = int(evaluation.duplicated(key).sum())
    add("evaluation_key_unique", duplicate_rows == 0, {"duplicate_rows": duplicate_rows, "key": key})
    add("condition_set", set(evaluation["condition_id"]) == set(conditions), sorted(evaluation["condition_id"].unique().tolist()))
    add("training_seed_set", set(evaluation["training_seed"].astype(int)) == set(seeds), sorted(evaluation["training_seed"].astype(int).unique().tolist()))
    add("checkpoint_set", set(evaluation["target_timesteps"].astype(int)) == set(targets), sorted(evaluation["target_timesteps"].astype(int).unique().tolist()))

    group_sizes = evaluation.groupby(["condition_id", "training_seed", "target_timesteps"]).size()
    add("episodes_per_checkpoint", bool((group_sizes == episodes).all()), {"minimum": int(group_sizes.min()), "maximum": int(group_sizes.max()), "expected": episodes})

    seed_sets = evaluation.groupby(["condition_id", "training_seed", "target_timesteps"])["seed"].apply(lambda values: sorted(values.astype(int).tolist()))
    paired = all(value == expected_eval_seeds for value in seed_sets)
    add("paired_evaluation_seeds", paired, expected_eval_seeds)

    reward_identity = np.abs(
        evaluation["proxy_return"]
        - evaluation["base_proxy_return"]
        - evaluation["reward_shaping_sum"]
    )
    add("observed_reward_decomposition", bool((reward_identity < 1e-6).all()), {"maximum_absolute_error": float(reward_identity.max())})

    shaping_columns = {"reward_forward_shaping_sum", "reward_lateral_shaping_sum"}
    if shaping_columns.issubset(evaluation.columns):
        shaping_component_error = np.abs(
            evaluation["reward_shaping_sum"]
            - evaluation["reward_forward_shaping_sum"]
            - evaluation["reward_lateral_shaping_sum"]
        )
        add("shaping_component_decomposition", bool((shaping_component_error < 1e-6).all()), {"maximum_absolute_error": float(shaping_component_error.max())})
        forward_formula_error = np.abs(
            evaluation["reward_forward_shaping_sum"]
            - evaluation["forward_progress_shaping_weight"]
            * evaluation["reward_forward_sum"]
        )
        add("forward_shaping_formula", bool((forward_formula_error < 1e-6).all()), {"maximum_absolute_error": float(forward_formula_error.max())})

    base_components = (
        evaluation["reward_forward_sum"]
        + evaluation["reward_ctrl_sum"]
        + evaluation["reward_contact_sum"]
        + evaluation["reward_survive_sum"]
    )
    component_error = np.abs(evaluation["base_proxy_return"] - base_components)
    add("ant_reward_component_decomposition", bool((component_error < 1e-6).all()), {"maximum_absolute_error": float(component_error.max())})

    forward_weights = (
        evaluation["forward_progress_shaping_weight"]
        if "forward_progress_shaping_weight" in evaluation
        else 0.0
    )
    unshaped = (
        (evaluation["lateral_drift_shaping_weight"] == 0.0)
        & (forward_weights == 0.0)
    )
    unshaped_max = float(evaluation.loc[unshaped, "reward_shaping_sum"].abs().max()) if unshaped.any() else 0.0
    add("unshaped_reward_is_zero", unshaped_max < 1e-12, {"maximum_absolute_shaping_sum": unshaped_max})

    ratio_nan = evaluation["control_effort_per_unit_distance"].isna()
    invalid_ratio_nan = int((ratio_nan & (evaluation["net_forward_progress"] > 1e-8)).sum())
    add("effort_ratio_nan_only_without_forward_progress", invalid_ratio_nan == 0, {"invalid_rows": invalid_ratio_nan, "allowed_nan_rows": int(ratio_nan.sum())})

    required_non_null = [
        "proxy_return",
        "base_proxy_return",
        "reward_shaping_sum",
        "net_forward_progress",
        "fall",
        "lateral_drift_final_abs",
        "lateral_drift_mean_abs",
        "torso_tilt_mean",
        "torso_tilt_std",
        "episode_length",
    ]
    null_counts = evaluation[required_non_null].isna().sum()
    add("required_metrics_complete", int(null_counts.sum()) == 0, {key: int(value) for key, value in null_counts.items()})

    rollout_overshoot = runtime["actual_model_timesteps"] - runtime["target_timesteps"]
    n_steps = int(config["ppo"]["n_steps"])
    add("ppo_rollout_overshoot_bounded", bool(((rollout_overshoot >= 0) & (rollout_overshoot < n_steps)).all()), {"minimum": int(rollout_overshoot.min()), "maximum": int(rollout_overshoot.max()), "exclusive_upper_bound": n_steps})

    model_count = len(list((output_root / "runs").rglob("checkpoint_*.zip")))
    add("model_checkpoint_count", model_count == expected_runtime_rows, {"actual": model_count, "expected": expected_runtime_rows})

    report = {
        "config_id": config["config_id"],
        "dataset_grain": "one deterministic evaluation episode per condition, training seed and checkpoint",
        "overall_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    report_path = output_root / "logs" / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
