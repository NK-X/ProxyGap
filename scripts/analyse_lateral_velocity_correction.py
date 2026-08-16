"""Audit the observable lateral-velocity shaping signal in a completed matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP_LOG_RE = re.compile(r"tr(?P<training_seed>\d+)_t(?P<timesteps>\d+)_ev(?P<evaluation_seed>\d+)\.csv\.gz$")


def episode_lateral_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Return lateral-velocity diagnostics for one evaluation episode."""

    velocity_error = frame["lateral_velocity"].to_numpy(float) - frame[
        "lateral_velocity_target"
    ].to_numpy(float)
    penalty = frame["lateral_penalty_step"].to_numpy(float)
    return {
        "mean_abs_lateral_velocity_error": float(np.mean(np.abs(velocity_error))),
        "rms_lateral_velocity_error": float(np.sqrt(np.mean(velocity_error**2))),
        "p95_abs_lateral_velocity_error": float(
            np.quantile(np.abs(velocity_error), 0.95)
        ),
        "mean_lateral_penalty": float(np.mean(penalty)),
        "max_lateral_penalty": float(np.max(penalty)),
        "logged_steps": int(len(frame)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/hybrid_guardrail_observability_correction_v1_20260816.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["execution"]["output_root"]
    analysis_root = output_root / "analysis" / "lateral_velocity"
    analysis_root.mkdir(parents=True, exist_ok=True)

    condition_weights = {
        row["condition_id"]: float(row["lateral_drift_shaping_weight"])
        for row in config["conditions"]
    }
    records: list[dict[str, object]] = []
    reward_errors: list[float] = []
    signal_values: set[str] = set()

    for path in sorted((output_root / "runs").rglob("*.csv.gz")):
        match = STEP_LOG_RE.match(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"Empty step log: {path}")
        condition_id = str(frame["condition_id"].iloc[0])
        signal_values.update(frame["lateral_shaping_signal"].dropna().astype(str))
        weight = condition_weights[condition_id]
        expected_reward = -weight * frame["lateral_penalty_step"].to_numpy(float)
        observed_reward = frame["reward_lateral_shaping_step"].to_numpy(float)
        reward_errors.append(float(np.max(np.abs(expected_reward - observed_reward))))
        records.append(
            {
                "condition_id": condition_id,
                "training_seed": int(match.group("training_seed")),
                "target_timesteps": int(match.group("timesteps")),
                "evaluation_seed": int(match.group("evaluation_seed")),
                "lateral_shaping_weight": weight,
                **episode_lateral_metrics(frame),
            }
        )

    episode = pd.DataFrame.from_records(records)
    expected_rows = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["checkpoint_timesteps"])
        * len(config["evaluation_seeds"])
    )
    primary_key = [
        "condition_id",
        "training_seed",
        "target_timesteps",
        "evaluation_seed",
    ]
    duplicate_rows = int(episode.duplicated(primary_key).sum())
    if len(episode) != expected_rows or duplicate_rows:
        raise RuntimeError(
            f"Lateral audit completeness failed: rows={len(episode)}, "
            f"expected={expected_rows}, duplicates={duplicate_rows}"
        )
    if signal_values != {"velocity_tanh_squared"}:
        raise RuntimeError(f"Unexpected shaping signals: {sorted(signal_values)}")
    max_reward_error = max(reward_errors, default=float("nan"))
    if not np.isfinite(max_reward_error) or max_reward_error > 1e-10:
        raise RuntimeError(f"Lateral reward reconciliation failed: {max_reward_error}")

    endpoint = int(config["development_gate"]["primary_endpoint_timesteps"])
    endpoint_episode = episode.loc[episode["target_timesteps"] == endpoint].copy()
    metric_columns = [
        "mean_abs_lateral_velocity_error",
        "rms_lateral_velocity_error",
        "p95_abs_lateral_velocity_error",
        "mean_lateral_penalty",
        "max_lateral_penalty",
    ]
    policy = (
        endpoint_episode.groupby(["condition_id", "training_seed"], as_index=False)[
            metric_columns
        ]
        .mean()
        .sort_values(["condition_id", "training_seed"])
    )
    condition = (
        policy.groupby("condition_id")[metric_columns]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    condition.columns = [
        "condition_id" if column[0] == "condition_id" else f"{column[0]}_{column[1]}"
        for column in condition.columns.to_flat_index()
    ]

    baseline_id = str(config["development_gate"]["baseline_condition_id"])
    baseline = endpoint_episode.loc[
        endpoint_episode["condition_id"] == baseline_id,
        ["training_seed", "evaluation_seed", *metric_columns],
    ]
    paired_rows: list[pd.DataFrame] = []
    for condition_id, candidate in endpoint_episode.groupby("condition_id"):
        if condition_id == baseline_id:
            continue
        merged = candidate.merge(
            baseline,
            on=["training_seed", "evaluation_seed"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        output = merged[["training_seed", "evaluation_seed"]].copy()
        output.insert(0, "condition_id", condition_id)
        for metric in metric_columns:
            output[f"delta_{metric}"] = (
                merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
            )
        paired_rows.append(output)
    paired_episode = pd.concat(paired_rows, ignore_index=True)
    paired_policy = (
        paired_episode.groupby(["condition_id", "training_seed"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["evaluation_seed"])
    )

    episode.to_csv(analysis_root / "episode_lateral_velocity_metrics.csv", index=False)
    policy.to_csv(analysis_root / "endpoint_policy_lateral_velocity_metrics.csv", index=False)
    condition.to_csv(
        analysis_root / "endpoint_condition_lateral_velocity_summary.csv", index=False
    )
    paired_episode.to_csv(
        analysis_root / "paired_episode_lateral_velocity_contrasts.csv", index=False
    )
    paired_policy.to_csv(
        analysis_root / "paired_policy_lateral_velocity_contrasts.csv", index=False
    )
    manifest = {
        "status": "passed",
        "role": "Post-run diagnostic of the predeclared observable shaping signal; not a new selection rule.",
        "episode_rows": len(episode),
        "expected_episode_rows": expected_rows,
        "duplicate_primary_keys": duplicate_rows,
        "signal_values": sorted(signal_values),
        "max_abs_lateral_reward_reconciliation_error": max_reward_error,
        "endpoint_timesteps": endpoint,
        "training_seed_is_replication_unit": True,
        "formal_inference": False,
    }
    (analysis_root / "lateral_velocity_qa.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
