"""Calibrate the frozen cosine-weight grid from baseline trajectory metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    replay_path = config["baseline_evidence"].get("replay_evaluation_csv")
    baseline_csv = (
        PROJECT_ROOT / replay_path
        if replay_path
        else run_root / "baseline_replay" / "logs" / "evaluation_metrics.csv"
    )
    if not baseline_csv.exists():
        raise FileNotFoundError(
            "Run reevaluate_orientation_pilot_baseline.py before calibration"
        )
    output_root = run_root / "offline_calibration"
    if output_root.exists():
        raise FileExistsError(f"Calibration output already exists: {output_root}")

    endpoint = int(config["pilot_gate"]["primary_endpoint"])
    expected_seeds = {
        int(value) for value in config["offline_calibration_training_seeds"]
    }
    source_rows = [
        row for row in read_rows(baseline_csv)
        if int(row["target_timesteps"]) == endpoint
    ]
    observed_seeds = {int(row["training_seed"]) for row in source_rows}
    expected_rows = len(expected_seeds) * int(config["eval_episodes_per_checkpoint"])
    if observed_seeds != expected_seeds or len(source_rows) != expected_rows:
        raise RuntimeError(
            f"Baseline endpoint mismatch: seeds={observed_seeds}, rows={len(source_rows)}"
        )

    detail_rows: list[dict] = []
    for row in source_rows:
        base_return = float(row["base_proxy_return"])
        penalty_sum = float(row["orientation_penalty_sum"])
        if not math.isfinite(base_return) or not math.isfinite(penalty_sum):
            raise RuntimeError("Non-finite baseline scale input")
        if not 0.0 <= penalty_sum <= int(config["eval_max_episode_steps"]):
            raise RuntimeError(f"Cosine penalty is outside its bounded sum: {penalty_sum}")
        for weight_value in config["orientation_shaping"]["candidate_weights"]:
            weight = float(weight_value)
            cumulative_penalty = weight * penalty_sum
            detail_rows.append(
                {
                    "training_seed": int(row["training_seed"]),
                    "evaluation_seed": int(row["seed"]),
                    "orientation_weight": weight,
                    "base_proxy_return": base_return,
                    "normalised_cosine_penalty_sum": penalty_sum,
                    "cumulative_weighted_penalty": cumulative_penalty,
                    "penalty_to_absolute_base_return": cumulative_penalty
                    / max(abs(base_return), 1e-8),
                    "counterfactual_rescored_return": base_return
                    - cumulative_penalty,
                }
            )

    summary_rows: list[dict] = []
    seed_summary_rows: list[dict] = []
    weights = [float(v) for v in config["orientation_shaping"]["candidate_weights"]]
    for weight in weights:
        subset = [row for row in detail_rows if row["orientation_weight"] == weight]
        ratios = [float(row["penalty_to_absolute_base_return"]) for row in subset]
        penalties = [float(row["cumulative_weighted_penalty"]) for row in subset]
        summary_rows.append(
            {
                "orientation_weight": weight,
                "episodes": len(subset),
                "weighted_penalty_median": statistics.median(penalties),
                "weighted_penalty_p05": percentile(penalties, 0.05),
                "weighted_penalty_p95": percentile(penalties, 0.95),
                "penalty_ratio_median": statistics.median(ratios),
                "penalty_ratio_p05": percentile(ratios, 0.05),
                "penalty_ratio_p95": percentile(ratios, 0.95),
            }
        )
        for training_seed in sorted(expected_seeds):
            seed_subset = [
                row
                for row in subset
                if int(row["training_seed"]) == training_seed
            ]
            seed_ratios = [
                float(row["penalty_to_absolute_base_return"])
                for row in seed_subset
            ]
            seed_penalties = [
                float(row["cumulative_weighted_penalty"])
                for row in seed_subset
            ]
            seed_summary_rows.append(
                {
                    "orientation_weight": weight,
                    "training_seed": training_seed,
                    "episodes": len(seed_subset),
                    "weighted_penalty_mean": statistics.mean(seed_penalties),
                    "weighted_penalty_median": statistics.median(seed_penalties),
                    "penalty_ratio_mean": statistics.mean(seed_ratios),
                    "penalty_ratio_median": statistics.median(seed_ratios),
                    "penalty_ratio_p95": percentile(seed_ratios, 0.95),
                }
            )

    medians = [float(row["weighted_penalty_median"]) for row in summary_rows]
    monotonic = all(right > left for left, right in zip(medians, medians[1:]))
    finite = all(
        math.isfinite(float(value))
        for row in detail_rows
        for value in row.values()
        if isinstance(value, (int, float))
    )
    largest_median_ratio = float(summary_rows[-1]["penalty_ratio_median"])
    stratified_gate = config.get("offline_calibration_gate")
    if stratified_gate:
        largest_weight = weights[-1]
        largest_rows = [
            row
            for row in seed_summary_rows
            if float(row["orientation_weight"]) == largest_weight
        ]
        by_seed = {int(row["training_seed"]): row for row in largest_rows}
        reference_seed = int(stratified_gate["reference_mode_training_seed"])
        adverse_seed = int(stratified_gate["adverse_mode_training_seed"])
        reference_ratio = float(by_seed[reference_seed]["penalty_ratio_mean"])
        adverse_ratio = float(by_seed[adverse_seed]["penalty_ratio_mean"])
        adverse_low, adverse_high = [
            float(value)
            for value in stratified_gate[
                "largest_weight_adverse_mean_ratio_interval"
            ]
        ]
        reference_max = float(
            stratified_gate["largest_weight_reference_mean_ratio_max"]
        )
        gate_checks = {
            "finite": finite,
            "strictly_monotonic_pooled_median_penalty": monotonic,
            "adverse_mode_ratio_in_interval": (
                adverse_low <= adverse_ratio <= adverse_high
            ),
            "reference_mode_ratio_below_max": reference_ratio <= reference_max,
        }
        status = "passed" if all(gate_checks.values()) else "blocked"
    else:
        reference_seed = None
        adverse_seed = None
        reference_ratio = None
        adverse_ratio = None
        gate_checks = {
            "finite": finite,
            "strictly_monotonic_pooled_median_penalty": monotonic,
            "pooled_median_ratio_in_interval": (
                0.01 <= largest_median_ratio <= 0.25
            ),
        }
        status = "passed" if all(gate_checks.values()) else "blocked"
    output_root.mkdir(parents=True)
    write_rows(output_root / "episode_scale_calibration.csv", detail_rows)
    write_rows(output_root / "weight_scale_summary.csv", summary_rows)
    write_rows(output_root / "seed_weight_scale_summary.csv", seed_summary_rows)
    adjudication = {
        "status": status,
        "finite": finite,
        "strictly_monotonic_median_penalty": monotonic,
        "largest_candidate_median_penalty_ratio": largest_median_ratio,
        "gate_checks": gate_checks,
        "reference_mode_training_seed": reference_seed,
        "adverse_mode_training_seed": adverse_seed,
        "largest_weight_reference_mean_ratio": reference_ratio,
        "largest_weight_adverse_mean_ratio": adverse_ratio,
        "candidate_weights": weights,
        "episodes": len(source_rows),
        "source_baseline_csv": str(baseline_csv),
        "config_path": str(config_path),
        "interpretation": (
            "Offline rescoring checks reward scale only. It cannot predict the "
            "policy learned under the modified reward."
        ),
    }
    (output_root / "calibration_adjudication.json").write_text(
        json.dumps(adjudication, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(adjudication, indent=2), flush=True)
    if status != "passed":
        raise RuntimeError("Offline calibration blocked training")


if __name__ == "__main__":
    main()
