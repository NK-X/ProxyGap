"""Evaluate whether development conclusions depend on one intent threshold set."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_CONFIG = ROOT / "configs" / "hybrid_guardrail_development_v2_20260816.json"
DEFAULT_SENSITIVITY_CONFIG = ROOT / "configs" / "hybrid_intent_sensitivity_v1_20260816.json"
DEFAULT_BASELINE = "R0_default__K0_none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_SENSITIVITY_CONFIG)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def compliance(frame: pd.DataFrame, threshold: dict[str, float]) -> pd.Series:
    tilt_degrees = np.degrees(pd.to_numeric(frame["torso_tilt_rms"]))
    return (
        as_bool(frame["full_horizon_completed"])
        & ~as_bool(frame["unhealthy_termination"])
        & ~as_bool(frame["sustained_inversion"])
        & pd.to_numeric(frame["fixed_horizon_mean_forward_velocity"]).between(
            threshold["forward_velocity_min"],
            threshold["forward_velocity_max"],
            inclusive="both",
        )
        & (tilt_degrees <= threshold["torso_tilt_rms_degrees_max"])
        & (
            pd.to_numeric(frame["net_displacement_direction_error_degrees"])
            <= threshold["direction_error_degrees_max"]
        )
        & (
            pd.to_numeric(frame["forward_path_efficiency"])
            >= threshold["forward_path_efficiency_min"]
        )
        & (
            pd.to_numeric(frame["normalised_action_roughness"])
            <= threshold["normalised_action_roughness_max"]
        )
        & (
            pd.to_numeric(frame["action_saturation_rate"])
            <= threshold["action_saturation_rate_max"]
        )
    )


def threshold_grid(config: dict) -> list[dict[str, float]]:
    grid = config["sensitivity_grid"]
    rows: list[dict[str, float]] = []
    for values in product(
        grid["forward_velocity_bands"],
        grid["torso_tilt_rms_degrees_max"],
        grid["direction_error_degrees_max"],
        grid["forward_path_efficiency_min"],
        grid["normalised_action_roughness_max"],
        grid["action_saturation_rate_max"],
    ):
        velocity, tilt, direction, efficiency, roughness, saturation = values
        rows.append(
            {
                "forward_velocity_min": float(velocity[0]),
                "forward_velocity_max": float(velocity[1]),
                "torso_tilt_rms_degrees_max": float(tilt),
                "direction_error_degrees_max": float(direction),
                "forward_path_efficiency_min": float(efficiency),
                "normalised_action_roughness_max": float(roughness),
                "action_saturation_rate_max": float(saturation),
            }
        )
    return rows


def domain_matrix(frame: pd.DataFrame, threshold: dict[str, float]) -> pd.DataFrame:
    work = frame.copy()
    work["horizon_and_health"] = (
        as_bool(work["full_horizon_completed"])
        & ~as_bool(work["unhealthy_termination"])
    )
    work["forward_tracking"] = pd.to_numeric(
        work["fixed_horizon_mean_forward_velocity"]
    ).between(
        threshold["forward_velocity_min"],
        threshold["forward_velocity_max"],
        inclusive="both",
    )
    work["no_sustained_inversion"] = ~as_bool(work["sustained_inversion"])
    work["torso_stability"] = (
        np.degrees(pd.to_numeric(work["torso_tilt_rms"]))
        <= threshold["torso_tilt_rms_degrees_max"]
    )
    work["directional_control"] = (
        pd.to_numeric(work["net_displacement_direction_error_degrees"])
        <= threshold["direction_error_degrees_max"]
    )
    work["path_directness"] = (
        pd.to_numeric(work["forward_path_efficiency"])
        >= threshold["forward_path_efficiency_min"]
    )
    work["action_smoothness"] = (
        pd.to_numeric(work["normalised_action_roughness"])
        <= threshold["normalised_action_roughness_max"]
    )
    work["low_saturation"] = (
        pd.to_numeric(work["action_saturation_rate"])
        <= threshold["action_saturation_rate_max"]
    )
    work["overall_intent"] = compliance(work, threshold)
    domains = [
        "horizon_and_health",
        "forward_tracking",
        "no_sustained_inversion",
        "torso_stability",
        "directional_control",
        "path_directness",
        "action_smoothness",
        "low_saturation",
        "overall_intent",
    ]
    policy = work.groupby(
        ["condition_id", "reward_id", "constraint_id", "training_seed"],
        as_index=False,
    )[domains].mean()
    return policy


def save_domain_heatmap(condition: pd.DataFrame, path: Path) -> None:
    identifiers = ["condition_id", "reward_id", "constraint_id"]
    domains = [column for column in condition.columns if column not in identifiers]
    values = condition[domains].to_numpy(float) * 100.0
    figure, axis = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    image = axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(domains)), [name.replace("_", "\n") for name in domains])
    axis.set_yticks(range(len(condition)), condition["condition_id"])
    axis.set_title("Intended-behaviour domain compliance at 300k (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "white" if values[row, column] < 30 or values[row, column] > 80 else "black"
            axis.text(column, row, f"{values[row, column]:.0f}", ha="center", va="center", color=colour, fontsize=8)
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.82)
    colour_bar.set_label("Mean episode compliance across three trained policies (%)")
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matrix_config = json.loads(args.matrix_config.resolve().read_text(encoding="utf-8"))
    sensitivity = json.loads(args.sensitivity_config.resolve().read_text(encoding="utf-8"))
    baseline_condition_id = str(
        matrix_config["development_gate"].get(
            "baseline_condition_id", DEFAULT_BASELINE
        )
    )
    run_root = ROOT / matrix_config["execution"]["output_root"]
    completion = json.loads((run_root / "parallel_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("Development matrix is not complete")
    endpoint = int(sensitivity["primary_endpoint_timesteps"])
    episode = pd.read_csv(run_root / "logs" / "evaluation_metrics.csv")
    episode = episode.loc[episode["target_timesteps"].astype(int) == endpoint].copy()
    expected = (
        len(matrix_config["conditions"])
        * len(matrix_config["training_seeds"])
        * int(matrix_config["eval_episodes_per_checkpoint"])
    )
    if len(episode) != expected:
        raise RuntimeError(f"Expected {expected} endpoint rows, found {len(episode)}")

    sensitivity_rows: list[dict] = []
    for grid_id, threshold in enumerate(threshold_grid(sensitivity), start=1):
        work = episode[["condition_id", "training_seed"]].copy()
        work["compliant"] = compliance(episode, threshold).astype(float)
        policy = work.groupby(["condition_id", "training_seed"], as_index=False)["compliant"].mean()
        baseline = policy.loc[
            policy["condition_id"] == baseline_condition_id
        ].set_index("training_seed")
        for condition_id, candidate in policy.groupby("condition_id"):
            if condition_id == baseline_condition_id:
                continue
            candidate = candidate.set_index("training_seed")
            shared = sorted(set(baseline.index) & set(candidate.index))
            deltas = candidate.loc[shared, "compliant"] - baseline.loc[shared, "compliant"]
            sensitivity_rows.append(
                {
                    "grid_id": grid_id,
                    **threshold,
                    "condition_id": condition_id,
                    "improved_seed_pairs": int((deltas > 0).sum()),
                    "replicated_in_two_of_three_seeds": bool((deltas > 0).sum() >= 2),
                    "mean_paired_compliance_delta": float(deltas.mean()),
                    "median_paired_compliance_delta": float(deltas.median()),
                }
            )
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    summary = sensitivity_table.groupby("condition_id", as_index=False).agg(
        grid_cells=("grid_id", "count"),
        replicated_grid_fraction=("replicated_in_two_of_three_seeds", "mean"),
        mean_paired_compliance_delta=("mean_paired_compliance_delta", "mean"),
        median_paired_compliance_delta=("median_paired_compliance_delta", "median"),
        minimum_paired_compliance_delta=("mean_paired_compliance_delta", "min"),
        maximum_paired_compliance_delta=("mean_paired_compliance_delta", "max"),
    )

    policy_domains = domain_matrix(episode, sensitivity["frozen_reference"])
    identifiers = ["condition_id", "reward_id", "constraint_id"]
    domains = [column for column in policy_domains.columns if column not in identifiers + ["training_seed"]]
    condition_domains = policy_domains.groupby(identifiers, as_index=False)[domains].mean()
    output = run_root / "analysis" / "intent_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    sensitivity_table.to_csv(output / "threshold_grid_results.csv", index=False)
    summary.to_csv(output / "threshold_robustness_summary.csv", index=False)
    policy_domains.to_csv(output / "policy_domain_compliance.csv", index=False)
    condition_domains.to_csv(output / "condition_domain_compliance.csv", index=False)
    save_domain_heatmap(condition_domains, output / "intent_domain_compliance_matrix")
    result = {
        "status": "complete",
        "threshold_grid_cells": len(threshold_grid(sensitivity)),
        "candidate_cell_evaluations": len(sensitivity_table),
        "primary_threshold_unchanged": True,
        "formal_inference": False,
        "baseline_condition_id": baseline_condition_id,
        "output_root": str(output),
    }
    (output / "sensitivity_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
