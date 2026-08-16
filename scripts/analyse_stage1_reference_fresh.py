"""Primary analysis for the V6 fresh-reference development diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.reference_baseline import (  # noqa: E402
    numeric,
    summarise_reference_endpoint,
    validate_reference_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"
DEFAULT_RUN = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_reference_fresh_1m_v6_20260814"
DEFAULT_ANALYSIS = PROJECT_ROOT / "artifacts" / "analysis" / "stage1_reference_fresh_1m_v6_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--analysis_root", type=Path, default=DEFAULT_ANALYSIS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty analysis table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows(raw: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for training_seed in config["training_seeds"]:
        for checkpoint in config["checkpoint_timesteps"]:
            rows = [
                row
                for row in raw
                if int(row["training_seed"]) == int(training_seed)
                and int(row["target_timesteps"]) == int(checkpoint)
            ]
            if len(rows) != int(config["eval_episodes_per_checkpoint"]):
                raise ValueError("Checkpoint episode count drifted")
            unhealthy = [numeric(row["unhealthy_termination"]) for row in rows]
            velocities = [numeric(row["mean_forward_velocity"]) for row in rows]
            output.append(
                {
                    "training_seed": int(training_seed),
                    "target_timesteps": int(checkpoint),
                    "evaluation_episodes": len(rows),
                    "unhealthy_termination_rate": float(np.mean(unhealthy)),
                    "mean_forward_velocity": float(np.mean(velocities)),
                    "mean_net_forward_progress": float(
                        np.mean([numeric(row["net_forward_progress"]) for row in rows])
                    ),
                    "mean_episode_length": float(
                        np.mean([numeric(row["episode_length"]) for row in rows])
                    ),
                }
            )
    return output


def create_plot(rows: list[dict[str, Any]], output: Path) -> None:
    colours = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    seeds = sorted({int(row["training_seed"]) for row in rows})
    for index, seed in enumerate(seeds):
        selected = sorted(
            (row for row in rows if int(row["training_seed"]) == seed),
            key=lambda row: int(row["target_timesteps"]),
        )
        x = [int(row["target_timesteps"]) / 1000 for row in selected]
        axes[0].plot(
            x,
            [float(row["mean_forward_velocity"]) for row in selected],
            marker="o",
            linewidth=1.7,
            color=colours(index),
            label=f"Seed {seed}",
        )
        axes[1].plot(
            x,
            [float(row["unhealthy_termination_rate"]) for row in selected],
            marker="s",
            linewidth=1.7,
            color=colours(index),
            label=f"Seed {seed}",
        )
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(0.2, color="black", linestyle="--", linewidth=1.2)
    axes[0].text(
        995,
        0.115,
        "Velocity gate: 0.10",
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    axes[1].text(
        995,
        0.215,
        "Health gate: 0.20",
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    axes[0].set(title="Forward velocity", xlabel="Training steps (thousands)", ylabel="Mean position units s$^{-1}$")
    axes[1].set(title="Unhealthy termination", xlabel="Training steps (thousands)", ylabel="Episode proportion")
    axes[1].set_ylim(-0.03, 1.03)
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    validate_reference_config(config)
    run_root = args.run_root.resolve()
    analysis_root = args.analysis_root.resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    raw = read_csv(run_root / "logs" / "evaluation_metrics.csv")
    runtime = read_csv(run_root / "logs" / "training_runtime.csv")
    completion = json.loads((run_root / "parallel_completion.json").read_text(encoding="utf-8"))

    expected_rows = 5 * 4 * 20
    keys = [
        (
            int(row["training_seed"]),
            int(row["target_timesteps"]),
            int(row["seed"]),
        )
        for row in raw
    ]
    if len(raw) != expected_rows or len(set(keys)) != expected_rows:
        raise ValueError("The raw evaluation matrix is incomplete or duplicated")
    if len(runtime) != 20:
        raise ValueError("The runtime manifest does not contain 20 checkpoints")
    models = sorted((run_root / "runs").rglob("checkpoint_*.zip"))
    if len(models) != 20:
        raise ValueError("The run does not contain 20 model checkpoints")
    if completion.get("failures") or int(completion.get("completed_policies", 0)) != 5:
        raise ValueError("The parallel completion record contains a failed policy")
    if any(
        not math.isfinite(numeric(row[field]))
        for row in raw
        for field in ("mean_forward_velocity", "unhealthy_termination")
    ):
        raise ValueError("A decision metric is non-finite")

    expected_eval = set(int(value) for value in config["evaluation_seeds"])
    for seed in config["training_seeds"]:
        for checkpoint in config["checkpoint_timesteps"]:
            selected = [
                row
                for row in raw
                if int(row["training_seed"]) == int(seed)
                and int(row["target_timesteps"]) == int(checkpoint)
            ]
            if {int(row["seed"]) for row in selected} != expected_eval:
                raise ValueError("Paired evaluation-seed contract failed")

    endpoint = summarise_reference_endpoint(raw, config)
    gate_rows = list(endpoint["policy_results"])
    trajectory_rows = checkpoint_rows(raw, config)
    write_csv(analysis_root / "reference_policy_gate.csv", gate_rows)
    write_csv(analysis_root / "reference_checkpoint_trajectory.csv", trajectory_rows)
    create_plot(trajectory_rows, analysis_root / "reference_competence_trajectory.png")

    classification = str(endpoint["classification"])
    next_action = (
        "Freeze a separate held-out stage-one candidate-confirmation protocol; do not start it automatically."
        if classification == "supported"
        else "Freeze a separate reference-configuration pilot before changing normalisation, architecture or optimisation settings."
    )
    adjudication = {
        "status": "primary_analysis_complete_independent_verification_pending",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "reference-only development diagnostic",
        "primary_endpoint": endpoint["primary_endpoint"],
        "independent_replication_unit": endpoint["independent_replication_unit"],
        "evaluation_episode_role": endpoint["evaluation_episode_role"],
        "passing_policies": endpoint["passing_policies"],
        "total_policies": endpoint["total_policies"],
        "configuration_classification": classification,
        "policy_results": gate_rows,
        "next_action": next_action,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
        "claim_boundary": config["claim_boundary"],
    }
    (analysis_root / "stage1_reference_adjudication.json").write_text(
        json.dumps(adjudication, indent=2) + "\n", encoding="utf-8"
    )
    qa = {
        "status": "pass",
        "raw_evaluation_rows": len(raw),
        "unique_evaluation_keys": len(set(keys)),
        "runtime_rows": len(runtime),
        "model_checkpoints": len(models),
        "training_seeds": config["training_seeds"],
        "checkpoints": config["checkpoint_timesteps"],
        "evaluation_seeds": config["evaluation_seeds"],
        "failed_policies": completion.get("failures", []),
    }
    (analysis_root / "qa_summary.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(adjudication, indent=2), flush=True)


if __name__ == "__main__":
    main()
