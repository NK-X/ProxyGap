"""Replay V6 reference policies and diagnose the high-z termination mechanism."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import evaluate_model  # noqa: E402
from proxygap.high_z_diagnostic import (  # noqa: E402
    finite_summary,
    select_common_high_z_seed,
    summarise_step_trace,
    truthy,
)
from proxygap.reference_baseline import validate_reference_config  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"
DEFAULT_RUN = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_reference_fresh_1m_v6_20260814"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "analysis" / "stage1_reference_high_z_diagnostic_v8_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse_existing_replay",
        action="store_true",
        help="Rebuild summaries and figures from a previously verified replay.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def model_path(run_root: Path, training_seed: int) -> Path:
    return (
        run_root
        / "runs"
        / f"seed_{training_seed}"
        / "reference"
        / "models"
        / "reference"
        / "checkpoint_1000000.zip"
    )


def compare_replay(
    original: list[dict[str, str]],
    replay: list[dict[str, Any]],
) -> dict[str, Any]:
    original_by_key = {
        (int(row["training_seed"]), int(row["seed"])): row for row in original
    }
    replay_by_key = {
        (int(row["training_seed"]), int(row["seed"])): row for row in replay
    }
    if original_by_key.keys() != replay_by_key.keys():
        raise ValueError("Replay and original endpoint keys differ")
    exact_fields = (
        "termination_category",
        "unhealthy_termination",
        "episode_length",
        "terminated",
        "truncated",
    )
    numeric_fields = (
        "condition_objective_return",
        "net_forward_progress",
        "mean_forward_velocity",
        "torso_tilt_rms",
    )
    exact_mismatches = 0
    maximum_errors = {field: 0.0 for field in numeric_fields}
    for key in sorted(original_by_key):
        left = original_by_key[key]
        right = replay_by_key[key]
        for field in exact_fields:
            if str(left[field]).lower() != str(right[field]).lower():
                exact_mismatches += 1
        for field in numeric_fields:
            error = abs(float(left[field]) - float(right[field]))
            maximum_errors[field] = max(maximum_errors[field], error)
    return {
        "original_rows": len(original),
        "replay_rows": len(replay),
        "exact_field_mismatches": exact_mismatches,
        "maximum_absolute_numeric_errors": maximum_errors,
        "replay_matches_original": exact_mismatches == 0
        and all(error <= 1e-8 for error in maximum_errors.values()),
    }


def plot_failure_matrix(
    endpoint_rows: list[dict[str, str]],
    *,
    training_seeds: list[int],
    evaluation_seeds: list[int],
    output: Path,
) -> None:
    lookup = {
        (int(row["training_seed"]), int(row["seed"])): (
            1 if str(row["high_z_termination"]).lower() == "true" else 0
        )
        for row in endpoint_rows
    }
    matrix = [
        [lookup[(training_seed, evaluation_seed)] for evaluation_seed in evaluation_seeds]
        for training_seed in training_seeds
    ]
    figure, axis = plt.subplots(figsize=(12.2, 3.2), constrained_layout=True)
    axis.imshow(matrix, aspect="auto", cmap=ListedColormap(["#E9EEF2", "#C44E52"]), vmin=0, vmax=1)
    axis.set_xticks(range(len(evaluation_seeds)), [str(seed) for seed in evaluation_seeds], rotation=55, ha="right")
    axis.set_yticks(range(len(training_seeds)), [str(seed) for seed in training_seeds])
    axis.set_xlabel("Evaluation seed")
    axis.set_ylabel("Training seed")
    axis.set_title("High-z termination pattern at the 1M checkpoint")
    axis.legend(
        handles=[
            Patch(facecolor="#E9EEF2", label="Reached time limit"),
            Patch(facecolor="#C44E52", label="High-z termination"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=2,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_matched_height_profiles(
    trace_root: Path,
    *,
    training_seeds: list[int],
    evaluation_seed: int,
    output: Path,
    dt: float,
) -> None:
    colours = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    for index, training_seed in enumerate(training_seeds):
        path = trace_root / f"seed_{training_seed}" / (
            f"reference__train_{training_seed}__target_1000000__eval_{evaluation_seed}.csv.gz"
        )
        rows = read_csv(path)
        times = [int(row["step_index"]) * dt for row in rows]
        heights = [float(row["torso_height"]) for row in rows]
        category = (
            "time_limit"
            if truthy(rows[-1]["truncated"])
            else rows[-1]["termination_category"]
        )
        axis.plot(
            times,
            heights,
            color=colours(index),
            linewidth=1.7,
            label=f"Seed {training_seed}: {category}",
        )
    axis.axhspan(0.2, 1.0, color="#4C9F70", alpha=0.08)
    axis.axhline(1.0, color="#C44E52", linestyle="--", linewidth=1.2)
    axis.axhline(0.2, color="#C44E52", linestyle="--", linewidth=1.2)
    axis.text(49.5, 1.015, "Upper health bound: 1.0", ha="right", va="bottom", fontsize=9)
    axis.set(
        title=f"Matched torso-height trajectories (evaluation seed {evaluation_seed})",
        xlabel="Simulated time (s)",
        ylabel="Torso height",
        xlim=(0, 50),
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_policy_posture_diagnostic(
    rows: list[dict[str, Any]],
    *,
    output: Path,
) -> None:
    seeds = [int(row["training_seed"]) for row in rows]
    positions = list(range(len(seeds)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    series = [
        (
            "High-z episode proportion",
            [float(row["high_z_terminations"]) / float(row["episodes"]) for row in rows],
            "#C44E52",
            -width,
        ),
        (
            "Inverted-step proportion",
            [float(row["step_weighted_inverted_proportion"]) for row in rows],
            "#4C72B0",
            0.0,
        ),
        (
            "Low-posture step proportion",
            [float(row["step_weighted_low_posture_proportion"]) for row in rows],
            "#55A868",
            width,
        ),
    ]
    for label, values, colour, offset in series:
        bars = axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            color=colour,
            label=label,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8)
    axis.set_xticks(positions, [str(seed) for seed in seeds])
    axis.set_ylim(0, 0.75)
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Proportion")
    axis.set_title("Post-hoc posture diagnostics at the 1M checkpoint")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    trace_root = output_root / "step_traces"
    output_root.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_reference_config(config)
    if config["normalisation_enabled"] or config["ctrl_cost_weights"] != [0.5]:
        raise ValueError("This diagnostic is restricted to the frozen unnormalised reference")
    endpoint_target = int(config["reference_competence_gate"]["checkpoint"])
    original_all = read_csv(run_root / "logs" / "evaluation_metrics.csv")
    original = [
        row for row in original_all if int(row["target_timesteps"]) == endpoint_target
    ]

    replay_csv = output_root / "replay_episode_metrics.csv"
    model_manifest_csv = output_root / "model_replay_manifest.csv"
    if args.reuse_existing_replay:
        if not replay_csv.is_file() or not model_manifest_csv.is_file():
            raise FileNotFoundError("Verified replay tables do not exist")
        replay_rows = read_csv(replay_csv)
        model_records = read_csv(model_manifest_csv)
        expected_traces = len(config["training_seeds"]) * len(config["evaluation_seeds"])
        if len(list(trace_root.rglob("*.csv.gz"))) != expected_traces:
            raise ValueError("Existing step-trace matrix is incomplete")
    else:
        replay_rows = []
        model_records = []
        for training_seed in config["training_seeds"]:
            path = model_path(run_root, int(training_seed))
            if not path.is_file():
                raise FileNotFoundError(path)
            model = PPO.load(path, device="cpu")
            if int(model.num_timesteps) != 1_001_472:
                raise ValueError("Unexpected final model timestep count")
            rows, elapsed = evaluate_model(
                model,
                condition_id="reference",
                ctrl_cost_weight=0.5,
                checkpoint_fraction=1.0,
                seed=int(config["evaluation_seed_base"]),
                episodes=int(config["eval_episodes_per_checkpoint"]),
                target_timesteps=endpoint_target,
                actual_model_timesteps=int(model.num_timesteps),
                training_seed=int(training_seed),
                max_episode_steps=int(config["eval_max_episode_steps"]),
                step_log_dir=trace_root / f"seed_{training_seed}",
            )
            replay_rows.extend(rows)
            model_records.append(
                {
                    "training_seed": int(training_seed),
                    "model_path": str(path),
                    "model_sha256": sha256(path),
                    "actual_model_timesteps": int(model.num_timesteps),
                    "evaluation_seconds": elapsed,
                }
            )
            del model

    replay_integrity = compare_replay(original, replay_rows)
    if not replay_integrity["replay_matches_original"]:
        raise RuntimeError("Deterministic replay did not reproduce the frozen endpoint")

    trace_summaries: list[dict[str, Any]] = []
    for training_seed in config["training_seeds"]:
        for evaluation_seed in config["evaluation_seeds"]:
            path = trace_root / f"seed_{training_seed}" / (
                f"reference__train_{training_seed}__target_1000000__eval_{evaluation_seed}.csv.gz"
            )
            summary = summarise_step_trace(
                read_csv(path), dt=float(config.get("environment_dt", 0.05))
            )
            if not finite_summary(summary):
                raise ValueError(f"Non-finite step diagnostic: {path}")
            trace_summaries.append(
                {
                    "training_seed": int(training_seed),
                    "evaluation_seed": int(evaluation_seed),
                    **summary,
                    "trace_path": str(path),
                    "trace_sha256": sha256(path),
                }
            )

    failing_seeds = sorted(
        {
            int(row["training_seed"])
            for row in original
            if str(row["high_z_termination"]).lower() == "true"
        }
    )
    matched_selection = select_common_high_z_seed(
        original, failing_training_seeds=failing_seeds
    )
    write_csv(replay_csv, replay_rows)
    write_csv(output_root / "step_trace_summary.csv", trace_summaries)
    write_csv(model_manifest_csv, model_records)

    policy_summaries: list[dict[str, Any]] = []
    for training_seed in config["training_seeds"]:
        selected_traces = [
            row for row in trace_summaries
            if int(row["training_seed"]) == int(training_seed)
        ]
        selected_episodes = [
            row for row in replay_rows
            if int(row["training_seed"]) == int(training_seed)
        ]
        total_steps = sum(int(row["episode_length"]) for row in selected_traces)
        inverted_steps = sum(
            float(row["proportion_steps_torso_tilt_ge_90_deg"])
            * int(row["episode_length"])
            for row in selected_traces
        )
        low_steps = sum(
            float(row["proportion_steps_torso_height_below_0p3"])
            * int(row["episode_length"])
            for row in selected_traces
        )
        policy_summaries.append(
            {
                "training_seed": int(training_seed),
                "episodes": len(selected_traces),
                "total_evaluation_steps": total_steps,
                "high_z_terminations": sum(
                    row["termination_category"] == "high_z_excursion"
                    for row in selected_traces
                ),
                "time_limit_episodes": sum(bool(row["truncated"]) for row in selected_traces),
                "step_weighted_inverted_proportion": inverted_steps / total_steps,
                "step_weighted_low_posture_proportion": low_steps / total_steps,
                "episodes_majority_inverted": sum(
                    float(row["proportion_steps_torso_tilt_ge_90_deg"]) > 0.5
                    for row in selected_traces
                ),
                "episodes_majority_below_height_0p3": sum(
                    float(row["proportion_steps_torso_height_below_0p3"]) > 0.5
                    for row in selected_traces
                ),
                "mean_proxy_return": sum(
                    float(row["condition_objective_return"])
                    for row in selected_episodes
                ) / len(selected_episodes),
                "mean_net_forward_progress": sum(
                    float(row["net_forward_progress"]) for row in selected_episodes
                ) / len(selected_episodes),
            }
        )
    write_csv(output_root / "policy_posture_diagnostic.csv", policy_summaries)
    plot_policy_posture_diagnostic(
        policy_summaries,
        output=output_root / "policy_posture_diagnostic.png",
    )
    plot_failure_matrix(
        original,
        training_seeds=[int(seed) for seed in config["training_seeds"]],
        evaluation_seeds=[int(seed) for seed in config["evaluation_seeds"]],
        output=output_root / "high_z_failure_matrix.png",
    )
    plot_matched_height_profiles(
        trace_root,
        training_seeds=[int(seed) for seed in config["training_seeds"]],
        evaluation_seed=int(matched_selection["evaluation_seed"]),
        output=output_root / "matched_torso_height_profiles.png",
        dt=0.05,
    )

    high_z_summaries = [
        row for row in trace_summaries if row["termination_category"] == "high_z_excursion"
    ]
    time_limit_summaries = [
        row for row in trace_summaries if bool(row["truncated"])
    ]
    result = {
        "status": "numerical_replay_complete_visual_adjudication_pending",
        "scope": "post-run diagnostic only; no training, coefficient selection or shaping",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "run_root": str(run_root),
        "replay_integrity": replay_integrity,
        "models_replayed": len(model_records),
        "episodes_replayed": len(replay_rows),
        "step_trace_files": len(trace_summaries),
        "failing_training_seeds": failing_seeds,
        "matched_video_selection": matched_selection,
        "termination_counts": {
            "high_z_excursion": len(high_z_summaries),
            "time_limit": len(time_limit_summaries),
            "other": len(trace_summaries) - len(high_z_summaries) - len(time_limit_summaries),
        },
        "policy_posture_diagnostic": policy_summaries,
        "high_z_descriptives": {
            "terminal_torso_height_range": [
                min(row["terminal_torso_height"] for row in high_z_summaries),
                max(row["terminal_torso_height"] for row in high_z_summaries),
            ],
            "terminal_vertical_velocity_range": [
                min(row["terminal_vertical_velocity"] for row in high_z_summaries),
                max(row["terminal_vertical_velocity"] for row in high_z_summaries),
            ],
            "torso_height_gain_last_second_range": [
                min(row["torso_height_gain_last_second"] for row in high_z_summaries),
                max(row["torso_height_gain_last_second"] for row in high_z_summaries),
            ],
        },
        "claim_boundary": (
            "Step traces identify the simulator event and kinematic pattern but do not "
            "alone decide whether the upper-z health rule matches human intent. Complete "
            "matched videos are required before construct adjudication."
        ),
    }
    (output_root / "high_z_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
