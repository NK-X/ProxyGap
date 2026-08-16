"""Audit the stage-one 300k-to-1M development budget extension."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.budget_extension import (  # noqa: E402
    LOCKED_EXTENSION_CHECKPOINTS,
    LOCKED_TRAINING_SEEDS,
    LOCKED_WEIGHTS,
    sha256,
    validate_budget_extension_config,
)
from proxygap.stage1 import screen_stage1_endpoint  # noqa: E402


SUMMARY_FIELDS = [
    "reward_forward_sum",
    "reward_survive_sum",
    "reward_contact_sum",
    "cumulative_squared_action",
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
    "action_saturation_rate",
    "normalised_action_roughness",
]
CONTRAST_FIELDS = [
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
SEED_COLOURS = {41101: "#0072B2", 41102: "#D55E00"}
CANDIDATE_STYLES = {
    0.21875: {"marker": "o", "linestyle": "-"},
    0.125: {"marker": "s", "linestyle": "--"},
}
REWARD_RECONCILIATION_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline_csv",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "stage1_bidirectional_development_v2_20260814"
            / "combined_episode_metrics.csv"
        ),
    )
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_actual_timesteps(
    source_actual_timesteps: int,
    targets: Iterable[int],
    rollout_steps: int,
) -> dict[int, int]:
    """Compute PPO rollout-aligned actual timesteps for sequential targets."""
    actual = int(source_actual_timesteps)
    result: dict[int, int] = {}
    for target in targets:
        target = int(target)
        rollouts = math.ceil((target - actual) / int(rollout_steps))
        if rollouts <= 0:
            raise ValueError("Every target must exceed the preceding actual timestep")
        actual += rollouts * int(rollout_steps)
        result[target] = actual
    return result


def validate_episode_data(
    extension: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    budget = config["budget_extension"]
    weights = tuple(float(value) for value in budget["ctrl_cost_weights"])
    seeds = tuple(int(value) for value in budget["training_seeds"])
    checkpoints = tuple(int(value) for value in budget["checkpoint_timesteps"])
    evaluation_seeds = tuple(int(value) for value in budget["evaluation_seeds"])
    required = {
        "ctrl_cost_weight",
        "training_seed",
        "target_timesteps",
        "actual_model_timesteps",
        "seed",
        "reward_shaping_sum",
        "base_reward_reconciliation_error",
        "ctrl_cost_reconciliation_error",
        "terminated",
        "truncated",
        *SUMMARY_FIELDS,
    }
    missing = sorted(required - set(extension.columns))
    if missing:
        raise ValueError(f"Extension metrics are missing columns: {missing}")
    expected_rows = len(weights) * len(seeds) * len(checkpoints) * len(evaluation_seeds)
    if len(extension) != expected_rows:
        raise ValueError(f"Expected {expected_rows} extension rows, found {len(extension)}")
    key = ["ctrl_cost_weight", "training_seed", "target_timesteps", "seed"]
    duplicate_count = int(extension.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate extension episode keys: {duplicate_count}")
    if set(extension["ctrl_cost_weight"].astype(float)) != set(weights):
        raise ValueError("Extension weights do not match the frozen config")
    if set(extension["training_seed"].astype(int)) != set(seeds):
        raise ValueError("Extension training seeds do not match the frozen config")
    if set(extension["target_timesteps"].astype(int)) != set(checkpoints):
        raise ValueError("Extension checkpoints do not match the frozen config")
    cell_sizes = extension.groupby(key[:3]).size()
    if not bool((cell_sizes == len(evaluation_seeds)).all()):
        raise ValueError("Not every extension policy/checkpoint has ten episodes")
    for cell, rows in extension.groupby(key[:3]):
        if set(rows["seed"].astype(int)) != set(evaluation_seeds):
            raise ValueError(f"Evaluation-seed pairing failed for cell {cell}")

    expected_actual = expected_actual_timesteps(
        int(budget["expected_source_model_timesteps"]),
        checkpoints,
        int(config["ppo"]["n_steps"]),
    )
    actual_by_target: dict[int, int] = {}
    for target in checkpoints:
        observed = sorted(
            set(
                extension.loc[
                    extension["target_timesteps"].astype(int) == target,
                    "actual_model_timesteps",
                ].astype(int)
            )
        )
        if observed != [expected_actual[target]]:
            raise ValueError(
                f"Target {target} actual timesteps {observed}; expected "
                f"{expected_actual[target]}"
            )
        actual_by_target[target] = observed[0]

    shaping_columns = [
        "reward_shaping_sum",
        "reward_forward_shaping_sum",
        "reward_lateral_shaping_sum",
        "reward_effort_shaping_sum",
        "reward_orientation_shaping_sum",
    ]
    shaping_max = {
        column: float(extension[column].abs().max()) for column in shaping_columns
    }
    if any(value > 1e-12 for value in shaping_max.values()):
        raise ValueError(f"Non-zero shaping was observed: {shaping_max}")
    reconciliation = {
        "base_reward_reconciliation_max_abs": float(
            extension["base_reward_reconciliation_error"].abs().max()
        ),
        "ctrl_cost_reconciliation_max_abs": float(
            extension["ctrl_cost_reconciliation_error"].abs().max()
        ),
    }
    if (
        reconciliation["base_reward_reconciliation_max_abs"]
        > REWARD_RECONCILIATION_TOLERANCE
    ):
        raise ValueError(f"Base reward reconciliation failed: {reconciliation}")
    if (
        reconciliation["ctrl_cost_reconciliation_max_abs"]
        > REWARD_RECONCILIATION_TOLERANCE
    ):
        raise ValueError(f"Control-cost reconciliation failed: {reconciliation}")
    finite_columns = [
        "reward_forward_sum",
        "reward_survive_sum",
        "reward_contact_sum",
        "cumulative_squared_action",
        "net_forward_progress",
        "mean_forward_velocity",
        "forward_path_efficiency",
        "lateral_drift_mean_abs",
        "torso_tilt_rms",
        "action_saturation_rate",
        "normalised_action_roughness",
    ]
    non_finite = {
        column: int((~np.isfinite(extension[column].astype(float))).sum())
        for column in finite_columns
    }
    if any(non_finite.values()):
        raise ValueError(f"Non-finite decision metrics were found: {non_finite}")
    duration_error = (
        extension["episode_duration_seconds"].astype(float)
        - 0.05 * extension["episode_length"].astype(float)
    ).abs()
    if float(duration_error.max()) > 1e-9:
        raise ValueError("Episode duration does not reconcile with 0.05 s timestep")

    baseline_selected = baseline[
        baseline["ctrl_cost_weight"].astype(float).isin(weights)
        & (baseline["target_timesteps"].astype(int) == 300_000)
        & baseline["training_seed"].astype(int).isin(seeds)
    ].copy()
    expected_baseline_rows = len(weights) * len(seeds) * len(evaluation_seeds)
    if len(baseline_selected) != expected_baseline_rows:
        raise ValueError(
            f"Expected {expected_baseline_rows} 300k rows, found {len(baseline_selected)}"
        )
    if int(baseline_selected.duplicated(key).sum()):
        raise ValueError("The selected 300k baseline contains duplicate episode keys")
    for cell, rows in baseline_selected.groupby(key[:3]):
        if set(rows["seed"].astype(int)) != set(evaluation_seeds):
            raise ValueError(f"300k evaluation-seed pairing failed for cell {cell}")

    runtime = pd.read_csv(run_root / "logs" / "training_runtime.csv")
    source_audit = pd.read_csv(run_root / "logs" / "source_model_audit.csv")
    if len(runtime) != len(weights) * len(seeds) * len(checkpoints):
        raise ValueError("Training-runtime row count is incomplete")
    if len(source_audit) != len(weights) * len(seeds):
        raise ValueError("Source-audit row count is incomplete")
    if not source_audit["source_hash_unchanged"].astype(str).str.lower().eq("true").all():
        raise ValueError("At least one source model hash changed")
    if not source_audit["loaded_model_audit"].astype(str).eq("pass").all():
        raise ValueError("At least one loaded source model failed its audit")
    completion = json.loads(
        (run_root / "parallel_completion.json").read_text(encoding="utf-8")
    )
    if completion.get("failures") or int(completion["completed_policies"]) != 6:
        raise ValueError(f"Parallel completion record is not clean: {completion}")

    return {
        "status": "pass",
        "extension_episode_rows": len(extension),
        "baseline_300k_episode_rows": len(baseline_selected),
        "duplicate_episode_keys": duplicate_count,
        "weights": list(weights),
        "training_seeds": list(seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "actual_timesteps_by_target": actual_by_target,
        "shaping_max_abs": shaping_max,
        "reconciliation": reconciliation,
        "non_finite_decision_metrics": non_finite,
        "duration_reconciliation_max_abs": float(duration_error.max()),
        "runtime_rows": len(runtime),
        "source_audit_rows": len(source_audit),
        "source_hashes_unchanged": True,
        "failed_policies": 0,
    }


def policy_summary(rows: pd.DataFrame, scoring_weight: float) -> pd.DataFrame:
    selected = rows.copy()
    selected["matched_proxy_return"] = (
        selected["reward_forward_sum"]
        + selected["reward_survive_sum"]
        + selected["reward_contact_sum"]
        - float(scoring_weight) * selected["cumulative_squared_action"]
    )
    fields = ["matched_proxy_return", *SUMMARY_FIELDS]
    fields = list(dict.fromkeys(fields))
    summary = selected.groupby(
        ["ctrl_cost_weight", "training_seed", "target_timesteps"],
        as_index=False,
    )[fields].mean()
    summary["scoring_weight"] = float(scoring_weight)
    summary["final_lateral_per_abs_forward"] = summary[
        "lateral_drift_final_abs"
    ] / summary["net_forward_progress"].abs().clip(lower=1e-8)
    summary["lateral_path_fraction"] = summary[
        "cumulative_lateral_path"
    ] / summary["cumulative_planar_path"].clip(lower=1e-8)
    return summary


def paired_candidate_contrasts(
    rows: pd.DataFrame,
    *,
    reference_weight: float = 0.5,
    candidate_weights: Iterable[float] = (0.21875, 0.125),
) -> pd.DataFrame:
    contrasts: list[pd.DataFrame] = []
    for candidate_weight in candidate_weights:
        rescored = policy_summary(rows, float(candidate_weight))
        reference = rescored[
            np.isclose(rescored["ctrl_cost_weight"].astype(float), reference_weight)
        ].copy()
        candidate = rescored[
            np.isclose(rescored["ctrl_cost_weight"].astype(float), candidate_weight)
        ].copy()
        merged = candidate.merge(
            reference,
            on=["training_seed", "target_timesteps"],
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        merged["candidate_weight"] = float(candidate_weight)
        for field in CONTRAST_FIELDS:
            merged[f"delta_{field}"] = (
                merged[f"{field}_candidate"] - merged[f"{field}_reference"]
            )
        contrasts.append(merged)
    return pd.concat(contrasts, ignore_index=True).sort_values(
        ["candidate_weight", "training_seed", "target_timesteps"],
        ascending=[False, True, True],
    )


def reference_competence(
    rows: pd.DataFrame,
    *,
    checkpoint: int = 1_000_000,
    unhealthy_max: float = 0.2,
    forward_velocity_min: float = 0.1,
) -> pd.DataFrame:
    endpoint = rows[
        np.isclose(rows["ctrl_cost_weight"].astype(float), 0.5)
        & (rows["target_timesteps"].astype(int) == checkpoint)
    ]
    summary = endpoint.groupby("training_seed", as_index=False)[
        ["unhealthy_termination", "mean_forward_velocity", "net_forward_progress"]
    ].mean()
    summary["health_gate_pass"] = (
        summary["unhealthy_termination"] <= unhealthy_max
    )
    summary["velocity_gate_pass"] = (
        summary["mean_forward_velocity"] >= forward_velocity_min
    )
    summary["joint_competence_gate_pass"] = (
        summary["health_gate_pass"] & summary["velocity_gate_pass"]
    )
    return summary


def flatten_screens(rows: pd.DataFrame, checkpoints: Iterable[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for screen in screen_stage1_endpoint(
            rows.to_dict("records"),
            checkpoint=int(checkpoint),
            proxy_relative_noninferiority_margin=0.05,
        ):
            records.append(
                {
                    "checkpoint": int(checkpoint),
                    "candidate_weight": screen.candidate_weight,
                    "positive_proxy_seed_count": screen.positive_proxy_seed_count,
                    "noninferior_proxy_seed_count": screen.noninferior_proxy_seed_count,
                    "consistently_harmed_domains": ";".join(
                        screen.consistently_harmed_domains
                    ),
                    "consistently_harmed_metrics_by_domain": json.dumps(
                        screen.consistently_harmed_metrics_by_domain,
                        sort_keys=True,
                    ),
                    "strong_development_candidate": screen.strong_development_candidate,
                    "noninferior_development_candidate": (
                        screen.noninferior_development_candidate
                    ),
                    "screen_json": json.dumps(screen.to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(records)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_root / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reference_competence(summary: pd.DataFrame, output_root: Path) -> None:
    reference = summary[np.isclose(summary["ctrl_cost_weight"], 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
    panels = [
        ("mean_forward_velocity", 0.1, "Mean forward velocity", "minimum"),
        ("unhealthy_termination", 0.2, "Unhealthy-termination rate", "maximum"),
    ]
    for axis, (field, threshold, title, threshold_role) in zip(axes, panels):
        for seed in LOCKED_TRAINING_SEEDS:
            selected = reference[reference["training_seed"].astype(int) == seed]
            axis.plot(
                selected["target_timesteps"] / 1000,
                selected[field],
                marker="o",
                color=SEED_COLOURS[seed],
                label=f"Training seed {seed}",
            )
        axis.axhline(threshold, color="#333333", linestyle=":", linewidth=1.2)
        axis.text(
            0.98,
            threshold,
            f" Operational {threshold_role}: {threshold:g}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
        )
        axis.set_title(title)
        axis.set_xlabel("Training checkpoint (k timesteps)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Position units s$^{-1}$")
    axes[1].set_ylabel("Proportion of 10 episodes")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Reference-policy competence trajectory", fontsize=12)
    save_figure(fig, output_root, "reference_competence_trajectory")


def plot_candidate_contrasts(contrasts: pd.DataFrame, output_root: Path) -> None:
    panels = [
        ("delta_matched_proxy_return", 0.0, "Matched proxy advantage", "Return"),
        (
            "delta_forward_path_efficiency",
            -0.10,
            "Path-efficiency difference",
            "Candidate minus reference",
        ),
        (
            "delta_lateral_drift_mean_abs",
            0.50,
            "Absolute lateral-drift difference",
            "Position units",
        ),
        (
            "delta_final_lateral_per_abs_forward",
            0.0,
            "Distance-normalised lateral difference",
            "Ratio difference",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True)
    for axis, (field, margin, title, ylabel) in zip(axes.flat, panels):
        for candidate_weight, style in CANDIDATE_STYLES.items():
            for seed in LOCKED_TRAINING_SEEDS:
                selected = contrasts[
                    np.isclose(contrasts["candidate_weight"], candidate_weight)
                    & (contrasts["training_seed"].astype(int) == seed)
                ]
                axis.plot(
                    selected["target_timesteps"] / 1000,
                    selected[field],
                    color=SEED_COLOURS[seed],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        if not math.isclose(margin, 0.0):
            axis.axhline(margin, color="#333333", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Training checkpoint (k timesteps)")
    legend_items = [
        Line2D([0], [0], color=SEED_COLOURS[seed], label=f"Training seed {seed}")
        for seed in LOCKED_TRAINING_SEEDS
    ] + [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=f"Candidate w={weight:g}",
        )
        for weight, style in CANDIDATE_STYLES.items()
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle("Matched proxy and diagnostic contrasts", y=1.07, fontsize=12)
    save_figure(fig, output_root, "candidate_contrast_trajectory")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    baseline_path = args.baseline_csv.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Analysis output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_errors = validate_budget_extension_config(
        config,
        project_root=PROJECT_ROOT,
        verify_source_files=True,
    )
    if config_errors:
        raise ValueError(f"Budget-extension config errors: {config_errors}")
    extension_path = run_root / "logs" / "evaluation_metrics.csv"
    extension = pd.read_csv(extension_path)
    baseline = pd.read_csv(baseline_path)
    qa = validate_episode_data(extension, baseline, config, run_root)
    baseline_selected = baseline[
        baseline["ctrl_cost_weight"].astype(float).isin(LOCKED_WEIGHTS)
        & (baseline["target_timesteps"].astype(int) == 300_000)
        & baseline["training_seed"].astype(int).isin(LOCKED_TRAINING_SEEDS)
    ].copy()
    timeline = pd.concat([baseline_selected, extension], ignore_index=True).sort_values(
        ["ctrl_cost_weight", "training_seed", "target_timesteps", "seed"],
        ascending=[False, True, True, True],
    )
    summaries = pd.concat(
        [policy_summary(timeline, weight) for weight in LOCKED_WEIGHTS],
        ignore_index=True,
    )
    # Retain each trained policy only under its own matched scoring weight in this table.
    policy_rows = summaries[
        np.isclose(summaries["ctrl_cost_weight"], summaries["scoring_weight"])
    ].copy()
    contrasts = paired_candidate_contrasts(timeline)
    competence = reference_competence(extension)
    screens = flatten_screens(timeline, [300_000, *LOCKED_EXTENSION_CHECKPOINTS])

    timeline.to_csv(output_root / "timeline_episode_metrics.csv", index=False)
    policy_rows.to_csv(output_root / "policy_checkpoint_summary.csv", index=False)
    contrasts.to_csv(output_root / "paired_candidate_contrasts.csv", index=False)
    competence.to_csv(output_root / "reference_competence_gate.csv", index=False)
    screens.drop(columns=["screen_json"]).to_csv(
        output_root / "checkpoint_screen_summary.csv", index=False
    )
    (output_root / "checkpoint_screens.json").write_text(
        json.dumps(
            [json.loads(value) for value in screens["screen_json"]],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint_screens = screens[screens["checkpoint"].astype(int) == 1_000_000]
    endpoint_contrasts = contrasts[
        contrasts["target_timesteps"].astype(int) == 1_000_000
    ]
    candidate_audits: dict[str, Any] = {}
    for weight in (0.21875, 0.125):
        screen = endpoint_screens[
            np.isclose(endpoint_screens["candidate_weight"], weight)
        ].iloc[0]
        paired = endpoint_contrasts[
            np.isclose(endpoint_contrasts["candidate_weight"], weight)
        ]
        candidate_audits[str(weight)] = {
            "screen": {
                "strict_proxy_gain_both_development_seeds": bool(
                    int(screen["positive_proxy_seed_count"]) == 2
                ),
                "proxy_noninferior_both_development_seeds": bool(
                    int(screen["noninferior_proxy_seed_count"]) == 2
                ),
                "consistently_harmed_domains": str(
                    screen["consistently_harmed_domains"]
                ).split(";")
                if str(screen["consistently_harmed_domains"])
                else [],
                "strong_development_candidate": bool(
                    screen["strong_development_candidate"]
                ),
                "noninferior_development_candidate": bool(
                    screen["noninferior_development_candidate"]
                ),
            },
            "alternative_explanations": {
                "absolute_lateral_margin_crossed_both_seeds": bool(
                    (paired["delta_lateral_drift_mean_abs"] >= 0.5).all()
                ),
                "distance_normalised_lateral_worse_both_seeds": bool(
                    (paired["delta_final_lateral_per_abs_forward"] > 0).all()
                ),
                "lateral_path_fraction_worse_both_seeds": bool(
                    (paired["delta_lateral_path_fraction"] > 0).all()
                ),
                "path_efficiency_margin_crossed_both_seeds": bool(
                    (paired["delta_forward_path_efficiency"] <= -0.10).all()
                ),
            },
            "seed_level_rows": paired[
                [
                    "training_seed",
                    "delta_matched_proxy_return",
                    "delta_net_forward_progress",
                    "delta_mean_forward_velocity",
                    "delta_forward_path_efficiency",
                    "delta_unhealthy_termination",
                    "delta_lateral_drift_mean_abs",
                    "delta_final_lateral_per_abs_forward",
                    "delta_lateral_path_fraction",
                    "delta_torso_tilt_rms",
                    "delta_action_saturation_rate",
                    "delta_normalised_action_roughness",
                ]
            ].to_dict("records"),
        }

    reference_pass = bool(competence["joint_competence_gate_pass"].all())
    blockers: list[str] = []
    if not reference_pass:
        blockers.append("reference_competence_gate_failed")
    primary = candidate_audits["0.21875"]
    if primary["alternative_explanations"][
        "absolute_lateral_margin_crossed_both_seeds"
    ] and not (
        primary["alternative_explanations"][
            "distance_normalised_lateral_worse_both_seeds"
        ]
        and primary["alternative_explanations"][
            "lateral_path_fraction_worse_both_seeds"
        ]
    ):
        blockers.append("absolute_lateral_drift_remains_exposure_sensitive")
    blockers.extend(
        [
            "development_seeds_are_not_held_out_confirmation",
            "formal_condition_matrix_and_seed_count_not_frozen",
            "accuracy_matrix_course_requirement_unresolved",
        ]
    )
    adjudication = {
        "status": "development_budget_extension_adjudicated",
        "reference_competence_passed_both_seeds": reference_pass,
        "candidate_audits": candidate_audits,
        "formal_protocol_freeze_ready": False,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
        "blockers": blockers,
        "claim_boundary": (
            "The results describe two continued development policies per condition. "
            "They cannot confirm reward hacking, a scalar true reward, a critical "
            "coefficient, or generalisation beyond this Ant-v5/PPO setting."
        ),
    }
    qa["input_sha256"] = {
        str(extension_path): hash_file(extension_path),
        str(baseline_path): hash_file(baseline_path),
        str(config_path): sha256(config_path),
        str(run_root / "run_config.json"): hash_file(run_root / "run_config.json"),
    }
    (output_root / "qa_summary.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "stage1_budget_extension_adjudication.json").write_text(
        json.dumps(adjudication, indent=2) + "\n", encoding="utf-8"
    )

    set_plot_style()
    plot_reference_competence(policy_rows, output_root)
    plot_candidate_contrasts(contrasts, output_root)
    print(json.dumps(adjudication, indent=2), flush=True)


if __name__ == "__main__":
    main()
