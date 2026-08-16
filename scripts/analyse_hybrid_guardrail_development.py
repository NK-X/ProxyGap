"""Validate and analyse the frozen hybrid guardrail development matrix."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "hybrid_guardrail_development_v1_20260816.json"
)
STEP_FILE_PATTERN = re.compile(
    r"tr(?P<train>\d+)_t(?P<target>\d+)_ev(?P<eval>\d+)\.csv\.gz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in output.columns
    ]
    return output.reset_index()


def validate_episode_table(
    episode: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_rows = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["checkpoint_timesteps"])
        * int(config["eval_episodes_per_checkpoint"])
    )
    key = ["condition_id", "training_seed", "target_timesteps", "seed"]
    duplicate_count = int(episode.duplicated(key).sum())
    required = [
        *key,
        "reward_id",
        "constraint_id",
        "intent_compliant",
        "fixed_horizon_mean_forward_velocity",
        "torso_tilt_rms",
        "net_displacement_direction_error_degrees",
        "forward_path_efficiency",
        "normalised_action_roughness",
        "unhealthy_termination",
        "sustained_inversion",
        "action_slew_intervention_rate",
        "base_reward_reconciliation_error",
        "ctrl_cost_reconciliation_error",
    ]
    missing_columns = sorted(set(required) - set(episode.columns))
    null_counts = (
        {column: int(episode[column].isna().sum()) for column in required}
        if not missing_columns
        else {}
    )
    expected_eval_seeds = set(config["evaluation_seeds"])
    bad_seed_groups: list[str] = []
    if not missing_columns:
        for identifiers, rows in episode.groupby(
            ["condition_id", "training_seed", "target_timesteps"],
            dropna=False,
        ):
            if set(rows["seed"].astype(int)) != expected_eval_seeds:
                bad_seed_groups.append("|".join(str(value) for value in identifiers))
    max_base_error = (
        float(episode["base_reward_reconciliation_error"].abs().max())
        if "base_reward_reconciliation_error" in episode
        else float("nan")
    )
    max_ctrl_error = (
        float(episode["ctrl_cost_reconciliation_error"].abs().max())
        if "ctrl_cost_reconciliation_error" in episode
        else float("nan")
    )
    critical_nulls = {
        column: count for column, count in null_counts.items() if count > 0
    }
    passed = bool(
        len(episode) == expected_rows
        and duplicate_count == 0
        and not missing_columns
        and not critical_nulls
        and not bad_seed_groups
        and max_base_error <= 1e-8
        and max_ctrl_error <= 1e-8
    )
    return {
        "status": "passed" if passed else "failed",
        "expected_rows": expected_rows,
        "observed_rows": int(len(episode)),
        "duplicate_primary_keys": duplicate_count,
        "missing_columns": missing_columns,
        "critical_null_counts": critical_nulls,
        "bad_evaluation_seed_groups": bad_seed_groups,
        "max_abs_base_reward_reconciliation_error": max_base_error,
        "max_abs_ctrl_cost_reconciliation_error": max_ctrl_error,
    }


def validate_step_logs(
    run_root: Path,
    episode: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in sorted(run_root.glob("runs/**/evaluation_steps/*.csv.gz")):
        match = STEP_FILE_PATTERN.search(path.name)
        if match is None:
            failures.append(f"unparseable:{path}")
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle)
        condition_id = str(frame["condition_id"].iloc[0])
        constraint_enabled = bool(as_bool(frame["action_constraint_enabled"]).iloc[0])
        applied_change = pd.to_numeric(
            frame["applied_action_change_l2_step"], errors="coerce"
        )
        correction = pd.to_numeric(
            frame["action_correction_l2_step"], errors="coerce"
        )
        interventions = as_bool(frame["action_slew_intervened_step"]).fillna(False)
        limit_values = pd.to_numeric(frame["action_slew_l2_limit"], errors="coerce")
        finite_limits = limit_values[np.isfinite(limit_values)]
        limit = float(finite_limits.iloc[0]) if len(finite_limits) else float("nan")
        max_applied_change = float(applied_change.max())
        max_correction = float(correction.max())
        if constraint_enabled and max_applied_change > limit + 1e-9:
            failures.append(f"slew_violation:{path}:{max_applied_change}>{limit}")
        if not constraint_enabled and (
            int(interventions.sum()) != 0 or max_correction > 1e-12
        ):
            failures.append(f"unexpected_k0_intervention:{path}")
        reward_error = (
            pd.to_numeric(frame["condition_objective_reward_step"])
            - pd.to_numeric(frame["base_proxy_reward_step"])
            - pd.to_numeric(frame["shaping_reward_step"])
        ).abs()
        records.append(
            {
                "condition_id": condition_id,
                "training_seed": int(match.group("train")),
                "target_timesteps": int(match.group("target")),
                "seed": int(match.group("eval")),
                "steps": int(len(frame)),
                "constraint_enabled": constraint_enabled,
                "action_slew_l2_limit": limit,
                "max_applied_action_change_l2": max_applied_change,
                "max_action_correction_l2": max_correction,
                "intervention_count": int(interventions.sum()),
                "max_step_reward_identity_error": float(reward_error.max()),
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    step_summary = pd.DataFrame(records)
    key = ["condition_id", "training_seed", "target_timesteps", "seed"]
    expected_log_files = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["checkpoint_timesteps"])
        * int(config["eval_episodes_per_checkpoint"])
    )
    duplicate_keys = int(step_summary.duplicated(key).sum()) if records else 0
    episode_keys = {
        tuple(row)
        for row in episode[key].itertuples(index=False, name=None)
    }
    step_keys = (
        {
            tuple(row)
            for row in step_summary[key].itertuples(index=False, name=None)
        }
        if records
        else set()
    )
    missing_keys = sorted(episode_keys - step_keys)
    extra_keys = sorted(step_keys - episode_keys)
    length_mismatches: list[str] = []
    if records and "episode_length" in episode.columns:
        expected_lengths = episode.set_index(key)["episode_length"].astype(int)
        for row in step_summary.itertuples(index=False):
            row_key = (
                row.condition_id,
                row.training_seed,
                row.target_timesteps,
                row.seed,
            )
            if row_key in expected_lengths.index and int(row.steps) != int(
                expected_lengths.loc[row_key]
            ):
                length_mismatches.append(
                    f"{'|'.join(str(value) for value in row_key)}:"
                    f"{row.steps}!={expected_lengths.loc[row_key]}"
                )
    passed = bool(
        len(records) == expected_log_files
        and duplicate_keys == 0
        and not failures
        and not missing_keys
        and not extra_keys
        and not length_mismatches
    )
    qa = {
        "status": "passed" if passed else "failed",
        "expected_step_log_files": expected_log_files,
        "step_log_files": len(records),
        "duplicate_primary_keys": duplicate_keys,
        "missing_episode_keys": [list(key_values) for key_values in missing_keys],
        "extra_episode_keys": [list(key_values) for key_values in extra_keys],
        "episode_length_mismatches": length_mismatches,
        "total_logged_steps": int(step_summary["steps"].sum()) if records else 0,
        "max_applied_action_change_l2": (
            float(step_summary["max_applied_action_change_l2"].max())
            if records
            else float("nan")
        ),
        "max_step_reward_identity_error": (
            float(step_summary["max_step_reward_identity_error"].max())
            if records
            else float("nan")
        ),
        "failures": failures,
    }
    return step_summary, qa


def build_policy_table(episode: pd.DataFrame, endpoint: int) -> pd.DataFrame:
    final = episode.loc[episode["target_timesteps"].astype(int) == endpoint].copy()
    for column in [
        "intent_compliant",
        "unhealthy_termination",
        "sustained_inversion",
        "full_horizon_completed",
    ]:
        final[column] = as_bool(final[column]).astype(float)
    final["torso_tilt_rms_degrees"] = np.degrees(final["torso_tilt_rms"])
    metrics = {
        "intent_compliant": "mean",
        "fixed_horizon_mean_forward_velocity": "mean",
        "net_forward_progress": "mean",
        "torso_tilt_rms_degrees": "mean",
        "net_displacement_direction_error_degrees": "mean",
        "forward_path_efficiency": "mean",
        "normalised_action_roughness": "mean",
        "action_saturation_rate": "mean",
        "unhealthy_termination": "mean",
        "sustained_inversion": "mean",
        "full_horizon_completed": "mean",
        "base_proxy_return": "mean",
        "condition_objective_return": "mean",
        "control_effort_per_unit_distance": "mean",
        "action_slew_intervention_rate": "mean",
        "mean_action_correction_l2": "mean",
        "proposed_normalised_action_roughness": "mean",
    }
    policy = (
        final.groupby(
            [
                "condition_id",
                "reward_id",
                "constraint_id",
                "training_seed",
            ],
            as_index=False,
        )
        .agg(metrics)
        .rename(columns={"intent_compliant": "intent_compliance_rate"})
    )
    return policy


def build_condition_table(policy: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in policy.columns
        if column
        not in {"condition_id", "reward_id", "constraint_id", "training_seed"}
    ]
    condition = policy.groupby(
        ["condition_id", "reward_id", "constraint_id"]
    )[metric_columns].agg(["mean", "median", "min", "max"])
    return flatten_columns(condition)


def build_paired_contrasts(
    policy: pd.DataFrame,
    baseline_condition_id: str,
) -> pd.DataFrame:
    baseline = policy.loc[policy["condition_id"] == baseline_condition_id].set_index(
        "training_seed"
    )
    metric_columns = [
        "intent_compliance_rate",
        "fixed_horizon_mean_forward_velocity",
        "torso_tilt_rms_degrees",
        "net_displacement_direction_error_degrees",
        "forward_path_efficiency",
        "normalised_action_roughness",
        "unhealthy_termination",
        "sustained_inversion",
        "base_proxy_return",
        "action_slew_intervention_rate",
    ]
    rows: list[dict[str, Any]] = []
    for _, candidate in policy.iterrows():
        seed = int(candidate["training_seed"])
        reference = baseline.loc[seed]
        row: dict[str, Any] = {
            "condition_id": candidate["condition_id"],
            "reward_id": candidate["reward_id"],
            "constraint_id": candidate["constraint_id"],
            "training_seed": seed,
        }
        for metric in metric_columns:
            row[f"delta_{metric}"] = float(candidate[metric] - reference[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def adjudicate_candidates(
    policy: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["development_gate"]
    baseline_condition_id = str(
        gate.get("baseline_condition_id", "R0_default__K0_none")
    )
    baseline = policy.loc[policy["condition_id"] == baseline_condition_id].set_index(
        "training_seed"
    )
    decisions: list[dict[str, Any]] = []
    for condition_id, candidate in policy.groupby("condition_id"):
        candidate = candidate.set_index("training_seed")
        shared = sorted(set(baseline.index) & set(candidate.index))
        compliance_delta = (
            candidate.loc[shared, "intent_compliance_rate"]
            - baseline.loc[shared, "intent_compliance_rate"]
        )
        velocity_ratio = float(
            candidate.loc[shared, "fixed_horizon_mean_forward_velocity"].median()
            / max(
                baseline.loc[shared, "fixed_horizon_mean_forward_velocity"].median(),
                1e-8,
            )
        )
        unhealthy_delta = float(
            candidate.loc[shared, "unhealthy_termination"].mean()
            - baseline.loc[shared, "unhealthy_termination"].mean()
        )
        improved_pairs = int((compliance_delta > 0).sum())
        passed = bool(
            condition_id != baseline_condition_id
            and improved_pairs >= int(gate["paired_training_seed_improvement_min"])
            and velocity_ratio
            >= float(gate["fixed_horizon_forward_velocity_retention_min"])
            and unhealthy_delta
            <= float(gate["unhealthy_termination_allowed_increase"])
        )
        decisions.append(
            {
                "condition_id": condition_id,
                "paired_training_seeds": shared,
                "seed_pairs_with_intent_compliance_improvement": improved_pairs,
                "median_fixed_horizon_velocity_retention": velocity_ratio,
                "mean_unhealthy_termination_rate_delta": unhealthy_delta,
                "development_gate_passed": passed,
                "interpretation": (
                    "eligible_for_extended_development_review"
                    if passed
                    else "does_not_pass_prespecified_development_gate"
                ),
            }
        )
    return {
        "gate_role": gate["role"],
        "decisions": decisions,
        "advanced_conditions": [
            row["condition_id"] for row in decisions if row["development_gate_passed"]
        ],
    }


def save_tradeoff_plot(policy: pd.DataFrame, path: Path) -> None:
    reward_ids = sorted(policy["reward_id"].unique())
    constraint_ids = sorted(policy["constraint_id"].unique())
    palette = plt.get_cmap("tab10")
    colours = {
        reward_id: palette(index % 10)
        for index, reward_id in enumerate(reward_ids)
    }
    marker_values = ["o", "s", "^", "D", "P", "X"]
    markers = {
        constraint_id: marker_values[index % len(marker_values)]
        for index, constraint_id in enumerate(constraint_ids)
    }
    figure, axis = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    for _, row in policy.iterrows():
        axis.scatter(
            row["fixed_horizon_mean_forward_velocity"],
            row["intent_compliance_rate"],
            color=colours[row["reward_id"]],
            marker=markers[row["constraint_id"]],
            s=70,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
    axis.axvspan(0.8, 1.2, color="#999999", alpha=0.10)
    axis.set_xlabel("Fixed-horizon forward velocity (m s$^{-1}$)")
    axis.set_ylabel("Intent-compliant evaluation episodes (proportion)")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colour, label=reward)
        for reward, colour in colours.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="#333333",
            label=constraint,
        )
        for constraint, marker in markers.items()
    ]
    axis.legend(handles=handles, frameon=False, ncol=2, loc="upper left")
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def save_effect_matrix(
    paired: pd.DataFrame,
    path: Path,
    baseline_condition_id: str,
) -> None:
    metrics = {
        "delta_intent_compliance_rate": "Intent compliance\n(proportion)",
        "delta_fixed_horizon_mean_forward_velocity": "Forward velocity\n(m s$^{-1}$)",
        "delta_torso_tilt_rms_degrees": "Torso tilt RMS\n(deg; lower better)",
        "delta_net_displacement_direction_error_degrees": "Direction error\n(deg; lower better)",
        "delta_forward_path_efficiency": "Path efficiency\n(proportion)",
        "delta_normalised_action_roughness": "Action roughness\n(lower better)",
        "delta_unhealthy_termination": "Unhealthy termination\n(proportion; lower better)",
    }
    candidate = paired.loc[paired["condition_id"] != baseline_condition_id]
    mean_delta = candidate.groupby("condition_id")[list(metrics)].mean()
    direction = np.array([1, 1, -1, -1, 1, -1, -1], dtype=float)
    signed = mean_delta.to_numpy(float) * direction
    scales = np.nanmax(np.abs(signed), axis=0)
    scales[scales < 1e-12] = 1.0
    colour_values = signed / scales
    figure, axis = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
    image = axis.imshow(colour_values, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(metrics)), labels=list(metrics.values()))
    axis.set_yticks(range(len(mean_delta.index)), labels=list(mean_delta.index))
    axis.tick_params(axis="x", rotation=25)
    for row_index in range(len(mean_delta.index)):
        for column_index, column in enumerate(metrics):
            value = mean_delta.iloc[row_index][column]
            axis.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.82)
    colour_bar.set_label("Direction-normalised improvement (column-scaled)")
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    completion = json.loads(
        (run_root / "parallel_completion.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete":
        raise RuntimeError("Development matrix is not complete")

    analysis_root = run_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    episode = pd.read_csv(run_root / "logs" / "evaluation_metrics.csv")
    episode_qa = validate_episode_table(episode, config)
    step_summary, step_qa = validate_step_logs(run_root, episode, config)
    qa = {
        "episode_table": episode_qa,
        "step_logs": step_qa,
        "status": (
            "passed"
            if episode_qa["status"] == "passed" and step_qa["status"] == "passed"
            else "failed"
        ),
    }
    (analysis_root / "data_quality_qa.json").write_text(
        json.dumps(qa, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    step_summary.to_csv(analysis_root / "step_log_qa.csv", index=False)
    if qa["status"] != "passed":
        raise RuntimeError(f"Data-quality gate failed: {qa}")

    endpoint = int(config["development_gate"]["primary_endpoint_timesteps"])
    policy = build_policy_table(episode, endpoint)
    condition = build_condition_table(policy)
    baseline_condition_id = str(
        config["development_gate"].get(
            "baseline_condition_id", "R0_default__K0_none"
        )
    )
    paired = build_paired_contrasts(policy, baseline_condition_id)
    adjudication = adjudicate_candidates(policy, config)
    policy.to_csv(analysis_root / "endpoint_policy_metrics.csv", index=False)
    condition.to_csv(analysis_root / "endpoint_condition_summary.csv", index=False)
    paired.to_csv(analysis_root / "paired_seed_contrasts.csv", index=False)
    (analysis_root / "development_gate_adjudication.json").write_text(
        json.dumps(adjudication, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    save_tradeoff_plot(policy, analysis_root / "intent_velocity_tradeoff")
    save_effect_matrix(
        paired,
        analysis_root / "paired_effect_matrix",
        baseline_condition_id,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "qa_status": qa["status"],
                "episode_rows": len(episode),
                "policy_rows": len(policy),
                "advanced_conditions": adjudication["advanced_conditions"],
                "analysis_root": str(analysis_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
