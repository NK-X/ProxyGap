"""Auditable analysis of the stage-one dense development grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DOMAINS,
    DiagnosticDomain,
    MetricMargin,
    screen_stage1_endpoint,
    validate_stage1_config,
)


SEED_COLOURS = {41101: "#0072B2", 41102: "#D55E00"}
SEED_MARKERS = {41101: "o", 41102: "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_csv", required=True)
    parser.add_argument("--dense_csv", required=True)
    parser.add_argument("--additional_csv", action="append", default=[])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_domains(scale: float) -> tuple[DiagnosticDomain, ...]:
    return tuple(
        DiagnosticDomain(
            name=domain.name,
            combination=domain.combination,
            metrics=tuple(
                MetricMargin(
                    field=metric.field,
                    direction=metric.direction,
                    practical_margin=metric.practical_margin * scale,
                )
                for metric in domain.metrics
            ),
        )
        for domain in DEFAULT_STAGE1_DOMAINS
    )


def screen_is_candidate(screen: Any, proxy_gate: str) -> bool:
    if proxy_gate == "strict_gain":
        return bool(screen.strong_development_candidate)
    if proxy_gate == "relative_noninferiority":
        return bool(screen.noninferior_development_candidate)
    raise ValueError(f"Unsupported primary_proxy_gate: {proxy_gate}")


def validate_data(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    development = config["development"]
    grid = [float(value) for value in development["ordered_analysis_grid"]]
    seeds = [int(value) for value in development["training_seeds"]]
    checkpoints = [int(value) for value in development["checkpoint_timesteps"]]
    episodes = int(development["evaluation_episodes_per_checkpoint"])
    required_columns = {
        "training_seed",
        "seed",
        "ctrl_cost_weight",
        "target_timesteps",
        "actual_model_timesteps",
        "reward_forward_sum",
        "reward_survive_sum",
        "reward_contact_sum",
        "cumulative_squared_action",
        "net_forward_progress",
        "forward_path_efficiency",
        "unhealthy_termination",
        "lateral_drift_mean_abs",
        "torso_tilt_rms",
        "action_saturation_rate",
        "normalised_action_roughness",
        "action_saturation_rate",
        "base_reward_reconciliation_error",
        "ctrl_cost_reconciliation_error",
        "environment_dt",
        "episode_duration_seconds",
        "mean_forward_velocity",
        "episode_length",
        "terminated",
        "truncated",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    key = ["training_seed", "ctrl_cost_weight", "target_timesteps", "seed"]
    duplicates = int(df.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"Duplicate episode keys: {duplicates}")
    expected_rows = len(grid) * len(seeds) * len(checkpoints) * episodes
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")
    if set(df["ctrl_cost_weight"].astype(float)) != set(grid):
        raise ValueError("Observed coefficient grid does not match the frozen config")
    if set(df["training_seed"].astype(int)) != set(seeds):
        raise ValueError("Observed training seeds do not match the frozen config")
    if set(df["target_timesteps"].astype(int)) != set(checkpoints):
        raise ValueError("Observed checkpoints do not match the frozen config")
    cell_sizes = df.groupby(key[:3]).size()
    if not bool((cell_sizes == episodes).all()):
        raise ValueError("Not every policy/checkpoint cell has the expected episodes")
    expected_eval_seeds = set(
        range(
            int(development["evaluation_seed_base"]),
            int(development["evaluation_seed_base"]) + episodes,
        )
    )
    for _, cell in df.groupby(key[:3]):
        if set(cell["seed"].astype(int)) != expected_eval_seeds:
            raise ValueError("Evaluation-seed pairing is incomplete")
    actual_timesteps_by_target: dict[str, int] = {}
    for target, target_rows in df.groupby("target_timesteps"):
        actual_values = sorted(
            set(target_rows["actual_model_timesteps"].astype(int).tolist())
        )
        if len(actual_values) != 1:
            raise ValueError(
                f"Target checkpoint {target} has inconsistent actual timesteps: "
                f"{actual_values}"
            )
        if actual_values[0] < int(target):
            raise ValueError(
                f"Actual timesteps {actual_values[0]} precede target {target}"
            )
        actual_timesteps_by_target[str(int(target))] = actual_values[0]
    shaping_columns = [
        "forward_progress_shaping_weight",
        "lateral_drift_shaping_weight",
        "effort_shaping_weight",
        "orientation_shaping_weight",
    ]
    nonzero_shaping = {
        column: float(df[column].abs().max())
        for column in shaping_columns
        if column in df and float(df[column].abs().max()) != 0.0
    }
    if nonzero_shaping:
        raise ValueError(f"Stage-one data contain shaping: {nonzero_shaping}")
    if set(np.round(df["environment_dt"].astype(float), 12)) != {0.05}:
        raise ValueError("Environment dt is not uniformly 0.05 seconds")
    decision_numeric = [
        "reward_forward_sum",
        "reward_survive_sum",
        "reward_contact_sum",
        "cumulative_squared_action",
        "net_forward_progress",
        "forward_path_efficiency",
        "unhealthy_termination",
        "lateral_drift_mean_abs",
        "torso_tilt_rms",
        "action_saturation_rate",
        "normalised_action_roughness",
    ]
    non_finite_counts = {
        column: int((~np.isfinite(df[column].astype(float))).sum())
        for column in decision_numeric
    }
    if any(non_finite_counts.values()):
        raise ValueError(f"Non-finite decision metrics: {non_finite_counts}")
    episode_lengths = df["episode_length"].astype(int)
    if not bool(episode_lengths.between(1, 1000).all()):
        raise ValueError("Episode length lies outside [1, 1000]")
    duration_error = (
        df["episode_duration_seconds"].astype(float)
        - episode_lengths * df["environment_dt"].astype(float)
    ).abs()
    if float(duration_error.max()) > 1e-10:
        raise ValueError("Episode duration is inconsistent with length and dt")
    velocity_error = (
        df["mean_forward_velocity"].astype(float)
        - df["net_forward_progress"].astype(float)
        / df["episode_duration_seconds"].astype(float)
    ).abs()
    if float(velocity_error.max()) > 1e-10:
        raise ValueError("Mean forward velocity is internally inconsistent")
    for bounded_metric in (
        "action_saturation_rate",
        "normalised_action_roughness",
    ):
        if not bool(df[bounded_metric].astype(float).between(0.0, 1.0).all()):
            raise ValueError(f"{bounded_metric} lies outside [0, 1]")
    if not bool(df["forward_path_efficiency"].astype(float).between(-1.000001, 1.000001).all()):
        raise ValueError("forward_path_efficiency lies outside its geometric bounds")

    def as_bool(series: pd.Series) -> pd.Series:
        return series.map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().lower() == "true"
        )

    terminated = as_bool(df["terminated"])
    truncated = as_bool(df["truncated"])
    invalid_end_states = int((terminated == truncated).sum())
    if invalid_end_states:
        raise ValueError(
            f"Episodes must end by exactly one route; invalid rows={invalid_end_states}"
        )
    base_reconciliation_max = float(
        df["base_reward_reconciliation_error"].abs().max()
    )
    ctrl_reconciliation_max = float(
        df["ctrl_cost_reconciliation_error"].abs().max()
    )
    if max(base_reconciliation_max, ctrl_reconciliation_max) > 1e-3:
        raise ValueError("Reward decomposition reconciliation exceeds 1e-3")
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_episode_keys": duplicates,
        "cell_count": int(cell_sizes.size),
        "episodes_per_cell": episodes,
        "weights": grid,
        "training_seeds": seeds,
        "checkpoints": checkpoints,
        "actual_timesteps_by_target": actual_timesteps_by_target,
        "evaluation_seeds": sorted(expected_eval_seeds),
        "non_finite_decision_metric_counts": non_finite_counts,
        "invalid_episode_end_state_count": invalid_end_states,
        "duration_reconciliation_max_abs": float(duration_error.max()),
        "velocity_reconciliation_max_abs": float(velocity_error.max()),
        "base_reward_reconciliation_max_abs": base_reconciliation_max,
        "ctrl_cost_reconciliation_max_abs": ctrl_reconciliation_max,
        "normalised_action_roughness_range": [
            float(df["normalised_action_roughness"].min()),
            float(df["normalised_action_roughness"].max()),
        ],
    }


def cross_rescore_rows(df: pd.DataFrame, scoring_weights: list[float]) -> pd.DataFrame:
    components = (
        df.groupby(["ctrl_cost_weight", "training_seed"], as_index=False)[
            [
                "reward_forward_sum",
                "reward_survive_sum",
                "reward_contact_sum",
                "cumulative_squared_action",
            ]
        ]
        .mean()
        .rename(columns={"ctrl_cost_weight": "trained_weight"})
    )
    rows: list[dict[str, float | int]] = []
    for item in components.to_dict("records"):
        for scoring_weight in scoring_weights:
            rows.append(
                {
                    "trained_weight": float(item["trained_weight"]),
                    "training_seed": int(item["training_seed"]),
                    "scoring_weight": scoring_weight,
                    "rescored_return": float(
                        item["reward_forward_sum"]
                        + item["reward_survive_sum"]
                        + item["reward_contact_sum"]
                        - scoring_weight * item["cumulative_squared_action"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def endpoint_effect_rows(screens: list[Any]) -> pd.DataFrame:
    margin_lookup = {
        metric.field: metric.practical_margin
        for domain in DEFAULT_STAGE1_DOMAINS
        for metric in domain.metrics
    }
    rows: list[dict[str, Any]] = []
    for screen in screens:
        for contrast in screen.contrasts:
            rows.append(
                {
                    "candidate_weight": screen.candidate_weight,
                    "training_seed": contrast.training_seed,
                    "quantity": "proxy_advantage_under_R_w",
                    "raw_delta": contrast.candidate_proxy_advantage_under_R_w,
                    "directed_harm": np.nan,
                    "practical_margin": np.nan,
                    "harm_to_margin_ratio": np.nan,
                }
            )
            for metric, raw_delta in contrast.raw_metric_deltas_candidate_minus_reference.items():
                margin = margin_lookup[metric]
                harm = contrast.practical_harm_amounts[metric]
                rows.append(
                    {
                        "candidate_weight": screen.candidate_weight,
                        "training_seed": contrast.training_seed,
                        "quantity": metric,
                        "raw_delta": raw_delta,
                        "directed_harm": harm,
                        "practical_margin": margin,
                        "harm_to_margin_ratio": harm / margin,
                    }
                )
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_root / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_contrasts(effects: pd.DataFrame, output_root: Path) -> None:
    metric_panels = [
        ("proxy_advantage_under_R_w", "Proxy advantage under matched $R_w$", False),
        ("net_forward_progress", "Net-progress harm / margin", True),
        ("forward_path_efficiency", "Path-efficiency harm / margin", True),
        ("unhealthy_termination", "Unhealthy-termination harm / margin", True),
        ("lateral_drift_mean_abs", "Lateral-drift harm / margin", True),
        ("torso_tilt_rms", "Torso-tilt harm / margin", True),
        ("action_saturation_rate", "Action-saturation harm / margin", True),
        ("normalised_action_roughness", "Action-roughness harm / margin", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.7), sharex=True)
    for axis, (metric, label, use_ratio) in zip(axes.flat, metric_panels):
        subset = effects[effects["quantity"] == metric].copy()
        ordered_weights = sorted(subset["candidate_weight"].unique(), reverse=True)
        weight_positions = {
            weight: index for index, weight in enumerate(ordered_weights)
        }
        for seed in sorted(subset["training_seed"].unique()):
            seed_rows = subset[subset["training_seed"] == seed].sort_values(
                "candidate_weight", ascending=False
            )
            y_column = "harm_to_margin_ratio" if use_ratio else "raw_delta"
            x_values = [
                weight_positions[weight]
                for weight in seed_rows["candidate_weight"].tolist()
            ]
            axis.plot(
                x_values,
                seed_rows[y_column],
                color=SEED_COLOURS[int(seed)],
                marker=SEED_MARKERS[int(seed)],
                linewidth=1.6,
                markersize=5,
                label=f"training seed {int(seed)}",
            )
            vertical_offset = 7 if int(seed) == 41101 else -12
            for x_value, y_value in zip(x_values, seed_rows[y_column]):
                axis.annotate(
                    f"{y_value:.2f}" if use_ratio else f"{y_value:.0f}",
                    (x_value, y_value),
                    xytext=(0, vertical_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.5,
                    color=SEED_COLOURS[int(seed)],
                )
        axis.axhline(1.0 if use_ratio else 0.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Candidate control-cost weight")
        axis.set_xticks(
            range(len(ordered_weights)),
            [f"{weight:g}" for weight in ordered_weights],
            rotation=35,
            ha="right",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0, 0].set_ylabel("Candidate minus reference")
    axes[1, 0].set_ylabel("Directed harm / practical margin")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save_figure(fig, output_root, "endpoint_seed_contrasts")


def plot_domain_matrix(screens: list[Any], output_root: Path) -> None:
    screens = sorted(screens, key=lambda screen: screen.candidate_weight, reverse=True)
    weights = [screen.candidate_weight for screen in screens]
    domains = [domain.name for domain in DEFAULT_STAGE1_DOMAINS]
    matrix = np.zeros((len(domains), len(weights)), dtype=int)
    for column, screen in enumerate(screens):
        for row, domain in enumerate(domains):
            matrix[row, column] = sum(
                domain in contrast.harmed_domains for contrast in screen.contrasts
            )
    cmap = ListedColormap(["#F2F2F2", "#E69F00", "#009E73"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axis = plt.subplots(figsize=(8.3, 3.4))
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(weights)), [f"{weight:g}" for weight in weights])
    axis.set_yticks(range(len(domains)), [name.replace("_", " ") for name in domains])
    axis.set_xlabel("Candidate control-cost weight")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=9)
    colourbar = fig.colorbar(image, ax=axis, ticks=[0, 1, 2], pad=0.02)
    colourbar.set_label("Number of training seeds (out of 2)")
    save_figure(fig, output_root, "domain_replication_matrix")


def plot_cross_rescore(cross: pd.DataFrame, output_root: Path) -> None:
    means = cross.groupby(["trained_weight", "scoring_weight"])["rescored_return"].mean().unstack()
    means = means.sort_index(ascending=False)
    means = means[sorted(means.columns, reverse=True)]
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    image = axis.imshow(means.to_numpy(), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(means.columns)), [f"{value:g}" for value in means.columns])
    axis.set_yticks(range(len(means.index)), [f"{value:g}" for value in means.index])
    axis.set_xlabel("Scoring coefficient in $R_w$")
    axis.set_ylabel("Policy training coefficient")
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means.iloc[row, column]
            colour = "white" if value < float(means.to_numpy().mean()) else "black"
            axis.text(column, row, f"{value:.0f}", ha="center", va="center", color=colour, fontsize=7.5)
    colourbar = fig.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("Mean rescored return across training seeds")
    save_figure(fig, output_root, "cross_rescore_matrix")


def plot_checkpoint_map(
    checkpoint_screens: dict[int, list[Any]],
    output_root: Path,
    *,
    proxy_gate: str,
) -> None:
    checkpoints = sorted(checkpoint_screens)
    weights = sorted(
        [screen.candidate_weight for screen in checkpoint_screens[checkpoints[-1]]],
        reverse=True,
    )
    matrix = np.zeros((len(weights), len(checkpoints)), dtype=int)
    for column, checkpoint in enumerate(checkpoints):
        by_weight = {
            screen.candidate_weight: screen
            for screen in checkpoint_screens[checkpoint]
        }
        for row, weight in enumerate(weights):
            screen = by_weight[weight]
            matrix[row, column] = sum(
                (
                    contrast.candidate_proxy_advantage_under_R_w > 0
                    if proxy_gate == "strict_gain"
                    else contrast.proxy_noninferior
                )
                and bool(contrast.harmed_domains)
                for contrast in screen.contrasts
            )
    cmap = ListedColormap(["#F2F2F2", "#56B4E9", "#0072B2"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axis = plt.subplots(figsize=(8.8, 4.1))
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(checkpoints)), [f"{value // 1000}k" for value in checkpoints])
    axis.set_yticks(range(len(weights)), [f"{value:g}" for value in weights])
    axis.set_xlabel("Training checkpoint")
    axis.set_ylabel("Candidate control-cost weight")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    colourbar = fig.colorbar(image, ax=axis, ticks=[0, 1, 2], pad=0.02)
    colourbar.set_label("Number of training seeds (out of 2)")
    save_figure(fig, output_root, "checkpoint_replication_matrix")


def plot_progress_effort_map(endpoint_data: pd.DataFrame, output_root: Path) -> None:
    means = (
        endpoint_data.groupby(["ctrl_cost_weight", "training_seed"], as_index=False)[
            ["cumulative_squared_action", "net_forward_progress"]
        ]
        .mean()
        .sort_values(["training_seed", "ctrl_cost_weight"], ascending=[True, False])
    )
    seeds = sorted(means["training_seed"].unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(11.0, 4.6), sharey=True)
    if len(seeds) == 1:
        axes = np.asarray([axes])
    offsets = [(6, 8), (6, -14), (6, 23), (6, -29), (6, 38), (6, -44), (6, 53)]
    for axis, seed in zip(axes, seeds):
        seed_rows = means[means["training_seed"] == seed]
        axis.plot(
            seed_rows["cumulative_squared_action"],
            seed_rows["net_forward_progress"],
            color=SEED_COLOURS[int(seed)],
            marker=SEED_MARKERS[int(seed)],
            linewidth=1.2,
            markersize=5,
        )
        for index, row in enumerate(seed_rows.itertuples(index=False)):
            axis.annotate(
                f"w={row.ctrl_cost_weight:g}",
                (row.cumulative_squared_action, row.net_forward_progress),
                xytext=offsets[index % len(offsets)],
                textcoords="offset points",
                fontsize=6.8,
                color=SEED_COLOURS[int(seed)],
                arrowprops={"arrowstyle": "-", "color": "#9AA3AA", "lw": 0.45},
            )
        axis.set_xlabel("Mean cumulative squared action")
        axis.set_title(f"Training seed {int(seed)}")
        axis.grid(color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Mean net forward progress")
    save_figure(fig, output_root, "progress_effort_map")


def main() -> None:
    args = parse_args()
    existing_csv = Path(args.existing_csv).resolve()
    dense_csv = Path(args.dense_csv).resolve()
    additional_csvs = [Path(value).resolve() for value in args.additional_csv]
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_errors = validate_stage1_config(config)
    if config_errors:
        raise ValueError(f"Stage-one config errors: {config_errors}")
    input_csvs = [existing_csv, dense_csv, *additional_csvs]
    data = pd.concat([pd.read_csv(path) for path in input_csvs], ignore_index=True)
    data = data.sort_values(
        ["training_seed", "ctrl_cost_weight", "target_timesteps", "seed"]
    ).reset_index(drop=True)
    quality = validate_data(data, config)

    checkpoints = config["development"]["checkpoint_timesteps"]
    development_screen = config.get("development_screen", {})
    proxy_gate = development_screen.get("primary_proxy_gate", "strict_gain")
    proxy_margin = float(
        development_screen.get("proxy_relative_noninferiority_margin", 0.0)
    )
    checkpoint_screens = {
        int(checkpoint): screen_stage1_endpoint(
            data.to_dict("records"),
            checkpoint=int(checkpoint),
            proxy_relative_noninferiority_margin=proxy_margin,
        )
        for checkpoint in checkpoints
    }
    endpoint_screens = checkpoint_screens[max(checkpoints)]
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_screens: dict[str, list[dict[str, Any]]] = {}
    for scale in (0.5, 1.0, 2.0):
        screens = screen_stage1_endpoint(
            data.to_dict("records"),
            checkpoint=max(checkpoints),
            domains=scaled_domains(scale),
            proxy_relative_noninferiority_margin=proxy_margin,
        )
        sensitivity_screens[str(scale)] = [screen.to_dict() for screen in screens]
        for screen in screens:
            sensitivity_rows.append(
                {
                    "margin_scale": scale,
                    "candidate_weight": screen.candidate_weight,
                    "positive_proxy_seed_count": screen.positive_proxy_seed_count,
                    "noninferior_proxy_seed_count": screen.noninferior_proxy_seed_count,
                    "consistently_harmed_domains": ";".join(
                        screen.consistently_harmed_domains
                    ),
                    "strong_development_candidate": screen.strong_development_candidate,
                    "noninferior_development_candidate": (
                        screen.noninferior_development_candidate
                    ),
                    "primary_gate_candidate": screen_is_candidate(screen, proxy_gate),
                }
            )

    proxy_sensitivity_rows: list[dict[str, Any]] = []
    proxy_sensitivity_screens: dict[str, list[dict[str, Any]]] = {}
    for margin in development_screen.get("proxy_margin_sensitivity", [proxy_margin]):
        margin_value = float(margin)
        screens = screen_stage1_endpoint(
            data.to_dict("records"),
            checkpoint=max(checkpoints),
            proxy_relative_noninferiority_margin=margin_value,
        )
        proxy_sensitivity_screens[str(margin_value)] = [
            screen.to_dict() for screen in screens
        ]
        for screen in screens:
            proxy_sensitivity_rows.append(
                {
                    "proxy_relative_noninferiority_margin": margin_value,
                    "candidate_weight": screen.candidate_weight,
                    "positive_proxy_seed_count": screen.positive_proxy_seed_count,
                    "noninferior_proxy_seed_count": screen.noninferior_proxy_seed_count,
                    "consistently_harmed_domains": ";".join(
                        screen.consistently_harmed_domains
                    ),
                    "noninferior_development_candidate": (
                        screen.noninferior_development_candidate
                    ),
                }
            )

    grid = [float(value) for value in config["development"]["ordered_analysis_grid"]]
    endpoint_by_weight = {
        screen.candidate_weight: screen for screen in endpoint_screens
    }
    onset_intervals: list[dict[str, Any]] = []
    exit_intervals: list[dict[str, Any]] = []
    status_sequence: list[dict[str, Any]] = []
    reference_weight = float(config["development"].get("reference_weight", 0.5))
    side_weights = {
        "lower": sorted((weight for weight in grid if weight < reference_weight), reverse=True),
        "upper": sorted(weight for weight in grid if weight > reference_weight),
    }
    selected_candidates: dict[str, float | None] = {"lower": None, "upper": None}
    for side, weights in side_weights.items():
        previous_weight = reference_weight
        previous_status = False
        for weight in weights:
            screen = endpoint_by_weight[weight]
            current_status = screen_is_candidate(screen, proxy_gate)
            status_sequence.append(
                {
                    "side": side,
                    "candidate_weight": weight,
                    "strong_development_candidate": screen.strong_development_candidate,
                    "noninferior_development_candidate": (
                        screen.noninferior_development_candidate
                    ),
                    "primary_gate_candidate": current_status,
                }
            )
            if current_status and selected_candidates[side] is None:
                selected_candidates[side] = weight
            if current_status and not previous_status:
                onset_intervals.append(
                    {
                        "side": side,
                        "nearer_noncandidate_weight": previous_weight,
                        "farther_candidate_weight": weight,
                    }
                )
            if previous_status and not current_status:
                exit_intervals.append(
                    {
                        "side": side,
                        "nearer_candidate_weight": previous_weight,
                        "farther_noncandidate_weight": weight,
                    }
                )
            previous_weight = weight
            previous_status = current_status

    late_checkpoints = [200000, 250000, 300000]
    persistence_rows: list[dict[str, Any]] = []
    for weight in [weight for weight in grid if weight != reference_weight]:
        statuses = [
            next(
                screen_is_candidate(screen, proxy_gate)
                for screen in checkpoint_screens[checkpoint]
                if screen.candidate_weight == weight
            )
            for checkpoint in late_checkpoints
        ]
        persistence_rows.append(
            {
                "candidate_weight": weight,
                "late_checkpoints_strong_count": int(sum(statuses)),
                "late_window_persistent": bool(sum(statuses) >= 2),
            }
        )

    endpoint_data = data[data["target_timesteps"] == max(checkpoints)].copy()
    cross = cross_rescore_rows(endpoint_data, grid)
    effects = endpoint_effect_rows(endpoint_screens)
    data.to_csv(output_root / "combined_episode_metrics.csv", index=False)
    effects.to_csv(output_root / "endpoint_seed_effects.csv", index=False)
    cross.to_csv(output_root / "cross_rescore_tidy.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(
        output_root / "margin_sensitivity.csv", index=False
    )
    pd.DataFrame(proxy_sensitivity_rows).to_csv(
        output_root / "proxy_margin_sensitivity.csv", index=False
    )
    pd.DataFrame(persistence_rows).to_csv(
        output_root / "late_checkpoint_persistence.csv", index=False
    )

    plot_endpoint_contrasts(effects, output_root)
    plot_domain_matrix(endpoint_screens, output_root)
    plot_cross_rescore(cross, output_root)
    plot_checkpoint_map(checkpoint_screens, output_root, proxy_gate=proxy_gate)
    plot_progress_effort_map(endpoint_data, output_root)

    code_paths = [
        PROJECT_ROOT / "src" / "proxygap" / "stage1.py",
        Path(__file__).resolve(),
    ]
    result = {
        "status": "development_analysis_complete_not_formal_confirmation",
        "input_files": {
            **{str(path): sha256(path) for path in input_csvs},
            str(config_path): sha256(config_path),
        },
        "analysis_code_sha256": {str(path): sha256(path) for path in code_paths},
        "data_quality": quality,
        "endpoint_screens": [screen.to_dict() for screen in endpoint_screens],
        "margin_sensitivity_screens": sensitivity_screens,
        "proxy_margin_sensitivity_screens": proxy_sensitivity_screens,
        "primary_proxy_gate": proxy_gate,
        "proxy_relative_noninferiority_margin": proxy_margin,
        "candidate_status_sequence": status_sequence,
        "selected_candidates_by_side": selected_candidates,
        "first_discrete_onset_interval": (
            onset_intervals[0] if onset_intervals else None
        ),
        "candidate_reentry_intervals": onset_intervals[1:],
        "candidate_exit_intervals": exit_intervals,
        "discrete_onset_intervals": onset_intervals,
        "late_checkpoint_persistence": persistence_rows,
        "claim_boundary": (
            "This result selects or rejects a development interval only. Two "
            "training seeds cannot confirm reproducibility, estimate a continuous "
            "change point or support a universal reward-hacking claim."
        ),
    }
    (output_root / "stage1_development_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    manifest_paths = sorted(path for path in output_root.rglob("*") if path.is_file())
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in manifest_paths
    ]
    pd.DataFrame(manifest_rows).to_csv(output_root / "output_manifest.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
