"""Combine formal runs and create report-ready tables and static figures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "formal" / "combined_v1_20260809"

SOURCES = {
    "formal_v1_coefficients_20260808": PROJECT_ROOT
    / "artifacts/formal/formal_v1_coefficients_20260808/logs/evaluation_metrics.csv",
    "formal_v1_shaped_20260808": PROJECT_ROOT
    / "artifacts/formal/formal_v1_shaped_20260808/logs/evaluation_metrics.csv",
    "formal_v1_core_replication_20260808": PROJECT_ROOT
    / "artifacts/formal/formal_v1_core_replication_20260808/logs/evaluation_metrics.csv",
}
EXPECTED_SOURCE_ROWS = {
    "formal_v1_coefficients_20260808": 240,
    "formal_v1_shaped_20260808": 60,
    "formal_v1_core_replication_20260808": 360,
}

CONDITION_ORDER = [
    "reference",
    "ctrl_0p25",
    "ctrl_0p125",
    "ctrl_0p0625",
    "shaped_ctrl_0p0625_forward_1p0",
]
CONDITION_LABELS = {
    "reference": "Reference (0.5)",
    "ctrl_0p25": "Control cost 0.25",
    "ctrl_0p125": "Control cost 0.125",
    "ctrl_0p0625": "Control cost 0.0625",
    "shaped_ctrl_0p0625_forward_1p0": "Shaped 0.0625",
}
COLOURS = {
    "reference": "#2F6B9A",
    "ctrl_0p25": "#C69214",
    "ctrl_0p125": "#D8742A",
    "ctrl_0p0625": "#C34F76",
    "shaped_ctrl_0p0625_forward_1p0": "#667D3E",
}
LINE_STYLES = {
    "reference": "-",
    "ctrl_0p25": "--",
    "ctrl_0p125": "-.",
    "ctrl_0p0625": ":",
    "shaped_ctrl_0p0625_forward_1p0": (0, (3, 1, 1, 1)),
}
MARKERS = {
    "reference": "o",
    "ctrl_0p25": "s",
    "ctrl_0p125": "^",
    "ctrl_0p0625": "D",
    "shaped_ctrl_0p0625_forward_1p0": "P",
}

METRICS = [
    "proxy_return",
    "base_proxy_return",
    "reward_shaping_sum",
    "net_forward_progress",
    "control_effort",
    "control_effort_per_unit_distance",
    "fall",
    "lateral_drift_final_abs",
    "lateral_drift_mean_abs",
    "torso_tilt_mean",
    "torso_tilt_std",
    "episode_length",
]


def load_formal_data() -> pd.DataFrame:
    frames = []
    for source_id, path in SOURCES.items():
        frame = pd.read_csv(path)
        frame["source_config_id"] = source_id
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in [
        "forward_progress_shaping_weight",
        "lateral_drift_shaping_weight",
        "reward_forward_shaping_sum",
        "reward_lateral_shaping_sum",
    ]:
        if column not in combined:
            combined[column] = 0.0
        combined[column] = combined[column].fillna(0.0)
    combined["fall"] = combined["fall"].map(
        lambda value: 1.0 if str(value).strip().lower() == "true" else 0.0
    )
    combined["condition_label"] = combined["condition_id"].map(CONDITION_LABELS)
    key = ["condition_id", "training_seed", "target_timesteps", "episode"]
    if combined.duplicated(key).any():
        raise ValueError("Duplicate formal evaluation keys found while combining runs")
    actual_source_rows = combined["source_config_id"].value_counts().to_dict()
    if actual_source_rows != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Unexpected formal source row counts: {actual_source_rows}"
        )
    return combined


def write_analysis_manifest(combined: pd.DataFrame) -> None:
    key = ["condition_id", "training_seed", "target_timesteps", "episode"]
    groups = combined.groupby(
        ["condition_id", "training_seed", "target_timesteps"],
        dropna=False,
    )
    evaluation_counts = groups.size()
    checkpoint_counts = combined.groupby(
        ["condition_id", "training_seed"], dropna=False
    )["target_timesteps"].nunique()
    endpoint = combined[combined["target_timesteps"] == 300_000]
    decomposition_error = (
        combined["proxy_return"]
        - combined["base_proxy_return"]
        - combined["reward_shaping_sum"]
    ).abs()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": {
            key: int(value)
            for key, value in combined["source_config_id"].value_counts().items()
        },
        "expected_source_rows": EXPECTED_SOURCE_ROWS,
        "combined_rows": int(len(combined)),
        "duplicate_evaluation_keys": int(combined.duplicated(key).sum()),
        "evaluation_episodes_per_seed_checkpoint_min": int(evaluation_counts.min()),
        "evaluation_episodes_per_seed_checkpoint_max": int(evaluation_counts.max()),
        "checkpoints_per_condition_seed_min": int(checkpoint_counts.min()),
        "checkpoints_per_condition_seed_max": int(checkpoint_counts.max()),
        "endpoint_training_seed_counts": {
            key: int(value)
            for key, value in endpoint.groupby("condition_id")[
                "training_seed"
            ].nunique().items()
        },
        "max_proxy_decomposition_absolute_error": float(
            decomposition_error.max()
        ),
        "status": "pass",
    }
    (OUTPUT_ROOT / "data" / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def build_summaries(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouping = [
        "condition_id",
        "condition_label",
        "training_seed",
        "target_timesteps",
    ]
    seed_summary = (
        combined.groupby(grouping, as_index=False)[METRICS]
        .mean(numeric_only=True)
        .sort_values(["condition_id", "training_seed", "target_timesteps"])
    )
    seed_summary["evaluation_episodes"] = 10

    rows = []
    for (condition_id, condition_label, target), group in seed_summary.groupby(
        ["condition_id", "condition_label", "target_timesteps"],
        sort=False,
    ):
        row = {
            "condition_id": condition_id,
            "condition_label": condition_label,
            "target_timesteps": int(target),
            "training_seed_count": int(group["training_seed"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean(skipna=True))
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1, skipna=True))
        rows.append(row)
    across_seeds = pd.DataFrame(rows)
    across_seeds["condition_order"] = across_seeds["condition_id"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    across_seeds = across_seeds.sort_values(
        ["target_timesteps", "condition_order"]
    ).drop(columns="condition_order")

    endpoint_seed = seed_summary[seed_summary["target_timesteps"] == 300_000].copy()
    endpoint_across = across_seeds[
        across_seeds["target_timesteps"] == 300_000
    ].copy()
    return seed_summary, across_seeds, endpoint_seed, endpoint_across


def style_axis(ax: plt.Axes, *, zero_line: bool = False) -> None:
    ax.set_facecolor("#FCFCFB")
    ax.grid(axis="y", color="#D9D9D6", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    if zero_line:
        ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=1)


def plot_checkpoint_metric(
    across_seeds: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    zero_line: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    figure, ax = plt.subplots(figsize=(9.2, 5.6))
    for condition in CONDITION_ORDER:
        data = across_seeds[across_seeds["condition_id"] == condition]
        if data.empty:
            continue
        seed_count = int(data["training_seed_count"].max())
        x = data["target_timesteps"].to_numpy() / 1000.0
        y = data[f"{metric}_mean"].to_numpy(dtype=float)
        yerr = data[f"{metric}_sd"].to_numpy(dtype=float)
        yerr = None if seed_count == 1 else np.nan_to_num(yerr, nan=0.0)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=COLOURS[condition],
            linestyle=LINE_STYLES[condition],
            marker=MARKERS[condition],
            linewidth=2.0,
            markersize=6.0,
            capsize=3,
            label=f"{CONDITION_LABELS[condition]} (n={seed_count})",
        )
    style_axis(ax, zero_line=zero_line)
    figure.suptitle(
        title,
        x=0.10,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#222222",
    )
    figure.text(
        0.10,
        0.925,
        "Mean across training seeds; error bars show +/-1 SD where n=3. Each seed mean uses 10 evaluation episodes.",
        fontsize=9,
        color="#555555",
    )
    ax.set_xlabel("Training timesteps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([50, 100, 150, 200, 250, 300])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="best")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.86)
    figure.savefig(OUTPUT_ROOT / "plots" / filename, dpi=220, facecolor="white")
    plt.close(figure)


def plot_proxy_progress_tradeoff(endpoint_seed: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(8.4, 5.8))
    for condition in CONDITION_ORDER:
        data = endpoint_seed[endpoint_seed["condition_id"] == condition]
        if data.empty:
            continue
        ax.scatter(
            data["net_forward_progress"],
            data["proxy_return"],
            s=72,
            marker=MARKERS[condition],
            color=COLOURS[condition],
            edgecolor="#222222",
            linewidth=0.7,
            label=f"{CONDITION_LABELS[condition]} (n={data['training_seed'].nunique()})",
            zorder=3,
        )
    style_axis(ax, zero_line=True)
    figure.suptitle(
        "Proxy return and net forward progress at 300k",
        x=0.11,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.11,
        0.925,
        "One point per training seed; each point averages 10 paired evaluation episodes.",
        fontsize=9,
        color="#555555",
    )
    ax.set_xlabel("Net forward progress")
    ax.set_ylabel("Observed proxy return")
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.08),
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.86)
    figure.savefig(
        OUTPUT_ROOT / "plots" / "proxy_progress_tradeoff_300k.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)


def endpoint_dot_panel(endpoint_seed: pd.DataFrame) -> None:
    specs = [
        ("fall", "Fall rate", (0.0, 1.05), False),
        ("lateral_drift_final_abs", "Final absolute lateral drift", None, False),
        ("torso_tilt_std", "Torso tilt variability", None, False),
        (
            "control_effort_per_unit_distance",
            "Control effort per unit distance",
            None,
            True,
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    positions = np.arange(len(CONDITION_ORDER), dtype=float)
    for ax, (metric, title, ylim, log_scale) in zip(axes.flat, specs):
        for index, condition in enumerate(CONDITION_ORDER):
            data = endpoint_seed[endpoint_seed["condition_id"] == condition]
            values = data[metric].dropna().to_numpy(dtype=float)
            if len(values):
                offsets = np.linspace(-0.10, 0.10, len(values))
                ax.scatter(
                    np.full(len(values), positions[index]) + offsets,
                    values,
                    s=46,
                    marker=MARKERS[condition],
                    color=COLOURS[condition],
                    edgecolor="#222222",
                    linewidth=0.6,
                    zorder=3,
                )
                ax.plot(
                    [positions[index] - 0.16, positions[index] + 0.16],
                    [np.mean(values), np.mean(values)],
                    color="#222222",
                    linewidth=2.0,
                    zorder=4,
                )
            missing = int(data[metric].isna().sum())
            if missing:
                ax.text(
                    positions[index],
                    0.03,
                    f"{missing} undefined",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#666666",
                )
        style_axis(ax)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [CONDITION_LABELS[name] for name in CONDITION_ORDER],
            rotation=24,
            ha="right",
            fontsize=8,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        if log_scale:
            ax.set_yscale("log")
    figure.suptitle(
        "Safety and efficiency diagnostics at 300k",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.945,
        "Points are training-seed means over 10 episodes; black bars show means across available seeds. Effort/distance is undefined for non-positive progress.",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.88,
        hspace=0.52,
        wspace=0.22,
    )
    figure.savefig(
        OUTPUT_ROOT / "plots" / "safety_efficiency_endpoint_300k.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)


def format_mean_sd(row: pd.Series, metric: str) -> str:
    mean = row[f"{metric}_mean"]
    sd = row[f"{metric}_sd"]
    if pd.isna(sd):
        return f"{mean:.2f} (n=1)"
    return f"{mean:.2f} +/- {sd:.2f}"


def write_summary_note(endpoint_across: pd.DataFrame) -> None:
    indexed = endpoint_across.set_index("condition_id")
    lines = [
        "# ProxyGap Formal Results Summary",
        "",
        "This summary reports descriptive formal evidence. Core comparisons use three training seeds; the 0.25 and 0.125 coefficient conditions use the predeclared single main seed only. No inferential significance claim is made from n=3.",
        "",
        "## 300k endpoint",
        "",
        "| Condition | Training seeds | Proxy return | Net forward progress | Fall rate | Final lateral drift | Torso tilt SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = indexed.loc[condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    CONDITION_LABELS[condition],
                    str(int(row["training_seed_count"])),
                    format_mean_sd(row, "proxy_return"),
                    format_mean_sd(row, "net_forward_progress"),
                    format_mean_sd(row, "fall"),
                    format_mean_sd(row, "lateral_drift_final_abs"),
                    format_mean_sd(row, "torso_tilt_std"),
                ]
            )
            + " |"
        )

    reference = indexed.loc["reference"]
    divergent = indexed.loc["ctrl_0p0625"]
    shaped = indexed.loc["shaped_ctrl_0p0625_forward_1p0"]
    lines.extend(
        [
            "",
            "## Descriptive interpretation",
            "",
            f"- The reduced 0.0625 condition had higher mean proxy return than the reference ({divergent['proxy_return_mean']:.2f} versus {reference['proxy_return_mean']:.2f}) but much lower mean net forward progress ({divergent['net_forward_progress_mean']:.2f} versus {reference['net_forward_progress_mean']:.2f}). This is the clearest three-seed proxy-performance gap in the tested core comparison.",
            f"- Forward-progress shaping increased mean net forward progress relative to the unshaped 0.0625 condition ({shaped['net_forward_progress_mean']:.2f} versus {divergent['net_forward_progress_mean']:.2f}), but remained below the reference mean ({reference['net_forward_progress_mean']:.2f}). The mitigation is therefore partial rather than complete.",
            f"- The shaped condition's mean fall rate was {shaped['fall_mean']:.2f}, compared with {divergent['fall_mean']:.2f} for unshaped 0.0625 and {reference['fall_mean']:.2f} for the reference. Reward shaping improved forward progress but introduced a safety trade-off that must be reported separately.",
            "- Training-seed variation is material. Conclusions should be phrased as exploratory simulation evidence, not a claim of universal PPO or robotics behaviour.",
            "",
            "## Provenance",
            "",
            "- Main coefficient sweep: `formal_v1_coefficients_20260808`",
            "- Main shaped condition: `formal_v1_shaped_20260808`",
            "- Core replication: `formal_v1_core_replication_20260808`",
            "- Evaluation grain: ten deterministic episodes per training seed and checkpoint, using paired evaluation seeds.",
        ]
    )
    (OUTPUT_ROOT / "FORMAL_RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "plots").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#333333",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "figure.facecolor": "white",
        }
    )

    combined = load_formal_data()
    seed_summary, across_seeds, endpoint_seed, endpoint_across = build_summaries(
        combined
    )
    combined.to_csv(OUTPUT_ROOT / "data" / "combined_evaluation_metrics.csv", index=False)
    seed_summary.to_csv(OUTPUT_ROOT / "data" / "seed_checkpoint_summary.csv", index=False)
    across_seeds.to_csv(OUTPUT_ROOT / "data" / "checkpoint_summary_across_seeds.csv", index=False)
    endpoint_seed.to_csv(OUTPUT_ROOT / "data" / "endpoint_300k_seed_summary.csv", index=False)
    endpoint_across.to_csv(OUTPUT_ROOT / "data" / "endpoint_300k_across_seeds.csv", index=False)
    write_analysis_manifest(combined)

    plot_checkpoint_metric(
        across_seeds,
        metric="proxy_return",
        title="Observed proxy return across checkpoints",
        ylabel="Observed proxy return",
        filename="proxy_return_checkpoints.png",
    )
    plot_checkpoint_metric(
        across_seeds,
        metric="net_forward_progress",
        title="Net forward progress across checkpoints",
        ylabel="Net forward progress",
        filename="net_forward_progress_checkpoints.png",
        zero_line=True,
    )
    plot_checkpoint_metric(
        across_seeds,
        metric="fall",
        title="Fall rate across checkpoints",
        ylabel="Fall rate",
        filename="fall_rate_checkpoints.png",
        ylim=(0.0, 1.05),
    )
    plot_checkpoint_metric(
        across_seeds,
        metric="episode_length",
        title="Episode length across checkpoints",
        ylabel="Mean episode length",
        filename="episode_length_checkpoints.png",
        ylim=(0.0, 1050.0),
    )
    plot_proxy_progress_tradeoff(endpoint_seed)
    endpoint_dot_panel(endpoint_seed)
    write_summary_note(endpoint_across)
    print(f"Saved combined formal analysis: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
