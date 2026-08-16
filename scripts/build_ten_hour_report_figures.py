"""Build compact, report-ready figures from the frozen development outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis"
OUTPUT = ROOT / "output" / "report_assets_20260816"
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)

CONDITION_LABELS = {
    "Rt0p1_Rvy0__K0": "posture 0.10\nlateral 0.00\nno guardrail",
    "Rt0p1_Rvy0__K1p1": "posture 0.10\nlateral 0.00\nslew 1.1",
    "Rt0p1_Rvy0p05__K0": "posture 0.10\nlateral 0.05\nno guardrail",
    "Rt0p1_Rvy0p05__K1p1": "posture 0.10\nlateral 0.05\nslew 1.1",
    "Rt0p1_Rvy0p1__K0": "posture 0.10\nlateral 0.10\nno guardrail",
    "Rt0p1_Rvy0p1__K1p1": "posture 0.10\nlateral 0.10\nslew 1.1",
}
CONDITION_ORDER = list(CONDITION_LABELS)
SEED_COLOURS = {41301: "#0072B2", 41302: "#D55E00", 41303: "#7A5195"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def domain_heatmap() -> None:
    frame = pd.read_csv(ANALYSIS / "intent_sensitivity" / "condition_domain_compliance.csv")
    frame = frame.set_index("condition_id").loc[CONDITION_ORDER]
    columns = [
        "horizon_and_health",
        "forward_tracking",
        "no_sustained_inversion",
        "torso_stability",
        "directional_control",
        "path_directness",
        "action_smoothness",
        "low_saturation",
    ]
    labels = [
        "Horizon +\nhealth",
        "Forward\ntracking",
        "No sustained\ninversion",
        "Torso\nstability",
        "Direction\ncontrol",
        "Path\ndirectness",
        "Applied-action\nsmoothness",
        "Low\nsaturation",
    ]
    values = frame[columns].to_numpy(float) * 100.0
    cmap = LinearSegmentedColormap.from_list(
        "compliance", ["#9D174D", "#F2C14E", "#F5F3C1", "#2A9D8F", "#005F56"]
    )
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    image = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(
        np.arange(len(CONDITION_ORDER)),
        [CONDITION_LABELS[item].replace("\n", " | ") for item in CONDITION_ORDER],
    )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            colour = "white" if value <= 22 or value >= 78 else "#111827"
            ax.text(column, row, f"{value:.0f}", ha="center", va="center", color=colour)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colourbar.set_label("Evaluation episodes passing each domain (%)")
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(
        0.0,
        -0.23,
        "Every condition had 0% overall intent compliance because no episode passed all domains.",
        transform=ax.transAxes,
        color="#5B6470",
    )
    fig.subplots_adjust(left=0.23, bottom=0.27, right=0.93, top=0.96)
    save(fig, "figure_domain_compliance")


def lateral_seed_plot() -> None:
    frame = pd.read_csv(
        ANALYSIS
        / "lateral_velocity"
        / "endpoint_policy_lateral_velocity_metrics.csv"
    )
    frame["lateral_weight"] = frame["condition_id"].map(
        {
            "Rt0p1_Rvy0__K0": 0.0,
            "Rt0p1_Rvy0__K1p1": 0.0,
            "Rt0p1_Rvy0p05__K0": 0.05,
            "Rt0p1_Rvy0p05__K1p1": 0.05,
            "Rt0p1_Rvy0p1__K0": 0.10,
            "Rt0p1_Rvy0p1__K1p1": 0.10,
        }
    )
    frame["guardrail"] = frame["condition_id"].str.endswith("K1p1")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True, sharey=True)
    for ax, guardrail, title in zip(
        axes,
        [False, True],
        ["No action guardrail", "Action-slew guardrail 1.1"],
        strict=True,
    ):
        subset = frame.loc[frame["guardrail"] == guardrail]
        marker = "s" if guardrail else "o"
        linestyle = "--" if guardrail else "-"
        for seed, colour in SEED_COLOURS.items():
            row = subset.loc[subset["training_seed"] == seed].sort_values(
                "lateral_weight"
            )
            ax.plot(
                row["lateral_weight"],
                row["mean_abs_lateral_velocity_error"],
                color=colour,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5,
                label=f"training seed {seed}",
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Lateral-velocity shaping weight")
        ax.set_xticks([0.0, 0.05, 0.10])
        ax.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    axes[0].set_ylabel("Mean absolute lateral velocity error (m/s)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    axes[0].text(-0.18, 1.08, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.12, 1.08, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.79, bottom=0.17, wspace=0.14)
    save(fig, "figure_lateral_velocity_by_seed")


def guardrail_mechanism_plot() -> None:
    frame = pd.read_csv(ANALYSIS / "endpoint_policy_metrics.csv")
    constrained = frame.loc[frame["constraint_id"] == "Kslew_1p1"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    x = np.arange(len(CONDITION_ORDER[1::2]))
    labels = ["lateral 0.00", "lateral 0.05", "lateral 0.10"]
    width = 0.22
    for index, (seed, colour) in enumerate(SEED_COLOURS.items()):
        row = constrained.loc[constrained["training_seed"] == seed].set_index(
            "condition_id"
        ).loc[CONDITION_ORDER[1::2]]
        bars = axes[0].bar(
            x + (index - 1) * width,
            row["proposed_normalised_action_roughness"],
            width,
            color=colour,
            alpha=0.88,
            label=f"training seed {seed}",
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    axes[0].axhline(0.04, color="#111827", linestyle=":", linewidth=1.5, label="intent threshold 0.04")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Proposed-action roughness")
    axes[0].set_title("Policy output before projection", loc="left")
    axes[0].grid(axis="y", color="#D8DEE6", linewidth=0.7)

    summary = pd.read_csv(ANALYSIS / "endpoint_condition_summary.csv").set_index(
        "condition_id"
    ).loc[CONDITION_ORDER[1::2]]
    applied = summary["normalised_action_roughness_mean"].to_numpy(float)
    intervention = summary["action_slew_intervention_rate_mean"].to_numpy(float) * 100
    bars = axes[1].bar(x, applied, color=["#2A9D8F", "#E69F00", "#7A5195"], width=0.55)
    axes[1].axhline(0.04, color="#111827", linestyle=":", linewidth=1.5)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Applied-action roughness")
    axes[1].set_title("Action after 1.1 projection", loc="left")
    axes[1].set_ylim(0, 0.045)
    axes[1].grid(axis="y", color="#D8DEE6", linewidth=0.7)
    for bar, rate in zip(bars, intervention, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.0015,
            f"{rate:.1f}%\nintervened",
            ha="center",
            va="top",
            color="white",
            fontsize=8,
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    axes[0].text(-0.18, 1.08, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.16, 1.08, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.79, bottom=0.17, wspace=0.25)
    save(fig, "figure_guardrail_mechanism")


def video_evidence_panel() -> None:
    frame_root = (
        ROOT
        / "artifacts"
        / "dev"
        / "hg_r3_obsfix_v1"
        / "videos"
        / "qa_frames"
    )
    items = [
        (
            frame_root / "Rt0p1_Rvy0__K0__tr41301__ev51301__t300000__10s.png",
            "a  Posture only: ended at 3.4 s",
        ),
        (
            frame_root
            / "Rt0p1_Rvy0p05__K1p1__tr41301__ev51301__t300000__49s.png",
            "b  Lateral 0.05 + slew 1.1: large lateral drift",
        ),
        (
            frame_root
            / "Rt0p1_Rvy0p1__K1p1__tr41301__ev51301__t300000__49s.png",
            "c  Lateral 0.10 + slew 1.1: inverted posture",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.15))
    for ax, (path, label) in zip(axes, items, strict=True):
        ax.imshow(plt.imread(path))
        ax.set_title(label, loc="left", fontsize=9)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.01)
    save(fig, "figure_video_evidence")


def equation_panel() -> None:
    fig = plt.figure(figsize=(10.4, 2.35))
    fig.patch.set_facecolor("white")
    equations = [
        r"$r_t^{base}=r_t^{forward}+r_t^{healthy}-0.5\|a_t\|_2^2-r_t^{contact}$",
        r"$\phi_{\theta}(\theta_t)=\frac{1-\cos\theta_t}{2},\qquad r_t^{\theta}=-0.1\phi_{\theta}(\theta_t)$",
        r"$\phi_y(v_{y,t})=\tanh\!\left[\left(\frac{v_{y,t}-0}{1\,\mathrm{m\,s^{-1}}}\right)^2\right],\qquad r_t^y=-\lambda_y\phi_y(v_{y,t})$",
        r"$\tilde a_t=\tilde a_{t-1}+\min\!\left(1,\frac{1.1}{\|a_t-\tilde a_{t-1}\|_2+\epsilon}\right)(a_t-\tilde a_{t-1})$",
    ]
    for index, equation in enumerate(equations):
        fig.text(0.04, 0.84 - index * 0.23, equation, fontsize=12, color="#172A3A")
    fig.text(0.012, 0.92, "a", fontsize=13, fontweight="bold")
    plt.axis("off")
    save(fig, "figure_experiment_equations")


def main() -> None:
    domain_heatmap()
    lateral_seed_plot()
    guardrail_mechanism_plot()
    video_evidence_panel()
    equation_panel()
    print(OUTPUT)


if __name__ == "__main__":
    main()
