"""Create publication-ready figures from the frozen multi-objective evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "artifacts/dev/v4_multiobjective_candidate_selection_v1_20260820/selection.json"
FORMAL = ROOT / "artifacts/dev/v4_pair0_multiobjective_full_map_final_v1_20260820/attempt_0"
HEIGHTS = ROOT / "artifacts/frozen/fixed_quad_terrain_v2_approved_20260818/map/scene/heights_m.npy"
OUTPUT = ROOT / "docs/figures/v4_multiobjective_final"

INK = "#20262E"
TEAL = "#168C8C"
BLUE = "#2E6E9E"
AMBER = "#D8902F"
RED = "#B84A3A"
GREY = "#87919B"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "#F4F1EA",
            "axes.facecolor": "#FCFBF7",
            "savefig.facecolor": "#F4F1EA",
        }
    )


def normalise_svg(path: Path) -> None:
    """Remove generator-only trailing spaces so repository checks stay clean."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def candidate_figure() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows = selection["scored_candidates"]
    times = np.asarray([float(row["elapsed_seconds"]) for row in rows])
    works = np.asarray([float(row["actuator_positive_mechanical_work_total_j"]) / 1000.0 for row in rows])
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.scatter(times, works, s=58, c=GREY, alpha=0.72, edgecolors="white", linewidths=0.8, label="15 feasible candidates")
    selected = {
        "time / balanced": selection["selections"]["time_priority"],
        "energy": selection["selections"]["energy_priority"],
    }
    colours = {"time / balanced": TEAL, "energy": AMBER}
    annotation_offsets = {"time / balanced": (9, 12), "energy": (9, -17)}
    for label, row in selected.items():
        x = float(row["elapsed_seconds"])
        y = float(row["positive_mechanical_work_j"]) / 1000.0
        ax.scatter([x], [y], s=145, marker="*", c=colours[label], edgecolors=INK, linewidths=0.7, zorder=5, label=f"selected: {label}")
        ax.annotate(
            row["selected_candidate"],
            (x, y),
            xytext=annotation_offsets[label],
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
            color=colours[label],
        )
    ax.set_title("Preference selection operated only within the feasible candidate bank", loc="left", weight="bold")
    ax.set_xlabel("Completion time (s)")
    ax.set_ylabel("Positive mechanical work proxy (kJ)")
    ax.set_ylim(float(works.min()) - 2.4, float(works.max()) + 1.4)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.text(
        0.50,
        0.005,
        "All points first satisfied arrival and safety gates. Mechanical work is not battery energy.",
        ha="center",
        fontsize=8,
        color="#59636C",
    )
    fig.savefig(OUTPUT / "candidate_time_work_scatter.png", dpi=220)
    fig.savefig(OUTPUT / "candidate_time_work_scatter.svg")
    normalise_svg(OUTPUT / "candidate_time_work_scatter.svg")
    plt.close(fig)


def formal_seed_figure() -> None:
    contracts = ["time_and_balanced", "energy_priority"]
    seeds = [690223864, 1864999454, 952993985]
    data: dict[str, list[dict[str, float]]] = {key: [] for key in contracts}
    for contract in contracts:
        for seed in seeds:
            row = json.loads((FORMAL / contract / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))
            data[contract].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), constrained_layout=True)
    metrics = [
        ("elapsed_seconds", "Completion time (s)"),
        ("actuator_positive_mechanical_work_total_j", "Positive work proxy (kJ)"),
        ("path_length_m", "Actual path length (m)"),
    ]
    x = np.arange(len(seeds))
    for ax, (key, label) in zip(axes, metrics, strict=True):
        for contract, colour, marker, name in (
            ("time_and_balanced", TEAL, "o", "time / balanced route"),
            ("energy_priority", AMBER, "s", "energy route"),
        ):
            values = np.asarray([float(row[key]) for row in data[contract]])
            if key == "actuator_positive_mechanical_work_total_j":
                values = values / 1000.0
            ax.plot(x, values, color=colour, marker=marker, markersize=7, linewidth=1.8, label=name)
            ax.axhline(float(np.mean(values)), color=colour, linewidth=1.0, alpha=0.35, linestyle="--")
        ax.set_title(label, loc="left", weight="bold")
        ax.set_xticks(x, ["seed 1", "seed 2", "seed 3"])
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Formal performance varied by reset seed; both route contracts completed 3/3", x=0.02, ha="left", fontsize=14, weight="bold")
    fig.savefig(OUTPUT / "formal_per_seed_outcomes.png", dpi=220)
    fig.savefig(OUTPUT / "formal_per_seed_outcomes.svg")
    normalise_svg(OUTPUT / "formal_per_seed_outcomes.svg")
    plt.close(fig)


def route_figure() -> None:
    heights = np.load(HEIGHTS, allow_pickle=False)
    half = 40.0
    representatives = [
        ("Time priority", "time_and_balanced", 690223864, TEAL),
        ("Balanced", "time_and_balanced", 1864999454, BLUE),
        ("Energy priority", "energy_priority", 952993985, AMBER),
    ]
    final_config = json.loads((FORMAL / "frozen_config.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), constrained_layout=True)
    for ax, (title, contract, seed, colour) in zip(axes, representatives, strict=True):
        trace = load_csv(FORMAL / contract / f"seed_{seed}" / "control_trace.csv")
        route = load_csv(ROOT / final_config["route_contracts"][contract]["route"])
        ax.imshow(heights, origin="upper", extent=(-half, half, -half, half), cmap="terrain", alpha=0.78)
        ax.plot([float(row["x_m"]) for row in route], [float(row["y_m"]) for row in route], color="#44D7E4", linewidth=1.6, label="planned")
        ax.plot([float(row["x_m"]) for row in trace], [float(row["y_m"]) for row in trace], color=colour, linewidth=1.2, label="actual")
        ax.scatter([-34, 34], [-34, 34], c=["white", RED], edgecolors=INK, s=42, zorder=5)
        ax.set_title(f"{title}\nseed {seed}", weight="bold")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Representative formal routes: planned waypoints versus executed trajectories", x=0.02, ha="left", fontsize=14, weight="bold")
    fig.savefig(OUTPUT / "representative_planned_vs_actual_routes.png", dpi=220)
    fig.savefig(OUTPUT / "representative_planned_vs_actual_routes.svg")
    normalise_svg(OUTPUT / "representative_planned_vs_actual_routes.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    candidate_figure()
    formal_seed_figure()
    route_figure()
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
