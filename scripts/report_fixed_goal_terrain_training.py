"""Create an auditable report bundle for the fixed-map pilot training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.metrics import quaternion_tilt_angle  # noqa: E402
from run_fixed_goal_terrain_training import make_task_env  # noqa: E402


DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_training_20260818"
    / "seed_62801"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--representative-seed", type=int, default=74803)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def sample_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def compact_row(row: dict[str, str]) -> dict[str, Any]:
    initial = float(row["fixed_goal_initial_distance_m"])
    minimum = float(row["fixed_goal_minimum_distance_m"])
    return {
        "evaluation_seed": int(row["evaluation_seed"]),
        "success": as_bool(row["fixed_goal_success"]),
        "qualified": as_bool(row["fixed_goal_qualified_no_fall_no_airborne_no_slip"]),
        "fall": as_bool(row["fall"]),
        "episode_steps": int(row["episode_length"]),
        "net_progress_m": float(row["fixed_goal_net_progress_m"]),
        "best_progress_m": initial - minimum,
        "final_distance_m": float(row["fixed_goal_final_distance_m"]),
        # The base contact monitor is authoritative for the historical run.
        # An earlier task-adapter field read a non-exported step key and
        # therefore incorrectly remained zero; use the directly accumulated
        # four-foot no-contact fraction retained in every raw episode row.
        "airborne_step_fraction": float(row["airborne_step_fraction"]),
        "contact_speed_exceedance_fraction": float(row["task_slip_violation_step_fraction"]),
        "maximum_contact_slip_speed_m_per_s": float(
            row["task_maximum_contact_slip_speed_m_per_s"]
        ),
        "maximum_torso_tilt_rad": float(row["terrain_relative_maximum_torso_tilt_rad"]),
        "termination_category": row["termination_category"],
    }


def summarise_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "qualified_episodes": sum(bool(row["qualified"]) for row in rows),
        "falls": sum(bool(row["fall"]) for row in rows),
        "net_progress_m": sample_summary([float(row["net_progress_m"]) for row in rows]),
        "best_progress_m": sample_summary([float(row["best_progress_m"]) for row in rows]),
        "airborne_step_fraction": sample_summary(
            [float(row["airborne_step_fraction"]) for row in rows]
        ),
        "contact_speed_exceedance_fraction": sample_summary(
            [float(row["contact_speed_exceedance_fraction"]) for row in rows]
        ),
        "maximum_contact_slip_speed_m_per_s": sample_summary(
            [float(row["maximum_contact_slip_speed_m_per_s"]) for row in rows]
        ),
        "maximum_torso_tilt_rad": sample_summary(
            [float(row["maximum_torso_tilt_rad"]) for row in rows]
        ),
    }


def run_trace(
    *,
    config: dict[str, Any],
    v22_config: dict[str, Any],
    scene_path: Path,
    model_path: Path,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = PPO.load(model_path, device="cpu")
    env = make_task_env(
        config,
        v22_config,
        xml_path=scene_path,
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=int(config["evaluation"]["full_route_max_episode_steps"]),
        cruise_speed=float(config["evaluation"]["cruise_speed_m_per_s"]),
        terminate_on_success=True,
    )
    observation, info = env.reset(seed=seed)
    dt = float(env.unwrapped.dt)

    def snapshot(step: int, reward: float, live_info: dict[str, Any]) -> dict[str, Any]:
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        position = qpos[:2].copy()
        contact_mask = np.asarray(
            live_info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool
        )
        slip_speeds = np.asarray(
            live_info.get(
                "proxygap_foot_contact_tangential_speeds_m_per_s_step", np.zeros(4)
            ),
            dtype=np.float64,
        )
        active = slip_speeds[contact_mask] if contact_mask.shape == (4,) else np.asarray([])
        max_contact_speed = float(active.max()) if active.size else 0.0
        return {
            "model": label,
            "evaluation_seed": seed,
            "step": step,
            "time_seconds": step * dt,
            "x_m": float(position[0]),
            "y_m": float(position[1]),
            "terrain_height_m": float(env._terrain_height(float(position[0]), float(position[1]))),
            "torso_z_m": float(qpos[2]),
            "torso_tilt_rad": float(quaternion_tilt_angle(qpos[3:7])),
            "distance_to_goal_m": float(np.linalg.norm(env.goal_xy - position)),
            "reward": float(reward),
            "airborne": bool(
                contact_mask.shape == (4,) and not np.any(contact_mask)
            ),
            "support_count": int(contact_mask.sum()) if contact_mask.shape == (4,) else 0,
            "maximum_contact_tangential_speed_m_per_s": max_contact_speed,
            "contact_speed_threshold_exceeded": bool(
                max_contact_speed > float(config["task_adapter"]["slip_speed_threshold_m_per_s"])
            ),
        }

    trace = [snapshot(0, 0.0, info)]
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=bool(config["evaluation"]["deterministic_policy"]),
        )
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        trace.append(snapshot(step, reward, info))
    summary = env.episode_summary()
    env.close()
    return trace, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_report(
    *,
    report_dir: Path,
    config: dict[str, Any],
    baseline: list[dict[str, Any]],
    final: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    monitor_rows: list[dict[str, str]],
) -> Path:
    approved = config["approved_map"]
    heights = np.load(ROOT / approved["heights_path"], allow_pickle=False)
    extent = float(approved["map_half_extent_m"])
    coordinates = np.linspace(-extent, extent, heights.shape[0])
    start = np.asarray(approved["start_xy_m"], dtype=np.float64)
    goal = np.asarray(approved["goal_xy_m"], dtype=np.float64)

    terrain_cmap = LinearSegmentedColormap.from_list(
        "terrain",
        ["#17324d", "#2b6f77", "#7ba05b", "#d0b46c", "#f2e5c4"],
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    figure.patch.set_facecolor("#f5f2eb")
    for axis in axes.ravel():
        axis.set_facecolor("#fbfaf6")

    baseline_xy = np.asarray([[row["x_m"], row["y_m"]] for row in baseline])
    final_xy = np.asarray([[row["x_m"], row["y_m"]] for row in final])

    ax = axes[0, 0]
    image = ax.imshow(
        heights,
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap=terrain_cmap,
        aspect="equal",
    )
    levels = np.linspace(float(heights.min()), float(heights.max()), 13)
    ax.contour(coordinates, coordinates, heights, levels=levels, colors="#f7f3e8", linewidths=0.45, alpha=0.65)
    ax.plot(baseline_xy[:, 0], baseline_xy[:, 1], color="#4f6bdc", lw=2.2, label="V22 baseline")
    ax.plot(final_xy[:, 0], final_xy[:, 1], color="#df5d4b", lw=2.2, label="Fine-tuned final")
    ax.scatter(*start, marker="o", s=90, color="#1c9b5f", edgecolor="white", zorder=5, label="Start")
    ax.scatter(*goal, marker="*", s=180, color="#f5bf42", edgecolor="#6a4d00", zorder=5, label="Goal")
    ax.set(title="A  Frozen 80 m x 80 m terrain and representative paths", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    figure.colorbar(image, ax=ax, label="Terrain height (m)", fraction=0.046)

    ax = axes[0, 1]
    local_margin = 8.0
    ax.imshow(
        heights,
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap=terrain_cmap,
        aspect="equal",
    )
    ax.contour(coordinates, coordinates, heights, levels=levels, colors="white", linewidths=0.55, alpha=0.75)
    ax.plot(baseline_xy[:, 0], baseline_xy[:, 1], color="#4f6bdc", lw=2.2)
    ax.plot(final_xy[:, 0], final_xy[:, 1], color="#df5d4b", lw=2.2)
    ax.scatter(*start, s=80, color="#1c9b5f", edgecolor="white", zorder=5)
    ax.set_xlim(start[0] - 1.5, start[0] + local_margin)
    ax.set_ylim(start[1] - 1.5, start[1] + local_margin)
    ax.set(title="B  Start-area detail: both policies stall near the first transition", xlabel="x (m)", ylabel="y (m)")

    ax = axes[0, 2]
    for trace, colour, label in (
        (baseline, "#4f6bdc", "V22 baseline"),
        (final, "#df5d4b", "Fine-tuned final"),
    ):
        ax.plot(
            [row["time_seconds"] for row in trace],
            [row["distance_to_goal_m"] for row in trace],
            color=colour,
            lw=2,
            label=label,
        )
    ax.set(title="C  Distance to goal for representative seed 74803", xlabel="Time (s)", ylabel="Distance to goal (m)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    seed_order = [int(row["evaluation_seed"]) for row in baseline_rows]
    x = np.arange(len(seed_order))
    baseline_progress = np.asarray([float(row["net_progress_m"]) for row in baseline_rows])
    final_lookup = {int(row["evaluation_seed"]): row for row in final_rows}
    final_progress = np.asarray([float(final_lookup[seed]["net_progress_m"]) for seed in seed_order])
    for index in range(len(x)):
        ax.plot([x[index] - 0.12, x[index] + 0.12], [baseline_progress[index], final_progress[index]], color="#9a948b", lw=1.3)
    ax.scatter(x - 0.12, baseline_progress, color="#4f6bdc", s=55, label="Baseline")
    ax.scatter(x + 0.12, final_progress, color="#df5d4b", s=55, label="Fine-tuned final")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, [str(seed) for seed in seed_order])
    ax.set(title="D  Paired net progress: mean improvement, no completion", xlabel="Evaluation reset seed", ylabel="Net goal progress (m)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    baseline_slip = 100.0 * np.asarray(
        [float(row["contact_speed_exceedance_fraction"]) for row in baseline_rows]
    )
    final_slip = 100.0 * np.asarray(
        [float(final_lookup[seed]["contact_speed_exceedance_fraction"]) for seed in seed_order]
    )
    for index in range(len(x)):
        ax.plot([x[index] - 0.12, x[index] + 0.12], [baseline_slip[index], final_slip[index]], color="#9a948b", lw=1.3)
    ax.scatter(x - 0.12, baseline_slip, color="#4f6bdc", s=55, label="Baseline")
    ax.scatter(x + 0.12, final_slip, color="#df5d4b", s=55, label="Fine-tuned final")
    ax.set_xticks(x, [str(seed) for seed in seed_order])
    ax.set(title="E  Contact-speed threshold exceedance remains high", xlabel="Evaluation reset seed", ylabel="Steps above 0.20 m/s (%)")
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    episode_lengths = np.asarray([int(row["l"]) for row in monitor_rows], dtype=np.int64)
    episode_return_per_step = np.asarray(
        [float(row["r"]) / max(1, int(row["l"])) for row in monitor_rows], dtype=np.float64
    )
    cumulative_steps = np.cumsum(episode_lengths)
    window = min(20, len(episode_return_per_step))
    if window > 1:
        smooth = np.convolve(episode_return_per_step, np.ones(window) / window, mode="valid")
        smooth_steps = cumulative_steps[window - 1 :]
        ax.plot(smooth_steps, smooth, color="#6b5ca5", lw=2.2, label=f"Rolling mean ({window} episodes)")
    ax.scatter(cumulative_steps, episode_return_per_step, color="#9b91bd", s=8, alpha=0.35, label="Training episodes")
    ax.axvline(131072, color="#444444", ls="--", lw=1, label="Stage boundary")
    ax.set(title="F  Training diagnostic (mixed spawn locations)", xlabel="Completed episode interaction steps", ylabel="Locomotion return per step")
    ax.legend(fontsize=8)

    figure.suptitle(
        "Fixed-map pilot: fine-tuning improves mean local progress but does not achieve corner-to-corner traversal",
        fontsize=16,
        fontweight="bold",
    )
    output = report_dir / "fixed_map_training_results.png"
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    report_dir = run_root / "report"
    report_dir.mkdir(exist_ok=True)
    config = json.loads((run_root / "frozen_run_config.json").read_text(encoding="utf-8"))
    v22_config = json.loads(
        (ROOT / config["base_policy"]["configuration"]).read_text(encoding="utf-8")
    )
    evaluation_rows = read_rows(run_root / "logs" / "evaluation_episodes.csv")
    baseline_rows = [
        compact_row(row) for row in evaluation_rows if row["checkpoint_label"] == "v22_baseline"
    ]
    final_rows = [
        compact_row(row) for row in evaluation_rows if row["checkpoint_label"] == "final_paired_test"
    ]
    if len(baseline_rows) != 5 or len(final_rows) != 5:
        raise RuntimeError("Expected five paired baseline and five final-test rows")
    final_by_seed = {int(row["evaluation_seed"]): row for row in final_rows}
    paired_rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        seed = int(baseline["evaluation_seed"])
        final = final_by_seed[seed]
        paired_rows.append(
            {
                "evaluation_seed": seed,
                "baseline_net_progress_m": baseline["net_progress_m"],
                "final_net_progress_m": final["net_progress_m"],
                "paired_progress_difference_m": float(final["net_progress_m"]) - float(baseline["net_progress_m"]),
                "baseline_fall": baseline["fall"],
                "final_fall": final["fall"],
                "baseline_airborne_step_fraction": baseline["airborne_step_fraction"],
                "final_airborne_step_fraction": final["airborne_step_fraction"],
                "baseline_contact_speed_exceedance_fraction": baseline["contact_speed_exceedance_fraction"],
                "final_contact_speed_exceedance_fraction": final["contact_speed_exceedance_fraction"],
            }
        )
    write_csv(report_dir / "paired_test_summary.csv", paired_rows)

    baseline_trace, baseline_trace_summary = run_trace(
        config=config,
        v22_config=v22_config,
        scene_path=run_root / "task_scenes" / "spawn_0_0.000.xml",
        model_path=ROOT / config["base_policy"]["model_path"],
        seed=args.representative_seed,
        label="v22_baseline",
    )
    final_trace, final_trace_summary = run_trace(
        config=config,
        v22_config=v22_config,
        scene_path=run_root / "task_scenes" / "spawn_0_0.000.xml",
        model_path=run_root / "models" / "checkpoint_2465792.zip",
        seed=args.representative_seed,
        label="final_paired_test",
    )
    write_csv(report_dir / "representative_trace.csv", baseline_trace + final_trace)

    expected_baseline = next(row for row in baseline_rows if row["evaluation_seed"] == args.representative_seed)
    expected_final = next(row for row in final_rows if row["evaluation_seed"] == args.representative_seed)
    trace_verification = {
        "representative_seed": args.representative_seed,
        "baseline_net_progress_difference_m": float(baseline_trace_summary["fixed_goal_net_progress_m"]) - float(expected_baseline["net_progress_m"]),
        "final_net_progress_difference_m": float(final_trace_summary["fixed_goal_net_progress_m"]) - float(expected_final["net_progress_m"]),
    }
    trace_verification["reproduced_within_1e_9_m"] = bool(
        abs(trace_verification["baseline_net_progress_difference_m"]) <= 1e-9
        and abs(trace_verification["final_net_progress_difference_m"]) <= 1e-9
    )

    monitor_path = run_root / "logs" / "training_vecmonitor.csv"
    with monitor_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle)
        monitor_rows = list(csv.DictReader(handle))
    figure_path = plot_report(
        report_dir=report_dir,
        config=config,
        baseline=baseline_trace,
        final=final_trace,
        baseline_rows=baseline_rows,
        final_rows=final_rows,
        monitor_rows=monitor_rows,
    )

    summary = {
        "claim_boundary": config["claim_boundary"],
        "map_height_sha256": config["approved_map"]["heights_sha256"],
        "training_seed": config["training"]["training_seed"],
        "added_training_timesteps": 262144,
        "paired_evaluation_seeds": config["evaluation"]["paired_test_seeds"],
        "baseline": summarise_condition(baseline_rows),
        "final": summarise_condition(final_rows),
        "paired_final_minus_baseline_net_progress_m": sample_summary(
            [float(row["paired_progress_difference_m"]) for row in paired_rows]
        ),
        "representative_trace_verification": trace_verification,
        "primary_outcome": "failed_to_complete_corner_to_corner_route",
        "figure": str(figure_path),
    }
    (report_dir / "training_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
