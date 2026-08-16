"""Analyse the frozen two-condition body-dynamics replication."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyse_body_smoothness_gsde_matrix as body_matrix  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "body_dynamics_replication_v1_20260817.json"
BASELINE = "B0__G0_REP"
SHAPED = "B1__G0_REP"


def paired_contrasts(
    policy: pd.DataFrame,
    metrics: list[str],
    prefix: str,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = policy.copy()
    if "target_timesteps" in endpoint:
        endpoint = endpoint.loc[
            endpoint["target_timesteps"] == endpoint["target_timesteps"].max()
        ]
    rows: list[dict] = []
    for training_seed, group in endpoint.groupby("training_seed"):
        indexed = group.set_index("condition_id")
        if set(indexed.index) != {BASELINE, SHAPED}:
            raise ValueError(f"Incomplete condition pair for training seed {training_seed}")
        for metric in metrics:
            rows.append(
                {
                    "training_seed": int(training_seed),
                    "metric": metric,
                    "contrast_shaped_minus_baseline": float(
                        indexed.loc[SHAPED, metric] - indexed.loc[BASELINE, metric]
                    ),
                }
            )
    contrasts = pd.DataFrame(rows)
    summary = (
        contrasts.groupby("metric")["contrast_shaped_minus_baseline"]
        .agg(["mean", "std", "min", "max", lambda values: int((values < 0).sum())])
        .reset_index()
        .rename(columns={"<lambda_0>": "seed_pairs_below_zero"})
    )
    contrasts.to_csv(output / f"{prefix}_paired_contrasts.csv", index=False)
    summary.to_csv(output / f"{prefix}_contrast_summary.csv", index=False)
    return contrasts, summary


def make_figure(policy: pd.DataFrame, body_policy: pd.DataFrame, output: Path) -> None:
    endpoint = policy.loc[policy["target_timesteps"] == policy["target_timesteps"].max()]
    merged = endpoint.merge(
        body_policy,
        on=["condition_id", "training_seed"],
        suffixes=("", "_body"),
    )
    conditions = [BASELINE, SHAPED]
    labels = ["Body term off", "Body term on"]
    colours = {41601: "#0072B2", 41602: "#D55E00", 41603: "#009E73"}
    markers = {41601: "o", 41602: "s", 41603: "^"}
    panels = [
        ("fixed_horizon_forward_velocity_m_per_s", "Forward velocity (m/s)"),
        ("forward_path_efficiency_body", "Path efficiency"),
        ("direction_error_degrees", "Direction error (degrees)"),
        ("rms_root_vertical_velocity_m_per_s", "RMS vertical velocity (m/s)"),
        (
            "rms_root_roll_pitch_angular_speed_rad_per_s",
            "RMS roll/pitch rate (rad/s)",
        ),
        ("no_floor_contact_step_fraction", "No-floor-contact fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        for seed in sorted(colours):
            values = (
                merged.loc[merged["training_seed"] == seed]
                .set_index("condition_id")
                .loc[conditions, metric]
            )
            axis.plot([0, 1], values, color=colours[seed], alpha=0.35, linewidth=1.2)
            axis.scatter(
                [0, 1],
                values,
                color=colours[seed],
                marker=markers[seed],
                edgecolor="white",
                linewidth=0.5,
                s=58,
                label=f"Training seed {seed}",
            )
        for x, condition_id in enumerate(conditions):
            mean_value = merged.loc[merged["condition_id"] == condition_id, metric].mean()
            axis.scatter(x, mean_value, color="black", marker="D", s=66, zorder=4)
            axis.annotate(
                f"{mean_value:.3f}",
                (x, mean_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        axis.set_xticks([0, 1], labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Body-dynamics shaping: paired final development replication", y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(output / "body_dynamics_replication_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_root = ROOT / config["execution"]["output_root"]
    output = run_root / "analysis"
    execution = json.loads((run_root / "execution_record.json").read_text(encoding="utf-8"))
    if execution.get("status") != "complete":
        raise RuntimeError("Replication training is not complete")
    output.mkdir(parents=True, exist_ok=True)

    body_matrix.CONFIG_PATH = CONFIG_PATH
    body_matrix.RUN_ROOT = run_root
    body_matrix.OUTPUT = output
    body_matrix.ENDPOINT = int(config["timesteps_per_condition"])

    evaluation = body_matrix.load_evaluation()
    expected_rows = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["checkpoint_timesteps"])
        * len(config["evaluation_seeds"])
    )
    if len(evaluation) != expected_rows:
        raise ValueError(f"Expected {expected_rows} evaluation rows, got {len(evaluation)}")
    policy, _ = body_matrix.evaluation_summaries(evaluation)
    evaluation_contrasts, _ = paired_contrasts(
        policy, body_matrix.EVALUATION_METRICS, "evaluation", output
    )

    body_episodes = body_matrix.replay_all(config)
    expected_replays = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["evaluation_seeds"])
    )
    if len(body_episodes) != expected_replays:
        raise ValueError(f"Expected {expected_replays} body replays, got {len(body_episodes)}")
    body_policy = body_episodes.groupby(
        ["condition_id", "training_seed"], as_index=False
    )[body_matrix.BODY_METRICS].mean()
    body_policy.to_csv(output / "endpoint_body_contact_policy_means.csv", index=False)
    body_contrasts, body_summary = paired_contrasts(
        body_policy, body_matrix.BODY_METRICS, "body_contact", output
    )

    primary = {
        "rms_root_vertical_velocity_m_per_s",
        "rms_root_roll_pitch_angular_speed_rad_per_s",
    }
    primary_rows = body_summary.loc[body_summary["metric"].isin(primary)]
    replication_supported = bool(
        len(primary_rows) == 2
        and (primary_rows["seed_pairs_below_zero"] >= 2).all()
    )
    make_figure(policy, body_policy, output)
    summary = {
        "status": "complete",
        "research_stage": "final_bounded_development_replication",
        "independent_policy_pairs": len(config["training_seeds"]),
        "evaluation_rows": len(evaluation),
        "endpoint_body_replays": len(body_episodes),
        "primary_replication_direction_rule_passed": replication_supported,
        "primary_rule": "Both RMS body-rate metrics must improve in at least two of three paired training seeds.",
        "claim_boundary": "Development replication only; a passed direction rule is not held-out formal confirmation or complete mitigation.",
    }
    (output / "body_dynamics_replication_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    qa = {
        "status": "PASS",
        "expected_evaluation_rows": expected_rows,
        "observed_evaluation_rows": len(evaluation),
        "expected_endpoint_body_replays": expected_replays,
        "observed_endpoint_body_replays": len(body_episodes),
        "evaluation_contrast_rows": len(evaluation_contrasts),
        "body_contrast_rows": len(body_contrasts),
    }
    (output / "data_quality_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
