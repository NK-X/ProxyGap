"""Build traceable evidence tables for the rigorous academic audit.

The script reads completed formal outputs without modifying them. Audit tables
are written to a separate directory and must not be merged into formal data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "audit" / "rigorous_academic_work_20260810"
)
FORMAL_ROOT = PROJECT_ROOT / "artifacts" / "formal"
COMBINED_ROOT = FORMAL_ROOT / "combined_v1_20260809" / "data"
RUNTIME_SOURCES = [
    "formal_v1_coefficients_20260808",
    "formal_v1_shaped_20260808",
    "formal_v1_core_replication_20260808",
]
CORE_CONDITIONS = [
    "reference",
    "ctrl_0p0625",
    "shaped_ctrl_0p0625_forward_1p0",
]
METRICS = [
    "proxy_return",
    "net_forward_progress",
    "control_effort",
    "control_effort_per_unit_distance",
    "fall",
    "lateral_drift_final_abs",
    "torso_tilt_std",
    "episode_length",
]


def build_paired_effects() -> pd.DataFrame:
    endpoint = pd.read_csv(COMBINED_ROOT / "endpoint_300k_seed_summary.csv")
    endpoint = endpoint[endpoint["condition_id"].isin(CORE_CONDITIONS)]
    pivot = endpoint.pivot(
        index="training_seed",
        columns="condition_id",
        values=METRICS,
    )
    comparisons = [
        ("ctrl_0p0625_minus_reference", "ctrl_0p0625", "reference"),
        (
            "shaped_minus_reference",
            "shaped_ctrl_0p0625_forward_1p0",
            "reference",
        ),
        (
            "shaped_minus_unshaped",
            "shaped_ctrl_0p0625_forward_1p0",
            "ctrl_0p0625",
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, treatment, comparator in comparisons:
        for seed in pivot.index:
            row: dict[str, float | int | str] = {
                "comparison": label,
                "training_seed": int(seed),
                "treatment": treatment,
                "comparator": comparator,
            }
            for metric in METRICS:
                row[f"{metric}_difference"] = float(
                    pivot.loc[seed, (metric, treatment)]
                    - pivot.loc[seed, (metric, comparator)]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_checkpoint_gaps() -> pd.DataFrame:
    seed_summary = pd.read_csv(COMBINED_ROOT / "seed_checkpoint_summary.csv")
    core = seed_summary[
        seed_summary["condition_id"].isin(["reference", "ctrl_0p0625"])
    ]
    means = (
        core.groupby(["condition_id", "target_timesteps"], as_index=False)[
            ["proxy_return", "net_forward_progress", "fall"]
        ]
        .mean()
        .pivot(index="target_timesteps", columns="condition_id")
    )
    return pd.DataFrame(
        {
            "target_timesteps": means.index.astype(int),
            "reference_proxy_return": means[("proxy_return", "reference")],
            "ctrl_0p0625_proxy_return": means[("proxy_return", "ctrl_0p0625")],
            "proxy_return_gap": (
                means[("proxy_return", "ctrl_0p0625")]
                - means[("proxy_return", "reference")]
            ),
            "reference_net_forward_progress": means[
                ("net_forward_progress", "reference")
            ],
            "ctrl_0p0625_net_forward_progress": means[
                ("net_forward_progress", "ctrl_0p0625")
            ],
            "net_forward_progress_gap": (
                means[("net_forward_progress", "ctrl_0p0625")]
                - means[("net_forward_progress", "reference")]
            ),
            "reference_fall_rate": means[("fall", "reference")],
            "ctrl_0p0625_fall_rate": means[("fall", "ctrl_0p0625")],
        }
    ).reset_index(drop=True)


def build_runtime_audit() -> tuple[pd.DataFrame, float, float]:
    frames = []
    for source in RUNTIME_SOURCES:
        frame = pd.read_csv(FORMAL_ROOT / source / "logs" / "training_runtime.csv")
        frame["source_config_id"] = source
        frames.append(frame)
    runtime = pd.concat(frames, ignore_index=True).sort_values(
        ["source_config_id", "training_seed", "condition_id", "target_timesteps"]
    )
    runtime["previous_actual_model_timesteps"] = runtime.groupby(
        ["source_config_id", "training_seed", "condition_id"]
    )["actual_model_timesteps"].shift(fill_value=0)
    runtime["actual_chunk_timesteps"] = (
        runtime["actual_model_timesteps"]
        - runtime["previous_actual_model_timesteps"]
    )
    runtime["actual_steps_per_sec"] = (
        runtime["actual_chunk_timesteps"] / runtime["train_elapsed_sec"]
    )
    # The two affected rows are over 46 times the median checkpoint duration.
    # This threshold identifies wall-clock pauses; it is not a scientific exclusion.
    runtime["wall_clock_outlier"] = runtime["train_elapsed_sec"] > 600.0
    clean = runtime[~runtime["wall_clock_outlier"]]
    clean_steps_per_sec = float(
        clean["actual_chunk_timesteps"].sum() / clean["train_elapsed_sec"].sum()
    )
    eval_seconds_per_episode = float(
        runtime["eval_elapsed_sec"].sum()
        / (runtime["eval_episodes"].sum())
    )
    return runtime, clean_steps_per_sec, eval_seconds_per_episode


def estimate_scenarios(
    clean_steps_per_sec: float,
    eval_seconds_per_episode: float,
) -> pd.DataFrame:
    model_paths = []
    for source in RUNTIME_SOURCES:
        model_paths.extend((FORMAL_ROOT / source / "runs").rglob("checkpoint_*.zip"))
    mean_model_bytes = float(np.mean([path.stat().st_size for path in model_paths]))
    scenarios = [
        {
            "scenario": "paired_timed_pilot",
            "conditions": 4,
            "training_seeds": 2,
            "actual_timesteps_per_run": 51_200,
            "checkpoints_per_run": 2,
            "evaluation_episodes_per_checkpoint": 5,
        },
        {
            "scenario": "core_confirmation_three_seeds",
            "conditions": 3,
            "training_seeds": 3,
            "actual_timesteps_per_run": 301_056,
            "checkpoints_per_run": 6,
            "evaluation_episodes_per_checkpoint": 10,
        },
        {
            "scenario": "core_confirmation_five_seeds",
            "conditions": 3,
            "training_seeds": 5,
            "actual_timesteps_per_run": 301_056,
            "checkpoints_per_run": 6,
            "evaluation_episodes_per_checkpoint": 10,
        },
        {
            "scenario": "core_confirmation_ten_seeds",
            "conditions": 3,
            "training_seeds": 10,
            "actual_timesteps_per_run": 301_056,
            "checkpoints_per_run": 6,
            "evaluation_episodes_per_checkpoint": 10,
        },
        {
            "scenario": "five_condition_shaping_ablation_five_seeds",
            "conditions": 5,
            "training_seeds": 5,
            "actual_timesteps_per_run": 301_056,
            "checkpoints_per_run": 6,
            "evaluation_episodes_per_checkpoint": 10,
        },
    ]
    rows = []
    for scenario in scenarios:
        runs = scenario["conditions"] * scenario["training_seeds"]
        model_count = runs * scenario["checkpoints_per_run"]
        actual_steps = runs * scenario["actual_timesteps_per_run"]
        eval_episodes = (
            model_count * scenario["evaluation_episodes_per_checkpoint"]
        )
        train_seconds = actual_steps / clean_steps_per_sec
        eval_seconds = eval_episodes * eval_seconds_per_episode
        rows.append(
            {
                **scenario,
                "independent_training_runs": runs,
                "model_checkpoints": model_count,
                "evaluation_episodes": eval_episodes,
                "estimated_train_minutes": train_seconds / 60.0,
                "estimated_evaluation_minutes": eval_seconds / 60.0,
                "estimated_total_minutes": (train_seconds + eval_seconds) / 60.0,
                "estimated_model_storage_mb": model_count
                * mean_model_bytes
                / (1024.0**2),
                "basis_actual_steps_per_sec": clean_steps_per_sec,
                "basis_eval_seconds_per_episode": eval_seconds_per_episode,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paired = build_paired_effects()
    checkpoints = build_checkpoint_gaps()
    runtime, clean_steps_per_sec, eval_seconds_per_episode = build_runtime_audit()
    scenarios = estimate_scenarios(clean_steps_per_sec, eval_seconds_per_episode)

    paired.to_csv(OUTPUT_ROOT / "paired_effects_300k.csv", index=False)
    checkpoints.to_csv(OUTPUT_ROOT / "checkpoint_gaps.csv", index=False)
    runtime.to_csv(OUTPUT_ROOT / "runtime_audit.csv", index=False)
    scenarios.to_csv(OUTPUT_ROOT / "resource_scenarios.csv", index=False)

    manifest = {
        "status": "pass",
        "role": "derived audit evidence; excluded from formal result rows",
        "paired_effect_rows": int(len(paired)),
        "checkpoint_rows": int(len(checkpoints)),
        "runtime_rows": int(len(runtime)),
        "wall_clock_outlier_rows": int(runtime["wall_clock_outlier"].sum()),
        "clean_actual_steps_per_sec": clean_steps_per_sec,
        "evaluation_seconds_per_episode": eval_seconds_per_episode,
        "source_combined_metrics": str(
            COMBINED_ROOT / "combined_evaluation_metrics.csv"
        ),
        "outlier_rule": "train_elapsed_sec > 600; used only for runtime estimation",
    }
    (OUTPUT_ROOT / "audit_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
