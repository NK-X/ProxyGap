"""Analyse the frozen orientation pilot without advancing to a second intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()

PRIMARY_METRICS = [
    "base_proxy_return",
    "condition_objective_return",
    "fixed_horizon_mean_forward_velocity",
    "unhealthy_termination",
    "inverted_step_fraction",
    "sustained_inversion",
    "torso_tilt_rms",
    "lateral_drift_mean_abs",
    "normalised_action_roughness",
    "action_saturation_rate",
    "control_effort_per_unit_distance",
]


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        if column not in converted:
            raise ValueError(f"Missing analysis column: {column}")
        if pd.api.types.is_bool_dtype(converted[column]):
            converted[column] = converted[column].astype(float)
        elif (
            pd.api.types.is_object_dtype(converted[column])
            or pd.api.types.is_string_dtype(converted[column])
        ):
            converted[column] = converted[column].replace(
                {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}
            )
        converted[column] = pd.to_numeric(converted[column], errors="raise")
    return converted


def bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(4000, finite.size))
    means = finite[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def relative_change(shaped: float, baseline: float) -> float:
    if not np.isfinite(shaped) or not np.isfinite(baseline):
        return float("nan")
    return float((shaped - baseline) / max(abs(baseline), 1e-8))


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small result table without pandas' optional tabulate package."""

    def format_value(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def save_tradeoff_figure(seed_level: pd.DataFrame, output: Path) -> None:
    colours = {0.1: "#0072B2", 0.25: "#E69F00", 0.5: "#009E73"}
    markers = {41201: "o", 41204: "s", 41205: "^"}
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    for row in seed_level.itertuples(index=False):
        weight = float(row.orientation_weight)
        seed = int(row.training_seed)
        ax.scatter(
            row.forward_velocity_retention,
            row.inverted_step_fraction_reduction,
            color=colours[weight],
            marker=markers[seed],
            s=80,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            f"{weight:g} / {seed}",
            (row.forward_velocity_retention, row.inverted_step_fraction_reduction),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.90, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(0.05, color="#555555", linestyle=":", linewidth=1)
    ax.set_xlabel("Fixed-horizon forward-velocity retention")
    ax.set_ylabel("Reduction in inverted-step fraction")
    ax.set_title("Orientation shaping: posture gain versus forward retention")
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_checkpoint_figure(
    baseline: pd.DataFrame,
    shaped: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    conditions: list[tuple[str, pd.DataFrame, str]] = [
        ("baseline", baseline, "#333333")
    ]
    palette = {0.1: "#0072B2", 0.25: "#E69F00", 0.5: "#009E73"}
    for weight in sorted(shaped["orientation_shaping_weight"].unique()):
        conditions.append(
            (
                f"lambda={weight:g}",
                shaped.loc[shaped["orientation_shaping_weight"] == weight],
                palette[float(weight)],
            )
        )
    for label, frame, colour in conditions:
        grouped = frame.groupby("target_timesteps", as_index=False)[
            ["inverted_step_fraction", "fixed_horizon_mean_forward_velocity"]
        ].mean()
        x = grouped["target_timesteps"] / 1000.0
        axes[0].plot(
            x,
            grouped["inverted_step_fraction"],
            marker="o",
            label=label,
            color=colour,
        )
        axes[1].plot(
            x,
            grouped["fixed_horizon_mean_forward_velocity"],
            marker="o",
            label=label,
            color=colour,
        )
    axes[0].set_ylabel("Mean inverted-step fraction")
    axes[1].set_ylabel("Fixed-horizon forward velocity")
    for axis in axes:
        axis.set_xlabel("Training checkpoint (thousand timesteps)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    completion = json.loads(
        (run_root / "parallel_completion.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete":
        raise RuntimeError("Pilot training is incomplete")
    analysis_root = run_root / "analysis"
    if analysis_root.exists():
        raise FileExistsError(f"Analysis output already exists: {analysis_root}")
    analysis_root.mkdir(parents=True)

    replay_path = config["baseline_evidence"].get("replay_evaluation_csv")
    baseline_csv = (
        PROJECT_ROOT / replay_path
        if replay_path
        else run_root / "baseline_replay" / "logs" / "evaluation_metrics.csv"
    )
    baseline = pd.read_csv(baseline_csv)
    shaped = pd.read_csv(run_root / "logs" / "evaluation_metrics.csv")
    numeric_columns = [
        "training_seed",
        "seed",
        "target_timesteps",
        "orientation_shaping_weight",
        *PRIMARY_METRICS,
    ]
    baseline = numeric_frame(baseline, numeric_columns)
    shaped = numeric_frame(shaped, numeric_columns)
    pilot_seeds = {int(value) for value in config["training_seeds"]}
    baseline = baseline.loc[baseline["training_seed"].isin(pilot_seeds)].copy()
    endpoint = int(config["pilot_gate"]["primary_endpoint"])
    baseline_endpoint = baseline.loc[baseline["target_timesteps"] == endpoint]
    shaped_endpoint = shaped.loc[shaped["target_timesteps"] == endpoint]

    expected_baseline = len(pilot_seeds) * int(config["eval_episodes_per_checkpoint"])
    expected_shaped = (
        expected_baseline
        * len(config["orientation_shaping"]["candidate_weights"])
    )
    if len(baseline_endpoint) != expected_baseline or len(shaped_endpoint) != expected_shaped:
        raise RuntimeError(
            f"Endpoint row mismatch: baseline={len(baseline_endpoint)}, "
            f"shaped={len(shaped_endpoint)}"
        )

    paired_frames: list[pd.DataFrame] = []
    seed_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for weight in sorted(shaped_endpoint["orientation_shaping_weight"].unique()):
        condition = shaped_endpoint.loc[
            shaped_endpoint["orientation_shaping_weight"] == weight
        ]
        merged = baseline_endpoint.merge(
            condition,
            on=["training_seed", "seed"],
            suffixes=("_baseline", "_shaped"),
            validate="one_to_one",
        )
        merged["orientation_weight"] = weight
        for metric in PRIMARY_METRICS:
            merged[f"{metric}_delta"] = (
                merged[f"{metric}_shaped"] - merged[f"{metric}_baseline"]
            )
        paired_frames.append(merged)

        for training_seed, seed_frame in merged.groupby("training_seed"):
            baseline_forward = float(
                seed_frame["fixed_horizon_mean_forward_velocity_baseline"].mean()
            )
            shaped_forward = float(
                seed_frame["fixed_horizon_mean_forward_velocity_shaped"].mean()
            )
            row = {
                "orientation_weight": float(weight),
                "training_seed": int(training_seed),
                "inverted_step_fraction_baseline": float(
                    seed_frame["inverted_step_fraction_baseline"].mean()
                ),
                "inverted_step_fraction_shaped": float(
                    seed_frame["inverted_step_fraction_shaped"].mean()
                ),
                "inverted_step_fraction_reduction": float(
                    -seed_frame["inverted_step_fraction_delta"].mean()
                ),
                "forward_velocity_baseline": baseline_forward,
                "forward_velocity_shaped": shaped_forward,
                "forward_velocity_retention": shaped_forward
                / max(abs(baseline_forward), 1e-8),
                "unhealthy_rate_baseline": float(
                    seed_frame["unhealthy_termination_baseline"].mean()
                ),
                "unhealthy_rate_shaped": float(
                    seed_frame["unhealthy_termination_shaped"].mean()
                ),
                "unhealthy_rate_increase": float(
                    seed_frame["unhealthy_termination_delta"].mean()
                ),
            }
            for metric in config["pilot_gate"]["guardrail_domains"]:
                base = float(seed_frame[f"{metric}_baseline"].mean())
                shaped_value = float(seed_frame[f"{metric}_shaped"].mean())
                row[f"{metric}_baseline"] = base
                row[f"{metric}_shaped"] = shaped_value
                row[f"{metric}_relative_change"] = relative_change(
                    shaped_value,
                    base,
                )
            seed_rows.append(row)

            for metric in [
                "inverted_step_fraction",
                "fixed_horizon_mean_forward_velocity",
                "unhealthy_termination",
                "torso_tilt_rms",
            ]:
                values = seed_frame[f"{metric}_delta"].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(
                    values,
                    seed=8152026 + int(training_seed) + int(round(weight * 1000)),
                )
                bootstrap_rows.append(
                    {
                        "orientation_weight": float(weight),
                        "training_seed": int(training_seed),
                        "metric": metric,
                        "paired_episode_mean_delta": float(np.nanmean(values)),
                        "paired_episode_bootstrap_ci95_low": low,
                        "paired_episode_bootstrap_ci95_high": high,
                        "episodes": len(values),
                        "scope": "initial-condition uncertainty within one trained policy; not training-seed uncertainty",
                    }
                )

    paired = pd.concat(paired_frames, ignore_index=True)
    seed_level = pd.DataFrame(seed_rows).sort_values(
        ["orientation_weight", "training_seed"]
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    gate = config["pilot_gate"]
    candidate_rows: list[dict] = []
    for weight, frame in seed_level.groupby("orientation_weight"):
        posture_count = int(
            (
                frame["inverted_step_fraction_reduction"]
                >= float(gate["posture_improvement_min_absolute"])
            ).sum()
        )
        median_retention = float(np.nanmedian(frame["forward_velocity_retention"]))
        unhealthy_bad_count = int(
            (
                frame["unhealthy_rate_increase"]
                > float(gate["unhealthy_rate_allowed_increase"])
            ).sum()
        )
        guardrail_medians: dict[str, float] = {}
        guardrail_bad = 0
        for metric in gate["guardrail_domains"]:
            value = float(np.nanmedian(frame[f"{metric}_relative_change"]))
            guardrail_medians[metric] = value
            if value > float(gate["guardrail_relative_worsening_threshold"]):
                guardrail_bad += 1
        passed = (
            posture_count >= int(gate["posture_improvement_min_seed_pairs"])
            and median_retention >= float(gate["median_forward_velocity_retention_min"])
            and unhealthy_bad_count <= int(gate["max_seed_pairs_with_unhealthy_increase"])
            and guardrail_bad < 2
        )
        candidate_rows.append(
            {
                "orientation_weight": float(weight),
                "posture_improved_seed_pairs": posture_count,
                "median_inverted_step_fraction_reduction": float(
                    np.nanmedian(frame["inverted_step_fraction_reduction"])
                ),
                "median_forward_velocity_retention": median_retention,
                "seed_pairs_with_excess_unhealthy_increase": unhealthy_bad_count,
                "guardrail_domains_worse_over_threshold": guardrail_bad,
                **{
                    f"median_relative_change_{metric}": value
                    for metric, value in guardrail_medians.items()
                },
                "pilot_gate_passed": bool(passed),
                "development_label": (
                    "promising_development_candidate" if passed else "not_supported_by_pilot_gate"
                ),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values("orientation_weight")

    paired.to_csv(analysis_root / "paired_episode_differences.csv", index=False)
    seed_level.to_csv(analysis_root / "seed_level_endpoint.csv", index=False)
    bootstrap.to_csv(analysis_root / "paired_episode_bootstrap_intervals.csv", index=False)
    candidates.to_csv(analysis_root / "candidate_adjudication.csv", index=False)
    save_tradeoff_figure(seed_level, analysis_root / "posture_forward_tradeoff.png")
    save_checkpoint_figure(
        baseline,
        shaped,
        analysis_root / "checkpoint_diagnostics.png",
    )

    passed_weights = candidates.loc[
        candidates["pilot_gate_passed"], "orientation_weight"
    ].tolist()
    outcome = {
        "status": "step_5_adjudicated",
        "passed_development_weights": passed_weights,
        "candidate_count": len(candidates),
        "training_seed_pairs": len(pilot_seeds),
        "evaluation_episodes_per_policy": int(config["eval_episodes_per_checkpoint"]),
        "second_intervention_started": False,
        "formal_experiment_started": False,
        "decision": (
            "At least one cosine weight passed the predeclared development gate."
            if passed_weights
            else "No cosine weight passed the predeclared development gate."
        ),
        "claim_boundary": (
            "These purpose-selected development seeds support a mechanism and "
            "design decision only. They do not estimate population-level "
            "mitigation efficacy or establish statistical confidence across training seeds."
        ),
    }
    (analysis_root / "pilot_adjudication.json").write_text(
        json.dumps(outcome, indent=2) + "\n",
        encoding="utf-8",
    )

    table_text = dataframe_to_markdown(candidates)
    report = f"""# 姿态余弦奖励塑形 Pilot 第五步裁决

## 结论边界

本轮完成的是 development pilot，不是正式 held-out 实验。训练 seed 是三个经目的性选择的已知行为模式；20 个 evaluation episodes 只能提高对固定策略的观测精度，不能替代更多独立训练 seed。

## 预声明矩阵

- 姿态项：$-\\lambda_\\theta(1-\\cos\\theta)/2$。
- $\\lambda_\\theta=0.10,0.25,0.50$。
- Training seeds：41201、41204、41205。
- 每个策略 1M timesteps，四个 checkpoints，每个 checkpoint 20 个匹配 evaluation seeds。
- Ant-v5 默认四项奖励权重、PPO、网络结构和 normalisation 均未改变。

## 候选裁决

{table_text}

## 当前决定

{outcome['decision']}

即使有候选通过，也只表示值得进入下一次 protocol-freeze 讨论；本脚本不会自动选择最终奖励、增加横向惩罚或启动正式 seeds。
"""
    report = f"""# 姿态余弦奖励塑形 Pilot 第五步裁决

## 结论边界

本轮完成的是 development pilot，不是正式 held-out 实验。三个 training
seeds 是为覆盖已知行为模式而有目的地选择；每个策略的 20 个 evaluation
episodes 只能提高对该固定策略的观测精度，不能替代更多独立 training seeds。

## 预声明矩阵

- 姿态项：$-\\lambda_\\theta(1-\\cos\\theta)/2$。
- $\\lambda_\\theta=0.10,0.25,0.50$。
- Training seeds：41201、41204、41205。
- 每个策略训练 1M timesteps，并保存 250k、500k、750k、1M checkpoints。
- 每个 checkpoint 使用 20 个配对 evaluation seeds。
- Ant-v5 默认四项奖励权重、PPO、网络结构和 normalisation 均未改变。

## 候选裁决

{table_text}

## 当前决定

{outcome['decision']}

即使有候选通过，也只表示值得进入下一次 protocol-freeze 讨论。本脚本不会
自动选择最终奖励、增加横向惩罚或启动正式 seeds。
"""
    (analysis_root / "PILOT_STEP5_RESULT_CN.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
