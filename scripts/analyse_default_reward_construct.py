"""Audit proxy-versus-intent ordering in the new default-reward comparator."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hybrid_guardrail_development_v2_20260816.json"
CONDITION = "R0_default__K0_none"
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def policy_table(episode: pd.DataFrame) -> pd.DataFrame:
    work = episode.loc[episode["condition_id"] == CONDITION].copy()
    for column in [
        "intent_compliant",
        "unhealthy_termination",
        "sustained_inversion",
    ]:
        work[column] = as_bool(work[column]).astype(float)
    work["target_velocity_absolute_error"] = (
        pd.to_numeric(work["fixed_horizon_mean_forward_velocity"]) - 1.0
    ).abs()
    work["path_inefficiency"] = 1.0 - pd.to_numeric(work["forward_path_efficiency"])
    work["torso_tilt_rms_degrees"] = np.degrees(pd.to_numeric(work["torso_tilt_rms"]))
    metrics = {
        "base_proxy_return": "mean",
        "intent_compliant": "mean",
        "target_velocity_absolute_error": "mean",
        "unhealthy_termination": "mean",
        "sustained_inversion": "mean",
        "torso_tilt_rms_degrees": "mean",
        "net_displacement_direction_error_degrees": "mean",
        "path_inefficiency": "mean",
        "normalised_action_roughness": "mean",
        "action_saturation_rate": "mean",
    }
    return (
        work.groupby(["training_seed", "target_timesteps"], as_index=False)
        .agg(metrics)
        .rename(columns={"intent_compliant": "intent_compliance_rate"})
        .sort_values(["training_seed", "target_timesteps"])
        .reset_index(drop=True)
    )


def pairwise_audit(policy: pd.DataFrame) -> pd.DataFrame:
    losses = [
        "target_velocity_absolute_error",
        "unhealthy_termination",
        "sustained_inversion",
        "torso_tilt_rms_degrees",
        "net_displacement_direction_error_degrees",
        "path_inefficiency",
        "normalised_action_roughness",
        "action_saturation_rate",
    ]
    rows: list[dict] = []
    for left_index, right_index in combinations(range(len(policy)), 2):
        left = policy.iloc[left_index]
        right = policy.iloc[right_index]
        proxy_difference = float(left["base_proxy_return"] - right["base_proxy_return"])
        if abs(proxy_difference) <= TOLERANCE:
            continue
        higher = left if proxy_difference > 0 else right
        lower = right if proxy_difference > 0 else left
        loss_differences = {metric: float(higher[metric] - lower[metric]) for metric in losses}
        intent_delta = float(
            higher["intent_compliance_rate"] - lower["intent_compliance_rate"]
        )
        weakly_worse_all = all(value >= -TOLERANCE for value in loss_differences.values())
        strictly_worse_any = any(value > TOLERANCE for value in loss_differences.values())
        rows.append(
            {
                "higher_proxy_training_seed": int(higher["training_seed"]),
                "higher_proxy_target_timesteps": int(higher["target_timesteps"]),
                "lower_proxy_training_seed": int(lower["training_seed"]),
                "lower_proxy_target_timesteps": int(lower["target_timesteps"]),
                "proxy_return_difference": abs(proxy_difference),
                "intent_compliance_difference": intent_delta,
                "proxy_intent_rank_inversion": bool(intent_delta < -TOLERANCE),
                "pareto_inversion": bool(weakly_worse_all and strictly_worse_any),
                **{f"higher_minus_lower_{metric}": value for metric, value in loss_differences.items()},
            }
        )
    return pd.DataFrame(rows)


def save_plot(policy: pd.DataFrame, path: Path) -> None:
    colours = {100000: "#56B4E9", 200000: "#E69F00", 300000: "#009E73"}
    markers = {41301: "o", 41302: "s", 41303: "^"}
    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for row in policy.itertuples(index=False):
        axis.scatter(
            row.base_proxy_return,
            row.intent_compliance_rate,
            color=colours[int(row.target_timesteps)],
            marker=markers[int(row.training_seed)],
            s=85,
        )
        axis.annotate(
            f"s{int(row.training_seed)} / {int(row.target_timesteps/1000)}k",
            (row.base_proxy_return, row.intent_compliance_rate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean default base proxy return")
    axis.set_ylabel("Intent-compliance rate")
    axis.set_ylim(-0.04, 1.04)
    axis.set_title("Default reward construct audit (development policies)")
    axis.grid(alpha=0.25)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    run_root = ROOT / config["execution"]["output_root"]
    completion = json.loads((run_root / "parallel_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("Development matrix is not complete")
    episode = pd.read_csv(run_root / "logs" / "evaluation_metrics.csv")
    policy = policy_table(episode)
    expected = len(config["training_seeds"]) * len(config["checkpoint_timesteps"])
    if len(policy) != expected:
        raise RuntimeError(f"Expected {expected} default policies, found {len(policy)}")
    pairs = pairwise_audit(policy)
    endpoint = policy.loc[
        policy["target_timesteps"].astype(int)
        == int(config["development_gate"]["primary_endpoint_timesteps"])
    ].copy()
    proxy_selected = endpoint.loc[endpoint["base_proxy_return"].idxmax()].to_dict()
    correlation = (
        float("nan")
        if policy["intent_compliance_rate"].nunique(dropna=True) < 2
        else policy["base_proxy_return"].rank(method="average").corr(
            policy["intent_compliance_rate"].rank(method="average")
        )
    )
    output = run_root / "analysis" / "default_reward_construct"
    output.mkdir(parents=True, exist_ok=True)
    policy.to_csv(output / "default_policy_checkpoint_metrics.csv", index=False)
    pairs.to_csv(output / "pairwise_proxy_intent_audit.csv", index=False)
    save_plot(policy, output / "proxy_return_vs_intent_compliance")
    domain_columns = [
        column
        for column in pairs.columns
        if column.startswith("higher_minus_lower_")
    ]
    domain_inversion_counts = {
        column.removeprefix("higher_minus_lower_"): int(
            (pd.to_numeric(pairs[column]) > TOLERANCE).sum()
        )
        for column in domain_columns
    }
    same_seed_pairs = pairs.loc[
        pairs["higher_proxy_training_seed"]
        == pairs["lower_proxy_training_seed"]
    ]
    result = {
        "status": "complete",
        "policy_checkpoints": len(policy),
        "unordered_non_tied_pairs": len(pairs),
        "proxy_intent_rank_inversions": int(pairs["proxy_intent_rank_inversion"].sum()),
        "pareto_inversions": int(pairs["pareto_inversion"].sum()),
        "domain_inversion_counts_all_pairs": domain_inversion_counts,
        "within_training_seed_pair_count": int(len(same_seed_pairs)),
        "within_training_seed_domain_inversion_counts": {
            column.removeprefix("higher_minus_lower_"): int(
                (pd.to_numeric(same_seed_pairs[column]) > TOLERANCE).sum()
            )
            for column in domain_columns
        },
        "spearman_proxy_vs_intent_rho_descriptive": float(correlation),
        "p_value": "not_calculated; repeated checkpoints and n=9 make asymptotic inference inappropriate",
        "proxy_selected_endpoint_policy": proxy_selected,
        "training_seed_is_replication_unit": True,
        "formal_claim_authorised": False,
    }
    (output / "construct_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
