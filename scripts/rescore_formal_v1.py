"""Counterfactually rescore formal v1 episodes under one fixed Ant reward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


REQUIRED_COLUMNS = {
    "condition_id",
    "training_seed",
    "seed",
    "target_timesteps",
    "ctrl_cost_weight",
    "proxy_return",
    "base_proxy_return",
    "reward_forward_sum",
    "reward_ctrl_sum",
    "reward_contact_sum",
    "reward_survive_sum",
    "control_effort",
    "net_forward_progress",
    "episode_length",
    "terminated",
    "truncated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--common-ctrl-weight", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    common_weight = float(args.common_ctrl_weight)
    if common_weight < 0:
        raise ValueError("common-ctrl-weight must be non-negative")
    data = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing formal v1 columns: {missing}")
    if data.empty:
        raise ValueError("Formal v1 input is empty")

    rescored = data.copy()
    rescored["condition_objective_return"] = rescored["proxy_return"]
    rescored["common_rescore_ctrl_cost_weight"] = common_weight
    rescored["cumulative_squared_action"] = rescored["control_effort"]
    rescored["common_rescored_return"] = (
        rescored["reward_forward_sum"]
        + rescored["reward_survive_sum"]
        + rescored["reward_contact_sum"]
        - common_weight * rescored["cumulative_squared_action"]
    )
    rescored["mean_squared_action_per_step"] = (
        rescored["cumulative_squared_action"] / rescored["episode_length"]
    )
    rescored["net_forward_progress_per_step"] = (
        rescored["net_forward_progress"] / rescored["episode_length"]
    )
    rescored["unhealthy_termination"] = rescored["terminated"].astype(bool)
    rescored["termination_subcategory_available"] = False
    rescored["ctrl_cost_reconciliation_error"] = (
        rescored["reward_ctrl_sum"]
        + rescored["ctrl_cost_weight"] * rescored["cumulative_squared_action"]
    )
    reconstructed_base = (
        rescored["reward_forward_sum"]
        + rescored["reward_survive_sum"]
        + rescored["reward_contact_sum"]
        - rescored["ctrl_cost_weight"] * rescored["cumulative_squared_action"]
    )
    rescored["base_reward_reconciliation_error"] = (
        rescored["base_proxy_return"] - reconstructed_base
    )

    numeric = rescored.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.drop(columns=["control_effort_per_unit_distance"])).all().all():
        raise ValueError("Unexpected non-finite value outside the legacy effort ratio")
    max_ctrl_error = float(rescored["ctrl_cost_reconciliation_error"].abs().max())
    max_base_error = float(rescored["base_reward_reconciliation_error"].abs().max())
    if max_ctrl_error > 1e-4 or max_base_error > 1e-4:
        raise ValueError(
            f"Reward reconstruction failed: ctrl={max_ctrl_error}, base={max_base_error}"
        )

    group_keys = ["condition_id", "training_seed", "target_timesteps"]
    metrics = [
        "condition_objective_return",
        "common_rescored_return",
        "net_forward_progress",
        "net_forward_progress_per_step",
        "cumulative_squared_action",
        "mean_squared_action_per_step",
        "unhealthy_termination",
        "episode_length",
    ]
    policy_summary = (
        rescored.groupby(group_keys, as_index=False)[metrics]
        .mean()
        .sort_values(group_keys)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "formal_v1_common_rescored_episodes.csv"
    summary_path = output_dir / "formal_v1_common_rescored_policy_summary.csv"
    rescored.to_csv(episode_path, index=False)
    policy_summary.to_csv(summary_path, index=False)

    final_target = int(rescored["target_timesteps"].max())
    final_summary = policy_summary[policy_summary["target_timesteps"] == final_target]
    core = final_summary[
        final_summary["condition_id"].isin(["reference", "ctrl_0p0625"])
    ]
    pivot = core.pivot(index="training_seed", columns="condition_id", values=metrics)
    paired = pd.DataFrame({"training_seed": pivot.index.to_numpy()})
    for metric in metrics:
        paired[f"ctrl_0p0625_minus_reference__{metric}"] = (
            pivot[(metric, "ctrl_0p0625")] - pivot[(metric, "reference")]
        ).to_numpy()
    paired_path = output_dir / "formal_v1_common_rescored_paired_300k.csv"
    paired.to_csv(paired_path, index=False)

    manifest = {
        "status": "pass",
        "role": "retrospective common rescoring; no training performed",
        "source": str(source),
        "source_sha256": sha256(source),
        "rows": int(len(rescored)),
        "policy_summary_rows": int(len(policy_summary)),
        "final_paired_rows": int(len(paired)),
        "common_rescore_ctrl_cost_weight": common_weight,
        "max_abs_ctrl_cost_reconciliation_error": max_ctrl_error,
        "max_abs_base_reward_reconciliation_error": max_base_error,
        "termination_subcategory_limitation": (
            "Formal v1 retained episode-level terminated flags but not torso-height trajectories; "
            "low-z, high-z and non-finite categories cannot be recovered retrospectively."
        ),
        "outputs": [str(episode_path), str(summary_path), str(paired_path)],
    }
    manifest_path = output_dir / "formal_v1_common_rescore_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
