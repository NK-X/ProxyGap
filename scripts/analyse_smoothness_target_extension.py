"""Analyse the frozen target-tracking budget extension.

This is development evidence. Evaluation episodes are averaged within each
trained policy before any cross-seed summary is computed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_jump_contact_gait import FOOT_GEOMS, contact_snapshot  # noqa: E402
from proxygap import make_proxygap_ant_env  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "smoothness_target_budget_extension_v1_20260816.json"
PILOT_ROOT = ROOT / "artifacts" / "dev" / "smoothness_mechanism_v1"
EXTENSION_ROOT = ROOT / "artifacts" / "dev" / "smoothness_target_extension_1m_v1"
OUTPUT = EXTENSION_ROOT / "analysis"
XML = ROOT / "assets" / "ant_render_large_floor.xml"
REPLAY_EVALUATION_SEED = 51401
ENDPOINT = 1_000_000
DT = 0.05


METRICS = [
    "condition_objective_return",
    "base_proxy_return",
    "fixed_horizon_mean_forward_velocity",
    "forward_path_efficiency",
    "net_displacement_direction_error_degrees",
    "torso_tilt_rms",
    "normalised_action_roughness",
    "proposed_normalised_action_roughness",
    "action_saturation_rate",
    "intent_compliant",
    "unhealthy_termination",
    "sustained_inversion",
]


def count_prominent_takeoffs(values: np.ndarray, threshold: float = 1.25) -> int:
    candidates = [
        index
        for index in range(1, len(values) - 1)
        if values[index] >= threshold
        and values[index] >= values[index - 1]
        and values[index] > values[index + 1]
    ]
    accepted: list[int] = []
    for index in candidates:
        if not accepted or index - accepted[-1] >= 12:
            accepted.append(index)
    return len(accepted)


def load_evaluation(config: dict) -> pd.DataFrame:
    conditions = set(config["conditions"])
    pilot = pd.read_csv(PILOT_ROOT / "logs" / "evaluation_metrics.csv")
    pilot = pilot.loc[pilot["condition_id"].isin(conditions)].copy()
    extension = pd.read_csv(EXTENSION_ROOT / "logs" / "evaluation_metrics.csv")
    frame = pd.concat([pilot, extension], ignore_index=True)
    for column in ("intent_compliant", "unhealthy_termination", "sustained_inversion"):
        frame[column] = (
            frame[column].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
        )
    frame[METRICS] = frame[METRICS].apply(pd.to_numeric)
    frame["target_timesteps"] = pd.to_numeric(frame["target_timesteps"]).astype(int)
    frame["training_seed"] = pd.to_numeric(frame["training_seed"]).astype(int)
    duplicated = frame.duplicated(
        ["condition_id", "training_seed", "target_timesteps", "seed", "episode"]
    )
    if duplicated.any():
        raise ValueError(f"Duplicate evaluation rows: {int(duplicated.sum())}")
    expected_checkpoints = {100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000}
    if set(frame["target_timesteps"].unique()) != expected_checkpoints:
        raise ValueError("Unexpected checkpoint set in combined evaluation data")
    return frame


def policy_and_condition_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = (
        frame.groupby(["condition_id", "training_seed", "target_timesteps"], as_index=False)[METRICS]
        .mean()
        .sort_values(["condition_id", "training_seed", "target_timesteps"])
    )
    counts = (
        frame.groupby(["condition_id", "training_seed", "target_timesteps"])
        .size()
        .unique()
    )
    if not np.array_equal(counts, np.array([10])):
        raise ValueError(f"Expected ten evaluation episodes per policy-checkpoint, got {counts}")
    policy.to_csv(OUTPUT / "checkpoint_policy_means.csv", index=False)

    condition = policy.groupby(["condition_id", "target_timesteps"])[METRICS].agg(
        ["mean", "std", "min", "max"]
    )
    condition.columns = [f"{metric}_{stat}" for metric, stat in condition.columns]
    condition = condition.reset_index()
    condition.to_csv(OUTPUT / "checkpoint_condition_summary.csv", index=False)
    return policy, condition


def paired_rate_contrasts(policy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (seed, checkpoint), group in policy.groupby(["training_seed", "target_timesteps"]):
        indexed = group.set_index("condition_id")
        for metric in METRICS:
            rows.append(
                {
                    "training_seed": int(seed),
                    "target_timesteps": int(checkpoint),
                    "metric": metric,
                    "Ar0p2_minus_Ar0": float(indexed.loc["Ftrack__Ar0p2", metric])
                    - float(indexed.loc["Ftrack__Ar0", metric]),
                }
            )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUTPUT / "paired_action_rate_contrasts.csv", index=False)
    return contrasts


def endpoint_model(config: dict, condition_id: str, training_seed: int) -> Path:
    return (
        EXTENSION_ROOT
        / "runs"
        / f"seed_{training_seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{ENDPOINT:07d}.zip"
    )


def replay_endpoint(config: dict, condition_id: str, training_seed: int) -> dict:
    shared = config["shared_reward"]
    rate_weight = float(config["condition_parameters"][condition_id]["action_rate_shaping_weight"])
    env = make_proxygap_ant_env(
        ctrl_cost_weight=float(shared["ctrl_cost_weight"]),
        condition_id=condition_id,
        seed=REPLAY_EVALUATION_SEED,
        xml_file=XML,
        max_episode_steps=1000,
        orientation_shaping_weight=float(shared["orientation_shaping_weight"]),
        orientation_shaping_scale=float(shared["orientation_shaping_scale"]),
        orientation_shaping_function=str(shared["orientation_shaping_function"]),
        lateral_drift_shaping_weight=float(shared["lateral_drift_shaping_weight"]),
        lateral_drift_shaping_scale=float(shared["lateral_drift_shaping_scale"]),
        lateral_shaping_signal=str(shared["lateral_shaping_signal"]),
        lateral_velocity_target=float(shared["lateral_velocity_target"]),
        replace_forward_reward_with_tracking=True,
        forward_velocity_target=float(shared["forward_velocity_target"]),
        forward_velocity_tracking_scale=float(shared["forward_velocity_tracking_scale"]),
        action_rate_shaping_weight=rate_weight,
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
    )
    model = PPO.load(endpoint_model(config, condition_id, training_seed), device="cpu")
    observation, _ = env.reset(seed=REPLAY_EVALUATION_SEED)
    start_x = float(env.unwrapped.data.qpos[0])
    start_y = float(env.unwrapped.data.qpos[1])
    previous_x, previous_y = start_x, start_y
    planar_path = 0.0
    rows: list[dict] = []
    for step_index in range(1, 1001):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        data = env.unwrapped.data
        x_position = float(data.qpos[0])
        y_position = float(data.qpos[1])
        planar_path += float(np.hypot(x_position - previous_x, y_position - previous_y))
        previous_x, previous_y = x_position, y_position
        row = {
            "step_index": step_index,
            "root_velocity_x": float(data.qvel[0]),
            "root_velocity_y": float(data.qvel[1]),
            "root_velocity_z": float(data.qvel[2]),
            "root_angular_velocity_norm": float(np.linalg.norm(data.qvel[3:6])),
            "objective_reward": float(reward),
            "action_rate_penalty": float(info.get("proxygap_action_rate_penalty_step", np.nan)),
        }
        row.update(contact_snapshot(env.unwrapped.model, data))
        rows.append(row)
        if terminated or truncated:
            break
    summary = env.episode_summary()
    final_x = float(env.unwrapped.data.qpos[0])
    final_y = float(env.unwrapped.data.qpos[1])
    env.close()

    frame = pd.DataFrame(rows)
    contact_columns = [f"{name}_contact" for name in FOOT_GEOMS]
    net_x = final_x - start_x
    net_y = final_y - start_y
    result = {
        "condition_id": condition_id,
        "training_seed": training_seed,
        "evaluation_seed": REPLAY_EVALUATION_SEED,
        "action_rate_shaping_weight": rate_weight,
        "episode_steps": len(frame),
        "net_forward_progress_m": net_x,
        "net_lateral_displacement_m": net_y,
        "fixed_horizon_forward_velocity_m_per_s": net_x / (1000 * DT),
        "direction_error_degrees": float(np.degrees(np.arctan2(abs(net_y), max(net_x, 1e-12)))),
        "forward_path_efficiency": net_x / max(planar_path, 1e-12),
        "objective_return": float(frame["objective_reward"].sum()),
        "proposed_normalised_action_roughness": float(summary["proposed_normalised_action_roughness"]),
        "applied_normalised_action_roughness": float(summary["normalised_action_roughness"]),
        "torso_tilt_rms_rad": float(summary["torso_tilt_rms"]),
        "unhealthy_termination": bool(summary["unhealthy_termination"]),
        "no_floor_contact_step_fraction": float((frame["floor_contact_count"] == 0).mean()),
        "no_foot_contact_step_fraction": float((frame[contact_columns].sum(axis=1) == 0).mean()),
        "prominent_takeoff_count_vz_ge_1p25": count_prominent_takeoffs(frame["root_velocity_z"].to_numpy()),
        "max_root_velocity_z_m_per_s": float(frame["root_velocity_z"].max()),
        "rms_root_velocity_z_m_per_s": float(np.sqrt(np.mean(np.square(frame["root_velocity_z"])))),
        "rms_root_angular_velocity_rad_per_s": float(np.sqrt(np.mean(np.square(frame["root_angular_velocity_norm"])))),
        "max_raw_floor_force_norm": float(frame["floor_total_force_norm"].max()),
        "p95_raw_floor_force_norm": float(frame["floor_total_force_norm"].quantile(0.95)),
        "same_side_pair_contact_fraction": float((
            ((frame["front_left_contact"] == 1) & (frame["hind_left_contact"] == 1))
            | ((frame["front_right_contact"] == 1) & (frame["hind_right_contact"] == 1))
        ).mean()),
        "diagonal_pair_contact_fraction": float((
            ((frame["front_left_contact"] == 1) & (frame["hind_right_contact"] == 1))
            | ((frame["front_right_contact"] == 1) & (frame["hind_left_contact"] == 1))
        ).mean()),
        "front_pair_simultaneous_contact_fraction": float((
            (frame["front_left_contact"] == 1) & (frame["front_right_contact"] == 1)
        ).mean()),
    }
    result["exploratory_hopping_dominant_flag"] = bool(
        result["no_floor_contact_step_fraction"] >= 0.25
        or result["prominent_takeoff_count_vz_ge_1p25"] >= 5
    )
    return result


def make_figure(policy: pd.DataFrame, contact: pd.DataFrame) -> None:
    colours = {"Ftrack__Ar0": "#0072B2", "Ftrack__Ar0p2": "#D55E00"}
    labels = {"Ftrack__Ar0": "Target tracking", "Ftrack__Ar0p2": "Target + action-rate penalty"}
    panels = [
        ("fixed_horizon_mean_forward_velocity", "Forward velocity (m/s)"),
        ("unhealthy_termination", "Unhealthy termination rate"),
        ("proposed_normalised_action_roughness", "Policy action roughness"),
        ("forward_path_efficiency", "Forward path efficiency"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for condition_id, group in policy.groupby("condition_id"):
        for axis, (metric, ylabel) in zip(axes.ravel()[:4], panels):
            pivot = group.pivot(index="target_timesteps", columns="training_seed", values=metric)
            x = pivot.index.to_numpy() / 1000
            for seed in pivot.columns:
                axis.plot(x, pivot[seed], color=colours[condition_id], alpha=0.22, linewidth=1)
            axis.plot(x, pivot.mean(axis=1), marker="o", color=colours[condition_id], linewidth=2.2, label=labels[condition_id])
            axis.set_xlabel("Training budget (k steps)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for condition_id, group in contact.groupby("condition_id"):
        axes[1, 1].scatter(
            group["no_floor_contact_step_fraction"],
            group["fixed_horizon_forward_velocity_m_per_s"],
            s=58,
            color=colours[condition_id],
            label=labels[condition_id],
        )
        axes[1, 2].scatter(
            group["prominent_takeoff_count_vz_ge_1p25"],
            group["max_raw_floor_force_norm"],
            s=58,
            color=colours[condition_id],
        )
    axes[1, 1].set_xlabel("No-floor-contact step fraction")
    axes[1, 1].set_ylabel("Forward velocity (m/s)")
    axes[1, 1].set_title("Endpoint flight-time diagnostic")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].set_xlabel("Prominent take-off count")
    axes[1, 2].set_ylabel("Maximum raw floor-force norm")
    axes[1, 2].set_title("Endpoint take-off and impact diagnostic")
    for axis in axes[1, 1:]:
        axis.grid(alpha=0.2)
    fig.suptitle("Target-tracking budget extension: development evidence", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "target_budget_extension_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "target_budget_extension_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluation = load_evaluation(config)
    policy, condition = policy_and_condition_summaries(evaluation)
    contrasts = paired_rate_contrasts(policy)
    contact_rows = []
    for condition_id in config["conditions"]:
        for seed in config["training_seeds"]:
            print(condition_id, seed, flush=True)
            contact_rows.append(replay_endpoint(config, condition_id, int(seed)))
    contact = pd.DataFrame(contact_rows)
    contact.to_csv(OUTPUT / "endpoint_contact_gait_matrix.csv", index=False)
    make_figure(policy, contact)

    endpoint_policy = policy.loc[policy["target_timesteps"] == ENDPOINT]
    summary = {
        "status": "complete",
        "research_stage": "development_budget_diagnostic",
        "evaluation_episode_rows": int(len(evaluation)),
        "policy_checkpoint_units": int(len(policy)),
        "independent_training_runs": int(policy[["condition_id", "training_seed"]].drop_duplicates().shape[0]),
        "endpoint_policies": int(len(contact)),
        "contact_replay_evaluation_seed": REPLAY_EVALUATION_SEED,
        "endpoint_condition_means": endpoint_policy.groupby("condition_id").mean(numeric_only=True).to_dict(orient="index"),
        "endpoint_contact_means": contact.groupby("condition_id").mean(numeric_only=True).to_dict(orient="index"),
        "endpoint_hopping_flag_fraction": contact.groupby("condition_id")["exploratory_hopping_dominant_flag"].mean().to_dict(),
        "rate_penalty_contrast_at_1m": contrasts.loc[contrasts["target_timesteps"] == ENDPOINT].groupby("metric")["Ar0p2_minus_Ar0"].mean().to_dict(),
        "claim_boundary": "Development evidence only. The six policy continuations are not bitwise-equivalent to uninterrupted 1M training. Contact replay uses one fixed evaluation seed per endpoint policy and is not a natural-gait classifier or prevalence estimate.",
    }
    (OUTPUT / "target_budget_extension_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
