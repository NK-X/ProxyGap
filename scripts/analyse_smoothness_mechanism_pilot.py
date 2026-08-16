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


CONFIG_PATH = ROOT / "configs" / "smoothness_mechanism_pilot_v1_20260816.json"
RUN_ROOT = ROOT / "artifacts" / "dev" / "smoothness_mechanism_v1"
OUTPUT = RUN_ROOT / "analysis"
XML = ROOT / "assets" / "ant_render_large_floor.xml"
ENDPOINT = 300_000
REPLAY_EVALUATION_SEED = 51401
DT = 0.05


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


def replay_endpoint(config: dict, condition: dict, training_seed: int) -> dict:
    shared = config["shared_reward"]
    condition_id = str(condition["condition_id"])
    model_path = (
        RUN_ROOT
        / "runs"
        / f"seed_{training_seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{ENDPOINT:06d}.zip"
    )
    env = make_proxygap_ant_env(
        ctrl_cost_weight=float(config["ctrl_cost_weight"]),
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
        replace_forward_reward_with_tracking=bool(
            condition["replace_forward_reward_with_tracking"]
        ),
        forward_velocity_target=float(shared["forward_velocity_target"]),
        forward_velocity_tracking_scale=float(shared["forward_velocity_tracking_scale"]),
        action_rate_shaping_weight=float(condition["action_rate_shaping_weight"]),
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
    )
    model = PPO.load(model_path, device="cpu")
    observation, _ = env.reset(seed=REPLAY_EVALUATION_SEED)
    start_x = float(env.unwrapped.data.qpos[0])
    start_y = float(env.unwrapped.data.qpos[1])
    previous_x = start_x
    previous_y = start_y
    planar_path = 0.0
    rows: list[dict] = []
    for step_index in range(1, 1001):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        data = env.unwrapped.data
        x_position = float(data.qpos[0])
        y_position = float(data.qpos[1])
        planar_path += float(np.hypot(x_position - previous_x, y_position - previous_y))
        previous_x = x_position
        previous_y = y_position
        row = {
            "step_index": step_index,
            "root_velocity_x": float(data.qvel[0]),
            "root_velocity_y": float(data.qvel[1]),
            "root_velocity_z": float(data.qvel[2]),
            "root_angular_velocity_norm": float(np.linalg.norm(data.qvel[3:6])),
            "objective_reward": float(reward),
            "base_forward_reward": float(info.get("reward_forward", np.nan)),
            "tracking_reward": float(info.get("reward_forward_tracking", np.nan)),
            "action_rate_penalty": float(info.get("action_rate_penalty", np.nan)),
            "proposed_action_change_l2": float(
                info.get("proxygap_proposed_action_change_l2_step", np.nan)
            ),
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
    direction_error = float(np.degrees(np.arctan2(abs(net_y), max(net_x, 1e-12))))
    result = {
        "condition_id": condition_id,
        "training_seed": training_seed,
        "evaluation_seed": REPLAY_EVALUATION_SEED,
        "replace_forward_reward_with_tracking": bool(
            condition["replace_forward_reward_with_tracking"]
        ),
        "action_rate_shaping_weight": float(condition["action_rate_shaping_weight"]),
        "episode_steps": len(frame),
        "net_forward_progress_m": net_x,
        "net_lateral_displacement_m": net_y,
        "fixed_horizon_forward_velocity_m_per_s": net_x / (1000 * DT),
        "direction_error_degrees": direction_error,
        "forward_path_efficiency": net_x / max(planar_path, 1e-12),
        "objective_return": float(frame["objective_reward"].sum()),
        "proposed_normalised_action_roughness": float(
            summary["proposed_normalised_action_roughness"]
        ),
        "applied_normalised_action_roughness": float(summary["normalised_action_roughness"]),
        "torso_tilt_rms_rad": float(summary["torso_tilt_rms"]),
        "unhealthy_termination": bool(summary["unhealthy_termination"]),
        "no_floor_contact_step_fraction": float((frame["floor_contact_count"] == 0).mean()),
        "no_foot_contact_step_fraction": float(
            (frame[contact_columns].sum(axis=1) == 0).mean()
        ),
        "prominent_takeoff_count_vz_ge_1p25": count_prominent_takeoffs(
            frame["root_velocity_z"].to_numpy()
        ),
        "max_root_velocity_z_m_per_s": float(frame["root_velocity_z"].max()),
        "rms_root_velocity_z_m_per_s": float(
            np.sqrt(np.mean(np.square(frame["root_velocity_z"])))
        ),
        "rms_root_angular_velocity_rad_per_s": float(
            np.sqrt(np.mean(np.square(frame["root_angular_velocity_norm"])))
        ),
        "max_raw_floor_force_norm": float(frame["floor_total_force_norm"].max()),
        "p95_raw_floor_force_norm": float(frame["floor_total_force_norm"].quantile(0.95)),
        "same_side_pair_contact_fraction": float(
            (
                ((frame["front_left_contact"] == 1) & (frame["hind_left_contact"] == 1))
                | ((frame["front_right_contact"] == 1) & (frame["hind_right_contact"] == 1))
            ).mean()
        ),
        "diagonal_pair_contact_fraction": float(
            (
                ((frame["front_left_contact"] == 1) & (frame["hind_right_contact"] == 1))
                | ((frame["front_right_contact"] == 1) & (frame["hind_left_contact"] == 1))
            ).mean()
        ),
        "front_pair_simultaneous_contact_fraction": float(
            ((frame["front_left_contact"] == 1) & (frame["front_right_contact"] == 1)).mean()
        ),
    }
    result["exploratory_hopping_dominant_flag"] = bool(
        result["no_floor_contact_step_fraction"] >= 0.25
        or result["prominent_takeoff_count_vz_ge_1p25"] >= 5
    )
    return result


def endpoint_evaluation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    endpoint = frame.loc[frame["target_timesteps"].astype(int) == ENDPOINT].copy()
    bool_columns = ["intent_compliant", "unhealthy_termination", "sustained_inversion"]
    for column in bool_columns:
        endpoint[column] = endpoint[column].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    numeric_columns = [
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
    endpoint[numeric_columns] = endpoint[numeric_columns].apply(pd.to_numeric)
    seed_means = (
        endpoint.groupby(["condition_id", "training_seed"], as_index=False)[numeric_columns]
        .mean()
        .sort_values(["condition_id", "training_seed"])
    )
    seed_means.to_csv(OUTPUT / "endpoint_seed_means.csv", index=False)
    condition_means = (
        seed_means.groupby("condition_id", as_index=False)[numeric_columns]
        .agg(["mean", "std"])
    )
    condition_means.columns = [
        "condition_id" if column[0] == "condition_id" else f"{column[0]}_{column[1]}"
        for column in condition_means.columns
    ]
    condition_means.to_csv(OUTPUT / "endpoint_condition_summary.csv", index=False)
    return seed_means


def factorial_contrasts(seed_means: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fixed_horizon_mean_forward_velocity",
        "forward_path_efficiency",
        "net_displacement_direction_error_degrees",
        "torso_tilt_rms",
        "normalised_action_roughness",
        "proposed_normalised_action_roughness",
        "unhealthy_termination",
    ]
    rows = []
    for seed, seed_frame in seed_means.groupby("training_seed"):
        indexed = seed_frame.set_index("condition_id")
        for metric in metrics:
            linear_0 = float(indexed.loc["Flinear__Ar0", metric])
            track_0 = float(indexed.loc["Ftrack__Ar0", metric])
            linear_rate = float(indexed.loc["Flinear__Ar0p2", metric])
            track_rate = float(indexed.loc["Ftrack__Ar0p2", metric])
            rows.append(
                {
                    "training_seed": int(seed),
                    "metric": metric,
                    "tracking_main_effect": ((track_0 + track_rate) - (linear_0 + linear_rate)) / 2,
                    "action_rate_main_effect": ((linear_rate + track_rate) - (linear_0 + track_0)) / 2,
                    "interaction": (track_rate - track_0) - (linear_rate - linear_0),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "factorial_seed_contrasts.csv", index=False)
    return result


def make_figure(seed_means: pd.DataFrame, contact: pd.DataFrame) -> None:
    colours = {
        "Flinear__Ar0": "#6B7280",
        "Ftrack__Ar0": "#0072B2",
        "Flinear__Ar0p2": "#D55E00",
        "Ftrack__Ar0p2": "#009E73",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for condition_id, group in seed_means.groupby("condition_id"):
        axes[0].scatter(
            group["proposed_normalised_action_roughness"],
            group["fixed_horizon_mean_forward_velocity"],
            color=colours[condition_id],
            label=condition_id,
            s=58,
        )
    for condition_id, group in contact.groupby("condition_id"):
        axes[1].scatter(
            group["no_floor_contact_step_fraction"],
            group["fixed_horizon_forward_velocity_m_per_s"],
            color=colours[condition_id],
            s=58,
        )
        axes[2].scatter(
            group["prominent_takeoff_count_vz_ge_1p25"],
            group["max_raw_floor_force_norm"],
            color=colours[condition_id],
            s=58,
        )
    axes[0].set_xlabel("Proposed normalised action roughness")
    axes[0].set_ylabel("Fixed-horizon forward velocity (m/s)")
    axes[0].set_title("Policy smoothness and command retention")
    axes[1].set_xlabel("No-floor-contact step fraction")
    axes[1].set_ylabel("Fixed-horizon forward velocity (m/s)")
    axes[1].set_title("Flight time and locomotion")
    axes[2].set_xlabel("Prominent take-off count")
    axes[2].set_ylabel("Maximum raw floor-force norm")
    axes[2].set_title("Take-off frequency and impact")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "smoothness_mechanism_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "smoothness_mechanism_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(RUN_ROOT / "logs" / "evaluation_metrics.csv")
    seed_means = endpoint_evaluation_summary(evaluation)
    contrasts = factorial_contrasts(seed_means)

    contact_rows = []
    for condition in config["conditions"]:
        for seed in config["training_seeds"]:
            print(condition["condition_id"], seed, flush=True)
            contact_rows.append(replay_endpoint(config, condition, int(seed)))
    contact = pd.DataFrame(contact_rows)
    contact.to_csv(OUTPUT / "endpoint_contact_gait_matrix.csv", index=False)
    make_figure(seed_means, contact)

    summary = {
        "status": "complete",
        "endpoint": ENDPOINT,
        "evaluation_episode_rows": int(
            (evaluation["target_timesteps"].astype(int) == ENDPOINT).sum()
        ),
        "policies": int(len(contact)),
        "conditions": int(contact["condition_id"].nunique()),
        "training_seeds": sorted(contact["training_seed"].unique().tolist()),
        "contact_replay_evaluation_seed": REPLAY_EVALUATION_SEED,
        "hopping_dominant_flag_fraction": float(
            contact["exploratory_hopping_dominant_flag"].mean()
        ),
        "condition_endpoint_means": seed_means.groupby("condition_id").mean(numeric_only=True).to_dict(orient="index"),
        "factorial_effect_means": contrasts.groupby("metric").mean(numeric_only=True).to_dict(orient="index"),
        "claim_boundary": "Development mechanism screen. Contact replay uses one fixed evaluation seed per endpoint policy; it is not a formal prevalence estimate or a natural-gait classifier.",
    }
    (OUTPUT / "smoothness_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
