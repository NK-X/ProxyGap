from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_jump_contact_gait import FOOT_GEOMS, contact_snapshot  # noqa: E402
from proxygap import make_proxygap_ant_env  # noqa: E402


CONFIG = ROOT / "configs" / "hybrid_guardrail_observability_correction_v1_20260816.json"
RUN_ROOT = ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1"
OUTPUT = RUN_ROOT / "analysis" / "contact_gait_matrix"
XML = ROOT / "assets" / "ant_render_large_floor.xml"
EVALUATION_SEED = 51301
CHECKPOINT = 300_000
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


def replay(condition: dict, training_seed: int) -> dict:
    condition_id = condition["condition_id"]
    model_path = (
        RUN_ROOT
        / "runs"
        / f"seed_{training_seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{CHECKPOINT}.zip"
    )
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id=condition_id,
        seed=EVALUATION_SEED,
        render_mode=None,
        max_episode_steps=1000,
        xml_file=XML,
        orientation_shaping_weight=float(condition["orientation_shaping_weight"]),
        orientation_shaping_scale=float(condition["orientation_shaping_scale"]),
        orientation_shaping_function=str(condition["orientation_shaping_function"]),
        lateral_drift_shaping_weight=float(condition["lateral_drift_shaping_weight"]),
        lateral_drift_shaping_scale=float(condition["lateral_drift_shaping_scale"]),
        lateral_shaping_signal=str(condition["lateral_shaping_signal"]),
        lateral_velocity_target=float(condition["lateral_velocity_target"]),
        augment_previous_applied_action=True,
        action_slew_l2_limit=condition["action_slew_l2_limit"],
    )
    model = PPO.load(model_path, device="cpu")
    observation, _ = env.reset(seed=EVALUATION_SEED)
    rows: list[dict] = []
    start_x = float(env.unwrapped.data.qpos[0])
    start_y = float(env.unwrapped.data.qpos[1])
    previous_x = start_x
    previous_y = start_y
    planar_path = 0.0
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
            "objective_reward": float(reward),
            "forward_reward": float(info.get("reward_forward", np.nan)),
            "control_reward": float(info.get("reward_ctrl", np.nan)),
            "contact_reward": float(info.get("reward_contact", np.nan)),
            "orientation_shaping_reward": -float(
                condition["orientation_shaping_weight"]
            )
            * float(info.get("proxygap_orientation_penalty_step", np.nan)),
            "applied_action_change_l2": float(
                info.get("proxygap_applied_action_change_l2_step", np.nan)
            ),
            "proposed_action_change_l2": float(
                info.get("proxygap_proposed_action_change_l2_step", np.nan)
            ),
        }
        row.update(contact_snapshot(env.unwrapped.model, data))
        rows.append(row)
        if terminated or truncated:
            break
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
        "evaluation_seed": EVALUATION_SEED,
        "episode_steps": len(frame),
        "net_forward_progress_m": net_x,
        "net_lateral_displacement_m": net_y,
        "mean_forward_velocity_m_per_s": net_x / (len(frame) * DT),
        "direction_error_degrees": direction_error,
        "forward_path_efficiency": net_x / max(planar_path, 1e-12),
        "objective_return": float(frame["objective_reward"].sum()),
        "no_floor_contact_step_fraction": float((frame["floor_contact_count"] == 0).mean()),
        "no_foot_contact_step_fraction": float(
            (frame[contact_columns].sum(axis=1) == 0).mean()
        ),
        "prominent_takeoff_count_vz_ge_1p25": count_prominent_takeoffs(
            frame["root_velocity_z"].to_numpy()
        ),
        "max_root_velocity_z_m_per_s": float(frame["root_velocity_z"].max()),
        "max_raw_floor_force_norm": float(frame["floor_total_force_norm"].max()),
        "p95_raw_floor_force_norm": float(frame["floor_total_force_norm"].quantile(0.95)),
        "same_side_pair_contact_fraction": float(
            (
                (
                    (frame["front_left_contact"] == 1)
                    & (frame["hind_left_contact"] == 1)
                )
                | (
                    (frame["front_right_contact"] == 1)
                    & (frame["hind_right_contact"] == 1)
                )
            ).mean()
        ),
        "diagonal_pair_contact_fraction": float(
            (
                (
                    (frame["front_left_contact"] == 1)
                    & (frame["hind_right_contact"] == 1)
                )
                | (
                    (frame["front_right_contact"] == 1)
                    & (frame["hind_left_contact"] == 1)
                )
            ).mean()
        ),
        "front_pair_simultaneous_contact_fraction": float(
            (
                (frame["front_left_contact"] == 1)
                & (frame["front_right_contact"] == 1)
            ).mean()
        ),
        "mean_applied_action_change_l2": float(
            frame["applied_action_change_l2"].iloc[1:].mean()
        ),
        "mean_proposed_action_change_l2": float(
            frame["proposed_action_change_l2"].iloc[1:].mean()
        ),
        "mean_forward_reward": float(frame["forward_reward"].mean()),
        "mean_control_reward": float(frame["control_reward"].mean()),
        "mean_contact_reward": float(frame["contact_reward"].mean()),
        "mean_orientation_shaping_reward": float(
            frame["orientation_shaping_reward"].mean()
        ),
    }
    result["exploratory_hopping_dominant_flag"] = bool(
        result["no_floor_contact_step_fraction"] >= 0.25
        or result["prominent_takeoff_count_vz_ge_1p25"] >= 5
    )
    return result


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for condition in config["conditions"]:
        for training_seed in config["training_seeds"]:
            print(condition["condition_id"], training_seed, flush=True)
            rows.append(replay(condition, int(training_seed)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "endpoint_contact_gait_matrix.csv", index=False)

    summary = {
        "status": "complete",
        "policy_count": len(frame),
        "conditions": int(frame["condition_id"].nunique()),
        "training_seeds": sorted(frame["training_seed"].unique().tolist()),
        "evaluation_seed": EVALUATION_SEED,
        "hopping_dominant_flag_count": int(frame["exploratory_hopping_dominant_flag"].sum()),
        "hopping_dominant_flag_fraction": float(
            frame["exploratory_hopping_dominant_flag"].mean()
        ),
        "no_floor_contact_fraction_range": [
            float(frame["no_floor_contact_step_fraction"].min()),
            float(frame["no_floor_contact_step_fraction"].max()),
        ],
        "prominent_takeoff_count_range": [
            int(frame["prominent_takeoff_count_vz_ge_1p25"].min()),
            int(frame["prominent_takeoff_count_vz_ge_1p25"].max()),
        ],
        "max_raw_floor_force_norm_range": [
            float(frame["max_raw_floor_force_norm"].min()),
            float(frame["max_raw_floor_force_norm"].max()),
        ],
        "claim_boundary": "One fixed evaluation seed per endpoint policy; development mechanism screen, not formal prevalence estimate or universal gait classification.",
    }
    (OUTPUT / "endpoint_contact_gait_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    colours = {41301: "#0072B2", 41302: "#D55E00", 41303: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for seed, group in frame.groupby("training_seed"):
        axes[0].scatter(
            group["no_floor_contact_step_fraction"],
            group["mean_forward_velocity_m_per_s"],
            label=f"training seed {seed}",
            color=colours[int(seed)],
            s=58,
        )
        axes[1].scatter(
            group["same_side_pair_contact_fraction"],
            group["diagonal_pair_contact_fraction"],
            label=f"training seed {seed}",
            color=colours[int(seed)],
            s=58,
        )
    axes[0].axvline(0.25, color="#A63A3A", linestyle="--", linewidth=1)
    axes[0].set_xlabel("No-floor-contact step fraction")
    axes[0].set_ylabel("Mean forward velocity (m/s)")
    axes[0].set_title("Flight time versus progress")
    axes[1].plot([0, 0.25], [0, 0.25], color="#566573", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Same-side pair contact fraction")
    axes[1].set_ylabel("Diagonal pair contact fraction")
    axes[1].set_title("Contact-pair structure")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "endpoint_contact_gait_matrix.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "endpoint_contact_gait_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
