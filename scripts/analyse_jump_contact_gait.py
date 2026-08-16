from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402


RUN_ROOT = ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1"
CONDITION = "Rt0p1_Rvy0__K1p1"
TRAINING_SEED = 41301
EVALUATION_SEED = 51301
TIMESTEPS = 300_000
DT = 0.05
MODEL = (
    RUN_ROOT
    / "runs"
    / f"seed_{TRAINING_SEED}"
    / CONDITION
    / "models"
    / CONDITION
    / f"checkpoint_{TIMESTEPS}.zip"
)
REFERENCE_LOG = (
    RUN_ROOT
    / "runs"
    / f"seed_{TRAINING_SEED}"
    / CONDITION
    / "logs"
    / "evaluation_steps"
    / f"tr{TRAINING_SEED}_t{TIMESTEPS}_ev{EVALUATION_SEED}.csv.gz"
)
OUTPUT = RUN_ROOT / "analysis" / "jump_contact_gait"
XML = ROOT / "assets" / "ant_render_large_floor.xml"

FOOT_GEOMS = {
    "front_left": "left_ankle_geom",
    "front_right": "right_ankle_geom",
    "hind_left": "third_ankle_geom",
    "hind_right": "fourth_ankle_geom",
}


def geom_name(model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def contact_snapshot(model, data) -> dict[str, float | int | str]:
    foot_normal = {name: 0.0 for name in FOOT_GEOMS}
    foot_tangent = {name: 0.0 for name in FOOT_GEOMS}
    floor_force_norms: list[float] = []
    floor_normal_forces: list[float] = []
    floor_contact_pairs: list[str] = []
    inverse = {geom: name for name, geom in FOOT_GEOMS.items()}
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        name1 = geom_name(model, contact.geom1)
        name2 = geom_name(model, contact.geom2)
        if "floor" not in {name1, name2}:
            continue
        other = name2 if name1 == "floor" else name1
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal = abs(float(force[0]))
        tangent = float(np.linalg.norm(force[1:3]))
        force_norm = float(np.linalg.norm(force[:3]))
        floor_normal_forces.append(normal)
        floor_force_norms.append(force_norm)
        floor_contact_pairs.append(other)
        if other in inverse:
            foot = inverse[other]
            foot_normal[foot] += normal
            foot_tangent[foot] += tangent
    contacts = [name for name, value in foot_normal.items() if value > 1e-9]
    values: dict[str, float | int | str] = {
        "mujoco_contact_count": int(data.ncon),
        "floor_contact_count": len(floor_force_norms),
        "floor_total_force_norm": float(sum(floor_force_norms)),
        "floor_max_force_norm": float(max(floor_force_norms, default=0.0)),
        "floor_total_normal_force": float(sum(floor_normal_forces)),
        "floor_max_normal_force": float(max(floor_normal_forces, default=0.0)),
        "foot_contact_pattern": "+".join(contacts) if contacts else "airborne",
    }
    for name in FOOT_GEOMS:
        values[f"{name}_contact"] = int(foot_normal[name] > 1e-9)
        values[f"{name}_normal_force"] = foot_normal[name]
        values[f"{name}_tangential_force"] = foot_tangent[name]
    return values


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id=CONDITION,
        seed=EVALUATION_SEED,
        render_mode=None,
        max_episode_steps=1000,
        xml_file=XML,
        orientation_shaping_weight=0.1,
        orientation_shaping_scale=1.0,
        orientation_shaping_function="cosine",
        lateral_drift_shaping_weight=0.0,
        lateral_drift_shaping_scale=1.0,
        lateral_shaping_signal="velocity_tanh_squared",
        lateral_velocity_target=0.0,
        augment_previous_applied_action=True,
        action_slew_l2_limit=1.1,
    )
    model = PPO.load(MODEL, device="cpu")
    observation, _ = env.reset(seed=EVALUATION_SEED)
    rows: list[dict] = []
    for step_index in range(1, 1001):
        proposed_action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(proposed_action)
        data = env.unwrapped.data
        row = {
            "step_index": step_index,
            "time_seconds": step_index * DT,
            "x_position": float(data.qpos[0]),
            "y_position": float(data.qpos[1]),
            "torso_height": float(data.qpos[2]),
            "root_velocity_x": float(data.qvel[0]),
            "root_velocity_y": float(data.qvel[1]),
            "root_velocity_z": float(data.qvel[2]),
            "condition_objective_reward_step": float(reward),
            "reward_forward_step": float(info.get("reward_forward", np.nan)),
            "reward_ctrl_step": float(info.get("reward_ctrl", np.nan)),
            "reward_contact_step": float(info.get("reward_contact", np.nan)),
            "reward_survive_step": float(info.get("reward_survive", np.nan)),
            "reward_orientation_shaping_step": -0.1
            * float(info.get("proxygap_orientation_penalty_step", np.nan)),
            "applied_action_change_l2_step": float(
                info.get("proxygap_applied_action_change_l2_step", np.nan)
            ),
            "proposed_action_change_l2_step": float(
                info.get("proxygap_proposed_action_change_l2_step", np.nan)
            ),
            "action_correction_l2_step": float(
                info.get("proxygap_action_correction_l2_step", np.nan)
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        row.update(contact_snapshot(env.unwrapped.model, data))
        rows.append(row)
        if terminated or truncated:
            break
    env.close()

    frame = pd.DataFrame(rows)
    reference = pd.read_csv(REFERENCE_LOG)
    merged = frame.merge(reference, on="step_index", suffixes=("_replay", "_logged"))
    replay_errors = {
        "max_abs_x_position_error": float(
            np.max(np.abs(merged["x_position_replay"] - merged["x_position_logged"]))
        ),
        "max_abs_y_position_error": float(
            np.max(np.abs(merged["y_position_replay"] - merged["y_position_logged"]))
        ),
        "max_abs_torso_height_error": float(
            np.max(np.abs(merged["torso_height_replay"] - merged["torso_height_logged"]))
        ),
        "max_abs_reward_error": float(
            np.max(
                np.abs(
                    merged["condition_objective_reward_step_replay"]
                    - merged["condition_objective_reward_step_logged"]
                )
            )
        ),
    }

    jump_row = frame.loc[frame["root_velocity_z"].idxmax()]
    jump_step = int(jump_row["step_index"])
    precursor = frame.loc[frame["step_index"].between(jump_step - 3, jump_step)]
    contact_columns = [f"{name}_contact" for name in FOOT_GEOMS]
    patterns = Counter(frame["foot_contact_pattern"])
    local_peak = (
        (frame["root_velocity_z"] >= frame["root_velocity_z"].shift(1))
        & (frame["root_velocity_z"] > frame["root_velocity_z"].shift(-1))
        & (frame["root_velocity_z"] >= 1.25)
    )
    peak_rows = frame.loc[local_peak].copy()
    prominent_takeoffs: list[dict] = []
    last_step = -100
    for _, peak in peak_rows.iterrows():
        step = int(peak["step_index"])
        if step - last_step < 12:
            continue
        precontact = frame.loc[frame["step_index"].between(step - 3, step)]
        prominent_takeoffs.append(
            {
                "step": step,
                "time_seconds": float(peak["time_seconds"]),
                "root_velocity_z_m_per_s": float(peak["root_velocity_z"]),
                "root_velocity_x_m_per_s": float(peak["root_velocity_x"]),
                "root_velocity_y_m_per_s": float(peak["root_velocity_y"]),
                "objective_reward": float(peak["condition_objective_reward_step"]),
                "preceding_four_step_max_floor_force_norm": float(
                    precontact["floor_total_force_norm"].max()
                ),
            }
        )
        last_step = step
    summary = {
        "status": "development_mechanism_diagnostic_complete",
        "condition_id": CONDITION,
        "training_seed": TRAINING_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "checkpoint_timesteps": TIMESTEPS,
        "replay_matches_logged_episode": all(value < 1e-8 for value in replay_errors.values()),
        "replay_errors": replay_errors,
        "jump_event": {
            "definition": "step with maximum MuJoCo root z velocity in this episode",
            "step": jump_step,
            "time_seconds": float(jump_row["time_seconds"]),
            "root_velocity_z_m_per_s": float(jump_row["root_velocity_z"]),
            "root_velocity_x_m_per_s": float(jump_row["root_velocity_x"]),
            "root_velocity_y_m_per_s": float(jump_row["root_velocity_y"]),
            "torso_height_m": float(jump_row["torso_height"]),
            "condition_objective_reward": float(jump_row["condition_objective_reward_step"]),
            "forward_reward": float(jump_row["reward_forward_step"]),
            "control_reward": float(jump_row["reward_ctrl_step"]),
            "contact_reward": float(jump_row["reward_contact_step"]),
            "orientation_shaping_reward": float(
                jump_row["reward_orientation_shaping_step"]
            ),
            "raw_floor_force_norm": float(jump_row["floor_total_force_norm"]),
            "raw_floor_normal_force": float(jump_row["floor_total_normal_force"]),
            "preceding_four_step_max_floor_force_norm": float(
                precursor["floor_total_force_norm"].max()
            ),
            "preceding_four_step_max_floor_normal_force": float(
                precursor["floor_total_normal_force"].max()
            ),
        },
        "prominent_takeoff_events_vz_at_least_1p25_m_per_s": prominent_takeoffs,
        "contact_and_gait": {
            "no_floor_contact_step_fraction": float(
                (frame["floor_contact_count"] == 0).mean()
            ),
            "no_foot_contact_step_fraction": float(
                (frame[contact_columns].sum(axis=1) == 0).mean()
            ),
            "front_pair_simultaneous_contact_fraction": float(
                (
                    (frame["front_left_contact"] == 1)
                    & (frame["front_right_contact"] == 1)
                ).mean()
            ),
            "hind_pair_simultaneous_contact_fraction": float(
                (
                    (frame["hind_left_contact"] == 1)
                    & (frame["hind_right_contact"] == 1)
                ).mean()
            ),
            "all_four_contact_fraction": float((frame[contact_columns].sum(axis=1) == 4).mean()),
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
            "per_foot_duty_factor": {
                name: float(frame[f"{name}_contact"].mean()) for name in FOOT_GEOMS
            },
            "contact_transition_count": {
                name: int(frame[f"{name}_contact"].diff().abs().fillna(0).sum())
                for name in FOOT_GEOMS
            },
            "most_common_contact_patterns": patterns.most_common(8),
        },
        "contact_cost_interpretation": {
            "minimum_logged_reward_contact_step": float(frame["reward_contact_step"].min()),
            "raw_force_units": "MuJoCo constraint-force units; development diagnostic only",
            "warning": "Gymnasium Ant contact cost clips external contact-force components before squaring. The logged contact reward is therefore not a linear measure of raw impact magnitude.",
        },
        "claim_boundary": "Single deterministic evaluation episode; identifies a plausible mechanism but does not estimate population frequency or prove causality.",
    }

    frame.to_csv(OUTPUT / "replayed_step_contact_gait.csv", index=False)
    precursor.to_csv(OUTPUT / "jump_window.csv", index=False)
    (OUTPUT / "jump_contact_gait_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    event = frame.loc[frame["step_index"].between(jump_step - 20, jump_step + 25)].copy()
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(event["time_seconds"], event["torso_height"], label="torso height", color="#173753")
    twin = axes[0].twinx()
    twin.plot(event["time_seconds"], event["root_velocity_z"], label="vertical velocity", color="#A63A3A")
    axes[0].set_ylabel("height (m)")
    twin.set_ylabel("vz (m/s)")
    axes[1].plot(event["time_seconds"], event["root_velocity_x"], label="vx", color="#0F766E")
    axes[1].plot(event["time_seconds"], event["root_velocity_y"], label="vy", color="#C99528")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[2].plot(event["time_seconds"], event["floor_total_force_norm"], label="raw floor force norm", color="#6D597A")
    axes[2].plot(event["time_seconds"], event["condition_objective_reward_step"], label="objective reward", color="#E76F51")
    axes[2].set_ylabel("force / reward")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    for index, name in enumerate(FOOT_GEOMS):
        axes[3].step(
            event["time_seconds"],
            event[f"{name}_contact"] + index * 1.35,
            where="post",
            label=name.replace("_", " "),
        )
    axes[3].set_yticks([])
    axes[3].set_ylabel("foot contact")
    axes[3].set_xlabel("simulation time (s)")
    axes[3].legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    for axis in axes:
        axis.axvline(jump_step * DT, color="#A63A3A", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
    fig.suptitle("Jump-event diagnostic: deterministic endpoint episode")
    fig.tight_layout()
    fig.savefig(OUTPUT / "jump_event_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "jump_event_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
