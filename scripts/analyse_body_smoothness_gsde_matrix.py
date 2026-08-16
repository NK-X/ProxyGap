"""Analyse the frozen body-smoothness and gSDE development matrix."""

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


CONFIG_PATH = ROOT / "configs" / "body_smoothness_gsde_matrix_v1_20260816.json"
RUN_ROOT = ROOT / "artifacts" / "dev" / "body_smoothness_gsde_matrix_v1"
OUTPUT = RUN_ROOT / "analysis"
XML = ROOT / "assets" / "ant_render_large_floor.xml"
ENDPOINT = 1_000_000
DT = 0.05


EVALUATION_METRICS = [
    "condition_objective_return",
    "base_proxy_return",
    "fixed_horizon_mean_forward_velocity",
    "forward_path_efficiency",
    "net_displacement_direction_error_degrees",
    "torso_tilt_rms",
    "normalised_action_roughness",
    "action_saturation_rate",
    "intent_compliant",
    "unhealthy_termination",
    "sustained_inversion",
    "reward_vertical_velocity_shaping_sum",
    "reward_roll_pitch_angular_velocity_shaping_sum",
]

BODY_METRICS = [
    "fixed_horizon_forward_velocity_m_per_s",
    "forward_path_efficiency",
    "direction_error_degrees",
    "torso_tilt_rms_rad",
    "normalised_action_roughness",
    "rms_root_vertical_velocity_m_per_s",
    "max_abs_root_vertical_velocity_m_per_s",
    "rms_root_roll_pitch_angular_speed_rad_per_s",
    "max_root_roll_pitch_angular_speed_rad_per_s",
    "no_floor_contact_step_fraction",
    "no_foot_contact_step_fraction",
    "prominent_takeoff_count_vz_ge_1p25",
    "max_raw_floor_force_norm",
    "p95_raw_floor_force_norm",
    "unhealthy_termination",
    "intent_compliant",
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


def condition(config: dict, condition_id: str) -> dict:
    return next(item for item in config["conditions"] if item["condition_id"] == condition_id)


def body_parameters(config: dict, condition_id: str) -> tuple[float, float]:
    enabled = bool(condition(config, condition_id)["body_dynamics_enabled"])
    body = config["body_dynamics"]
    if not enabled:
        return 0.0, 0.0
    return (
        float(body["vertical_velocity_shaping_weight"]),
        float(body["roll_pitch_angular_velocity_shaping_weight"]),
    )


def make_env(config: dict, condition_id: str, evaluation_seed: int):
    shared = config["shared_reward"]
    body = config["body_dynamics"]
    vertical_weight, angular_weight = body_parameters(config, condition_id)
    return make_proxygap_ant_env(
        ctrl_cost_weight=float(shared["ctrl_cost_weight"]),
        condition_id=condition_id,
        seed=evaluation_seed,
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
        action_rate_shaping_weight=float(shared["action_rate_shaping_weight"]),
        vertical_velocity_shaping_weight=vertical_weight,
        vertical_velocity_shaping_scale=float(body["vertical_velocity_shaping_scale"]),
        roll_pitch_angular_velocity_shaping_weight=angular_weight,
        roll_pitch_angular_velocity_shaping_scale=float(
            body["roll_pitch_angular_velocity_shaping_scale"]
        ),
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
    )


def model_path(condition_id: str, training_seed: int) -> Path:
    return (
        RUN_ROOT
        / "runs"
        / f"seed_{training_seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{ENDPOINT}.zip"
    )


def load_evaluation() -> pd.DataFrame:
    frame = pd.read_csv(RUN_ROOT / "logs" / "evaluation_metrics.csv")
    for column in ("intent_compliant", "unhealthy_termination", "sustained_inversion"):
        frame[column] = frame[column].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    frame[EVALUATION_METRICS] = frame[EVALUATION_METRICS].apply(pd.to_numeric)
    frame["training_seed"] = pd.to_numeric(frame["training_seed"]).astype(int)
    frame["target_timesteps"] = pd.to_numeric(frame["target_timesteps"]).astype(int)
    duplicate = frame.duplicated(
        ["condition_id", "training_seed", "target_timesteps", "seed", "episode"]
    )
    if duplicate.any():
        raise ValueError(f"Duplicate evaluation rows: {int(duplicate.sum())}")
    return frame


def evaluation_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame.groupby(["condition_id", "training_seed", "target_timesteps"]).size()
    if set(counts.unique()) != {10}:
        raise ValueError(f"Expected ten episodes per policy-checkpoint, got {counts.unique()}")
    policy = (
        frame.groupby(["condition_id", "training_seed", "target_timesteps"], as_index=False)[EVALUATION_METRICS]
        .mean()
        .sort_values(["condition_id", "training_seed", "target_timesteps"])
    )
    condition_summary = policy.groupby(["condition_id", "target_timesteps"])[EVALUATION_METRICS].agg(
        ["mean", "std", "min", "max"]
    )
    condition_summary.columns = [f"{metric}_{stat}" for metric, stat in condition_summary.columns]
    condition_summary = condition_summary.reset_index()
    policy.to_csv(OUTPUT / "checkpoint_policy_means.csv", index=False)
    condition_summary.to_csv(OUTPUT / "checkpoint_condition_summary.csv", index=False)
    return policy, condition_summary


def replay_episode(config: dict, condition_id: str, training_seed: int, evaluation_seed: int, model: PPO) -> dict:
    env = make_env(config, condition_id, evaluation_seed)
    observation, _ = env.reset(seed=evaluation_seed)
    start_x = float(env.unwrapped.data.qpos[0])
    start_y = float(env.unwrapped.data.qpos[1])
    previous_x, previous_y = start_x, start_y
    path = 0.0
    rows: list[dict] = []
    for step_index in range(1, 1001):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        data = env.unwrapped.data
        x_position = float(data.qpos[0])
        y_position = float(data.qpos[1])
        path += float(np.hypot(x_position - previous_x, y_position - previous_y))
        previous_x, previous_y = x_position, y_position
        row = {
            "root_vertical_velocity": float(data.qvel[2]),
            "root_roll_pitch_angular_speed": float(np.linalg.norm(data.qvel[3:5])),
        }
        row.update(contact_snapshot(env.unwrapped.model, data))
        rows.append(row)
        if terminated or truncated:
            break
    frame = pd.DataFrame(rows)
    summary = env.episode_summary()
    final_x = float(env.unwrapped.data.qpos[0])
    final_y = float(env.unwrapped.data.qpos[1])
    env.close()
    net_x = final_x - start_x
    net_y = final_y - start_y
    contact_columns = [f"{name}_contact" for name in FOOT_GEOMS]
    return {
        "condition_id": condition_id,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "episode_steps": len(frame),
        "fixed_horizon_forward_velocity_m_per_s": net_x / (1000 * DT),
        "forward_path_efficiency": net_x / max(path, 1e-12),
        "direction_error_degrees": float(np.degrees(np.arctan2(abs(net_y), max(net_x, 1e-12)))),
        "torso_tilt_rms_rad": float(summary["torso_tilt_rms"]),
        "normalised_action_roughness": float(summary["normalised_action_roughness"]),
        "rms_root_vertical_velocity_m_per_s": float(np.sqrt(np.mean(np.square(frame["root_vertical_velocity"])))),
        "max_abs_root_vertical_velocity_m_per_s": float(frame["root_vertical_velocity"].abs().max()),
        "rms_root_roll_pitch_angular_speed_rad_per_s": float(np.sqrt(np.mean(np.square(frame["root_roll_pitch_angular_speed"])))),
        "max_root_roll_pitch_angular_speed_rad_per_s": float(frame["root_roll_pitch_angular_speed"].max()),
        "no_floor_contact_step_fraction": float((frame["floor_contact_count"] == 0).mean()),
        "no_foot_contact_step_fraction": float((frame[contact_columns].sum(axis=1) == 0).mean()),
        "prominent_takeoff_count_vz_ge_1p25": count_prominent_takeoffs(frame["root_vertical_velocity"].to_numpy()),
        "max_raw_floor_force_norm": float(frame["floor_total_force_norm"].max()),
        "p95_raw_floor_force_norm": float(frame["floor_total_force_norm"].quantile(0.95)),
        "unhealthy_termination": float(bool(summary["unhealthy_termination"])),
        "intent_compliant": float(bool(summary["intent_compliant"])),
    }


def replay_all(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for item in config["conditions"]:
        condition_id = str(item["condition_id"])
        for training_seed in config["training_seeds"]:
            print(condition_id, training_seed, flush=True)
            model = PPO.load(model_path(condition_id, int(training_seed)), device="cpu")
            for evaluation_seed in config["evaluation_seeds"]:
                rows.append(
                    replay_episode(
                        config,
                        condition_id,
                        int(training_seed),
                        int(evaluation_seed),
                        model,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "endpoint_body_contact_episode_metrics.csv", index=False)
    return frame


def factorial_contrasts(policy: pd.DataFrame, metrics: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = policy.copy()
    if "target_timesteps" in endpoint:
        endpoint = endpoint.loc[endpoint["target_timesteps"] == ENDPOINT]
    rows: list[dict] = []
    for seed, group in endpoint.groupby("training_seed"):
        indexed = group.set_index("condition_id")
        for metric in metrics:
            b_at_g0 = float(indexed.loc["B1__G0", metric] - indexed.loc["B0__G0", metric])
            b_at_g8 = float(indexed.loc["B1__G8", metric] - indexed.loc["B0__G8", metric])
            g_at_b0 = float(indexed.loc["B0__G8", metric] - indexed.loc["B0__G0", metric])
            g_at_b1 = float(indexed.loc["B1__G8", metric] - indexed.loc["B1__G0", metric])
            rows.extend(
                [
                    {"training_seed": int(seed), "metric": metric, "effect": "body_main", "contrast": (b_at_g0 + b_at_g8) / 2},
                    {"training_seed": int(seed), "metric": metric, "effect": "gsde_main", "contrast": (g_at_b0 + g_at_b1) / 2},
                    {"training_seed": int(seed), "metric": metric, "effect": "interaction", "contrast": b_at_g8 - b_at_g0},
                ]
            )
    contrasts = pd.DataFrame(rows)
    summary = contrasts.groupby(["metric", "effect"])["contrast"].agg(["mean", "std", "min", "max"]).reset_index()
    contrasts.to_csv(OUTPUT / f"{prefix}_paired_factorial_contrasts.csv", index=False)
    summary.to_csv(OUTPUT / f"{prefix}_factorial_contrast_summary.csv", index=False)
    return contrasts, summary


def make_figure(policy: pd.DataFrame, body_policy: pd.DataFrame) -> None:
    endpoint = policy.loc[policy["target_timesteps"] == ENDPOINT]
    merged = endpoint.merge(
        body_policy,
        on=["condition_id", "training_seed"],
        suffixes=("", "_body"),
    )
    condition_order = ["B0__G0", "B1__G0", "B0__G8", "B1__G8"]
    condition_labels = ["B0 / G0", "B1 / G0", "B0 / G8", "B1 / G8"]
    seed_colours = {41501: "#0072B2", 41502: "#D55E00", 41503: "#009E73"}
    seed_markers = {41501: "o", 41502: "s", 41503: "^"}
    panels = [
        ("fixed_horizon_forward_velocity_m_per_s", "Forward velocity (m/s)"),
        ("forward_path_efficiency_body", "Forward path efficiency"),
        ("direction_error_degrees", "Direction error (degrees)"),
        ("rms_root_vertical_velocity_m_per_s", "RMS vertical velocity (m/s)"),
        ("rms_root_roll_pitch_angular_speed_rad_per_s", "RMS roll/pitch rate (rad/s)"),
        ("no_floor_contact_step_fraction", "No-floor-contact fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        for position in (0.5, 2.5):
            axis.axvline(position, color="#CBD5E1", linewidth=0.8, zorder=0)
        for seed in sorted(seed_colours):
            group = (
                merged.loc[merged["training_seed"] == seed]
                .set_index("condition_id")
                .loc[condition_order]
            )
            axis.plot(
                range(len(condition_order)),
                group[metric],
                color=seed_colours[seed],
                alpha=0.28,
                linewidth=1.0,
                zorder=1,
            )
            axis.scatter(
                range(len(condition_order)),
                group[metric],
                color=seed_colours[seed],
                marker=seed_markers[seed],
                edgecolor="white",
                linewidth=0.5,
                s=55,
                label=f"Training seed {seed}",
                zorder=2,
            )
        for x, condition_id in enumerate(condition_order):
            mean_value = merged.loc[merged["condition_id"] == condition_id, metric].mean()
            axis.scatter(
                x,
                mean_value,
                color="black",
                marker="D",
                edgecolor="white",
                linewidth=0.7,
                s=72,
                label="Condition mean" if x == 0 else None,
                zorder=3,
            )
            axis.annotate(
                f"{mean_value:.2f}",
                (x, mean_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#111827",
            )
        axis.set_xticks(range(len(condition_order)), condition_labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Body-dynamics shaping and gSDE: paired development endpoints", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.015,
        "B0/B1: body-dynamics penalty absent/present; G0/G8: ordinary Gaussian exploration/gSDE resampled every 8 steps.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUTPUT / "body_smoothness_gsde_endpoint_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    execution = json.loads((RUN_ROOT / "execution_record.json").read_text(encoding="utf-8"))
    if execution.get("status") != "complete":
        raise RuntimeError("Development matrix is not complete")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluation = load_evaluation()
    expected_evaluation_rows = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["checkpoint_timesteps"])
        * len(config["evaluation_seeds"])
    )
    if len(evaluation) != expected_evaluation_rows:
        raise ValueError(
            f"Expected {expected_evaluation_rows} evaluation rows, got {len(evaluation)}"
        )
    if int(evaluation[EVALUATION_METRICS].isna().sum().sum()) != 0:
        raise ValueError("Missing values detected in evaluation metrics")
    policy, _ = evaluation_summaries(evaluation)
    _, evaluation_effects = factorial_contrasts(policy, EVALUATION_METRICS, "evaluation")
    body_episodes = replay_all(config)
    expected_body_episodes = (
        len(config["conditions"])
        * len(config["training_seeds"])
        * len(config["evaluation_seeds"])
    )
    if len(body_episodes) != expected_body_episodes:
        raise ValueError(
            f"Expected {expected_body_episodes} body replays, got {len(body_episodes)}"
        )
    if int(body_episodes[BODY_METRICS].isna().sum().sum()) != 0:
        raise ValueError("Missing values detected in body/contact metrics")
    body_policy = body_episodes.groupby(["condition_id", "training_seed"], as_index=False)[BODY_METRICS].mean()
    body_policy.to_csv(OUTPUT / "endpoint_body_contact_policy_means.csv", index=False)
    body_condition = body_policy.groupby("condition_id")[BODY_METRICS].agg(["mean", "std", "min", "max"])
    body_condition.columns = [f"{metric}_{stat}" for metric, stat in body_condition.columns]
    body_condition.reset_index().to_csv(OUTPUT / "endpoint_body_contact_condition_summary.csv", index=False)
    _, body_effects = factorial_contrasts(body_policy, BODY_METRICS, "body_contact")
    make_figure(policy, body_policy)
    summary = {
        "status": "complete",
        "research_stage": "development_mechanism_matrix",
        "independent_training_runs": int(body_policy.shape[0]),
        "evaluation_episode_rows": int(len(evaluation)),
        "endpoint_body_replay_episodes": int(len(body_episodes)),
        "endpoint_condition_means": body_policy.groupby("condition_id").mean(numeric_only=True).to_dict(orient="index"),
        "evaluation_factorial_effect_means": evaluation_effects.set_index(["metric", "effect"])["mean"].to_dict(),
        "body_factorial_effect_means": body_effects.set_index(["metric", "effect"])["mean"].to_dict(),
        "claim_boundary": "Development mechanism evidence only. Contact and body replay spans ten paired initial-state seeds per trained endpoint but does not validate a biological gait or hardware safety.",
    }
    # JSON requires string keys rather than metric/effect tuples.
    summary["evaluation_factorial_effect_means"] = {
        f"{metric}|{effect}": float(value)
        for (metric, effect), value in summary["evaluation_factorial_effect_means"].items()
    }
    summary["body_factorial_effect_means"] = {
        f"{metric}|{effect}": float(value)
        for (metric, effect), value in summary["body_factorial_effect_means"].items()
    }
    (OUTPUT / "body_smoothness_gsde_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    qa = {
        "status": "PASS",
        "expected_evaluation_rows": expected_evaluation_rows,
        "observed_evaluation_rows": int(len(evaluation)),
        "expected_policy_checkpoint_units": (
            len(config["conditions"])
            * len(config["training_seeds"])
            * len(config["checkpoint_timesteps"])
        ),
        "observed_policy_checkpoint_units": int(len(policy)),
        "expected_endpoint_body_replays": expected_body_episodes,
        "observed_endpoint_body_replays": int(len(body_episodes)),
        "evaluation_duplicate_keys": 0,
        "evaluation_missing_metric_values": int(
            evaluation[EVALUATION_METRICS].isna().sum().sum()
        ),
        "body_missing_metric_values": int(body_episodes[BODY_METRICS].isna().sum().sum()),
        "evaluation_group_sizes": sorted(
            evaluation.groupby(
                ["condition_id", "training_seed", "target_timesteps"]
            ).size().unique().tolist()
        ),
        "body_group_sizes": sorted(
            body_episodes.groupby(["condition_id", "training_seed"]).size().unique().tolist()
        ),
    }
    if qa["observed_policy_checkpoint_units"] != qa["expected_policy_checkpoint_units"]:
        raise ValueError("Policy-checkpoint unit count mismatch")
    if qa["evaluation_group_sizes"] != [len(config["evaluation_seeds"])]:
        raise ValueError(f"Unexpected evaluation group sizes: {qa['evaluation_group_sizes']}")
    if qa["body_group_sizes"] != [len(config["evaluation_seeds"])]:
        raise ValueError(f"Unexpected body replay group sizes: {qa['body_group_sizes']}")
    (OUTPUT / "data_quality_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
