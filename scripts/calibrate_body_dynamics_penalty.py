"""Calibrate bounded body-dynamics penalty scales from development policies.

This script does not select a successful policy and does not alter any reward.
It replays every frozen 1M endpoint policy on the paired development evaluation
seeds, records body-level motion, and reports transparent scale candidates for a
later, separately frozen mechanism experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "smoothness_target_budget_extension_v1_20260816.json"
RUN_ROOT = ROOT / "artifacts" / "dev" / "smoothness_target_extension_1m_v1"
OUTPUT = RUN_ROOT / "analysis" / "body_dynamics_calibration"
ENDPOINT = 1_000_000


def model_path(condition_id: str, training_seed: int) -> Path:
    return (
        RUN_ROOT
        / "runs"
        / f"seed_{training_seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{ENDPOINT:07d}.zip"
    )


def make_env(config: dict, condition_id: str, evaluation_seed: int):
    shared = config["shared_reward"]
    return make_proxygap_ant_env(
        ctrl_cost_weight=float(shared["ctrl_cost_weight"]),
        condition_id=condition_id,
        seed=evaluation_seed,
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
        action_rate_shaping_weight=float(
            config["condition_parameters"][condition_id]["action_rate_shaping_weight"]
        ),
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
    )


def replay_policy(config: dict, condition_id: str, training_seed: int) -> pd.DataFrame:
    path = model_path(condition_id, training_seed)
    if not path.exists():
        raise FileNotFoundError(path)
    model = PPO.load(path, device="cpu")
    frames: list[pd.DataFrame] = []
    for evaluation_seed in config["evaluation_seeds"]:
        env = make_env(config, condition_id, int(evaluation_seed))
        observation, _ = env.reset(seed=int(evaluation_seed))
        rows: list[dict] = []
        for step_index in range(1, 1001):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
            rows.append(
                {
                    "condition_id": condition_id,
                    "training_seed": training_seed,
                    "evaluation_seed": int(evaluation_seed),
                    "step_index": step_index,
                    "abs_root_vertical_velocity": abs(float(qvel[2])),
                    "root_roll_pitch_angular_speed": float(np.linalg.norm(qvel[3:5])),
                    "root_yaw_angular_speed_abs": abs(float(qvel[5])),
                    "objective_reward_step": float(reward),
                }
            )
            if terminated or truncated:
                break
        env.close()
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def quantile_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    signals = [
        "abs_root_vertical_velocity",
        "root_roll_pitch_angular_speed",
        "root_yaw_angular_speed_abs",
    ]
    for signal in signals:
        values = frame[signal].to_numpy(dtype=float)
        for quantile in (0.50, 0.75, 0.90, 0.95, 0.99):
            rows.append(
                {
                    "signal": signal,
                    "quantile": quantile,
                    "value": float(np.quantile(values, quantile)),
                    "step_rows": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def penalty_scale_rows(frame: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for signal in ("abs_root_vertical_velocity", "root_roll_pitch_angular_speed"):
        values = frame[signal].to_numpy(dtype=float)
        for source_quantile in (0.75, 0.90, 0.95):
            scale = float(
                quantiles.loc[
                    (quantiles["signal"] == signal)
                    & np.isclose(quantiles["quantile"], source_quantile),
                    "value",
                ].iloc[0]
            )
            bounded_penalty = np.tanh(np.square(values / max(scale, 1e-12)))
            rows.append(
                {
                    "signal": signal,
                    "scale_source_quantile": source_quantile,
                    "scale": scale,
                    "mean_bounded_penalty": float(bounded_penalty.mean()),
                    "median_bounded_penalty": float(np.median(bounded_penalty)),
                    "p95_bounded_penalty": float(np.quantile(bounded_penalty, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for condition_id in config["conditions"]:
        for training_seed in config["training_seeds"]:
            print(condition_id, training_seed, flush=True)
            frames.append(replay_policy(config, condition_id, int(training_seed)))
    frame = pd.concat(frames, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "body_dynamics_step_samples.csv.gz", index=False)
    quantiles = quantile_rows(frame)
    quantiles.to_csv(OUTPUT / "body_dynamics_quantiles.csv", index=False)
    scales = penalty_scale_rows(frame, quantiles)
    scales.to_csv(OUTPUT / "bounded_penalty_scale_candidates.csv", index=False)
    summary = {
        "status": "development_calibration_complete",
        "endpoint_policies": int(
            frame[["condition_id", "training_seed"]].drop_duplicates().shape[0]
        ),
        "evaluation_episodes": int(
            frame[["condition_id", "training_seed", "evaluation_seed"]]
            .drop_duplicates()
            .shape[0]
        ),
        "step_rows": int(len(frame)),
        "penalty_family": "tanh((signal/scale)^2)",
        "claim_boundary": (
            "Development calibration only. Quantiles describe the current endpoint "
            "policies and do not define natural gait, safety, or an optimal reward."
        ),
    }
    (OUTPUT / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
