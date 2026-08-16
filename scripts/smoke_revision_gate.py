"""Run a tiny, non-scientific PPO smoke for the revised v2 logging path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    runtime_rows, evaluation_rows = train_condition(
        output_root=output,
        condition_id="revision_gate_smoke",
        ctrl_cost_weight=0.0625,
        effort_shaping_weight=0.1,
        effort_shaping_scale=2.0,
        orientation_shaping_weight=0.1,
        orientation_shaping_scale=0.5,
        common_rescore_ctrl_cost_weight=0.5,
        effort_distance_min=0.1,
        action_saturation_threshold=0.95,
        total_timesteps=256,
        checkpoint_timesteps=[256],
        seed=20260810,
        evaluation_seed_base=90260810,
        eval_episodes=1,
        eval_max_episode_steps=10,
        ppo_n_steps=128,
        ppo_batch_size=64,
        ppo_n_epochs=1,
        ppo_learning_rate=3e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_range=0.2,
        ppo_ent_coef=0.0,
        ppo_vf_coef=0.5,
        ppo_max_grad_norm=0.5,
        ppo_normalize_advantage=True,
        ppo_policy="MlpPolicy",
        ppo_policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": "Tanh",
        },
        ppo_device="cpu",
        record_evaluation_steps=True,
    )
    write_standard_outputs(
        output,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    row = evaluation_rows[0]
    required = {
        "condition_objective_return",
        "common_rescored_return",
        "unhealthy_termination",
        "termination_category",
        "cumulative_squared_action",
        "torso_tilt_rms",
    }
    missing = sorted(required - set(row))
    if missing:
        raise AssertionError(f"Missing revised evaluation fields: {missing}")
    if abs(float(row["base_reward_reconciliation_error"])) > 1e-4:
        raise AssertionError("Base reward reconciliation failed")
    monitor = output / "logs" / "training.monitor.csv"
    step_logs = list((output / "logs" / "evaluation_steps").glob("*.csv.gz"))
    models = list((output / "models").rglob("*.zip"))
    if not monitor.exists() or len(step_logs) != 1 or len(models) != 1:
        raise AssertionError("Monitor, step log or model checkpoint is missing")
    model = PPO.load(models[0], device="cpu")
    if int(model.num_timesteps) < 256:
        raise AssertionError("Saved checkpoint has fewer timesteps than requested")
    values = [
        float(row["condition_objective_return"]),
        float(row["common_rescored_return"]),
        float(row["cumulative_squared_action"]),
    ]
    if not np.isfinite(values).all():
        raise AssertionError("Revised smoke produced non-finite required metrics")
    result = {
        "status": "pass",
        "role": "pipeline smoke only; excluded from pilot and formal evidence",
        "requested_timesteps": 256,
        "actual_model_timesteps": int(model.num_timesteps),
        "evaluation_rows": len(evaluation_rows),
        "monitor": str(monitor),
        "evaluation_step_log": str(step_logs[0]),
        "model": str(models[0]),
    }
    (output / "smoke_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
