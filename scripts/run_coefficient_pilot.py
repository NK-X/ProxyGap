"""Run a small coefficient pilot for parameter locking evidence.

This is not formal training. The default coefficient list is a halving sequence
around the Ant-v5 reference control-cost weight, used only to estimate runtime
and identify whether a plausible divergence appears.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    make_run_id,
    save_run_config,
    summarise_evaluation,
    train_condition,
    write_rows,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--output_root", default="artifacts/pilot")
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--eval_max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--ctrl_cost_weights", nargs="+", type=float, default=[0.5, 0.25, 0.125, 0.0625])
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=1)
    return parser.parse_args()


def condition_id(weight: float) -> str:
    token = str(weight).replace(".", "p")
    return f"ctrl_{token}"


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id("coefficient_pilot")
    output_root = Path(args.output_root) / run_id

    config = {
        "status": "pilot_not_formal_experiment",
        "run_id": run_id,
        "timesteps_per_condition": args.timesteps,
        "eval_episodes_per_checkpoint": args.eval_episodes,
        "eval_max_episode_steps": args.eval_max_episode_steps,
        "seed": args.seed,
        "ctrl_cost_weights": args.ctrl_cost_weights,
        "checkpoints": [0.25, 0.5, 0.75, 1.0],
        "ppo": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "device": "cpu"
        },
        "interpretation_warning": "Pilot outputs support parameter locking and runtime estimates only."
    }
    save_run_config(output_root, config)

    all_runtime_rows = []
    all_eval_rows = []
    for index, weight in enumerate(args.ctrl_cost_weights):
        cid = "reference" if weight == 0.5 else condition_id(weight)
        runtime_rows, eval_rows = train_condition(
            output_root=output_root,
            condition_id=cid,
            ctrl_cost_weight=weight,
            total_timesteps=args.timesteps,
            seed=args.seed + index * 100,
            eval_episodes=args.eval_episodes,
            eval_max_episode_steps=args.eval_max_episode_steps,
            ppo_n_steps=args.n_steps,
            ppo_batch_size=args.batch_size,
            ppo_n_epochs=args.n_epochs,
        )
        all_runtime_rows.extend(runtime_rows)
        all_eval_rows.extend(eval_rows)

    summary_rows = summarise_evaluation(all_eval_rows)
    write_standard_outputs(
        output_root,
        runtime_rows=all_runtime_rows,
        eval_rows=all_eval_rows,
        summary_rows=summary_rows,
    )
    write_runtime_estimates(output_root, all_runtime_rows, config)
    print(f"Saved pilot run: {output_root}")


def write_runtime_estimates(output_root: Path, runtime_rows: list[dict], config: dict) -> None:
    step_column = (
        "chunk_timesteps_requested"
        if runtime_rows and "chunk_timesteps_requested" in runtime_rows[0]
        else "chunk_timesteps"
    )
    total_train_steps = sum(int(row[step_column]) for row in runtime_rows)
    total_train_sec = sum(float(row["train_elapsed_sec"]) for row in runtime_rows)
    total_eval_episodes = sum(int(row["eval_episodes"]) for row in runtime_rows)
    total_eval_sec = sum(float(row["eval_elapsed_sec"]) for row in runtime_rows)
    train_sps = total_train_steps / max(total_train_sec, 1e-8)
    eval_sec_per_episode = total_eval_sec / max(total_eval_episodes, 1)

    rows = []
    for formal_steps in [50_000, 100_000, 200_000]:
        for eval_episodes in [5, 10]:
            conditions = 5
            checkpoints = 4
            train_sec = conditions * formal_steps / train_sps
            eval_sec = conditions * checkpoints * eval_episodes * eval_sec_per_episode
            rows.append(
                {
                    "scenario": f"{formal_steps}_steps_{eval_episodes}_eval_eps",
                    "conditions": conditions,
                    "timesteps_per_condition": formal_steps,
                    "checkpoints": checkpoints,
                    "eval_episodes_per_checkpoint": eval_episodes,
                    "estimated_train_sec": round(train_sec, 1),
                    "estimated_eval_sec": round(eval_sec, 1),
                    "estimated_total_sec": round(train_sec + eval_sec, 1),
                    "pilot_train_steps_per_sec": round(train_sps, 2),
                    "pilot_eval_sec_per_episode": round(eval_sec_per_episode, 3),
                    "warning": "Scenario estimate only; not a locked formal budget."
                }
            )
    write_rows(output_root / "logs" / "runtime_estimate.csv", rows)


if __name__ == "__main__":
    main()
