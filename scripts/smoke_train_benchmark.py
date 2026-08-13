"""Short Ant-v5/PPO smoke benchmark.

This script is for feasibility checks only. Its output must not be interpreted
as a formal research result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap import make_proxygap_ant_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ctrl_cost_weight", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/logs/smoke_train_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_proxygap_ant_env(
        ctrl_cost_weight=args.ctrl_cost_weight,
        condition_id="smoke_reference",
        seed=args.seed,
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=1,
        learning_rate=3e-4,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.perf_counter() - start
    steps_per_sec = model.num_timesteps / elapsed
    env.close()

    row = {
        "timesteps": int(model.num_timesteps),
        "elapsed_sec": round(elapsed, 3),
        "steps_per_sec": round(steps_per_sec, 2),
        "ctrl_cost_weight": args.ctrl_cost_weight,
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
