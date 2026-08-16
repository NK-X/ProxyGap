"""Run a small coefficient pilot for parameter locking evidence.

This is not formal training. The default coefficient list is a halving sequence
around the Ant-v5 reference control-cost weight, used only to estimate runtime
and identify whether a plausible divergence appears.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import ctypes
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialise_only", action="store_true")
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--eval_max_episode_steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41001, 41002])
    parser.add_argument("--evaluation_seed_base", type=int, default=51001)
    parser.add_argument(
        "--ctrl_cost_weights",
        nargs="+",
        type=float,
        default=[0.5, 0.375, 0.25, 0.125, 0.0625],
    )
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--checkpoint_timesteps", nargs="+", type=int, default=None)
    return parser.parse_args()


def condition_id(weight: float) -> str:
    token = str(weight).replace(".", "p")
    return f"ctrl_{token}"


def request_windows_awake() -> None:
    """Keep Windows awake while this process owns a development run."""
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        es_continuous | es_system_required
    )
    if result == 0:
        raise OSError("Windows rejected the development-run sleep prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        es_continuous,
    )


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_is_complete(root: Path, *, checkpoints: int, eval_rows: int) -> bool:
    models = list((root / "models").rglob("checkpoint_*.zip"))
    rows = read_csv_rows(root / "logs" / "evaluation_metrics.csv")
    return len(models) == checkpoints and len(rows) == eval_rows


def main() -> None:
    args = parse_args()
    request_windows_awake()
    run_id = args.run_id or make_run_id("coefficient_pilot")
    output_root = Path(args.output_root) / run_id

    checkpoint_values = args.checkpoint_timesteps or [
        int(round(args.timesteps * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)
    ]

    config = {
        "status": "pilot_not_formal_experiment",
        "run_id": run_id,
        "timesteps_per_condition": args.timesteps,
        "eval_episodes_per_checkpoint": args.eval_episodes,
        "eval_max_episode_steps": args.eval_max_episode_steps,
        "training_seeds": args.seeds,
        "evaluation_seed_base": args.evaluation_seed_base,
        "ctrl_cost_weights": args.ctrl_cost_weights,
        "checkpoint_timesteps": checkpoint_values,
        "ppo": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "torch_num_threads": args.torch_num_threads,
            "device": "cpu"
        },
        "interpretation_warning": "Pilot outputs support parameter locking and runtime estimates only."
    }
    saved_config = output_root / "run_config.json"
    if saved_config.exists() and not args.resume:
        raise FileExistsError(f"Development run exists; use --resume: {output_root}")
    if not saved_config.exists():
        save_run_config(output_root, config)
    if args.initialise_only:
        print(f"Initialised development run: {output_root}", flush=True)
        return

    all_runtime_rows = []
    all_eval_rows = []
    for training_seed in args.seeds:
        for weight in args.ctrl_cost_weights:
            cid = "reference" if weight == 0.5 else condition_id(weight)
            condition_root = output_root / "runs" / f"seed_{training_seed}" / cid
            expected_eval_rows = len(checkpoint_values) * args.eval_episodes
            if condition_is_complete(
                condition_root,
                checkpoints=len(checkpoint_values),
                eval_rows=expected_eval_rows,
            ):
                print(f"Skipping completed {cid}, seed={training_seed}", flush=True)
                all_runtime_rows.extend(
                    read_csv_rows(condition_root / "logs" / "training_runtime.csv")
                )
                all_eval_rows.extend(
                    read_csv_rows(condition_root / "logs" / "evaluation_metrics.csv")
                )
                continue
            if condition_root.exists() and any(condition_root.rglob("*")):
                raise RuntimeError(
                    f"Incomplete condition requires manual audit before resume: {condition_root}"
                )
            print(f"Starting {cid}, seed={training_seed}", flush=True)
            runtime_rows, eval_rows = train_condition(
                output_root=condition_root,
                condition_id=cid,
                ctrl_cost_weight=weight,
                total_timesteps=args.timesteps,
                seed=training_seed,
                evaluation_seed_base=args.evaluation_seed_base,
                eval_episodes=args.eval_episodes,
                eval_max_episode_steps=args.eval_max_episode_steps,
                ppo_n_steps=args.n_steps,
                ppo_batch_size=args.batch_size,
                ppo_n_epochs=args.n_epochs,
                ppo_torch_num_threads=args.torch_num_threads,
                checkpoint_timesteps=checkpoint_values,
            )
            write_standard_outputs(
                condition_root,
                runtime_rows=runtime_rows,
                eval_rows=eval_rows,
                summary_rows=summarise_evaluation(eval_rows),
            )
            all_runtime_rows.extend(runtime_rows)
            all_eval_rows.extend(eval_rows)
            write_standard_outputs(
                output_root,
                runtime_rows=all_runtime_rows,
                eval_rows=all_eval_rows,
                summary_rows=summarise_evaluation(all_eval_rows),
            )
            print(f"Completed {cid}, seed={training_seed}", flush=True)

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
            conditions = len(config["ctrl_cost_weights"])
            training_seeds = len(config["training_seeds"])
            policies = conditions * training_seeds
            checkpoints = len(config["checkpoint_timesteps"])
            train_sec = policies * formal_steps / train_sps
            eval_sec = policies * checkpoints * eval_episodes * eval_sec_per_episode
            rows.append(
                {
                    "scenario": f"{formal_steps}_steps_{eval_episodes}_eval_eps",
                    "conditions": conditions,
                    "training_seeds": training_seeds,
                    "policies": policies,
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
