"""Fill missing whole policies in a development matrix.

Despite the historical filename, this script does not continue an individual
saved checkpoint. Use ``run_stage1_budget_extension.py`` for audited model
continuation.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    return parser.parse_args()


def request_windows_awake() -> None:
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        es_continuous | es_system_required
    )
    if result == 0:
        raise OSError("Windows rejected the parallel-run sleep prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        es_continuous,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_id(weight: float) -> str:
    if weight == 0.5:
        return "reference"
    return f"ctrl_{str(weight).replace('.', 'p')}"


def condition_complete(root: Path, config: dict[str, Any]) -> bool:
    models = list((root / "models").rglob("checkpoint_*.zip"))
    rows = read_rows(root / "logs" / "evaluation_metrics.csv")
    expected_models = len(config["checkpoint_timesteps"])
    expected_rows = expected_models * int(config["eval_episodes_per_checkpoint"])
    return len(models) == expected_models and len(rows) == expected_rows


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    ppo = config["ppo"]
    root = Path(task["condition_root"])
    runtime_rows, eval_rows = train_condition(
        output_root=root,
        condition_id=task["condition_id"],
        ctrl_cost_weight=float(task["ctrl_cost_weight"]),
        total_timesteps=int(config["timesteps_per_condition"]),
        checkpoint_timesteps=config["checkpoint_timesteps"],
        seed=int(task["training_seed"]),
        evaluation_seed_base=int(config["evaluation_seed_base"]),
        eval_episodes=int(config["eval_episodes_per_checkpoint"]),
        eval_max_episode_steps=int(config["eval_max_episode_steps"]),
        ppo_n_steps=int(ppo["n_steps"]),
        ppo_batch_size=int(ppo["batch_size"]),
        ppo_n_epochs=int(ppo["n_epochs"]),
        ppo_learning_rate=float(ppo.get("learning_rate", 3e-4)),
        ppo_gamma=float(ppo.get("gamma", 0.99)),
        ppo_gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        ppo_clip_range=float(ppo.get("clip_range", 0.2)),
        ppo_ent_coef=float(ppo.get("ent_coef", 0.0)),
        ppo_vf_coef=float(ppo.get("vf_coef", 0.5)),
        ppo_max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        ppo_normalize_advantage=bool(ppo.get("normalize_advantage", True)),
        ppo_policy=str(ppo.get("policy", "MlpPolicy")),
        ppo_policy_kwargs=dict(ppo.get("policy_kwargs", {})),
        ppo_device=str(config.get("device", "cpu")),
        ppo_torch_num_threads=int(ppo.get("torch_num_threads", 1)),
    )
    write_standard_outputs(
        root,
        runtime_rows=runtime_rows,
        eval_rows=eval_rows,
        summary_rows=summarise_evaluation(eval_rows),
    )
    return {
        "condition_id": task["condition_id"],
        "training_seed": task["training_seed"],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(eval_rows),
    }


def collect_completed(run_root: Path, config: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    runtime_rows: list[dict] = []
    eval_rows: list[dict] = []
    for seed in config["training_seeds"]:
        for weight in config["ctrl_cost_weights"]:
            root = run_root / "runs" / f"seed_{seed}" / condition_id(float(weight))
            if condition_complete(root, config):
                runtime_rows.extend(read_rows(root / "logs" / "training_runtime.csv"))
                eval_rows.extend(read_rows(root / "logs" / "evaluation_metrics.csv"))
    return runtime_rows, eval_rows


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    request_windows_awake()
    run_root = Path(args.run_root).resolve()
    config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for seed in config["training_seeds"]:
        for weight_value in config["ctrl_cost_weights"]:
            weight = float(weight_value)
            cid = condition_id(weight)
            root = run_root / "runs" / f"seed_{seed}" / cid
            if condition_complete(root, config):
                print(f"Skipping completed {cid}, seed={seed}", flush=True)
                continue
            if root.exists() and any(root.rglob("*")):
                raise RuntimeError(f"Incomplete condition requires manual audit: {root}")
            tasks.append(
                {
                    "condition_id": cid,
                    "ctrl_cost_weight": weight,
                    "training_seed": int(seed),
                    "condition_root": str(root),
                    "config": config,
                }
            )

    task_order_seed = int(config.get("task_order_seed", 0))
    random.Random(task_order_seed).shuffle(tasks)

    execution_record = {
        "max_workers": args.max_workers,
        "torch_threads_per_worker": config["ppo"].get("torch_num_threads", 1),
        "parallel_unit": "independent trained policy",
        "scientific_effect": "execution scheduling only; policy budgets and seeds unchanged",
        "task_count_at_start": len(tasks),
        "task_order_seed": task_order_seed,
        "submitted_task_order": [
            {
                "condition_id": task["condition_id"],
                "training_seed": task["training_seed"],
            }
            for task in tasks
        ],
    }
    (run_root / "parallel_execution.json").write_text(
        json.dumps(execution_record, indent=2) + "\n", encoding="utf-8"
    )

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Completed {result}", flush=True)
            except Exception as error:  # Preserve every other completed task.
                failures.append(
                    f"{task['condition_id']}, seed={task['training_seed']}: {error!r}"
                )
                print(f"FAILED {failures[-1]}", flush=True)
            runtime_rows, eval_rows = collect_completed(run_root, config)
            write_standard_outputs(
                run_root,
                runtime_rows=runtime_rows,
                eval_rows=eval_rows,
                summary_rows=summarise_evaluation(eval_rows),
            )

    runtime_rows, eval_rows = collect_completed(run_root, config)
    completion = {
        "completed_policies": len(runtime_rows) // len(config["checkpoint_timesteps"]),
        "expected_policies": len(config["training_seeds"]) * len(config["ctrl_cost_weights"]),
        "evaluation_rows": len(eval_rows),
        "failures": failures,
    }
    (run_root / "parallel_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    if failures or completion["completed_policies"] != completion["expected_policies"]:
        raise RuntimeError(f"Parallel development run incomplete: {completion}")
    print(f"Parallel development run complete: {run_root}", flush=True)


if __name__ == "__main__":
    main()
