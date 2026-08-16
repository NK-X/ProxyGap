"""Run the frozen reward-by-guardrail development matrix."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import hashlib
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


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "hybrid_guardrail_development_v1_20260816.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def request_windows_awake() -> None:
    if sys.platform != "win32":
        return
    continuous = 0x80000000
    system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        continuous | system_required
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        continuous,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_authorised_development_matrix":
        raise ValueError("The development matrix is not authorised and frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("Formal launch must remain prohibited")
    if set(config["training_seeds"]) & set(config["reserved_formal_training_seeds"]):
        raise ValueError("Development and reserved formal seeds overlap")
    if not config["observation"][
        "augment_previous_applied_action_for_all_conditions"
    ]:
        raise ValueError("All conditions must share the augmented observation")
    condition_ids = [condition["condition_id"] for condition in config["conditions"]]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Condition IDs must be unique")
    limits = {condition["action_slew_l2_limit"] for condition in config["conditions"]}
    expected_limits = set(config.get("expected_action_slew_limits", [None, 1.4]))
    if limits != expected_limits:
        raise ValueError(
            f"Observed action-slew limits {limits} do not match frozen "
            f"limits {expected_limits}"
        )
    if config["evaluation_seeds"] != list(
        range(
            int(config["evaluation_seed_base"]),
            int(config["evaluation_seed_base"])
            + int(config["eval_episodes_per_checkpoint"]),
        )
    ):
        raise ValueError("Evaluation seed list does not match the paired sequence")


def condition_complete(root: Path, *, checkpoints: int, episodes: int) -> bool:
    models = list((root / "models").rglob("checkpoint_*.zip"))
    rows = read_rows(root / "logs" / "evaluation_metrics.csv")
    return len(models) == checkpoints and len(rows) == checkpoints * episodes


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    condition = task["condition"]
    ppo = config["ppo"]
    root = Path(task["condition_root"])
    runtime_rows, eval_rows = train_condition(
        output_root=root,
        condition_id=str(condition["condition_id"]),
        ctrl_cost_weight=float(config["ctrl_cost_weight"]),
        total_timesteps=int(task["timesteps"]),
        checkpoint_timesteps=task["checkpoints"],
        seed=int(task["training_seed"]),
        evaluation_seed_base=int(config["evaluation_seed_base"]),
        eval_episodes=int(task["eval_episodes"]),
        eval_max_episode_steps=int(config["eval_max_episode_steps"]),
        orientation_shaping_weight=float(condition["orientation_shaping_weight"]),
        orientation_shaping_scale=float(condition["orientation_shaping_scale"]),
        orientation_shaping_function=str(condition["orientation_shaping_function"]),
        lateral_drift_shaping_weight=float(
            condition.get("lateral_drift_shaping_weight", 0.0)
        ),
        lateral_drift_shaping_scale=float(
            condition.get("lateral_drift_shaping_scale", 1.0)
        ),
        lateral_shaping_signal=str(
            condition.get("lateral_shaping_signal", "offset_tanh")
        ),
        lateral_velocity_target=float(
            condition.get("lateral_velocity_target", 0.0)
        ),
        common_rescore_ctrl_cost_weight=float(
            config["reward"]["common_rescore_ctrl_cost_weight"]
        ),
        effort_distance_min=float(config["metrics"]["effort_distance_min"]),
        action_saturation_threshold=float(
            config["metrics"]["action_saturation_threshold"]
        ),
        augment_previous_applied_action=True,
        action_slew_l2_limit=condition["action_slew_l2_limit"],
        record_evaluation_steps=bool(task["record_evaluation_steps"]),
        ppo_n_steps=int(ppo["n_steps"]),
        ppo_batch_size=int(ppo["batch_size"]),
        ppo_n_epochs=int(ppo["n_epochs"]),
        ppo_learning_rate=float(ppo["learning_rate"]),
        ppo_gamma=float(ppo["gamma"]),
        ppo_gae_lambda=float(ppo["gae_lambda"]),
        ppo_clip_range=float(ppo["clip_range"]),
        ppo_ent_coef=float(ppo["ent_coef"]),
        ppo_vf_coef=float(ppo["vf_coef"]),
        ppo_max_grad_norm=float(ppo["max_grad_norm"]),
        ppo_normalize_advantage=bool(ppo["normalize_advantage"]),
        ppo_policy=str(ppo["policy"]),
        ppo_policy_kwargs=dict(ppo["policy_kwargs"]),
        ppo_device=str(config["device"]),
        ppo_torch_num_threads=int(ppo["torch_num_threads"]),
    )
    for row in runtime_rows:
        row["reward_id"] = condition["reward_id"]
        row["constraint_id"] = condition["constraint_id"]
    for row in eval_rows:
        row["reward_id"] = condition["reward_id"]
        row["constraint_id"] = condition["constraint_id"]
    write_standard_outputs(
        root,
        runtime_rows=runtime_rows,
        eval_rows=eval_rows,
        summary_rows=summarise_evaluation(eval_rows),
    )
    return {
        "condition_id": condition["condition_id"],
        "training_seed": task["training_seed"],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(eval_rows),
    }


def collect_completed(
    run_root: Path,
    config: dict[str, Any],
    *,
    checkpoints: int,
    episodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for seed in config["training_seeds"]:
        for condition in config["conditions"]:
            root = run_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            if condition_complete(root, checkpoints=checkpoints, episodes=episodes):
                runtime_rows.extend(read_rows(root / "logs" / "training_runtime.csv"))
                eval_rows.extend(read_rows(root / "logs" / "evaluation_metrics.csv"))
    return runtime_rows, eval_rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    request_windows_awake()

    smoke = bool(args.smoke)
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    if smoke:
        run_root = (
            PROJECT_ROOT
            / "artifacts"
            / "smoke"
            / f"{config['config_id']}__smoke"
        )
        conditions = [config["conditions"][0], config["conditions"][3]]
        seeds = [int(config["training_seeds"][0])]
        timesteps = 4096
        checkpoint_timesteps = [4096]
        eval_episodes = 2
        record_evaluation_steps = True
        max_workers = 2
    else:
        conditions = config["conditions"]
        seeds = config["training_seeds"]
        timesteps = int(config["timesteps_per_condition"])
        checkpoint_timesteps = [int(value) for value in config["checkpoint_timesteps"]]
        eval_episodes = int(config["eval_episodes_per_checkpoint"])
        record_evaluation_steps = bool(
            config["record_evaluation_steps_during_training"]
        )
        max_workers = int(config["execution"]["max_workers"])

    run_root.mkdir(parents=True, exist_ok=True)
    frozen_config_path = run_root / "frozen_run_config.json"
    if frozen_config_path.exists():
        if frozen_config_path.read_bytes() != config_path.read_bytes():
            raise RuntimeError("Frozen run config differs from the source config")
    else:
        frozen_config_path.write_bytes(config_path.read_bytes())

    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        for condition in conditions:
            root = run_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            if condition_complete(
                root,
                checkpoints=len(checkpoint_timesteps),
                episodes=eval_episodes,
            ):
                print(
                    f"Skipping complete {condition['condition_id']}, seed={seed}",
                    flush=True,
                )
                continue
            if root.exists() and any(root.rglob("*")):
                raise RuntimeError(f"Incomplete task requires manual audit: {root}")
            tasks.append(
                {
                    "condition": condition,
                    "training_seed": int(seed),
                    "condition_root": str(root),
                    "config": config,
                    "timesteps": timesteps,
                    "checkpoints": checkpoint_timesteps,
                    "eval_episodes": eval_episodes,
                    "record_evaluation_steps": record_evaluation_steps,
                }
            )

    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    execution_record = {
        "status": "started",
        "smoke": smoke,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "max_workers": max_workers,
        "task_count_at_start": len(tasks),
        "submitted_task_order": [
            {
                "condition_id": task["condition"]["condition_id"],
                "training_seed": task["training_seed"],
            }
            for task in tasks
        ],
        "source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in [
                PROJECT_ROOT / "src" / "proxygap" / "ant_wrapper.py",
                PROJECT_ROOT / "src" / "proxygap" / "metrics.py",
                PROJECT_ROOT / "src" / "proxygap" / "experiment.py",
                Path(__file__).resolve(),
            ]
        },
    }
    (run_root / "parallel_execution.json").write_text(
        json.dumps(execution_record, indent=2) + "\n",
        encoding="utf-8",
    )

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                message = (
                    f"{task['condition']['condition_id']}, "
                    f"seed={task['training_seed']}: {error!r}"
                )
                failures.append(message)
                print(f"FAILED {message}", flush=True)

    runtime_rows, eval_rows = collect_completed(
        run_root,
        {**config, "conditions": conditions, "training_seeds": seeds},
        checkpoints=len(checkpoint_timesteps),
        episodes=eval_episodes,
    )
    write_standard_outputs(
        run_root,
        runtime_rows=runtime_rows,
        eval_rows=eval_rows,
        summary_rows=summarise_evaluation(eval_rows),
    )
    expected_policies = len(conditions) * len(seeds)
    completion = {
        "status": "complete" if not failures else "failed",
        "smoke": smoke,
        "completed_policies": len(runtime_rows) // len(checkpoint_timesteps),
        "expected_policies": expected_policies,
        "evaluation_rows": len(eval_rows),
        "failures": failures,
    }
    (run_root / "parallel_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures or completion["completed_policies"] != expected_policies:
        raise RuntimeError(f"Development run incomplete: {completion}")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
