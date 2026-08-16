from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import hashlib
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


DEFAULT_CONFIG = ROOT / "configs" / "smoothness_mechanism_pilot_v1_20260816.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep_windows_awake() -> None:
    continuous = 0x80000000
    system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        continuous | system_required
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")


def validate_config(config: dict) -> None:
    if config.get("status") != "frozen_authorised_development_matrix":
        raise ValueError("Development configuration is not frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("Formal launch must remain prohibited")
    if len(config["conditions"]) != 4:
        raise ValueError("The mechanism screen requires exactly four conditions")
    factors = {
        (
            bool(condition["replace_forward_reward_with_tracking"]),
            float(condition["action_rate_shaping_weight"]),
        )
        for condition in config["conditions"]
    }
    if factors != {(False, 0.0), (True, 0.0), (False, 0.2), (True, 0.2)}:
        raise ValueError("The frozen two-by-two factor matrix has changed")
    if set(config["training_seeds"]) & set(config["reserved_formal_training_seeds"]):
        raise ValueError("Development and formal training seeds overlap")
    expected_eval = list(
        range(
            int(config["evaluation_seed_base"]),
            int(config["evaluation_seed_base"]) + int(config["eval_episodes_per_checkpoint"]),
        )
    )
    if config["evaluation_seeds"] != expected_eval:
        raise ValueError("Evaluation seeds do not match the frozen paired sequence")


def run_task(task: dict) -> dict:
    config = task["config"]
    condition = task["condition"]
    shared = config["shared_reward"]
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
        lateral_drift_shaping_weight=float(shared["lateral_drift_shaping_weight"]),
        lateral_drift_shaping_scale=float(shared["lateral_drift_shaping_scale"]),
        lateral_shaping_signal=str(shared["lateral_shaping_signal"]),
        lateral_velocity_target=float(shared["lateral_velocity_target"]),
        orientation_shaping_weight=float(shared["orientation_shaping_weight"]),
        orientation_shaping_scale=float(shared["orientation_shaping_scale"]),
        orientation_shaping_function=str(shared["orientation_shaping_function"]),
        replace_forward_reward_with_tracking=bool(
            condition["replace_forward_reward_with_tracking"]
        ),
        forward_velocity_target=float(shared["forward_velocity_target"]),
        forward_velocity_tracking_scale=float(shared["forward_velocity_tracking_scale"]),
        action_rate_shaping_weight=float(condition["action_rate_shaping_weight"]),
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    keep_windows_awake()

    smoke = bool(args.smoke)
    output_root = ROOT / config["execution"]["output_root"]
    if smoke:
        output_root = ROOT / "artifacts" / "smoke" / "smoothness_mechanism_v1"
        conditions = config["conditions"]
        seeds = [int(config["training_seeds"][0])]
        timesteps = 4096
        checkpoints = [4096]
        eval_episodes = 2
        max_workers = 2
    else:
        conditions = config["conditions"]
        seeds = [int(seed) for seed in config["training_seeds"]]
        timesteps = int(config["timesteps_per_condition"])
        checkpoints = [int(value) for value in config["checkpoint_timesteps"]]
        eval_episodes = int(config["eval_episodes_per_checkpoint"])
        max_workers = int(config["execution"]["max_workers"])

    output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = output_root / "frozen_run_config.json"
    if frozen_config.exists() and frozen_config.read_bytes() != config_path.read_bytes():
        raise RuntimeError("Output root contains a different frozen configuration")
    frozen_config.write_bytes(config_path.read_bytes())

    tasks = []
    for seed in seeds:
        for condition in conditions:
            condition_root = output_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            if condition_root.exists() and any(condition_root.rglob("*")):
                raise RuntimeError(f"Refusing to overwrite an existing task: {condition_root}")
            tasks.append(
                {
                    "config": config,
                    "condition": condition,
                    "training_seed": seed,
                    "condition_root": str(condition_root),
                    "timesteps": timesteps,
                    "checkpoints": checkpoints,
                    "eval_episodes": eval_episodes,
                    "record_evaluation_steps": True,
                }
            )
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    execution = {
        "status": "started",
        "smoke": smoke,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "task_count": len(tasks),
        "max_workers": max_workers,
        "task_order": [
            {"condition_id": task["condition"]["condition_id"], "seed": task["training_seed"]}
            for task in tasks
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                ROOT / "src" / "proxygap" / "ant_wrapper.py",
                ROOT / "src" / "proxygap" / "experiment.py",
                ROOT / "src" / "proxygap" / "metrics.py",
                Path(__file__).resolve(),
            ]
        },
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")

    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                failure = {
                    "condition_id": task["condition"]["condition_id"],
                    "seed": task["training_seed"],
                    "error": repr(error),
                }
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)

    runtime_rows = []
    evaluation_rows = []
    for seed in seeds:
        for condition in conditions:
            condition_root = output_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            runtime_rows.extend(read_csv_rows(condition_root / "logs" / "training_runtime.csv"))
            evaluation_rows.extend(read_csv_rows(condition_root / "logs" / "evaluation_metrics.csv"))
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    execution.update(
        {
            "status": "failed" if failures else "complete",
            "failures": failures,
            "runtime_rows": len(runtime_rows),
            "evaluation_rows": len(evaluation_rows),
        }
    )
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} pilot tasks failed")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
