"""Run the frozen three-seed by three-weight orientation development pilot."""

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
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
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


def weight_label(weight: float) -> str:
    return str(weight).replace(".", "p")


def condition_id(weight: float) -> str:
    return f"orientation_cosine_lambda_{weight_label(weight)}"


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
        ctrl_cost_weight=float(config["ctrl_cost_weight"]),
        total_timesteps=int(config["timesteps_per_condition"]),
        checkpoint_timesteps=config["checkpoint_timesteps"],
        seed=int(task["training_seed"]),
        evaluation_seed_base=int(config["evaluation_seed_base"]),
        eval_episodes=int(config["eval_episodes_per_checkpoint"]),
        eval_max_episode_steps=int(config["eval_max_episode_steps"]),
        orientation_shaping_weight=float(task["orientation_weight"]),
        orientation_shaping_scale=float(config["orientation_shaping"]["scale"]),
        orientation_shaping_function=str(
            config["orientation_shaping"]["function"]
        ),
        common_rescore_ctrl_cost_weight=float(
            config["reward"]["common_rescore_ctrl_cost_weight"]
        ),
        effort_distance_min=float(config["metrics"]["effort_distance_min"]),
        action_saturation_threshold=float(
            config["metrics"]["action_saturation_threshold"]
        ),
        record_evaluation_steps=bool(
            config["record_evaluation_steps_during_training"]
        ),
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
        "condition_id": task["condition_id"],
        "training_seed": task["training_seed"],
        "orientation_weight": task["orientation_weight"],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(eval_rows),
    }


def collect_completed(
    run_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    runtime_rows: list[dict] = []
    eval_rows: list[dict] = []
    weights = config["orientation_shaping"]["candidate_weights"]
    for seed in config["training_seeds"]:
        for weight_value in weights:
            weight = float(weight_value)
            root = run_root / "runs" / f"seed_{seed}" / condition_id(weight)
            if condition_complete(root, config):
                runtime_rows.extend(read_rows(root / "logs" / "training_runtime.csv"))
                eval_rows.extend(read_rows(root / "logs" / "evaluation_metrics.csv"))
    return runtime_rows, eval_rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    request_windows_awake()
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    calibration_path = (
        run_root / "offline_calibration" / "calibration_adjudication.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise RuntimeError("Frozen offline calibration has not passed")
    run_root.mkdir(parents=True, exist_ok=True)

    frozen_config_path = run_root / "frozen_run_config.json"
    if frozen_config_path.exists():
        if frozen_config_path.read_bytes() != config_path.read_bytes():
            raise RuntimeError("Frozen run config differs from the source config")
    else:
        frozen_config_path.write_bytes(config_path.read_bytes())

    tasks: list[dict[str, Any]] = []
    for seed in config["training_seeds"]:
        for weight_value in config["orientation_shaping"]["candidate_weights"]:
            weight = float(weight_value)
            cid = condition_id(weight)
            root = run_root / "runs" / f"seed_{seed}" / cid
            if condition_complete(root, config):
                print(f"Skipping complete {cid}, seed={seed}", flush=True)
                continue
            if root.exists() and any(root.rglob("*")):
                raise RuntimeError(f"Incomplete task requires manual audit: {root}")
            tasks.append(
                {
                    "condition_id": cid,
                    "orientation_weight": weight,
                    "training_seed": int(seed),
                    "condition_root": str(root),
                    "config": config,
                }
            )

    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    execution_record = {
        "status": "started",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "max_workers": int(config["execution"]["max_workers"]),
        "torch_threads_per_worker": int(config["ppo"]["torch_num_threads"]),
        "task_count_at_start": len(tasks),
        "submitted_task_order": [
            {
                "condition_id": task["condition_id"],
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
    max_workers = int(config["execution"]["max_workers"])
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Completed {result}", flush=True)
            except Exception as error:
                message = (
                    f"{task['condition_id']}, seed={task['training_seed']}: "
                    f"{error!r}"
                )
                failures.append(message)
                print(f"FAILED {message}", flush=True)
            runtime_rows, eval_rows = collect_completed(run_root, config)
            write_standard_outputs(
                run_root,
                runtime_rows=runtime_rows,
                eval_rows=eval_rows,
                summary_rows=summarise_evaluation(eval_rows),
            )

    runtime_rows, eval_rows = collect_completed(run_root, config)
    checkpoint_count = len(config["checkpoint_timesteps"])
    expected_policies = (
        len(config["training_seeds"])
        * len(config["orientation_shaping"]["candidate_weights"])
    )
    completion = {
        "status": "complete" if not failures else "failed",
        "completed_policies": len(runtime_rows) // checkpoint_count,
        "expected_policies": expected_policies,
        "evaluation_rows": len(eval_rows),
        "failures": failures,
    }
    (run_root / "parallel_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures or completion["completed_policies"] != expected_policies:
        raise RuntimeError(f"Pilot run incomplete: {completion}")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
