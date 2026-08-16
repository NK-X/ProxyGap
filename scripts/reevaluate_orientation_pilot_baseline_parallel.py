"""Parallel metric-schema replay of the frozen baseline model checkpoints."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from stable_baselines3 import PPO
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    evaluate_model,
    summarise_evaluation,
    write_standard_outputs,
)


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    model_path = Path(task["model_path"])
    model = PPO.load(model_path, device="cpu")
    start = time.perf_counter()
    rows, elapsed = evaluate_model(
        model,
        condition_id="reference",
        ctrl_cost_weight=float(config["ctrl_cost_weight"]),
        checkpoint_fraction=int(task["target_timesteps"])
        / int(config["timesteps_per_condition"]),
        target_timesteps=int(task["target_timesteps"]),
        actual_model_timesteps=int(model.num_timesteps),
        training_seed=int(task["training_seed"]),
        seed=int(config["evaluation_seed_base"]),
        episodes=int(config["eval_episodes_per_checkpoint"]),
        max_episode_steps=int(config["eval_max_episode_steps"]),
        orientation_shaping_weight=0.0,
        orientation_shaping_scale=1.0,
        orientation_shaping_function="cosine",
        common_rescore_ctrl_cost_weight=float(
            config["reward"]["common_rescore_ctrl_cost_weight"]
        ),
        effort_distance_min=float(config["metrics"]["effort_distance_min"]),
        action_saturation_threshold=float(
            config["metrics"]["action_saturation_threshold"]
        ),
    )
    return {
        "rows": rows,
        "runtime": {
            "condition_id": "reference",
            "training_seed": int(task["training_seed"]),
            "target_timesteps": int(task["target_timesteps"]),
            "evaluation_rows": len(rows),
            "elapsed_sec": elapsed,
            "wall_elapsed_sec": time.perf_counter() - start,
            "model_path": str(model_path),
        },
        "model_record": {
            "training_seed": int(task["training_seed"]),
            "target_timesteps": int(task["target_timesteps"]),
            "path": str(model_path),
            "sha256": sha256(model_path),
        },
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["execution"]["output_root"] / "baseline_replay"
    if output_root.exists():
        raise FileExistsError(f"Baseline replay already exists: {output_root}")
    output_root.mkdir(parents=True)

    tasks: list[dict[str, Any]] = []
    pattern = config["baseline_evidence"]["model_pattern"]
    for training_seed in config["offline_calibration_training_seeds"]:
        for target_timesteps in config["checkpoint_timesteps"]:
            model_path = PROJECT_ROOT / pattern.format(
                training_seed=int(training_seed),
                target_timesteps=int(target_timesteps),
            )
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            tasks.append(
                {
                    "training_seed": int(training_seed),
                    "target_timesteps": int(target_timesteps),
                    "model_path": str(model_path),
                    "config": config,
                }
            )
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    execution = {
        "status": "started",
        "execution_only_change": "parallel checkpoint evaluation",
        "max_workers": int(config["execution"]["max_workers"]),
        "torch_threads_per_worker": int(config["ppo"]["torch_num_threads"]),
        "task_order": [
            {
                "training_seed": task["training_seed"],
                "target_timesteps": task["target_timesteps"],
            }
            for task in tasks
        ],
    }
    (output_root / "parallel_execution.json").write_text(
        json.dumps(execution, indent=2) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict] = []
    runtime_rows: list[dict] = []
    model_records: list[dict] = []
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=int(config["execution"]["max_workers"])
    ) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                all_rows.extend(result["rows"])
                runtime_rows.append(result["runtime"])
                model_records.append(result["model_record"])
                print(
                    "Re-evaluated baseline "
                    f"seed={task['training_seed']}, target={task['target_timesteps']}",
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    f"seed={task['training_seed']}, target={task['target_timesteps']}: "
                    f"{error!r}"
                )
            write_standard_outputs(
                output_root,
                runtime_rows=runtime_rows,
                eval_rows=all_rows,
                summary_rows=summarise_evaluation(all_rows),
            )

    expected_rows = len(tasks) * int(config["eval_episodes_per_checkpoint"])
    manifest = {
        "status": (
            "complete"
            if not failures and len(all_rows) == expected_rows
            else "incomplete"
        ),
        "role": "metric-schema replay only; no policy training",
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "evaluation_rows": len(all_rows),
        "expected_rows": expected_rows,
        "failures": failures,
        "model_records": sorted(
            model_records,
            key=lambda row: (row["training_seed"], row["target_timesteps"]),
        ),
    }
    (output_root / "baseline_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if manifest["status"] != "complete":
        raise RuntimeError(f"Baseline replay incomplete: {manifest}")
    print(json.dumps(manifest | {"model_records": "20 records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
