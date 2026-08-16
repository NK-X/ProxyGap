from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402
from proxygap.experiment import (  # noqa: E402
    evaluate_model,
    summarise_evaluation,
    write_standard_outputs,
)


DEFAULT_CONFIG = ROOT / "configs" / "smoothness_target_budget_extension_v1_20260816.json"


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
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        0x80000000 | 0x00000001
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")


def validate_config(config: dict) -> None:
    if config.get("status") != "frozen_authorised_development_extension":
        raise ValueError("Extension configuration is not frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("Formal launch must remain prohibited")
    if config["conditions"] != ["Ftrack__Ar0", "Ftrack__Ar0p2"]:
        raise ValueError("Only the two frozen target-tracking conditions may continue")
    if config["checkpoint_timesteps"] != [500000, 750000, 1000000]:
        raise ValueError("Continuation checkpoints changed")
    if config["evaluation_seeds"] != list(range(51401, 51411)):
        raise ValueError("Paired evaluation seeds changed")
    if config["observation"]["dimensions"] != 113:
        raise ValueError("Continuation observation dimension changed")


def source_path(config: dict, condition_id: str, seed: int, *, smoke: bool) -> Path:
    if smoke:
        return (
            ROOT
            / "artifacts"
            / "smoke"
            / "smoothness_mechanism_v1"
            / "runs"
            / f"seed_{seed}"
            / condition_id
            / "models"
            / condition_id
            / "checkpoint_004096.zip"
        )
    return (
        ROOT
        / config["source_run_root"]
        / "runs"
        / f"seed_{seed}"
        / condition_id
        / "models"
        / condition_id
        / "checkpoint_300000.zip"
    )


def make_env(config: dict, condition_id: str, seed: int):
    shared = config["shared_reward"]
    return make_proxygap_ant_env(
        ctrl_cost_weight=float(shared["ctrl_cost_weight"]),
        condition_id=condition_id,
        seed=seed,
        orientation_shaping_weight=float(shared["orientation_shaping_weight"]),
        orientation_shaping_function=str(shared["orientation_shaping_function"]),
        orientation_shaping_scale=float(shared["orientation_shaping_scale"]),
        lateral_drift_shaping_weight=float(shared["lateral_drift_shaping_weight"]),
        lateral_drift_shaping_scale=float(shared["lateral_drift_shaping_scale"]),
        lateral_shaping_signal=str(shared["lateral_shaping_signal"]),
        lateral_velocity_target=float(shared["lateral_velocity_target"]),
        replace_forward_reward_with_tracking=bool(
            shared["replace_forward_reward_with_tracking"]
        ),
        forward_velocity_target=float(shared["forward_velocity_target"]),
        forward_velocity_tracking_scale=float(shared["forward_velocity_tracking_scale"]),
        action_rate_shaping_weight=float(
            config["condition_parameters"][condition_id]["action_rate_shaping_weight"]
        ),
        augment_previous_applied_action=True,
        action_slew_l2_limit=None,
    )


def audit_loaded_model(model: PPO, config: dict, seed: int, expected_steps: int) -> None:
    ppo = config["ppo"]
    checks = {
        "num_timesteps": (int(model.num_timesteps), expected_steps),
        "seed": (int(model.seed), seed),
        "n_steps": (int(model.n_steps), int(ppo["n_steps"])),
        "batch_size": (int(model.batch_size), int(ppo["batch_size"])),
        "n_epochs": (int(model.n_epochs), int(ppo["n_epochs"])),
        "use_sde": (bool(model.use_sde), bool(ppo["use_sde"])),
    }
    failures = [f"{name}: {actual!r} != {expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
    if tuple(model.observation_space.shape or ()) != (113,):
        failures.append(f"observation shape {model.observation_space.shape} != (113,)")
    if tuple(model.action_space.shape or ()) != (8,):
        failures.append(f"action shape {model.action_space.shape} != (8,)")
    if type(model.policy.optimizer).__name__ != "Adam":
        failures.append("optimiser is not Adam")
    if failures:
        raise ValueError(f"Loaded source failed audit: {failures}")


def continue_task(task: dict) -> dict:
    config = task["config"]
    condition_id = str(task["condition_id"])
    seed = int(task["training_seed"])
    source = Path(task["source_path"])
    source_hash = sha256(source)
    output_root = Path(task["output_root"])
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    raw_env = make_env(config, condition_id, seed)
    monitor_path = output_root / "logs" / "training_extension.monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(raw_env, filename=str(monitor_path))
    model = PPO.load(source, env=env, device="cpu")
    audit_loaded_model(model, config, seed, int(task["expected_source_steps"]))
    model.set_random_seed(seed)
    runtime_rows = []
    evaluation_rows = []
    final_target = int(task["targets"][-1])
    try:
        for target in task["targets"]:
            start_steps = int(model.num_timesteps)
            started = time.perf_counter()
            model.learn(total_timesteps=int(target) - start_steps, reset_num_timesteps=False)
            train_elapsed = time.perf_counter() - started
            actual_steps = int(model.num_timesteps)
            model_path = output_root / "models" / condition_id / f"checkpoint_{int(target):07d}.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            rows, eval_elapsed = evaluate_model(
                model,
                condition_id=condition_id,
                ctrl_cost_weight=float(config["shared_reward"]["ctrl_cost_weight"]),
                checkpoint_fraction=int(target) / final_target,
                seed=int(config["evaluation_seed_base"]),
                episodes=int(task["eval_episodes"]),
                target_timesteps=int(target),
                actual_model_timesteps=actual_steps,
                training_seed=seed,
                max_episode_steps=int(config["evaluation_max_episode_steps"]),
                orientation_shaping_weight=float(config["shared_reward"]["orientation_shaping_weight"]),
                orientation_shaping_function=str(config["shared_reward"]["orientation_shaping_function"]),
                orientation_shaping_scale=float(config["shared_reward"]["orientation_shaping_scale"]),
                lateral_drift_shaping_weight=float(config["shared_reward"]["lateral_drift_shaping_weight"]),
                lateral_drift_shaping_scale=float(config["shared_reward"]["lateral_drift_shaping_scale"]),
                lateral_shaping_signal=str(config["shared_reward"]["lateral_shaping_signal"]),
                lateral_velocity_target=float(config["shared_reward"]["lateral_velocity_target"]),
                replace_forward_reward_with_tracking=True,
                forward_velocity_target=float(config["shared_reward"]["forward_velocity_target"]),
                forward_velocity_tracking_scale=float(config["shared_reward"]["forward_velocity_tracking_scale"]),
                action_rate_shaping_weight=float(config["condition_parameters"][condition_id]["action_rate_shaping_weight"]),
                augment_previous_applied_action=True,
                action_slew_l2_limit=None,
                step_log_dir=output_root / "logs" / "evaluation_steps",
            )
            evaluation_rows.extend(rows)
            runtime_rows.append(
                {
                    "condition_id": condition_id,
                    "training_seed": seed,
                    "source_model_path": str(source),
                    "source_model_sha256": source_hash,
                    "source_model_timesteps": start_steps,
                    "target_timesteps": int(target),
                    "actual_model_timesteps": actual_steps,
                    "train_elapsed_sec": round(train_elapsed, 3),
                    "train_steps_per_sec": round((int(target) - start_steps) / max(train_elapsed, 1e-9), 2),
                    "eval_episodes": int(task["eval_episodes"]),
                    "eval_elapsed_sec": round(eval_elapsed, 3),
                    "model_path": str(model_path),
                    "model_sha256": sha256(model_path),
                    "environment_state_restored": False,
                    "random_stream_restored_bitwise": False,
                }
            )
    finally:
        env.close()
    if sha256(source) != source_hash:
        raise RuntimeError("Immutable source model changed during continuation")
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    return {
        "condition_id": condition_id,
        "training_seed": seed,
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(evaluation_rows),
        "source_sha256": source_hash,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
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
    output_root = (
        ROOT / "artifacts" / "smoke" / "smoothness_target_extension_1m_v1"
        if smoke
        else ROOT / config["output_root"]
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True)

    conditions = config["conditions"]
    seeds = [int(config["training_seeds"][0])] if smoke else [int(seed) for seed in config["training_seeds"]]
    targets = [8192] if smoke else [int(value) for value in config["checkpoint_timesteps"]]
    expected_source_steps = 4096 if smoke else int(config["expected_source_model_timesteps"])
    eval_episodes = 2 if smoke else int(config["evaluation_episodes_per_checkpoint"])
    tasks = []
    resolved_sources = []
    for seed in seeds:
        for condition_id in conditions:
            source = source_path(config, condition_id, seed, smoke=smoke)
            if not source.is_file():
                raise FileNotFoundError(source)
            resolved_sources.append(
                {
                    "condition_id": condition_id,
                    "training_seed": seed,
                    "path": str(source),
                    "sha256": sha256(source),
                    "expected_num_timesteps": expected_source_steps,
                }
            )
            tasks.append(
                {
                    "config": config,
                    "condition_id": condition_id,
                    "training_seed": seed,
                    "source_path": str(source),
                    "expected_source_steps": expected_source_steps,
                    "targets": targets,
                    "eval_episodes": eval_episodes,
                    "output_root": str(output_root / "runs" / f"seed_{seed}" / condition_id),
                }
            )
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    resolved_config = {**config, "smoke": smoke, "resolved_source_policies": resolved_sources}
    (output_root / "resolved_frozen_run_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    record = {
        "status": "started",
        "smoke": smoke,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "tasks": len(tasks),
        "max_workers": min(int(config["execution"]["max_workers"]), len(tasks)),
        "source_policies": resolved_sources,
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    failures = []
    with ProcessPoolExecutor(max_workers=record["max_workers"]) as executor:
        futures = {executor.submit(continue_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                failure = {
                    "condition_id": task["condition_id"],
                    "training_seed": task["training_seed"],
                    "error": repr(error),
                }
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
    runtime_rows = []
    evaluation_rows = []
    for task in tasks:
        task_root = Path(task["output_root"])
        runtime_rows.extend(read_rows(task_root / "logs" / "training_runtime.csv"))
        evaluation_rows.extend(read_rows(task_root / "logs" / "evaluation_metrics.csv"))
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    record.update(
        {
            "status": "failed" if failures else "complete",
            "failures": failures,
            "runtime_rows": len(runtime_rows),
            "evaluation_rows": len(evaluation_rows),
        }
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} extension tasks failed")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
