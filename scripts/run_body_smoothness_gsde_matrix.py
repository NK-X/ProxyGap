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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


DEFAULT_CONFIG = ROOT / "configs" / "body_smoothness_gsde_matrix_v1_20260816.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Execution-only worker override; reward and PPO settings are unchanged.",
    )
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
    allowed_statuses = {
        "frozen_authorised_development_matrix",
        "frozen_authorised_final_development_replication",
        "frozen_user_requested_development",
    }
    if config.get("status") not in allowed_statuses:
        raise ValueError("Development configuration is not frozen")
    if config.get("formal_launch") != "prohibited":
        raise ValueError("Formal launch must remain prohibited")
    if config["checkpoint_timesteps"] != [250000, 500000, 750000, 1000000]:
        raise ValueError("Frozen checkpoint schedule changed")
    design = config.get("design_type", "body_by_gsde_factorial")
    if design == "body_by_gsde_factorial":
        factors = {
            (
                bool(condition["body_dynamics_enabled"]),
                bool(condition["use_sde"]),
                int(condition["sde_sample_freq"]),
            )
            for condition in config["conditions"]
        }
        expected = {
            (False, False, -1),
            (True, False, -1),
            (False, True, 8),
            (True, True, 8),
        }
    elif design == "ordinary_exploration_body_replication":
        factors = {
            (
                bool(condition["body_dynamics_enabled"]),
                bool(condition["use_sde"]),
                int(condition["sde_sample_freq"]),
            )
            for condition in config["conditions"]
        }
        expected = {
            (False, False, -1),
            (True, False, -1),
        }
    elif design == "foot_landing_velocity_ablation":
        factors = {
            (
                bool(condition["body_dynamics_enabled"]),
                bool(condition["foot_landing_enabled"]),
                bool(condition["use_sde"]),
                int(condition["sde_sample_freq"]),
            )
            for condition in config["conditions"]
        }
        expected = {
            (True, False, False, -1),
            (True, True, False, -1),
        }
    elif design == "pitch_balance_single":
        factors = {
            (
                bool(condition["body_dynamics_enabled"]),
                bool(condition["foot_landing_enabled"]),
                bool(condition["pitch_balance_enabled"]),
                bool(condition["use_sde"]),
                int(condition["sde_sample_freq"]),
            )
            for condition in config["conditions"]
        }
        expected = {(True, True, True, False, -1)}
    else:
        raise ValueError(f"Unknown development design_type: {design}")
    if factors != expected:
        raise ValueError("Frozen mechanism condition set changed")
    if set(config["training_seeds"]) & set(config["reserved_formal_training_seeds"]):
        raise ValueError("Development and reserved formal seeds overlap")
    evaluation_seed_base = int(config["evaluation_seed_base"])
    expected_evaluation_seeds = list(
        range(evaluation_seed_base, evaluation_seed_base + int(config["eval_episodes_per_checkpoint"]))
    )
    if config["evaluation_seeds"] != expected_evaluation_seeds:
        raise ValueError("Paired evaluation seeds are inconsistent with the declared base and episode count")


def run_task(task: dict) -> dict:
    config = task["config"]
    condition = task["condition"]
    shared = config["shared_reward"]
    body = config["body_dynamics"]
    foot = config.get("foot_landing", {})
    pitch_balance = config.get("pitch_balance", {})
    ppo = config["ppo"]
    enabled = bool(condition["body_dynamics_enabled"])
    foot_enabled = bool(condition.get("foot_landing_enabled", False))
    pitch_balance_enabled = bool(condition.get("pitch_balance_enabled", False))
    root = Path(task["condition_root"])
    runtime_rows, eval_rows = train_condition(
        output_root=root,
        condition_id=str(condition["condition_id"]),
        ctrl_cost_weight=float(shared["ctrl_cost_weight"]),
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
        replace_forward_reward_with_tracking=True,
        forward_velocity_target=float(shared["forward_velocity_target"]),
        forward_velocity_tracking_scale=float(shared["forward_velocity_tracking_scale"]),
        forward_velocity_tracking_weight=float(
            shared.get("forward_velocity_tracking_weight", 1.0)
        ),
        action_rate_shaping_weight=float(shared["action_rate_shaping_weight"]),
        vertical_velocity_shaping_weight=(
            float(body["vertical_velocity_shaping_weight"]) if enabled else 0.0
        ),
        vertical_velocity_shaping_scale=float(body["vertical_velocity_shaping_scale"]),
        roll_pitch_angular_velocity_shaping_weight=(
            float(body["roll_pitch_angular_velocity_shaping_weight"])
            if enabled
            else 0.0
        ),
        roll_pitch_angular_velocity_shaping_scale=float(
            body["roll_pitch_angular_velocity_shaping_scale"]
        ),
        foot_landing_height_threshold=float(
            foot.get("height_threshold_m", 0.03)
        ),
        foot_lateral_velocity_shaping_weight=(
            float(foot.get("lateral_velocity_weight_per_foot", 0.0))
            if foot_enabled
            else 0.0
        ),
        foot_lateral_velocity_shaping_scale=float(
            foot.get("lateral_velocity_scale_m_per_s", 1.0)
        ),
        foot_vertical_velocity_shaping_weight=(
            float(foot.get("vertical_velocity_weight_per_foot", 0.0))
            if foot_enabled
            else 0.0
        ),
        foot_vertical_velocity_shaping_scale=float(
            foot.get("vertical_velocity_scale_m_per_s", 1.0)
        ),
        pitch_balance_shaping_weight=(
            float(pitch_balance.get("shaping_weight", 0.0))
            if pitch_balance_enabled
            else 0.0
        ),
        foot_geom_names=tuple(
            foot.get(
                "foot_geom_names",
                (
                    "left_ankle_geom",
                    "right_ankle_geom",
                    "third_ankle_geom",
                    "fourth_ankle_geom",
                ),
            )
        ),
        augment_previous_applied_action=bool(shared["augment_previous_applied_action"]),
        action_slew_l2_limit=None,
        record_evaluation_steps=False,
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
        ppo_use_sde=bool(condition["use_sde"]),
        ppo_sde_sample_freq=int(condition["sde_sample_freq"]),
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
        ROOT / "artifacts" / "smoke" / str(config["config_id"])
        if smoke
        else ROOT / config["execution"]["output_root"]
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True)
    conditions = config["conditions"]
    seeds = [int(config["training_seeds"][0])] if smoke else [int(v) for v in config["training_seeds"]]
    timesteps = 4096 if smoke else int(config["timesteps_per_condition"])
    checkpoints = [4096] if smoke else [int(v) for v in config["checkpoint_timesteps"]]
    eval_episodes = 2 if smoke else int(config["eval_episodes_per_checkpoint"])
    max_workers = (
        2
        if smoke
        else (
            int(args.max_workers)
            if args.max_workers is not None
            else int(config["execution"]["max_workers"])
        )
    )
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())

    tasks = []
    for seed in seeds:
        for condition in conditions:
            condition_root = output_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            tasks.append(
                {
                    "config": config,
                    "condition": condition,
                    "training_seed": seed,
                    "condition_root": str(condition_root),
                    "timesteps": timesteps,
                    "checkpoints": checkpoints,
                    "eval_episodes": eval_episodes,
                }
            )
    random.Random(int(config["execution"]["task_order_seed"])).shuffle(tasks)
    sources = [
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
        ROOT / "src" / "proxygap" / "experiment.py",
        ROOT / "src" / "proxygap" / "metrics.py",
        Path(__file__).resolve(),
    ]
    execution = {
        "status": "started",
        "smoke": smoke,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "tasks": len(tasks),
        "max_workers": max_workers,
        "max_workers_configured": int(config["execution"]["max_workers"]),
        "max_workers_overridden": args.max_workers is not None,
        "task_order": [
            {"condition_id": t["condition"]["condition_id"], "seed": t["training_seed"]}
            for t in tasks
        ],
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    record = output_root / "execution_record.json"
    record.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
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

    runtime_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    for seed in seeds:
        for condition in conditions:
            condition_root = output_root / "runs" / f"seed_{seed}" / condition["condition_id"]
            runtime_rows.extend(read_rows(condition_root / "logs" / "training_runtime.csv"))
            evaluation_rows.extend(read_rows(condition_root / "logs" / "evaluation_metrics.csv"))
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
    record.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} development tasks failed")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
