"""Run a locked ProxyGap configuration without mixing it with pilot outputs."""

from __future__ import annotations

import argparse
import atexit
import csv
import ctypes
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    resolve_ppo_config,
    save_run_config,
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "config_id",
        "status",
        "timesteps_per_condition",
        "checkpoint_timesteps",
        "eval_episodes_per_checkpoint",
        "eval_max_episode_steps",
        "training_seeds",
        "evaluation_seed_base",
        "conditions",
        "ppo",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing formal config keys: {missing}")
    if not str(config["status"]).startswith("formal_locked"):
        raise ValueError("Formal runner requires a status beginning with 'formal_locked'")
    require_complete = int(config.get("schema_version", 1)) >= 2
    config["ppo_resolved"] = resolve_ppo_config(
        config["ppo"],
        require_complete=require_complete,
    )
    return config


def request_windows_awake() -> None:
    """Keep Windows awake while this process owns a formal training run."""
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        es_continuous | es_system_required
    )
    if result == 0:
        raise OSError("Windows rejected the formal-run sleep prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        es_continuous,
    )


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_is_complete(condition_root: Path, config: dict[str, Any]) -> bool:
    expected_models = len(config["checkpoint_timesteps"])
    expected_eval_rows = expected_models * int(config["eval_episodes_per_checkpoint"])
    models = list((condition_root / "models").rglob("checkpoint_*.zip"))
    eval_rows = read_csv_rows(condition_root / "logs" / "evaluation_metrics.csv")
    return len(models) == expected_models and len(eval_rows) == expected_eval_rows


def collect_completed_rows(
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for training_seed in config["training_seeds"]:
        for condition in config["conditions"]:
            root = (
                output_root
                / "runs"
                / f"seed_{training_seed}"
                / str(condition["condition_id"])
            )
            if condition_is_complete(root, config):
                runtime_rows.extend(read_csv_rows(root / "logs" / "training_runtime.csv"))
                eval_rows.extend(read_csv_rows(root / "logs" / "evaluation_metrics.csv"))
    return runtime_rows, eval_rows


def main() -> None:
    args = parse_args()
    request_windows_awake()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_root = PROJECT_ROOT / "artifacts" / "formal" / str(config["config_id"])
    if (output_root / "completed.json").exists():
        raise FileExistsError(f"Formal run already completed: {output_root}")

    saved_config_path = output_root / "run_config.json"
    if saved_config_path.exists() and not args.resume:
        raise FileExistsError(f"Partial formal run exists; use --resume: {output_root}")
    if not saved_config_path.exists():
        saved_config = dict(config)
        saved_config["source_config"] = str(config_path)
        save_run_config(output_root, saved_config)

    all_runtime_rows, all_eval_rows = collect_completed_rows(output_root, config)
    for training_seed in config["training_seeds"]:
        for condition in config["conditions"]:
            condition_id = str(condition["condition_id"])
            condition_root = output_root / "runs" / f"seed_{training_seed}" / condition_id
            if condition_is_complete(condition_root, config):
                print(f"Skipping completed {condition_id}, seed={training_seed}", flush=True)
                continue
            if condition_root.exists() and any(condition_root.rglob("*")):
                raise RuntimeError(
                    f"Incomplete condition has files and requires manual audit: {condition_root}"
                )
            print(
                f"Starting {condition_id}, seed={training_seed}, "
                f"timesteps={config['timesteps_per_condition']}",
                flush=True,
            )
            runtime_rows, eval_rows = train_condition(
                output_root=condition_root,
                condition_id=condition_id,
                ctrl_cost_weight=float(condition["ctrl_cost_weight"]),
                forward_progress_shaping_weight=float(
                    condition.get("forward_progress_shaping_weight", 0.0)
                ),
                lateral_drift_shaping_weight=float(
                    condition.get("lateral_drift_shaping_weight", 0.0)
                ),
                lateral_drift_shaping_scale=float(
                    condition.get("lateral_drift_shaping_scale", 1.0)
                ),
                effort_shaping_weight=float(condition.get("effort_shaping_weight", 0.0)),
                effort_shaping_scale=float(condition.get("effort_shaping_scale", 1.0)),
                orientation_shaping_weight=float(
                    condition.get("orientation_shaping_weight", 0.0)
                ),
                orientation_shaping_scale=float(
                    condition.get("orientation_shaping_scale", 1.0)
                ),
                common_rescore_ctrl_cost_weight=float(
                    config.get("metric_parameters", {}).get(
                        "common_rescore_ctrl_cost_weight", 0.5
                    )
                ),
                effort_distance_min=float(
                    config.get("metric_parameters", {}).get("effort_distance_min", 1e-8)
                ),
                action_saturation_threshold=float(
                    config.get("metric_parameters", {}).get(
                        "action_saturation_threshold", 0.95
                    )
                ),
                record_evaluation_steps=bool(
                    config.get("records", {}).get("evaluation_step_logs_gzip", False)
                ),
                total_timesteps=int(config["timesteps_per_condition"]),
                checkpoint_timesteps=config["checkpoint_timesteps"],
                seed=int(training_seed),
                evaluation_seed_base=int(config["evaluation_seed_base"]),
                eval_episodes=int(config["eval_episodes_per_checkpoint"]),
                eval_max_episode_steps=int(config["eval_max_episode_steps"]),
                ppo_n_steps=int(config["ppo_resolved"]["n_steps"]),
                ppo_batch_size=int(config["ppo_resolved"]["batch_size"]),
                ppo_n_epochs=int(config["ppo_resolved"]["n_epochs"]),
                ppo_learning_rate=float(config["ppo_resolved"]["learning_rate"]),
                ppo_gamma=float(config["ppo_resolved"]["gamma"]),
                ppo_gae_lambda=float(config["ppo_resolved"]["gae_lambda"]),
                ppo_clip_range=float(config["ppo_resolved"]["clip_range"]),
                ppo_ent_coef=float(config["ppo_resolved"]["ent_coef"]),
                ppo_vf_coef=float(config["ppo_resolved"]["vf_coef"]),
                ppo_max_grad_norm=float(config["ppo_resolved"]["max_grad_norm"]),
                ppo_normalize_advantage=bool(
                    config["ppo_resolved"]["normalize_advantage"]
                ),
                ppo_policy=str(config["ppo_resolved"]["policy"]),
                ppo_policy_kwargs=dict(config["ppo_resolved"]["policy_kwargs"]),
                ppo_device=str(config["ppo_resolved"]["device"]),
                ppo_torch_num_threads=int(config["ppo"].get("torch_num_threads", 1)),
            )
            condition_summary = summarise_evaluation(eval_rows)
            write_standard_outputs(
                condition_root,
                runtime_rows=runtime_rows,
                eval_rows=eval_rows,
                summary_rows=condition_summary,
            )
            all_runtime_rows.extend(runtime_rows)
            all_eval_rows.extend(eval_rows)
            write_standard_outputs(
                output_root,
                runtime_rows=all_runtime_rows,
                eval_rows=all_eval_rows,
                summary_rows=summarise_evaluation(all_eval_rows),
            )
            print(f"Completed {condition_id}, seed={training_seed}", flush=True)

    write_standard_outputs(
        output_root,
        runtime_rows=all_runtime_rows,
        eval_rows=all_eval_rows,
        summary_rows=summarise_evaluation(all_eval_rows),
    )
    completion = {
        "config_id": config["config_id"],
        "status": "completed",
        "conditions": len(config["conditions"]),
        "training_seeds": config["training_seeds"],
        "evaluation_rows": len(all_eval_rows),
    }
    (output_root / "completed.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved formal run: {output_root}", flush=True)


if __name__ == "__main__":
    main()
