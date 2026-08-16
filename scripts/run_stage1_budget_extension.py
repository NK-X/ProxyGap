"""Run the authorised 300k-to-1M stage-one development extension."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import ctypes
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.budget_extension import (  # noqa: E402
    condition_id,
    continue_policy,
    sha256,
    validate_budget_extension_config,
)
from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    write_rows,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
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
        raise OSError("Windows rejected the sleep-prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        es_continuous,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_root(output_root: Path, source: dict[str, Any]) -> Path:
    return (
        output_root
        / "runs"
        / f"seed_{int(source['training_seed'])}"
        / str(source["condition_id"])
    )


def task_complete(root: Path, checkpoints: list[int], episodes: int) -> bool:
    models = list((root / "models").rglob("checkpoint_*.zip"))
    runtime_rows = read_rows(root / "logs" / "training_runtime.csv")
    evaluation_rows = read_rows(root / "logs" / "evaluation_metrics.csv")
    source_rows = read_rows(root / "logs" / "source_model_audit.csv")
    return (
        len(models) == len(checkpoints)
        and len(runtime_rows) == len(checkpoints)
        and len(evaluation_rows) == len(checkpoints) * episodes
        and len(source_rows) == 1
    )


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["output_root"])
    runtime_rows, evaluation_rows, source_audit = continue_policy(
        project_root=Path(task["project_root"]),
        output_root=root,
        source=task["source"],
        config=task["config"],
    )
    write_standard_outputs(
        root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    write_rows(root / "logs" / "source_model_audit.csv", [source_audit])
    return {
        "condition_id": task["source"]["condition_id"],
        "training_seed": task["source"]["training_seed"],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(evaluation_rows),
    }


def collect_completed(
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    runtime_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    source_rows: list[dict] = []
    extension = config["budget_extension"]
    for source in config["source_policies"]:
        root = task_root(output_root, source)
        if not task_complete(
            root,
            extension["checkpoint_timesteps"],
            len(extension["evaluation_seeds"]),
        ):
            continue
        runtime_rows.extend(read_rows(root / "logs" / "training_runtime.csv"))
        evaluation_rows.extend(read_rows(root / "logs" / "evaluation_metrics.csv"))
        source_rows.extend(read_rows(root / "logs" / "source_model_audit.csv"))
    runtime_rows.sort(
        key=lambda row: (
            int(row["training_seed"]),
            float(row["ctrl_cost_weight"]),
            int(row["target_timesteps"]),
        )
    )
    evaluation_rows.sort(
        key=lambda row: (
            int(row["training_seed"]),
            float(row["ctrl_cost_weight"]),
            int(row["target_timesteps"]),
            int(row["seed"]),
        )
    )
    source_rows.sort(
        key=lambda row: (int(row["training_seed"]), float(row["ctrl_cost_weight"]))
    )
    return runtime_rows, evaluation_rows, source_rows


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_budget_extension_config(
        config,
        project_root=PROJECT_ROOT,
        verify_source_files=True,
    )
    if errors:
        raise ValueError(f"Budget-extension config errors: {errors}")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (PROJECT_ROOT / config["output_root"]).resolve()
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "validated_dry_run",
                    "config": str(config_path),
                    "output_root": str(output_root),
                    "max_workers": args.max_workers,
                    "task_count": len(config["source_policies"]),
                    "tasks": [
                        {
                            "condition_id": source["condition_id"],
                            "training_seed": source["training_seed"],
                            "source_sha256": source["sha256"],
                        }
                        for source in config["source_policies"]
                    ],
                    "formal_launch": "prohibited",
                    "shaping_launch": "prohibited",
                },
                indent=2,
            ),
            flush=True,
        )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    request_windows_awake()

    protocol_path = (PROJECT_ROOT / config["protocol_document"]).resolve()
    code_paths = [
        PROJECT_ROOT / "src" / "proxygap" / "budget_extension.py",
        PROJECT_ROOT / "src" / "proxygap" / "experiment.py",
        PROJECT_ROOT / "src" / "proxygap" / "ant_wrapper.py",
        Path(__file__).resolve(),
    ]
    run_config = {
        **config,
        "execution_started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_config": str(config_path),
        "protocol_config_sha256": sha256(config_path),
        "protocol_document_absolute": str(protocol_path),
        "protocol_document_sha256": sha256(protocol_path),
        "code_sha256": {str(path): file_hash(path) for path in code_paths},
        "max_workers": args.max_workers,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    tasks: list[dict[str, Any]] = []
    extension = config["budget_extension"]
    for source in config["source_policies"]:
        root = task_root(output_root, source)
        if task_complete(
            root,
            extension["checkpoint_timesteps"],
            len(extension["evaluation_seeds"]),
        ):
            continue
        if root.exists() and any(root.rglob("*")):
            raise RuntimeError(f"Incomplete task requires forensic audit: {root}")
        tasks.append(
            {
                "project_root": str(PROJECT_ROOT),
                "output_root": str(root),
                "source": source,
                "config": config,
            }
        )

    random.Random(int(config["task_order_seed"])).shuffle(tasks)
    execution_record = {
        "parallel_unit": "continued trained policy",
        "task_order_seed": int(config["task_order_seed"]),
        "task_count_at_start": len(tasks),
        "submitted_task_order": [
            {
                "condition_id": task["source"]["condition_id"],
                "training_seed": task["source"]["training_seed"],
            }
            for task in tasks
        ],
        "environment_state_restored": False,
        "scientific_role": "development budget sufficiency only",
    }
    (output_root / "parallel_execution.json").write_text(
        json.dumps(execution_record, indent=2) + "\n",
        encoding="utf-8",
    )

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Completed {result}", flush=True)
            except Exception as error:  # Preserve all completed sibling policies.
                message = (
                    f"{task['source']['condition_id']}, "
                    f"seed={task['source']['training_seed']}: {error!r}"
                )
                failures.append(message)
                print(f"FAILED {message}", flush=True)
            runtime_rows, evaluation_rows, source_rows = collect_completed(
                output_root, config
            )
            write_standard_outputs(
                output_root,
                runtime_rows=runtime_rows,
                eval_rows=evaluation_rows,
                summary_rows=summarise_evaluation(evaluation_rows),
            )
            write_rows(output_root / "logs" / "source_model_audit.csv", source_rows)

    runtime_rows, evaluation_rows, source_rows = collect_completed(output_root, config)
    completed_policies = len(source_rows)
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_policies": completed_policies,
        "expected_policies": len(config["source_policies"]),
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(evaluation_rows),
        "source_audit_rows": len(source_rows),
        "failures": failures,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
    }
    (output_root / "parallel_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures or completed_policies != len(config["source_policies"]):
        raise RuntimeError(f"Budget extension incomplete: {completion}")
    print(f"Budget extension complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
