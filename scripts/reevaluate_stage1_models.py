"""Re-evaluate existing stage-one models under one versioned metric contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import gymnasium
import numpy
import stable_baselines3
from stable_baselines3 import PPO
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    evaluate_model,
    summarise_evaluation,
    write_rows,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", action="append", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--evaluation_seed_base", type=int, default=51_101)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=1_000)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 150_000, 200_000, 250_000, 300_000],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_weight(condition_id: str) -> float:
    if condition_id == "reference":
        return 0.5
    if not condition_id.startswith("ctrl_"):
        raise ValueError(f"Cannot infer coefficient from {condition_id}")
    return float(condition_id.removeprefix("ctrl_").replace("p", "."))


def discover_models(
    source_roots: list[Path], checkpoints: set[int]
) -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    seen: set[tuple[int, float, int]] = set()
    for source_root in source_roots:
        for model_path in sorted((source_root / "runs").rglob("checkpoint_*.zip")):
            checkpoint = int(model_path.stem.removeprefix("checkpoint_"))
            if checkpoint not in checkpoints:
                continue
            seed_part = next(
                (part for part in model_path.parts if part.startswith("seed_")),
                None,
            )
            if seed_part is None:
                raise ValueError(f"Training seed is absent from path: {model_path}")
            training_seed = int(seed_part.removeprefix("seed_"))
            condition_id = model_path.parent.name
            weight = condition_weight(condition_id)
            key = (training_seed, weight, checkpoint)
            if key in seen:
                raise ValueError(f"Duplicate model cell discovered: {key}")
            seen.add(key)
            models.append(
                {
                    "source_root": str(source_root),
                    "source_model": str(model_path.resolve()),
                    "source_model_sha256": sha256(model_path),
                    "condition_id": condition_id,
                    "ctrl_cost_weight": weight,
                    "training_seed": training_seed,
                    "target_timesteps": checkpoint,
                }
            )
    return sorted(
        models,
        key=lambda item: (
            int(item["training_seed"]),
            float(item["ctrl_cost_weight"]),
            int(item["target_timesteps"]),
        ),
    )


def evaluate_one_model(
    item: dict[str, object],
    *,
    evaluation_seed_base: int,
    episodes: int,
    max_episode_steps: int,
    final_checkpoint: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate one immutable model checkpoint in an isolated CPU process."""
    torch.set_num_threads(1)
    model_path = Path(str(item["source_model"]))
    model = PPO.load(model_path, device="cpu")
    checkpoint = int(item["target_timesteps"])
    actual_model_timesteps = int(model.num_timesteps)
    rows, elapsed = evaluate_model(
        model,
        condition_id=str(item["condition_id"]),
        ctrl_cost_weight=float(item["ctrl_cost_weight"]),
        checkpoint_fraction=checkpoint / final_checkpoint,
        seed=evaluation_seed_base,
        episodes=episodes,
        target_timesteps=checkpoint,
        actual_model_timesteps=actual_model_timesteps,
        training_seed=int(item["training_seed"]),
        max_episode_steps=max_episode_steps,
    )
    runtime = {
        **item,
        "actual_model_timesteps": actual_model_timesteps,
        "evaluation_seed_base": evaluation_seed_base,
        "episodes": episodes,
        "evaluation_elapsed_sec": elapsed,
    }
    return rows, runtime


def main() -> None:
    args = parse_args()
    source_roots = [Path(value).resolve() for value in args.source_root]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = set(args.checkpoints)
    manifest_rows = discover_models(source_roots, checkpoints)
    if not manifest_rows:
        raise FileNotFoundError("No source models were discovered")

    expected_weights = {0.5, 0.375, 0.25, 0.125}
    expected_seeds = {41101, 41102}
    found_weights = {float(row["ctrl_cost_weight"]) for row in manifest_rows}
    found_seeds = {int(row["training_seed"]) for row in manifest_rows}
    expected_count = len(expected_weights) * len(expected_seeds) * len(checkpoints)
    if found_weights != expected_weights or found_seeds != expected_seeds:
        raise ValueError(
            f"Unexpected model coverage: weights={found_weights}, seeds={found_seeds}"
        )
    if len(manifest_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} model checkpoints, found {len(manifest_rows)}"
        )

    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    evaluation_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_one_model,
                item,
                evaluation_seed_base=args.evaluation_seed_base,
                episodes=args.episodes,
                max_episode_steps=args.max_episode_steps,
                final_checkpoint=max(checkpoints),
            ): item
            for item in manifest_rows
        }
        for future in as_completed(futures):
            item = futures[future]
            rows, runtime = future.result()
            evaluation_rows.extend(rows)
            runtime_rows.append(runtime)
            evaluation_rows.sort(
                key=lambda row: (
                    int(row["training_seed"]),
                    float(row["ctrl_cost_weight"]),
                    int(row["target_timesteps"]),
                    int(row["seed"]),
                )
            )
            runtime_rows.sort(
                key=lambda row: (
                    int(row["training_seed"]),
                    float(row["ctrl_cost_weight"]),
                    int(row["target_timesteps"]),
                )
            )
            write_standard_outputs(
                output_root,
                runtime_rows=runtime_rows,
                eval_rows=evaluation_rows,
                summary_rows=summarise_evaluation(evaluation_rows),
            )
            print(
                f"Evaluated {item['condition_id']} seed={item['training_seed']} "
                f"checkpoint={item['target_timesteps']}",
                flush=True,
            )

    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    write_rows(output_root / "source_model_manifest.csv", manifest_rows)

    code_paths = [
        PROJECT_ROOT / "src" / "proxygap" / "metrics.py",
        PROJECT_ROOT / "src" / "proxygap" / "ant_wrapper.py",
        PROJECT_ROOT / "src" / "proxygap" / "experiment.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "status": "completed_development_reevaluation_not_formal_evidence",
        "source_roots": [str(path) for path in source_roots],
        "source_model_count": len(manifest_rows),
        "evaluation_row_count": len(evaluation_rows),
        "evaluation_seed_base": args.evaluation_seed_base,
        "episodes_per_model": args.episodes,
        "max_workers": args.max_workers,
        "checkpoints": sorted(checkpoints),
        "elapsed_seconds": time.perf_counter() - started,
        "code_sha256": {str(path): sha256(path) for path in code_paths},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": numpy.__version__,
        },
        "claim_boundary": (
            "This run harmonises measurement of existing development policies. "
            "It does not convert them into held-out confirmation evidence."
        ),
    }
    (output_root / "reevaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
