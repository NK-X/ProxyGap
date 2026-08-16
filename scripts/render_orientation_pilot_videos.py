"""Render the predeclared complete-episode MP4 panel for the pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def weight_label(weight: float) -> str:
    return str(weight).replace(".", "p")


def condition_id(weight: float) -> str:
    return f"orientation_cosine_lambda_{weight_label(weight)}"


def shaped_model_path(
    run_root: Path,
    *,
    seed: int,
    weight: float,
    target: int,
) -> Path:
    cid = condition_id(weight)
    return (
        run_root
        / "runs"
        / f"seed_{seed}"
        / cid
        / "models"
        / cid
        / f"checkpoint_{target:06d}.zip"
    )


def render_job(job: dict) -> dict:
    subprocess.run(job["command"], cwd=PROJECT_ROOT, check=True)
    output = Path(job["output"])
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    summary = manifest["episode_summary"]
    tags: list[str] = []
    if summary.get("sustained_inversion"):
        tags.append("sustained_inversion")
    category = str(summary.get("termination_category", "none"))
    if category != "none":
        tags.append(category)
    if summary.get("full_horizon_completed") and not summary.get(
        "sustained_inversion"
    ):
        tags.append("full_horizon_no_sustained_inversion")
    if not tags:
        tags.append("other_complete_episode")
    return {
        "condition_id": job["condition_id"],
        "orientation_weight": job["orientation_weight"],
        "training_seed": job["training_seed"],
        "evaluation_seed": job["evaluation_seed"],
        "target_timesteps": job["target_timesteps"],
        "frames": manifest["frames"],
        "duration_seconds": manifest["video_duration_seconds"],
        "behaviour_tags": "|".join(tags),
        "video_path": str(output),
        "video_sha256": manifest["video_sha256"],
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    completion = json.loads(
        (run_root / "parallel_completion.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete":
        raise RuntimeError("Pilot training is not complete")
    video_root = run_root / "videos"
    if video_root.exists():
        raise FileExistsError(f"Video output already exists: {video_root}")
    video_root.mkdir(parents=True)

    plan = config["video_plan"]
    fixed_eval_seed = int(plan["matched_evaluation_seed"])
    progression_seed = int(plan["progression_training_seed"])
    final_target = int(config["pilot_gate"]["primary_endpoint"])
    baseline_pattern = config["baseline_evidence"]["model_pattern"]

    tasks: dict[tuple[str, int, int], dict] = {}
    conditions = [{"kind": "baseline", "weight": 0.0}]
    conditions.extend(
        {"kind": "shaped", "weight": float(weight)}
        for weight in config["orientation_shaping"]["candidate_weights"]
    )
    for target in plan["progression_checkpoints"]:
        for condition in conditions:
            key = (condition["kind"] + str(condition["weight"]), progression_seed, int(target))
            tasks[key] = {
                **condition,
                "training_seed": progression_seed,
                "target": int(target),
            }
    for seed in plan["additional_final_training_seeds"]:
        for condition in conditions:
            key = (condition["kind"] + str(condition["weight"]), int(seed), final_target)
            tasks[key] = {
                **condition,
                "training_seed": int(seed),
                "target": final_target,
            }

    jobs: list[dict] = []
    renderer = PROJECT_ROOT / "scripts" / "render_stage1_full_video.py"
    for task in tasks.values():
        seed = int(task["training_seed"])
        target = int(task["target"])
        weight = float(task["weight"])
        if task["kind"] == "baseline":
            cid = "reference"
            model_path = PROJECT_ROOT / baseline_pattern.format(
                training_seed=seed,
                target_timesteps=target,
            )
        else:
            cid = condition_id(weight)
            model_path = shaped_model_path(
                run_root,
                seed=seed,
                weight=weight,
                target=target,
            )
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        output = video_root / (
            f"{cid}__train_{seed}__eval_{fixed_eval_seed}__target_{target}.mp4"
        )
        command = [
            sys.executable,
            str(renderer),
            "--model",
            str(model_path),
            "--condition_id",
            cid,
            "--ctrl_cost_weight",
            str(config["ctrl_cost_weight"]),
            "--orientation_shaping_weight",
            str(weight),
            "--orientation_shaping_function",
            "cosine",
            "--orientation_shaping_scale",
            "1.0",
            "--training_seed",
            str(seed),
            "--evaluation_seed",
            str(fixed_eval_seed),
            "--target_timesteps",
            str(target),
            "--max_steps",
            str(plan["max_steps"]),
            "--fps",
            str(plan["fps"]),
            "--output",
            str(output),
        ]
        jobs.append(
            {
                "command": command,
                "output": str(output),
                "condition_id": cid,
                "orientation_weight": weight,
                "training_seed": seed,
                "evaluation_seed": fixed_eval_seed,
                "target_timesteps": target,
            }
        )

    index_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_job, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            index_rows.append(row)
            print(
                f"Rendered {Path(row['video_path']).name}: "
                f"{row['behaviour_tags']}",
                flush=True,
            )
    index_rows.sort(
        key=lambda row: (
            int(row["training_seed"]),
            float(row["orientation_weight"]),
            int(row["target_timesteps"]),
            str(row["condition_id"]),
        )
    )

    csv_path = video_root / "VIDEO_INDEX.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    (video_root / "VIDEO_INDEX.json").write_text(
        json.dumps(index_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(index_rows)} complete videos to {video_root}", flush=True)


if __name__ == "__main__":
    main()
