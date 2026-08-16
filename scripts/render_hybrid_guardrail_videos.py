"""Render the predeclared fixed-seed hybrid development video panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_CONFIG = ROOT / "configs" / "hybrid_guardrail_development_v2_20260816.json"
DEFAULT_VIDEO_CONFIG = ROOT / "configs" / "hybrid_video_plan_v1_20260816.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--video-config", type=Path, default=DEFAULT_VIDEO_CONFIG)
    return parser.parse_args()


def model_path(run_root: Path, condition_id: str, seed: int, target: int) -> Path:
    return (
        run_root
        / "runs"
        / f"seed_{seed}"
        / condition_id
        / "models"
        / condition_id
        / f"checkpoint_{target:06d}.zip"
    )


def render(command: list[str], output: Path) -> dict:
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "condition_id": manifest["condition_id"],
        "training_seed": manifest["training_seed"],
        "evaluation_seed": manifest["evaluation_seed"],
        "target_timesteps": manifest["target_timesteps"],
        "trajectory_frames": manifest["trajectory_frames"],
        "video_frames": manifest["frames"],
        "video_duration_seconds": manifest["video_duration_seconds"],
        "padded_frames": manifest["padded_frames"],
        "intent_compliant": manifest["episode_summary"]["intent_compliant"],
        "video_path": manifest["video_path"],
        "video_sha256": manifest["video_sha256"],
    }


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix_config.resolve().read_text(encoding="utf-8"))
    plan = json.loads(args.video_config.resolve().read_text(encoding="utf-8"))
    run_root = ROOT / matrix["execution"]["output_root"]
    completion = json.loads((run_root / "parallel_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("Development matrix is not complete")
    adjudication = json.loads(
        (run_root / "analysis" / "development_gate_adjudication.json").read_text(encoding="utf-8")
    )
    video_root = run_root / "videos"
    if video_root.exists():
        raise FileExistsError(video_root)
    video_root.mkdir(parents=True)
    seed = int(plan["fixed_training_seed"])
    evaluation_seed = int(plan["fixed_evaluation_seed"])
    endpoint = int(plan["endpoint_timesteps"])
    jobs: list[tuple[dict, int]] = [(condition, endpoint) for condition in matrix["conditions"]]
    baseline_id = matrix["development_gate"]["baseline_condition_id"]
    baseline = next(
        row for row in matrix["conditions"] if row["condition_id"] == baseline_id
    )
    jobs.extend((baseline, int(target)) for target in plan["baseline_progression_checkpoints"])
    advanced = sorted(adjudication["advanced_conditions"])
    if advanced:
        selected = next(row for row in matrix["conditions"] if row["condition_id"] == advanced[0])
        jobs.extend((selected, int(target)) for target in plan["baseline_progression_checkpoints"])

    renderer = ROOT / "scripts" / "render_stage1_full_video.py"
    xml_file = ROOT / plan["render_xml_file"]
    rows: list[dict] = []
    for condition, target in jobs:
        model = model_path(run_root, condition["condition_id"], seed, target)
        if not model.exists():
            raise FileNotFoundError(model)
        output = video_root / f"{condition['condition_id']}__tr{seed}__ev{evaluation_seed}__t{target}.mp4"
        command = [
            sys.executable,
            str(renderer),
            "--model", str(model),
            "--condition_id", condition["condition_id"],
            "--ctrl_cost_weight", str(matrix["ctrl_cost_weight"]),
            "--orientation_shaping_weight", str(condition["orientation_shaping_weight"]),
            "--orientation_shaping_function", str(condition["orientation_shaping_function"]),
            "--orientation_shaping_scale", str(condition["orientation_shaping_scale"]),
            "--lateral_drift_shaping_weight", str(condition.get("lateral_drift_shaping_weight", 0.0)),
            "--lateral_drift_shaping_scale", str(condition.get("lateral_drift_shaping_scale", 1.0)),
            "--lateral_shaping_signal", str(condition.get("lateral_shaping_signal", "offset_tanh")),
            "--lateral_velocity_target", str(condition.get("lateral_velocity_target", 0.0)),
            "--training_seed", str(seed),
            "--evaluation_seed", str(evaluation_seed),
            "--target_timesteps", str(target),
            "--max_steps", str(plan["max_steps"]),
            "--fps", str(plan["fps"]),
            "--augment_previous_applied_action",
            "--xml_file", str(xml_file),
            "--output", str(output),
        ]
        if condition["action_slew_l2_limit"] is not None:
            command.extend(["--action_slew_l2_limit", str(condition["action_slew_l2_limit"])])
        if plan["pad_early_termination_to_horizon"]:
            command.append("--pad_to_horizon")
        rows.append(render(command, output))
        print(f"Rendered {output.name}", flush=True)

    index = video_root / "VIDEO_INDEX.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (video_root / "VIDEO_INDEX.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "videos": len(rows), "root": str(video_root)}, indent=2))


if __name__ == "__main__":
    main()
