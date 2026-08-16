"""Render all six endpoint conditions for all three development training seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hybrid_guardrail_observability_correction_v1_20260816.json"
VIDEO_PLAN = ROOT / "configs" / "hybrid_video_plan_v1_20260816.json"


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


def main() -> None:
    matrix = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = json.loads(VIDEO_PLAN.read_text(encoding="utf-8"))
    run_root = ROOT / matrix["execution"]["output_root"]
    video_root = run_root / "8_16_trials_1"
    if video_root.exists():
        raise FileExistsError(video_root)
    video_root.mkdir(parents=True)
    renderer = ROOT / "scripts" / "render_stage1_full_video.py"
    xml_file = ROOT / plan["render_xml_file"]
    evaluation_seed = int(plan["fixed_evaluation_seed"])
    target = int(plan["endpoint_timesteps"])
    rows: list[dict] = []
    for condition in matrix["conditions"]:
        for seed in matrix["training_seeds"]:
            seed = int(seed)
            model = model_path(run_root, condition["condition_id"], seed, target)
            output = video_root / (
                f"{condition['condition_id']}__tr{seed}__ev{evaluation_seed}__t{target}.mp4"
            )
            command = [
                sys.executable,
                str(renderer),
                "--model",
                str(model),
                "--condition_id",
                condition["condition_id"],
                "--ctrl_cost_weight",
                str(matrix["ctrl_cost_weight"]),
                "--orientation_shaping_weight",
                str(condition["orientation_shaping_weight"]),
                "--orientation_shaping_function",
                str(condition["orientation_shaping_function"]),
                "--orientation_shaping_scale",
                str(condition["orientation_shaping_scale"]),
                "--lateral_drift_shaping_weight",
                str(condition.get("lateral_drift_shaping_weight", 0.0)),
                "--lateral_drift_shaping_scale",
                str(condition.get("lateral_drift_shaping_scale", 1.0)),
                "--lateral_shaping_signal",
                str(condition.get("lateral_shaping_signal", "offset_tanh")),
                "--lateral_velocity_target",
                str(condition.get("lateral_velocity_target", 0.0)),
                "--training_seed",
                str(seed),
                "--evaluation_seed",
                str(evaluation_seed),
                "--target_timesteps",
                str(target),
                "--max_steps",
                str(plan["max_steps"]),
                "--fps",
                str(plan["fps"]),
                "--augment_previous_applied_action",
                "--xml_file",
                str(xml_file),
                "--output",
                str(output),
                "--pad_to_horizon",
            ]
            if condition["action_slew_l2_limit"] is not None:
                command.extend(
                    ["--action_slew_l2_limit", str(condition["action_slew_l2_limit"])]
                )
            subprocess.run(command, cwd=ROOT, check=True)
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "condition_id": manifest["condition_id"],
                    "training_seed": manifest["training_seed"],
                    "evaluation_seed": manifest["evaluation_seed"],
                    "target_timesteps": manifest["target_timesteps"],
                    "trajectory_frames": manifest["trajectory_frames"],
                    "video_frames": manifest["frames"],
                    "video_duration_seconds": manifest["video_duration_seconds"],
                    "padded_frames": manifest["padded_frames"],
                    "termination_category": manifest["episode_summary"]["termination_category"],
                    "intent_compliant": manifest["episode_summary"]["intent_compliant"],
                    "intent_failure_reasons": manifest["episode_summary"]["intent_failure_reasons"],
                    "video_path": manifest["video_path"],
                    "video_sha256": manifest["video_sha256"],
                }
            )
            print(f"Rendered {output.name}", flush=True)
    with (video_root / "VIDEO_INDEX.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (video_root / "VIDEO_INDEX.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "videos": len(rows), "root": str(video_root)}, indent=2))


if __name__ == "__main__":
    main()
