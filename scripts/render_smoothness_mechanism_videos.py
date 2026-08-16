"""Render one fixed 50-second endpoint trajectory for every pilot policy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "smoothness_mechanism_pilot_v1_20260816.json"
EVALUATION_SEED = 51401
ENDPOINT = 300_000


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    shared = config["shared_reward"]
    run_root = ROOT / config["execution"]["output_root"]
    video_root = run_root / "8_16_trials_2"
    if video_root.exists():
        raise FileExistsError(video_root)
    video_root.mkdir(parents=True)
    renderer = ROOT / "scripts" / "render_stage1_full_video.py"
    xml_file = ROOT / "assets" / "ant_render_large_floor.xml"
    rows = []
    for condition in config["conditions"]:
        condition_id = str(condition["condition_id"])
        for training_seed in config["training_seeds"]:
            training_seed = int(training_seed)
            model = (
                run_root
                / "runs"
                / f"seed_{training_seed}"
                / condition_id
                / "models"
                / condition_id
                / f"checkpoint_{ENDPOINT:06d}.zip"
            )
            output = video_root / (
                f"{condition_id}__tr{training_seed}__ev{EVALUATION_SEED}__t{ENDPOINT}.mp4"
            )
            command = [
                sys.executable,
                str(renderer),
                "--model",
                str(model),
                "--condition_id",
                condition_id,
                "--ctrl_cost_weight",
                str(config["ctrl_cost_weight"]),
                "--orientation_shaping_weight",
                str(shared["orientation_shaping_weight"]),
                "--orientation_shaping_function",
                str(shared["orientation_shaping_function"]),
                "--orientation_shaping_scale",
                str(shared["orientation_shaping_scale"]),
                "--lateral_drift_shaping_weight",
                str(shared["lateral_drift_shaping_weight"]),
                "--lateral_drift_shaping_scale",
                str(shared["lateral_drift_shaping_scale"]),
                "--lateral_shaping_signal",
                str(shared["lateral_shaping_signal"]),
                "--lateral_velocity_target",
                str(shared["lateral_velocity_target"]),
                "--forward_velocity_target",
                str(shared["forward_velocity_target"]),
                "--forward_velocity_tracking_scale",
                str(shared["forward_velocity_tracking_scale"]),
                "--action_rate_shaping_weight",
                str(condition["action_rate_shaping_weight"]),
                "--training_seed",
                str(training_seed),
                "--evaluation_seed",
                str(EVALUATION_SEED),
                "--target_timesteps",
                str(ENDPOINT),
                "--max_steps",
                "1000",
                "--fps",
                "20",
                "--augment_previous_applied_action",
                "--xml_file",
                str(xml_file),
                "--output",
                str(output),
                "--pad_to_horizon",
            ]
            if condition["replace_forward_reward_with_tracking"]:
                command.append("--replace_forward_reward_with_tracking")
            subprocess.run(command, cwd=ROOT, check=True)
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "condition_id": condition_id,
                    "training_seed": training_seed,
                    "evaluation_seed": EVALUATION_SEED,
                    "replace_forward_reward_with_tracking": condition[
                        "replace_forward_reward_with_tracking"
                    ],
                    "action_rate_shaping_weight": condition["action_rate_shaping_weight"],
                    "trajectory_frames": manifest["trajectory_frames"],
                    "video_frames": manifest["frames"],
                    "padded_frames": manifest["padded_frames"],
                    "termination_category": manifest["episode_summary"]["termination_category"],
                    "intent_compliant": manifest["episode_summary"]["intent_compliant"],
                    "video_path": manifest["video_path"],
                    "video_sha256": manifest["video_sha256"],
                }
            )
            print(f"Rendered {output.name}", flush=True)
    with (video_root / "VIDEO_INDEX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (video_root / "VIDEO_INDEX.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "videos": len(rows), "root": str(video_root)}, indent=2))


if __name__ == "__main__":
    main()
