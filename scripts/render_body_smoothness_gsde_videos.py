"""Render one fixed full-horizon endpoint replay for a frozen body experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "body_smoothness_gsde_matrix_v1_20260816.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output-dir-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    shared = config["shared_reward"]
    body = config["body_dynamics"]
    run_root = ROOT / config["execution"]["output_root"]
    evaluation_seed = (
        int(args.evaluation_seed)
        if args.evaluation_seed is not None
        else int(config["evaluation_seeds"][0])
    )
    endpoint = int(config["timesteps_per_condition"])
    default_output_name = (
        "8_16_trials_4"
        if config["config_id"] == "body_smoothness_gsde_matrix_v1_20260816"
        else "full_horizon_videos"
    )
    video_root = run_root / (args.output_dir_name or default_output_name)
    video_root.mkdir(parents=True, exist_ok=True)
    renderer = ROOT / "scripts" / "render_stage1_full_video.py"
    xml_file = ROOT / "assets" / "ant_render_large_floor.xml"
    rows = []
    for item in config["conditions"]:
        condition_id = item["condition_id"]
        enabled = bool(item["body_dynamics_enabled"])
        vertical_weight = body["vertical_velocity_shaping_weight"] if enabled else 0.0
        angular_weight = (
            body["roll_pitch_angular_velocity_shaping_weight"] if enabled else 0.0
        )
        for training_seed in config["training_seeds"]:
            model = (
                run_root
                / "runs"
                / f"seed_{training_seed}"
                / condition_id
                / "models"
                / condition_id
                / f"checkpoint_{endpoint}.zip"
            )
            output = video_root / (
                f"{condition_id}__tr{training_seed}__ev{evaluation_seed}__t{endpoint}.mp4"
            )
            command = [
                sys.executable,
                str(renderer),
                "--model", str(model),
                "--condition_id", condition_id,
                "--ctrl_cost_weight", str(shared["ctrl_cost_weight"]),
                "--orientation_shaping_weight", str(shared["orientation_shaping_weight"]),
                "--orientation_shaping_function", str(shared["orientation_shaping_function"]),
                "--orientation_shaping_scale", str(shared["orientation_shaping_scale"]),
                "--lateral_drift_shaping_weight", str(shared["lateral_drift_shaping_weight"]),
                "--lateral_drift_shaping_scale", str(shared["lateral_drift_shaping_scale"]),
                "--lateral_shaping_signal", str(shared["lateral_shaping_signal"]),
                "--lateral_velocity_target", str(shared["lateral_velocity_target"]),
                "--replace_forward_reward_with_tracking",
                "--forward_velocity_target", str(shared["forward_velocity_target"]),
                "--forward_velocity_tracking_scale", str(shared["forward_velocity_tracking_scale"]),
                "--action_rate_shaping_weight", str(shared["action_rate_shaping_weight"]),
                "--vertical_velocity_shaping_weight", str(vertical_weight),
                "--vertical_velocity_shaping_scale", str(body["vertical_velocity_shaping_scale"]),
                "--roll_pitch_angular_velocity_shaping_weight", str(angular_weight),
                "--roll_pitch_angular_velocity_shaping_scale", str(body["roll_pitch_angular_velocity_shaping_scale"]),
                "--training_seed", str(training_seed),
                "--evaluation_seed", str(evaluation_seed),
                "--target_timesteps", str(endpoint),
                "--max_steps", "1000",
                "--fps", "20",
                "--augment_previous_applied_action",
                "--xml_file", str(xml_file),
                "--output", str(output),
                "--pad_to_horizon",
            ]
            manifest_path = output.with_suffix(".json")
            if output.exists() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                print(f"Reused {output.name}", flush=True)
            else:
                subprocess.run(command, cwd=ROOT, check=True)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                print(f"Rendered {output.name}", flush=True)
            rows.append(
                {
                    "condition_id": condition_id,
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "body_dynamics_enabled": enabled,
                    "use_sde": bool(item["use_sde"]),
                    "trajectory_frames": manifest["trajectory_frames"],
                    "video_frames": manifest["frames"],
                    "padded_frames": manifest["padded_frames"],
                    "termination_category": manifest["episode_summary"]["termination_category"],
                    "intent_compliant": manifest["episode_summary"]["intent_compliant"],
                    "video_path": manifest["video_path"],
                    "video_sha256": manifest["video_sha256"],
                }
            )
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
