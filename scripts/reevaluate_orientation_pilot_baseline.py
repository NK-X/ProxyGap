"""Re-evaluate frozen baseline models with the orientation-pilot metric schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    evaluate_model,
    summarise_evaluation,
    write_standard_outputs,
)


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["execution"]["output_root"] / "baseline_replay"
    if output_root.exists():
        raise FileExistsError(f"Baseline replay already exists: {output_root}")
    output_root.mkdir(parents=True)

    model_pattern = config["baseline_evidence"]["model_pattern"]
    checkpoints = [int(value) for value in config["checkpoint_timesteps"]]
    total_timesteps = int(config["timesteps_per_condition"])
    all_rows: list[dict] = []
    runtime_rows: list[dict] = []
    model_records: list[dict] = []

    for training_seed in config["offline_calibration_training_seeds"]:
        for target_timesteps in checkpoints:
            relative_model = model_pattern.format(
                training_seed=int(training_seed),
                target_timesteps=int(target_timesteps),
            )
            model_path = PROJECT_ROOT / relative_model
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = PPO.load(model_path, device="cpu")
            start = time.perf_counter()
            rows, elapsed = evaluate_model(
                model,
                condition_id="reference",
                ctrl_cost_weight=float(config["ctrl_cost_weight"]),
                checkpoint_fraction=target_timesteps / total_timesteps,
                target_timesteps=target_timesteps,
                actual_model_timesteps=int(model.num_timesteps),
                training_seed=int(training_seed),
                seed=int(config["evaluation_seed_base"]),
                episodes=int(config["eval_episodes_per_checkpoint"]),
                max_episode_steps=int(config["eval_max_episode_steps"]),
                orientation_shaping_weight=0.0,
                orientation_shaping_scale=1.0,
                orientation_shaping_function="cosine",
                common_rescore_ctrl_cost_weight=float(
                    config["reward"]["common_rescore_ctrl_cost_weight"]
                ),
                effort_distance_min=float(config["metrics"]["effort_distance_min"]),
                action_saturation_threshold=float(
                    config["metrics"]["action_saturation_threshold"]
                ),
            )
            all_rows.extend(rows)
            runtime_rows.append(
                {
                    "condition_id": "reference",
                    "training_seed": int(training_seed),
                    "target_timesteps": target_timesteps,
                    "evaluation_rows": len(rows),
                    "elapsed_sec": elapsed,
                    "wall_elapsed_sec": time.perf_counter() - start,
                    "model_path": str(model_path),
                }
            )
            model_records.append(
                {
                    "training_seed": int(training_seed),
                    "target_timesteps": target_timesteps,
                    "path": str(model_path),
                    "sha256": sha256(model_path),
                }
            )
            print(
                f"Re-evaluated baseline seed={training_seed}, target={target_timesteps}",
                flush=True,
            )

    expected_rows = (
        len(config["offline_calibration_training_seeds"])
        * len(checkpoints)
        * int(config["eval_episodes_per_checkpoint"])
    )
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, received {len(all_rows)}")
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=all_rows,
        summary_rows=summarise_evaluation(all_rows),
    )
    manifest = {
        "status": "complete",
        "role": "metric-schema replay only; no policy training",
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "evaluation_rows": len(all_rows),
        "model_records": model_records,
    }
    (output_root / "baseline_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved baseline replay: {output_root}", flush=True)


if __name__ == "__main__":
    main()
