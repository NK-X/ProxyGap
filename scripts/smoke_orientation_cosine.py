"""Engineering smoke test for the cosine reward path and PPO integration."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


def main() -> None:
    output_root = (
        PROJECT_ROOT
        / "artifacts"
        / "smoke"
        / "orientation_cosine_training_smoke_20260815"
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    runtime_rows, eval_rows = train_condition(
        output_root=output_root,
        condition_id="orientation_cosine_lambda_0p25_smoke",
        ctrl_cost_weight=0.5,
        total_timesteps=4096,
        checkpoint_timesteps=[4096],
        seed=81501,
        evaluation_seed_base=81551,
        eval_episodes=2,
        eval_max_episode_steps=100,
        orientation_shaping_weight=0.25,
        orientation_shaping_scale=1.0,
        orientation_shaping_function="cosine",
        ppo_n_steps=2048,
        ppo_batch_size=64,
        ppo_n_epochs=1,
        ppo_torch_num_threads=2,
    )
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=eval_rows,
        summary_rows=summarise_evaluation(eval_rows),
    )
    if not all(row["orientation_shaping_function"] == "cosine" for row in eval_rows):
        raise RuntimeError("Evaluation did not retain the cosine function label")
    if not all(float(row["orientation_penalty_sum"]) > 0 for row in eval_rows):
        raise RuntimeError("Cosine penalty was not accumulated")
    if not all(float(row["reward_orientation_shaping_sum"]) < 0 for row in eval_rows):
        raise RuntimeError("Cosine shaping did not alter the observed reward")
    manifest = {
        "status": "passed",
        "training_timesteps": 4096,
        "evaluation_rows": len(eval_rows),
        "scientific_role": "engineering smoke only; no scientific inference",
    }
    (output_root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
