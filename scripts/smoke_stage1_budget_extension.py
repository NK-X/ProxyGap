"""One-rollout smoke test for checkpoint continuation; not scientific evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.budget_extension import continue_policy, sha256  # noqa: E402
from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    write_rows,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source = next(
        item
        for item in config["source_policies"]
        if float(item["ctrl_cost_weight"]) == 0.5
        and int(item["training_seed"]) == 41101
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Smoke output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = PROJECT_ROOT / source["path"]
    source_hash_before = sha256(source_path)

    smoke_config = json.loads(json.dumps(config))
    smoke_config["budget_extension"]["checkpoint_timesteps"] = [303104]
    smoke_config["budget_extension"]["target_timesteps"] = 303104
    smoke_config["budget_extension"]["evaluation_seeds"] = [61101, 61102]
    smoke_config["budget_extension"]["evaluation_max_episode_steps"] = 100
    runtime_rows, evaluation_rows, source_audit = continue_policy(
        project_root=PROJECT_ROOT,
        output_root=output_root,
        source=source,
        config=smoke_config,
    )
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )
    write_rows(output_root / "logs" / "source_model_audit.csv", [source_audit])

    model_paths = list((output_root / "models").rglob("*.zip"))
    if len(model_paths) != 1:
        raise AssertionError("Smoke test did not create exactly one continued model")
    if int(runtime_rows[0]["start_actual_timesteps"]) != 301056:
        raise AssertionError("Smoke test did not start from the audited 300k model")
    if int(runtime_rows[0]["actual_model_timesteps"]) != 303104:
        raise AssertionError("Smoke test did not complete exactly one PPO rollout")
    if len(evaluation_rows) != 2:
        raise AssertionError("Smoke test evaluation row count is incorrect")
    if any(abs(float(row["reward_shaping_sum"])) > 1e-12 for row in evaluation_rows):
        raise AssertionError("Smoke test unexpectedly applied reward shaping")
    if sha256(source_path) != source_hash_before:
        raise AssertionError("Smoke test modified the immutable source model")
    manifest = {
        "status": "pass",
        "role": "checkpoint-continuation smoke only; excluded from scientific evidence",
        "source_model": str(source_path),
        "source_model_sha256": source_hash_before,
        "start_actual_timesteps": runtime_rows[0]["start_actual_timesteps"],
        "final_actual_timesteps": runtime_rows[0]["actual_model_timesteps"],
        "evaluation_rows": len(evaluation_rows),
        "source_hash_unchanged": True,
        "environment_state_restored": False,
        "shaping_applied": False,
    }
    (output_root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
