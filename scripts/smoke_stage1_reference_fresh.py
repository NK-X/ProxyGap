"""Run a bounded V6 engineering smoke test before the 1M reference run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)
from proxygap.reference_baseline import numeric, validate_reference_config  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "smoke" / "stage1_reference_fresh_v6_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    validate_reference_config(config)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Smoke output already exists: {output_root}")
        shutil.rmtree(output_root)

    ppo = config["ppo"]
    runtime_rows, evaluation_rows = train_condition(
        output_root=output_root,
        condition_id="reference_smoke",
        ctrl_cost_weight=0.5,
        total_timesteps=4096,
        checkpoint_timesteps=[2048, 4096],
        seed=61201,
        evaluation_seed_base=61211,
        eval_episodes=2,
        eval_max_episode_steps=1000,
        ppo_n_steps=int(ppo["n_steps"]),
        ppo_batch_size=int(ppo["batch_size"]),
        ppo_n_epochs=int(ppo["n_epochs"]),
        ppo_learning_rate=float(ppo["learning_rate"]),
        ppo_gamma=float(ppo["gamma"]),
        ppo_gae_lambda=float(ppo["gae_lambda"]),
        ppo_clip_range=float(ppo["clip_range"]),
        ppo_ent_coef=float(ppo["ent_coef"]),
        ppo_vf_coef=float(ppo["vf_coef"]),
        ppo_max_grad_norm=float(ppo["max_grad_norm"]),
        ppo_normalize_advantage=bool(ppo["normalize_advantage"]),
        ppo_policy=str(ppo["policy"]),
        ppo_policy_kwargs=dict(ppo["policy_kwargs"]),
        ppo_device=str(config["device"]),
        ppo_torch_num_threads=int(ppo["torch_num_threads"]),
    )
    write_standard_outputs(
        output_root,
        runtime_rows=runtime_rows,
        eval_rows=evaluation_rows,
        summary_rows=summarise_evaluation(evaluation_rows),
    )

    models = sorted((output_root / "models").rglob("checkpoint_*.zip"))
    if len(runtime_rows) != 2 or len(evaluation_rows) != 4 or len(models) != 2:
        raise AssertionError("Smoke artifact counts do not match the 2-checkpoint contract")
    max_base_error = max(abs(numeric(row["base_reward_reconciliation_error"])) for row in evaluation_rows)
    max_ctrl_error = max(abs(numeric(row["ctrl_cost_reconciliation_error"])) for row in evaluation_rows)
    max_shaping = max(abs(numeric(row["reward_shaping_sum"])) for row in evaluation_rows)
    if not all(math.isfinite(value) for value in (max_base_error, max_ctrl_error)):
        raise AssertionError("Smoke reward reconciliation is non-finite")
    if max_base_error > 1e-3 or max_ctrl_error > 1e-3 or max_shaping > 1e-12:
        raise AssertionError("Smoke reward or shaping contract failed")

    manifest = {
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "engineering smoke only; no scientific inference",
        "training_seed": 61201,
        "evaluation_seeds": [61211, 61212],
        "runtime_rows": len(runtime_rows),
        "evaluation_rows": len(evaluation_rows),
        "models": [
            {"path": str(path), "sha256": sha256(path)} for path in models
        ],
        "max_abs_base_reward_reconciliation_error": max_base_error,
        "max_abs_ctrl_cost_reconciliation_error": max_ctrl_error,
        "max_abs_shaping_sum": max_shaping,
    }
    (output_root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
