"""Evaluate an existing curved-gait checkpoint without retraining it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.experiment import write_rows  # noqa: E402
from run_curved_gait_training import (  # noqa: E402
    evaluate,
    final_selection_summary,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint-timesteps", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing evaluation: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, require_local_base=True)
    model = PPO.load(model_path, device=args.device)
    rows = evaluate(
        model,
        config,
        training_seed=args.training_seed,
        checkpoint_timesteps=args.checkpoint_timesteps,
        smoke=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_path, rows)
    record = {
        "status": "complete",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "training_seed": args.training_seed,
        "checkpoint_timesteps": args.checkpoint_timesteps,
        "device": args.device,
        "selection_summary": final_selection_summary(rows),
        "metrics_path": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
