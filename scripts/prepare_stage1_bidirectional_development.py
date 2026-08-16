"""Prepare the immutable execution record for the upper-side development run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import save_run_config  # noqa: E402
from proxygap.stage1 import validate_stage1_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_run_config(
    protocol_config: dict[str, Any],
    *,
    config_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    development = protocol_config["development"]
    return {
        "schema_version": 2,
        "run_id": "stage1_upper_development_300k_20260814",
        "status": "development_only_not_formal_confirmation",
        "stage_scope": "stage_1_detection_only",
        "protocol_config": str(config_path),
        "protocol_config_sha256": sha256(config_path),
        "protocol_document": str(protocol_path),
        "protocol_document_sha256": sha256(protocol_path),
        "environment": protocol_config["environment"],
        "algorithm": protocol_config["algorithm"],
        "device": protocol_config["device"],
        "ctrl_cost_weights": development["new_upper_weights"],
        "training_seeds": development["training_seeds"],
        "evaluation_seed_base": development["evaluation_seed_base"],
        "eval_episodes_per_checkpoint": development[
            "evaluation_episodes_per_checkpoint"
        ],
        "eval_max_episode_steps": 1000,
        "timesteps_per_condition": development["total_timesteps"],
        "checkpoint_timesteps": development["checkpoint_timesteps"],
        "task_order_seed": 8142027,
        "ppo": protocol_config["ppo"],
        "reward_constants": {
            **protocol_config["reward"],
            "all_shaping_weights": 0.0,
        },
        "proxy_relative_noninferiority_margin": protocol_config[
            "development_screen"
        ]["proxy_relative_noninferiority_margin"],
        "claim_boundary": (
            "Upper-side development evidence only; it cannot confirm the "
            "hypothesis or authorise shaping."
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_stage1_config(config)
    if errors:
        raise ValueError(f"Stage-one config errors: {errors}")

    protocol_path = PROJECT_ROOT / config["protocol_document"]
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else PROJECT_ROOT / config["development"]["upper_output_root"]
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")

    run_config = build_run_config(
        config,
        config_path=config_path,
        protocol_path=protocol_path,
    )
    save_run_config(output_root, run_config)
    print(json.dumps(run_config, indent=2), flush=True)


if __name__ == "__main__":
    main()
