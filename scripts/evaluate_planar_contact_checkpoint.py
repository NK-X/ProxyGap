"""Evaluate contact stability of a frozen planar-transition checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.planar_transition import make_planar_transition_env  # noqa: E402
from run_planar_translation_transition import environment_kwargs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=[61001, 61002, 61003]
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    *,
    model: PPO,
    config: dict[str, Any],
    seed: int,
    steps: int,
) -> dict[str, Any]:
    switch_step = int(config["commands"]["evaluation_switch_step"])
    if steps > switch_step:
        raise ValueError(
            "steps must not exceed the configured command-switch step; "
            "this evaluator measures the initial straight segment only"
        )
    env = make_planar_transition_env(
        condition_id="PLANAR_CONTACT_DIAGNOSTIC",
        seed=seed,
        max_episode_steps=steps,
        **environment_kwargs(config, evaluation=True),
    )
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
    summary = env.episode_summary()
    env.close()
    return {
        "evaluation_seed": seed,
        "requested_steps": steps,
        "full_horizon_completed": int(summary["episode_length"]) == steps,
        **summary,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = PPO.load(model_path, device=args.device)
    rows = [
        evaluate(model=model, config=config, seed=seed, steps=args.steps)
        for seed in args.evaluation_seeds
    ]
    payload = {
        "schema_version": 1,
        "status": "development_diagnostic",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "steps": args.steps,
        "evaluation_seeds": args.evaluation_seeds,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
