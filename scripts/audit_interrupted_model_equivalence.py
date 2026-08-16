"""Compare interrupted and restarted PPO checkpoints at tensor level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interrupted_root", required=True)
    parser.add_argument("--restarted_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        array = state[key].detach().cpu().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def relative_models(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("checkpoint_*.zip")
    }


def main() -> None:
    args = parse_args()
    interrupted_root = Path(args.interrupted_root).resolve()
    restarted_root = Path(args.restarted_root).resolve()
    output = Path(args.output).resolve()
    interrupted = relative_models(interrupted_root)
    restarted = relative_models(restarted_root)
    shared = sorted(set(interrupted) & set(restarted))
    if not shared:
        raise FileNotFoundError("No matching checkpoint paths were found")

    comparisons: list[dict[str, Any]] = []
    for relative_path in shared:
        path_a = interrupted[relative_path]
        path_b = restarted[relative_path]
        model_a = PPO.load(path_a, device="cpu")
        model_b = PPO.load(path_b, device="cpu")
        state_a = model_a.policy.state_dict()
        state_b = model_b.policy.state_dict()
        if set(state_a) != set(state_b):
            raise ValueError(f"State-dict keys differ for {relative_path}")
        maxima = {
            key: float(
                np.max(
                    np.abs(
                        state_a[key].detach().cpu().numpy()
                        - state_b[key].detach().cpu().numpy()
                    )
                )
            )
            for key in state_a
        }
        max_abs = max(maxima.values(), default=0.0)
        comparisons.append(
            {
                "relative_path": str(relative_path),
                "interrupted_zip_sha256": sha256(path_a),
                "restarted_zip_sha256": sha256(path_b),
                "zip_hash_equal": sha256(path_a) == sha256(path_b),
                "interrupted_policy_tensor_sha256": tensor_digest(state_a),
                "restarted_policy_tensor_sha256": tensor_digest(state_b),
                "policy_tensor_hash_equal": tensor_digest(state_a)
                == tensor_digest(state_b),
                "maximum_absolute_parameter_difference": max_abs,
                "interrupted_num_timesteps": int(model_a.num_timesteps),
                "restarted_num_timesteps": int(model_b.num_timesteps),
            }
        )

    result = {
        "status": "interruption_forensic_comparison_complete",
        "interrupted_root": str(interrupted_root),
        "restarted_root": str(restarted_root),
        "shared_checkpoint_count": len(shared),
        "all_policy_tensors_equal": all(
            item["policy_tensor_hash_equal"] for item in comparisons
        ),
        "all_zip_hashes_equal": all(item["zip_hash_equal"] for item in comparisons),
        "comparisons": comparisons,
        "interpretation_rule": (
            "A differing ZIP hash alone is not evidence of a differing policy. "
            "Tensor equality is assessed separately."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
