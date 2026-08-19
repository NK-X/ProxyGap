"""Exploratory full-map compatibility run for the archived V4 turn expert.

The policy is never trained or saved.  It receives the leading 118 columns of
the current 135D observation, which exactly match its native observation
contract.  The environment, PAIR0 contact contract, map and success audit are
the post-seal formal contracts.  This one-seed result is a screening result,
not a retained final evaluation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_post_seal_full_map_v1 as full_map  # noqa: E402


OUTPUT = ROOT / "artifacts/dev/post_seal_full_map_v4_expert_screen_v1_20260819"
CHECKPOINT = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"


class V4CompatibilityPolicy:
    def __init__(self) -> None:
        self.model = PPO.load(CHECKPOINT, device="cpu")
        self.model.policy.set_training_mode(False)
        self.num_timesteps = int(self.model.num_timesteps)
        self.observation_space = Box(-np.inf, np.inf, shape=(135,), dtype=np.float32)
        self.action_space = self.model.action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (135,):
            raise RuntimeError(f"V4 compatibility observation changed: {vector.shape}")
        return self.model.predict(vector[:118], deterministic=deterministic)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    # The canonical validator intentionally requires its own exact import
    # closure, so this additional screening driver cannot call it in-process.
    # The formal config and fixed-map declaration are read unchanged here; the
    # prepared scene still revalidates the frozen asset and PAIR0 hashes.
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads(
        (ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8")
    )
    seed = int(config["evaluation"]["formal_seed"])
    horizon = int(config["evaluation"]["horizon_control_steps"])
    with tempfile.TemporaryDirectory(prefix="proxygap_v4_map_") as temp:
        scene, audit, _ = full_map.prepare_pair0_scene(config, fixed, Path(temp))
        row, control, _, events = full_map.evaluate_episode(
            config=config,
            fixed=fixed,
            model=V4CompatibilityPolicy(),
            scene=scene,
            seed=seed,
            horizon=horizon,
            mode="exploratory_v4_turn_expert_direct_goal",
        )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": "exploratory_read_only_v4_full_map_screen_complete",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "seed": seed,
        "pair0_audit": audit,
        "claim_boundary": "One archived policy, one seen map and one seed; no training, selection or final claim.",
        "episode": row,
        "corrected_slip_events": events,
    }
    (OUTPUT / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    full_map.write_csv(OUTPUT / "control_trace.csv", control)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
