"""Exploratory direct-goal run of a directional PAIR0/V4 policy mixture."""

from __future__ import annotations

import json
import math
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

OUTPUT = ROOT / "artifacts/dev/post_seal_full_map_directional_hybrid_v1_20260819"
PAIR0 = ROOT / "artifacts/dev/pair0_l2b_v3_20260819/attempt_0/pair0_adapt/models/checkpoint_2727936.zip"
V4 = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"


class DirectionalHybridPolicy:
    """Use PAIR0 normally and blend V4 only for positive heading correction."""

    def __init__(self) -> None:
        self.pair0 = PPO.load(PAIR0, device="cpu")
        self.v4 = PPO.load(V4, device="cpu")
        self.num_timesteps = int(self.pair0.num_timesteps)
        self.observation_space = Box(-np.inf, np.inf, shape=(135,), dtype=np.float32)
        self.action_space = self.pair0.action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        vector = np.asarray(observation, dtype=np.float32)
        pair0_action, _ = self.pair0.predict(vector, deterministic=True)
        heading_error = math.atan2(float(vector[116]), float(vector[117]))
        if heading_error <= 0.04:
            return np.asarray(pair0_action), None
        v4_action, _ = self.v4.predict(vector[:118], deterministic=True)
        alpha = min(0.85, 0.55 + 1.5 * heading_error)
        action = (1.0 - alpha) * np.asarray(pair0_action) + alpha * np.asarray(v4_action)
        return np.clip(action, -1.0, 1.0), None


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    seed = int(config["evaluation"]["formal_seed"])
    with tempfile.TemporaryDirectory(prefix="proxygap_hybrid_map_") as temp:
        scene, audit, _ = full_map.prepare_pair0_scene(config, fixed, Path(temp))
        row, control, _, events = full_map.evaluate_episode(
            config=config,
            fixed=fixed,
            model=DirectionalHybridPolicy(),
            scene=scene,
            seed=seed,
            horizon=int(config["evaluation"]["horizon_control_steps"]),
            mode="exploratory_directional_pair0_v4_hybrid_direct_goal",
        )
    OUTPUT.mkdir(parents=True)
    payload = {
        "status": "exploratory_directional_hybrid_complete",
        "policy_rule": "PAIR0 unless heading error >0.04 rad; then smooth V4 blend alpha=min(0.85,0.55+1.5*error)",
        "pair0_audit": audit,
        "episode": row,
        "corrected_slip_events": events,
        "claim_boundary": "One seen map and one seed; screening only.",
    }
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    full_map.write_csv(OUTPUT / "control_trace.csv", control)
    print(json.dumps({"status": payload["status"], "episode": row}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
