"""Short PAIR0-flat screen of smooth PAIR0/V4 action blending."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_standard_pair0_flat_turn_diagnostic as diagnostic  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/post_seal_hybrid_turn_blend_screen_v1_20260819"
PAIR0 = ROOT / "artifacts/dev/pair0_l2b_v3_20260819/attempt_0/pair0_adapt/models/checkpoint_2727936.zip"
V4 = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"


class BlendPolicy:
    def __init__(self, alpha: float) -> None:
        self.pair0 = PPO.load(PAIR0, device="cpu")
        self.v4 = PPO.load(V4, device="cpu")
        self.alpha = float(alpha)
        self.num_timesteps = int(self.pair0.num_timesteps)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        vector = np.asarray(observation, dtype=np.float32)
        pair0_action, _ = self.pair0.predict(vector, deterministic=True)
        v4_action, _ = self.v4.predict(vector[:118], deterministic=True)
        action = (1.0 - self.alpha) * np.asarray(pair0_action) + self.alpha * np.asarray(v4_action)
        return np.clip(action, -1.0, 1.0), None


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    protocol, reward, _ = diagnostic.validate_config(config)
    local = copy.deepcopy(config)
    local["evaluation"]["max_episode_steps"] = 240
    conditions = {row["condition_name"]: row for row in diagnostic.condition_specs(local)}
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="proxygap_blend_") as temp:
        scene = diagnostic.prepare_flat_scene(local, protocol, Path(temp))
        for alpha in (0.25, 0.5, 0.75):
            policy = BlendPolicy(alpha)
            for condition_name in ("straight_055", "curve_left_010", "curve_right_010"):
                row, _ = diagnostic.evaluate_episode(
                    policy, local, protocol, reward, scene, conditions[condition_name], 96187
                )
                rows.append(
                    {
                        "alpha_v4": alpha,
                        "condition": condition_name,
                        "actual_yaw_rad": row["actual_cumulative_yaw_change_rad"],
                        "target_yaw_rad": row["target_cumulative_yaw_change_rad"],
                        "yaw_ratio": row["yaw_change_target_ratio"],
                        "progress_m": row["signed_initial_heading_progress_m"],
                        "fall": row["fall"],
                        "zero_foot_fraction": row["full_interval_zero_foot_count"] / row["control_steps"],
                        "slip_events": row["corrected_slip_event_count"],
                    }
                )
    OUTPUT.mkdir(parents=True)
    payload = {"status": "exploratory_blend_screen_complete", "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
