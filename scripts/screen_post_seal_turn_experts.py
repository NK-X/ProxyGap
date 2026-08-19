"""Read-only short screen of archived locomotion checkpoints as turn experts.

This is an exploratory engineering screen, not a promotion experiment.  Every
candidate is evaluated in the same PAIR0 numerical-flat environment.  Legacy
policies receive the exact leading observation columns of their native space;
no weights are changed and no checkpoint is written.
"""

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


OUTPUT = ROOT / "artifacts/dev/post_seal_turn_expert_screen_v1_20260819"
SEED = 96181
HORIZON = 240

CANDIDATES = (
    ("v4_curve_expert", "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"),
    ("v22_contact_curve", "artifacts/dev/curved_gait_tangent_v22_contact_observation_pilot_20260818/runs/seed_43812/models/checkpoint_2203648.zip"),
    ("local_preview_final", "artifacts/dev/fixed_quad_terrain_v2_local_preview_pilot_v1_20260819/seed_62802/models/checkpoint_2727936.zip"),
    ("pair0_final", "artifacts/dev/pair0_l2b_v3_20260819/attempt_0/pair0_adapt/models/checkpoint_2727936.zip"),
    ("v5_c0", "artifacts/dev/tb_v5_20260819/a0/c0_straight_continue/models/checkpoint_2793472.zip"),
    ("v5_c1", "artifacts/dev/tb_v5_20260819/a0/c1_balanced_turn/models/checkpoint_2793472.zip"),
)

CONDITION_NAMES = (
    "straight_055",
    "curve_left_010",
    "curve_right_010",
    "curve_left_020",
    "curve_right_020",
)


class NativeObservationPolicy:
    def __init__(self, checkpoint: Path) -> None:
        self.model = PPO.load(checkpoint, device="cpu")
        self.dimension = int(self.model.observation_space.shape[0])
        self.num_timesteps = int(self.model.num_timesteps)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (135,):
            raise RuntimeError(f"Screen observation changed shape: {vector.shape}")
        return self.model.predict(vector[: self.dimension], deterministic=deterministic)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    protocol, reward, _ = diagnostic.validate_config(config)
    screen_config = copy.deepcopy(config)
    screen_config["evaluation"]["max_episode_steps"] = HORIZON
    conditions = {
        row["condition_name"]: row for row in diagnostic.condition_specs(screen_config)
    }
    with tempfile.TemporaryDirectory(prefix="proxygap_turn_screen_") as temp:
        scene = diagnostic.prepare_flat_scene(
            screen_config, protocol, Path(temp)
        )
        rows: list[dict[str, Any]] = []
        for candidate_id, relative_path in CANDIDATES:
            checkpoint = ROOT / relative_path
            policy = NativeObservationPolicy(checkpoint)
            for condition_name in CONDITION_NAMES:
                row, _ = diagnostic.evaluate_episode(
                    policy,
                    screen_config,
                    protocol,
                    reward,
                    scene,
                    conditions[condition_name],
                    SEED,
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "checkpoint": relative_path,
                        "observation_dimension": policy.dimension,
                        "condition": condition_name,
                        "target_yaw_rad": row["target_cumulative_yaw_change_rad"],
                        "actual_yaw_rad": row["actual_cumulative_yaw_change_rad"],
                        "yaw_ratio": row["yaw_change_target_ratio"],
                        "signed_progress_m": row["signed_initial_heading_progress_m"],
                        "fall": row["fall"],
                        "torso": row["torso_ground_any"],
                        "sustained_nonfoot": row["sustained_nonfoot_contact"],
                        "corrected_slip_events": row["corrected_slip_event_count"],
                        "full_interval_zero_foot_fraction": row[
                            "full_interval_zero_foot_count"
                        ] / row["control_steps"],
                    }
                )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": "exploratory_read_only_screen_complete",
        "seed": SEED,
        "horizon_control_steps": HORIZON,
        "claim_boundary": "One-seed short engineering screen; not a promotion or robustness result.",
        "rows": rows,
    }
    (OUTPUT / "screen_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
