"""One-seed read-only V4 compatibility screen on PAIR0 standard slopes."""

from __future__ import annotations

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

import evaluate_fixed_standard_pair0_slope_capability_boundary as slope  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/v4_pair0_standard_slope_screen_v1_20260819"
V4 = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"


class V4Policy:
    def __init__(self) -> None:
        self.model = PPO.load(V4, device="cpu")
        self.num_timesteps = int(self.model.num_timesteps)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        return self.model.predict(np.asarray(observation)[:118], deterministic=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(slope.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    protocol, reward, _ = slope.validate_config(config)
    with tempfile.TemporaryDirectory(prefix="proxygap_v4_slopes_") as temp:
        scenes, _ = slope.prepare_scenes(config, protocol, Path(temp))
        rows = []
        for name in ("flat", "uphill_8deg", "downhill_8deg", "uphill_12deg"):
            row, _, _ = slope.l2.evaluate_episode(
                V4Policy(), config, protocol, reward, scenes[name],
                condition_id=slope.PAIR0_ID, seed=94131,
                checkpoint_additional_timesteps=0, max_episode_steps=600,
                retain_substeps=False,
            )
            rows.append(row)
    OUTPUT.mkdir(parents=True)
    payload = {"status": "exploratory_v4_pair0_slope_screen_complete", "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
