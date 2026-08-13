"""Render representative 300k trajectories for the three core conditions."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

from PIL import Image
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "formal" / "combined_v1_20260809" / "videos"
EVALUATION_SEED = 30260808
MAX_STEPS = 1000
FRAME_STRIDE = 4

CONDITIONS = [
    {
        "condition_id": "reference",
        "ctrl_cost_weight": 0.5,
        "forward_progress_shaping_weight": 0.0,
        "model": PROJECT_ROOT
        / "artifacts/formal/formal_v1_coefficients_20260808/runs/seed_20260808/reference/models/reference/checkpoint_300000.zip",
    },
    {
        "condition_id": "ctrl_0p0625",
        "ctrl_cost_weight": 0.0625,
        "forward_progress_shaping_weight": 0.0,
        "model": PROJECT_ROOT
        / "artifacts/formal/formal_v1_coefficients_20260808/runs/seed_20260808/ctrl_0p0625/models/ctrl_0p0625/checkpoint_300000.zip",
    },
    {
        "condition_id": "shaped_ctrl_0p0625_forward_1p0",
        "ctrl_cost_weight": 0.0625,
        "forward_progress_shaping_weight": 1.0,
        "model": PROJECT_ROOT
        / "artifacts/formal/formal_v1_shaped_20260808/runs/seed_20260808/shaped_ctrl_0p0625_forward_1p0/models/shaped_ctrl_0p0625_forward_1p0/checkpoint_300000.zip",
    },
]


def make_json_safe(value):
    """Replace non-finite floats with JSON-compatible null values."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for condition in CONDITIONS:
        env = make_proxygap_ant_env(
            ctrl_cost_weight=condition["ctrl_cost_weight"],
            condition_id=condition["condition_id"],
            forward_progress_shaping_weight=condition[
                "forward_progress_shaping_weight"
            ],
            seed=EVALUATION_SEED,
            render_mode="rgb_array",
            max_episode_steps=MAX_STEPS,
        )
        model = PPO.load(condition["model"], env=env, device="cpu")
        observation, _ = env.reset(seed=EVALUATION_SEED)
        frames = []
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            if step % FRAME_STRIDE == 0:
                frames.append(Image.fromarray(env.render()))
            step += 1
        summary = env.episode_summary()
        summary["model_path"] = str(condition["model"])
        summary["evaluation_seed"] = EVALUATION_SEED
        summaries.append(summary)
        env.close()

        output = OUTPUT_ROOT / f"{condition['condition_id']}_300k.gif"
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            optimize=False,
        )
        print(f"Saved: {output}")

    (OUTPUT_ROOT / "representative_trajectory_metrics.json").write_text(
        json.dumps(make_json_safe(summaries), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
