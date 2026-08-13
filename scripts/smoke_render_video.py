"""Render a very short Ant-v5 GIF to validate the local video pathway."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap import make_proxygap_ant_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", default=None, help="Example: glfw")
    parser.add_argument("--output", default="artifacts/videos/ant_v5_render_smoke.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        os.environ["MUJOCO_GL"] = args.backend

    env = make_proxygap_ant_env(
        ctrl_cost_weight=0.5,
        condition_id="render_smoke_reference",
        seed=args.seed,
        render_mode="rgb_array",
    )
    frames = []
    env.reset(seed=args.seed)
    for _ in range(args.frames):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))
        if terminated or truncated:
            env.reset()
    env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
