"""Print local Ant-v5 reference settings."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym


def main() -> None:
    env = gym.make("Ant-v5")
    ant = env.unwrapped
    data = {
        "environment": env.spec.id,
        "ctrl_cost_weight": ant._ctrl_cost_weight,
        "forward_reward_weight": ant._forward_reward_weight,
        "healthy_reward": ant._healthy_reward,
        "terminate_when_unhealthy": ant._terminate_when_unhealthy,
        "healthy_z_range": list(ant._healthy_z_range),
        "contact_cost_weight": ant._contact_cost_weight,
        "frame_skip": ant.frame_skip,
        "dt": ant.dt,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
    }
    env.close()
    print(json.dumps(data, indent=2))
    output = Path("artifacts/metadata/ant_v5_reference_observed.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
