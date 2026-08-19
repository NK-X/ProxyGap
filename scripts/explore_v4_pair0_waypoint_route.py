"""Exploratory V4 route-following run on the frozen PAIR0 full map."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_map_waypoint_route as route_eval  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/v4_pair0_waypoint_route_screen_v1_20260819"
CHECKPOINT = ROOT / "artifacts/dev/curved_gait_tangent_v4_canonical_frame_20260818/runs/seed_43301/models/checkpoint_1024000.zip"
SCENE = ROOT / "artifacts/dev/post_seal_full_map_eval_v1_20260819/attempt_0/task_scenes/spawn_0_0.000_pair0.xml"


class V4RoutePolicy:
    def __init__(self) -> None:
        self.model = PPO.load(CHECKPOINT, device="cpu")
        self.num_timesteps = int(self.model.num_timesteps)
        self.observation_space = Box(-np.inf, np.inf, shape=(122,), dtype=np.float32)
        self.action_space = self.model.action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        return self.model.predict(np.asarray(observation, dtype=np.float32)[:118], deterministic=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    route_config = json.loads(route_eval.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / route_config["fixed_map_config"]).read_text(encoding="utf-8"))
    policy_config = json.loads((ROOT / "configs/curved_gait_tangent_v22_contact_observation_pilot_20260818.json").read_text(encoding="utf-8"))
    approved = fixed["approved_map"]
    heights = np.load(ROOT / approved["heights_path"], allow_pickle=False)
    settings = dict(route_config["route_reconstruction"])
    settings.update(
        {
            "maximum_abs_directional_slope_degrees": 16.0,
            "slope_cost_weight": 1.0,
            "turn_cost_m_per_rad": 1.0,
            "start_heading_degrees": 45.0,
            "line_of_sight_max_span_m": 8.0,
        }
    )
    planner = route_eval.RouteReconstructor(
        heights,
        half_extent=float(approved["map_half_extent_m"]),
        settings=settings,
    )
    waypoints, metrics = planner.plan(
        np.asarray(approved["start_xy_m"], dtype=np.float64),
        np.asarray(approved["goal_xy_m"], dtype=np.float64),
    )
    route = route_eval.Polyline(waypoints)
    spacing = 2.0 * float(approved["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    condition = {
        "condition_id": "V4_PAIR0_ROUTE_SCREEN",
        "route_mode": "waypoint_route",
        "initial_heading_degrees": 45.0,
        "cruise_speed_m_per_s": 0.45,
        "lookahead_m": 3.0,
        "maximum_abs_curvature_per_m": 0.2,
        "yaw_gain_per_second": 0.75,
        "high_level_speed_schedule": {
            "type": "maximum_current_and_lookahead_local_slope",
            "full_speed_below_degrees": 6.0,
            "minimum_speed_at_degrees": 16.0,
            "full_speed_m_per_s": 0.45,
            "minimum_speed_m_per_s": 0.28,
        },
    }
    local_settings = dict(route_config)
    local_settings["evaluation"] = dict(route_config["evaluation"])
    row, trace, audit = route_eval.evaluate_episode(
        fixed_config=fixed,
        policy_config=policy_config,
        model=V4RoutePolicy(),
        scene=SCENE,
        heights=heights,
        terrain_gradient_x=gradient_x,
        terrain_gradient_y=gradient_y,
        route=route,
        condition=condition,
        seed=1763594348,
        settings=local_settings,
    )
    OUTPUT.mkdir(parents=True)
    payload = {
        "status": "exploratory_v4_pair0_waypoint_route_complete",
        "route_settings": settings,
        "route_metrics": metrics,
        "waypoints": waypoints.tolist(),
        "episode": row,
        "audit": audit,
        "claim_boundary": "One reconstructed route on one seen map and seed; screening only.",
    }
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    route_eval.write_csv(OUTPUT / "trace.csv", trace)
    print(json.dumps({"status": payload["status"], "route": metrics, "episode": row}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
