"""Small one-seed screen of V4 waypoint-follower parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_map_waypoint_route as route_eval  # noqa: E402
from explore_v4_pair0_waypoint_route import SCENE, V4RoutePolicy  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/v4_pair0_waypoint_follower_matrix_v1_20260819"

CONDITIONS = (
    ("L100_V040_K025_Y100", 1.0, 0.40, 0.25, 1.00),
    ("L050_V040_K025_Y100", 0.5, 0.40, 0.25, 1.00),
    ("L100_V035_K020_Y075", 1.0, 0.35, 0.20, 0.75),
    ("L050_V035_K025_Y100", 0.5, 0.35, 0.25, 1.00),
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(route_eval.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map_config"]).read_text(encoding="utf-8"))
    policy_config = json.loads((ROOT / "configs/curved_gait_tangent_v22_contact_observation_pilot_20260818.json").read_text(encoding="utf-8"))
    approved = fixed["approved_map"]
    heights = np.load(ROOT / approved["heights_path"], allow_pickle=False)
    planner_settings = dict(config["route_reconstruction"])
    planner_settings.update({
        "maximum_abs_directional_slope_degrees": 16.0,
        "slope_cost_weight": 1.0,
        "turn_cost_m_per_rad": 1.0,
        "start_heading_degrees": 45.0,
        "line_of_sight_max_span_m": 8.0,
    })
    planner = route_eval.RouteReconstructor(heights, half_extent=float(approved["map_half_extent_m"]), settings=planner_settings)
    waypoints, metrics = planner.plan(np.asarray(approved["start_xy_m"]), np.asarray(approved["goal_xy_m"]))
    route = route_eval.Polyline(waypoints)
    spacing = 2.0 * float(approved["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    rows = []
    for name, lookahead, speed, curvature, yaw_gain in CONDITIONS:
        condition = {
            "condition_id": name,
            "route_mode": "waypoint_route",
            "initial_heading_degrees": 45.0,
            "cruise_speed_m_per_s": speed,
            "lookahead_m": lookahead,
            "maximum_abs_curvature_per_m": curvature,
            "yaw_gain_per_second": yaw_gain,
            "high_level_speed_schedule": {
                "type": "maximum_current_and_lookahead_local_slope",
                "full_speed_below_degrees": 6.0,
                "minimum_speed_at_degrees": 16.0,
                "full_speed_m_per_s": speed,
                "minimum_speed_m_per_s": min(0.28, speed),
            },
        }
        row, trace, audit = route_eval.evaluate_episode(
            fixed_config=fixed, policy_config=policy_config, model=V4RoutePolicy(),
            scene=SCENE, heights=heights, terrain_gradient_x=gradient_x,
            terrain_gradient_y=gradient_y, route=route, condition=condition,
            seed=1763594348, settings=config,
        )
        rows.append({"condition": condition, "episode": row, "audit": audit})
        (OUTPUT / name).mkdir(parents=True, exist_ok=False)
        route_eval.write_csv(OUTPUT / name / "trace.csv", trace)
    payload = {"status": "exploratory_follower_matrix_complete", "route_metrics": metrics, "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "route_metrics": metrics, "episodes": [row["episode"] for row in rows]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
