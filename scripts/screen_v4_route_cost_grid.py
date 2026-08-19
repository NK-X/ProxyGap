"""Intermediate route-cost grid at the retained 0.50 m/s schedule."""

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
import evaluate_post_seal_full_map_v1 as full_map  # noqa: E402
import evaluate_v4_pair0_multiobjective_routes_engineering as engineering  # noqa: E402

OUTPUT = ROOT / "artifacts/dev/v4_route_cost_grid_v1_20260819"
GRID = ((0.50, 0.75), (0.75, 1.00), (1.50, 1.00), (2.00, 1.25))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    heights = np.load(ROOT / fixed["approved_map"]["heights_path"], allow_pickle=False)
    spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    start = np.asarray(fixed["approved_map"]["start_xy_m"])
    goal = np.asarray(fixed["approved_map"]["goal_xy_m"])
    results = []
    for slope_weight, turn_weight in GRID:
        candidate = f"s{slope_weight:.2f}_t{turn_weight:.2f}".replace(".", "p")
        settings = json.loads(route_eval.DEFAULT_CONFIG.read_text(encoding="utf-8"))["route_reconstruction"]
        settings.update({
            "maximum_abs_directional_slope_degrees": 16.0,
            "slope_cost_weight": slope_weight,
            "turn_cost_m_per_rad": turn_weight,
            "start_heading_degrees": 45.0,
            "line_of_sight_max_span_m": 8.0,
        })
        planner = route_eval.RouteReconstructor(heights, half_extent=float(fixed["approved_map"]["map_half_extent_m"]), settings=settings)
        points, metrics = planner.plan(start, goal)
        regime = {"id": candidate, "weights": None, "speed": 0.50, "minimum_speed": 0.28}
        result, controls, substeps = engineering.evaluate_route(
            canonical_config=config, fixed=fixed, route=route_eval.Polyline(points),
            regime=regime, heights=heights, gradient_x=gradient_x,
            gradient_y=gradient_y, seed=int(config["evaluation"]["formal_seed"]),
        )
        result["route_metrics"] = metrics
        result["route_cost_settings"] = {"slope_weight": slope_weight, "turn_weight": turn_weight}
        results.append(result)
        root = OUTPUT / candidate
        root.mkdir(parents=True, exist_ok=False)
        engineering.write_csv(root / "route_waypoints.csv", [{"index": i, "x_m": p[0], "y_m": p[1]} for i, p in enumerate(points)])
        engineering.write_csv(root / "control_trace.csv", controls)
        engineering.write_csv(root / "substep_trace.csv", substeps)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    (OUTPUT / "summary.json").write_text(json.dumps({"status": "route_cost_grid_complete", "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
