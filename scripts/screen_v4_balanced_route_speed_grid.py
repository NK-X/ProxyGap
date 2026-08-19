"""Fine speed grid on the safest balanced V4 route."""

from __future__ import annotations

import csv
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

SOURCE = ROOT / "artifacts/dev/v4_pair0_multiobjective_routes_engineering_v1_20260819/balanced/route_waypoints.csv"
OUTPUT = ROOT / "artifacts/dev/v4_balanced_route_speed_grid_v1_20260819"
SPEEDS = (0.38, 0.42, 0.48, 0.50)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    route = route_eval.Polyline(np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows]))
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    heights = np.load(ROOT / fixed["approved_map"]["heights_path"], allow_pickle=False)
    spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    results = []
    for speed in SPEEDS:
        candidate = f"balanced_speed_{speed:.2f}".replace(".", "p")
        regime = {"id": candidate, "weights": None, "speed": speed, "minimum_speed": min(0.28, speed)}
        result, controls, substeps = engineering.evaluate_route(
            canonical_config=config, fixed=fixed, route=route, regime=regime,
            heights=heights, gradient_x=gradient_x, gradient_y=gradient_y,
            seed=int(config["evaluation"]["formal_seed"]),
        )
        results.append(result)
        root = OUTPUT / candidate
        root.mkdir(parents=True, exist_ok=False)
        engineering.write_csv(root / "control_trace.csv", controls)
        engineering.write_csv(root / "substep_trace.csv", substeps)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    (OUTPUT / "summary.json").write_text(json.dumps({"status": "speed_grid_complete", "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
