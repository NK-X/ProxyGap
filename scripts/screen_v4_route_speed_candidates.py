"""Evaluate additional speed schedules on frozen multi-objective routes."""

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

SOURCE = ROOT / "artifacts/dev/v4_pair0_multiobjective_routes_engineering_v1_20260819"
OUTPUT = ROOT / "artifacts/dev/v4_route_speed_candidate_screen_v1_20260819"

CANDIDATES = (
    ("time_route_fast", "time_priority", 0.55, 0.32),
    ("balanced_route_fast", "balanced", 0.55, 0.32),
    ("balanced_route_economy", "balanced", 0.32, 0.22),
    ("energy_route_economy", "energy_priority", 0.32, 0.22),
)


def read_route(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float64)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    heights = np.load(ROOT / fixed["approved_map"]["heights_path"], allow_pickle=False)
    spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    seed = int(config["evaluation"]["formal_seed"])
    results = []
    for candidate_id, route_id, speed, minimum_speed in CANDIDATES:
        points = read_route(SOURCE / route_id / "route_waypoints.csv")
        regime = {"id": candidate_id, "weights": None, "speed": speed, "minimum_speed": minimum_speed}
        result, controls, substeps = engineering.evaluate_route(
            canonical_config=config, fixed=fixed, route=route_eval.Polyline(points),
            regime=regime, heights=heights, gradient_x=gradient_x,
            gradient_y=gradient_y, seed=seed,
        )
        result["source_route_id"] = route_id
        results.append(result)
        root = OUTPUT / candidate_id
        root.mkdir(parents=True, exist_ok=False)
        engineering.write_csv(root / "control_trace.csv", controls)
        engineering.write_csv(root / "substep_trace.csv", substeps)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    (OUTPUT / "summary.json").write_text(json.dumps({"status": "candidate_screen_complete", "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
