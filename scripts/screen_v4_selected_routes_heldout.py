"""Held-out three-seed screen of the two selected route contracts."""

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

OUTPUT = ROOT / "artifacts/dev/v4_selected_routes_heldout_screen_v1_20260820"
SEEDS = (1305785918, 172486292, 696250711)
CONTRACTS = (
    ("time_and_balanced", ROOT / "artifacts/dev/v4_route_cost_grid_v1_20260819/s1p50_t1p00/route_waypoints.csv"),
    ("energy_priority", ROOT / "artifacts/dev/v4_pair0_multiobjective_routes_engineering_v1_20260819/balanced/route_waypoints.csv"),
)


def load_route(path: Path) -> route_eval.Polyline:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return route_eval.Polyline(np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows]))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(full_map.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8"))
    heights = np.load(ROOT / fixed["approved_map"]["heights_path"], allow_pickle=False)
    spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
    results = []
    for contract_id, route_path in CONTRACTS:
        route = load_route(route_path)
        for seed in SEEDS:
            regime = {"id": contract_id, "weights": None, "speed": 0.50, "minimum_speed": 0.28}
            result, controls, substeps = engineering.evaluate_route(
                canonical_config=config, fixed=fixed, route=route, regime=regime,
                heights=heights, gradient_x=gradient_x, gradient_y=gradient_y, seed=seed,
            )
            result["evaluation_seed"] = seed
            result["route_path"] = route_path.relative_to(ROOT).as_posix()
            results.append(result)
            root = OUTPUT / contract_id / f"seed_{seed}"
            root.mkdir(parents=True, exist_ok=False)
            engineering.write_csv(root / "control_trace.csv", controls)
            engineering.write_csv(root / "substep_trace.csv", substeps)
            (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
    (OUTPUT / "summary.json").write_text(json.dumps({"status": "heldout_screen_complete", "all_passed": all(r["safety_qualified_completion"] for r in results), "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
