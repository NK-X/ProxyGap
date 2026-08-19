"""Select safety-feasible route candidates under three frozen weightings."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/dev/v4_multiobjective_candidate_selection_v1_20260820"
SOURCES = (
    ROOT / "artifacts/dev/v4_pair0_multiobjective_routes_engineering_v1_20260819/summary.json",
    ROOT / "artifacts/dev/v4_route_speed_candidate_screen_v1_20260819/summary.json",
    ROOT / "artifacts/dev/v4_balanced_route_speed_grid_v1_20260819/summary.json",
    ROOT / "artifacts/dev/v4_route_cost_grid_v1_20260819/summary.json",
)
OBJECTIVES = {
    "time_priority": (0.8, 0.2),
    "balanced": (0.5, 0.5),
    "energy_priority": (0.2, 0.8),
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    candidates = []
    for source in SOURCES:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for row in payload["results"]:
            record = dict(row)
            record["source_summary"] = source.relative_to(ROOT).as_posix()
            candidates.append(record)
    feasible = [
        row for row in candidates
        if row["safety_qualified_completion"]
        and row["duration_corrected_slip_event_count"] == 0
        and not row["fall"]
    ]
    time_reference = min(float(row["elapsed_seconds"]) for row in feasible)
    energy_reference = min(float(row["actuator_positive_mechanical_work_total_j"]) for row in feasible)
    scored = []
    for row in feasible:
        item = dict(row)
        item["normalised_time"] = float(row["elapsed_seconds"]) / time_reference
        item["normalised_positive_mechanical_work"] = float(row["actuator_positive_mechanical_work_total_j"]) / energy_reference
        item["objective_scores"] = {
            name: wt * item["normalised_time"] + we * item["normalised_positive_mechanical_work"]
            for name, (wt, we) in OBJECTIVES.items()
        }
        scored.append(item)
    selections = {}
    for name, weights in OBJECTIVES.items():
        selected = min(scored, key=lambda row: (row["objective_scores"][name], row["regime"]))
        selections[name] = {
            "weights_time_energy": list(weights),
            "selected_candidate": selected["regime"],
            "source_summary": selected["source_summary"],
            "score": selected["objective_scores"][name],
            "elapsed_seconds": selected["elapsed_seconds"],
            "positive_mechanical_work_j": selected["actuator_positive_mechanical_work_total_j"],
            "path_length_m": selected["path_length_m"],
            "minimum_goal_distance_m": selected["minimum_goal_distance_m"],
            "slip_events": selected["duration_corrected_slip_event_count"],
        }
    payload = {
        "status": "multiobjective_selection_complete",
        "normalisation": {
            "time_reference_seconds": time_reference,
            "positive_mechanical_work_reference_j": energy_reference,
            "formula": "w_time*(T/T_min)+w_energy*(W_positive/W_positive_min)",
            "energy_claim_boundary": "mechanical-work proxy, not electrical battery energy",
        },
        "feasible_candidate_count": len(feasible),
        "objectives": OBJECTIVES,
        "selections": selections,
        "scored_candidates": scored,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "selection.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "feasible": len(feasible), "selections": selections}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
