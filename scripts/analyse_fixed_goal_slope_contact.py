"""Relate four-foot no-contact events to local terrain slope in one video trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_training_20260818"
    / "seed_62801"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--evaluation-seed", type=int, default=74803)
    return parser.parse_args()


def sample_grid(array: np.ndarray, x: float, y: float, extent: float) -> float:
    rows, columns = array.shape
    column_f = np.clip((x + extent) / (2.0 * extent) * (columns - 1), 0, columns - 1)
    row_f = np.clip((y + extent) / (2.0 * extent) * (rows - 1), 0, rows - 1)
    column0 = min(int(math.floor(column_f)), columns - 2)
    row0 = min(int(math.floor(row_f)), rows - 2)
    tx = float(column_f - column0)
    ty = float(row_f - row0)
    return float(
        (1.0 - ty)
        * ((1.0 - tx) * array[row0, column0] + tx * array[row0, column0 + 1])
        + ty
        * ((1.0 - tx) * array[row0 + 1, column0] + tx * array[row0 + 1, column0 + 1])
    )


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    config = json.loads((run_root / "frozen_run_config.json").read_text(encoding="utf-8"))
    trace_dir = run_root / "videos" / f"representative_seed_{args.evaluation_seed}"
    trace_path = trace_dir / f"fixed_map_final_policy_seed_{args.evaluation_seed}_trace.csv"
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    approved = config["approved_map"]
    heights = np.load(ROOT / approved["heights_path"], allow_pickle=False)
    extent = float(approved["map_half_extent_m"])
    spacing = 2.0 * extent / (heights.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(heights, spacing, spacing)

    x = np.asarray([float(row["x_m"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["y_m"]) for row in rows], dtype=np.float64)
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=np.float64)
    airborne = np.asarray([row["airborne"].strip().lower() == "true" for row in rows])
    gx = np.asarray(
        [sample_grid(gradient_x, px, py, extent) for px, py in zip(x, y, strict=True)]
    )
    gy = np.asarray(
        [sample_grid(gradient_y, px, py, extent) for px, py in zip(x, y, strict=True)]
    )
    slope_degrees = np.degrees(np.arctan(np.hypot(gx, gy)))

    displacement = np.column_stack((np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
    displacement_norm = np.linalg.norm(displacement, axis=1)
    moving = displacement_norm > 1e-9
    motion_direction = np.zeros_like(displacement)
    motion_direction[moving] = displacement[moving] / displacement_norm[moving, None]
    signed_grade_actual = gx * motion_direction[:, 0] + gy * motion_direction[:, 1]
    signed_actual_slope_degrees = np.degrees(np.arctan(signed_grade_actual))

    goal_direction = np.asarray(approved["goal_xy_m"], dtype=np.float64) - np.asarray(
        approved["start_xy_m"], dtype=np.float64
    )
    goal_direction /= np.linalg.norm(goal_direction)
    signed_grade_goal = gx * goal_direction[0] + gy * goal_direction[1]
    signed_goal_slope_degrees = np.degrees(np.arctan(signed_grade_goal))

    bins = [0.0, 5.0, 10.0, 15.0, 20.0, float("inf")]
    by_slope: list[dict[str, float | int | str]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (slope_degrees >= lower) & (slope_degrees < upper)
        count = int(mask.sum())
        by_slope.append(
            {
                "slope_bin_degrees": f"[{lower:g}, {upper:g})",
                "steps": count,
                "fraction_of_video_steps": count / len(rows),
                "airborne_fraction_within_bin": float(airborne[mask].mean()) if count else None,
            }
        )

    early = times <= 5.0
    result = {
        "schema_version": "proxygap-slope-contact-diagnostic-v1",
        "trace": str(trace_path),
        "steps": len(rows),
        "terrain_slope_degrees_all_steps": stats(slope_degrees),
        "terrain_slope_degrees_airborne_steps": stats(slope_degrees[airborne]),
        "terrain_slope_degrees_contact_steps": stats(slope_degrees[~airborne]),
        "signed_goal_direction_slope_degrees": stats(signed_goal_slope_degrees),
        "signed_actual_motion_slope_degrees": stats(signed_actual_slope_degrees[moving]),
        "airborne_fraction_all_steps": float(airborne.mean()),
        "airborne_fraction_first_5_seconds": float(airborne[early].mean()),
        "airborne_fraction_after_5_seconds": float(airborne[~early].mean()),
        "airborne_steps_on_local_slope_below_5_degrees": int(
            np.logical_and(airborne, slope_degrees < 5.0).sum()
        ),
        "airborne_fraction_occurring_below_5_degrees": float(
            np.logical_and(airborne, slope_degrees < 5.0).sum() / max(1, airborne.sum())
        ),
        "airborne_steps_on_local_slope_below_10_degrees": int(
            np.logical_and(airborne, slope_degrees < 10.0).sum()
        ),
        "airborne_fraction_occurring_below_10_degrees": float(
            np.logical_and(airborne, slope_degrees < 10.0).sum() / max(1, airborne.sum())
        ),
        "by_local_slope_bin": by_slope,
        "interpretation_boundary": (
            "This is a descriptive single-episode association. Local heightfield slope does not "
            "identify the causal contribution of friction, controller commands, reset transients "
            "or contact-detection rules."
        ),
    }
    output_path = trace_dir / f"fixed_map_final_policy_seed_{args.evaluation_seed}_slope_contact.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
