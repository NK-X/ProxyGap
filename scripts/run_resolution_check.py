"""Compare 257, 513 and 1025 heightfields on one development recipe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import gymnasium as gym  # noqa: E402

from mujoco_heightfield import (  # noqa: E402
    build_ant_heightfield_xml,
    install_heightfield_data,
    run_ant_smoke_test,
)
from terrain_generator import generate_terrain, load_config  # noqa: E402
from terrain_queries import TerrainQueries  # noqa: E402
from terrain_validation import assert_terrain_valid, differential_metrics  # noqa: E402


RESOLUTIONS = (257, 513, 1025)


def flat_plane_baseline(reset_seed: int = 202_608_018, steps: int = 10) -> dict[str, Any]:
    """Record default-plane soft contacts under the same zero-action smoke protocol."""

    env = gym.make("Ant-v5")
    try:
        observation, _ = env.reset(seed=reset_seed)
        initial_warning_counts = [int(item.number) for item in env.unwrapped.data.warning]
        initial_contact_count = int(env.unwrapped.data.ncon)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        minimum_distances: list[float] = []
        maximum_contact_count = initial_contact_count
        for _ in range(steps):
            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated or not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                raise AssertionError("default Ant-v5 plane failed the matched smoke baseline")
            warning_counts = [int(item.number) for item in env.unwrapped.data.warning]
            if any(warning_counts):
                raise AssertionError(f"MuJoCo warning in flat-plane baseline: {warning_counts}")
            maximum_contact_count = max(maximum_contact_count, int(env.unwrapped.data.ncon))
            minimum_distances.extend(
                float(env.unwrapped.data.contact[index].dist)
                for index in range(int(env.unwrapped.data.ncon))
            )
        return {
            "reset_seed": reset_seed,
            "steps": steps,
            "initial_contact_count": initial_contact_count,
            "initial_mujoco_warning_counts": initial_warning_counts,
            "maximum_contact_count_during_smoke": maximum_contact_count,
            "minimum_contact_distance_during_smoke_m": min(minimum_distances)
            if minimum_distances
            else None,
        }
    finally:
        env.close()


def distribution_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "values_s": values,
        "median_s": float(statistics.median(values)),
        "minimum_s": float(min(values)),
        "maximum_s": float(max(values)),
        "iqr_s": float(np.percentile(ordered, 75) - np.percentile(ordered, 25)),
    }


def error_summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "maximum_absolute": float(np.max(absolute)),
        "p95_absolute": float(np.percentile(absolute, 95)),
    }


def faceting_diagnostic(terrain) -> dict[str, float | bool]:
    dx = float(terrain.x_coordinates_m[1] - terrain.x_coordinates_m[0])
    dy = float(terrain.y_coordinates_m[1] - terrain.y_coordinates_m[0])
    metrics = differential_metrics(terrain.height_m, dx, dy)
    normal = np.stack(
        (-metrics["dh_dx"], -metrics["dh_dy"], np.ones_like(terrain.height_m)), axis=-1
    )
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    dots = np.concatenate(
        (
            np.sum(normal[:, :-1] * normal[:, 1:], axis=-1).ravel(),
            np.sum(normal[:-1, :] * normal[1:, :], axis=-1).ravel(),
        )
    )
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    maximum = float(np.max(angles))
    return {
        "maximum_adjacent_normal_angle_deg": maximum,
        "p99_adjacent_normal_angle_deg": float(np.percentile(angles, 99)),
        "obvious_local_faceting_heuristic": maximum > 5.0,
        "heuristic_threshold_deg": 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "terrain_development.json",
    )
    parser.add_argument("--load-repetitions", type=int, default=3)
    parser.add_argument("--step-blocks", type=int, default=5)
    parser.add_argument("--steps-per-block", type=int, default=50)
    args = parser.parse_args()
    if args.load_repetitions < 2 or args.step_blocks < 2 or args.steps_per_block < 10:
        raise ValueError("resolution timing requires at least two repetitions and 10 steps per block")

    base = load_config(args.config)
    calibration_native_scales: dict[int, float] = {}
    calibration_uncapped_applied_scales: dict[int, float] = {}
    for resolution in RESOLUTIONS:
        calibration_terrain = generate_terrain(
            replace(base, nrow=resolution, ncol=resolution)
        )
        assert_terrain_valid(calibration_terrain)
        calibration_native_scales[resolution] = float(
            calibration_terrain.metadata["native_constraint_scale"]
        )
        calibration_uncapped_applied_scales[resolution] = float(
            calibration_terrain.metadata["applied_constraint_scale"]
        )
        del calibration_terrain
        gc.collect()

    initial_common_scale_cap = min(calibration_native_scales.values())
    common_scale_cap = initial_common_scale_cap
    scale_cap_attempts: list[dict[str, Any]] = []
    terrains: dict[int, Any] = {}
    generation_times: dict[int, float] = {}
    validations: dict[int, Any] = {}
    xml_paths: dict[int, Path] = {}
    constraint_scales_identical = False
    for cap_attempt in range(8):
        candidate_terrains: dict[int, Any] = {}
        candidate_generation_times: dict[int, float] = {}
        candidate_validations: dict[int, Any] = {}
        for resolution in RESOLUTIONS:
            config = replace(base, nrow=resolution, ncol=resolution)
            started = time.perf_counter()
            terrain = generate_terrain(
                config,
                stochastic_residual_scale_cap=common_scale_cap,
            )
            candidate_generation_times[resolution] = time.perf_counter() - started
            candidate_validations[resolution] = assert_terrain_valid(terrain)
            candidate_terrains[resolution] = terrain
        applied_scales = {
            resolution: float(terrain.metadata["applied_constraint_scale"])
            for resolution, terrain in candidate_terrains.items()
        }
        scale_cap_attempts.append(
            {
                "attempt": cap_attempt + 1,
                "scale_cap": common_scale_cap,
                "applied_scales": applied_scales,
            }
        )
        constraint_scales_identical = len(set(applied_scales.values())) == 1
        if constraint_scales_identical:
            terrains = candidate_terrains
            generation_times = candidate_generation_times
            validations = candidate_validations
            break

        minimum_applied = min(applied_scales.values())
        common_scale_cap = (
            0.0 if minimum_applied == 0.0 else minimum_applied * (1.0 - 1e-9)
        )
        del candidate_terrains
        gc.collect()
    else:
        raise RuntimeError(
            "could not obtain an identical applied stochastic-residual scale across resolutions"
        )

    for resolution, terrain in terrains.items():
        xml_path = ROOT / "outputs" / "mujoco" / f"resolution_{resolution}.xml"
        build_ant_heightfield_xml(terrain, xml_path)
        xml_paths[resolution] = xml_path

    component_json = {
        resolution: json.dumps(terrain.metadata["components"], sort_keys=True)
        for resolution, terrain in terrains.items()
    }
    recipe_components_identical = len(set(component_json.values())) == 1

    rng = np.random.Generator(np.random.PCG64(91_731))
    margin = base.minimum_feature_width_m
    probe_x = rng.uniform(
        -0.5 * base.terrain_length_m + margin,
        0.5 * base.terrain_length_m - margin,
        size=384,
    )
    probe_y = rng.uniform(
        -0.5 * base.terrain_width_m + margin,
        0.5 * base.terrain_width_m - margin,
        size=384,
    )
    query_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for resolution, terrain in terrains.items():
        queries = TerrainQueries(terrain)
        heights = np.asarray(
            [queries.height(float(x), float(y)) for x, y in zip(probe_x, probe_y)]
        )
        gradients = np.asarray(
            [queries.gradient(float(x), float(y)) for x, y in zip(probe_x, probe_y)]
        )
        query_values[resolution] = heights, gradients
    reference_heights, reference_gradients = query_values[1025]

    load_times: dict[int, list[float]] = {resolution: [] for resolution in RESOLUTIONS}
    schedule = [
        (resolution, repetition)
        for repetition in range(args.load_repetitions)
        for resolution in RESOLUTIONS
    ]
    schedule_rng = np.random.Generator(np.random.PCG64(44_081))
    schedule_rng.shuffle(schedule)
    for resolution, _ in schedule:
        started = time.perf_counter()
        model = mujoco.MjModel.from_xml_path(str(xml_paths[resolution].resolve()))
        install_heightfield_data(model, terrains[resolution])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        load_times[resolution].append(time.perf_counter() - started)
        del data, model
        gc.collect()

    step_times: dict[int, list[float]] = {resolution: [] for resolution in RESOLUTIONS}
    step_warning_counts: dict[int, list[list[int]]] = {
        resolution: [] for resolution in RESOLUTIONS
    }
    for resolution in (513, 1025, 257):
        model = mujoco.MjModel.from_xml_path(str(xml_paths[resolution].resolve()))
        install_heightfield_data(model, terrains[resolution])
        for _ in range(args.step_blocks):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for _ in range(10):
                mujoco.mj_step(model, data)
            started = time.perf_counter()
            for _ in range(args.steps_per_block):
                mujoco.mj_step(model, data)
            elapsed = time.perf_counter() - started
            step_times[resolution].append(elapsed / args.steps_per_block)
            warning_counts = [int(item.number) for item in data.warning]
            step_warning_counts[resolution].append(warning_counts)
            if any(warning_counts):
                raise AssertionError(
                    f"MuJoCo warning during resolution {resolution} timing: {warning_counts}"
                )
        del model
        gc.collect()

    smoke_results: dict[int, Any] = {}
    for resolution in RESOLUTIONS:
        smoke_results[resolution] = run_ant_smoke_test(
            terrains[resolution],
            ROOT / "outputs" / "mujoco" / f"resolution_{resolution}_smoke.xml",
            steps=10,
        )
    plane_baseline = flat_plane_baseline()

    rows: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        terrain = terrains[resolution]
        heights, gradients = query_values[resolution]
        height_error = error_summary(heights - reference_heights)
        gradient_norm_error = np.linalg.norm(gradients - reference_gradients, axis=1)
        gradient_error = error_summary(gradient_norm_error)
        smoke = smoke_results[resolution]
        minimum_contacts = [
            item["minimum_contact_distance_m"]
            for item in smoke["records"]
            if item["minimum_contact_distance_m"] is not None
        ]
        maximum_contact_count = max(
            [smoke["initial_contacts"]["contact_count"]]
            + [item["contact_count"] for item in smoke["records"]]
        )
        faceting = faceting_diagnostic(terrain)
        row = {
            "resolution": resolution,
            "nrow": terrain.config.nrow,
            "ncol": terrain.config.ncol,
            "dx_m": float(terrain.x_coordinates_m[1] - terrain.x_coordinates_m[0]),
            "dy_m": float(terrain.y_coordinates_m[1] - terrain.y_coordinates_m[0]),
            "generation_time_s": generation_times[resolution],
            "native_constraint_scale": float(terrain.metadata["native_constraint_scale"]),
            "applied_constraint_scale": float(terrain.metadata["applied_constraint_scale"]),
            "stochastic_residual_scale_cap": terrain.metadata[
                "stochastic_residual_scale_cap"
            ],
            "height_error_vs_1025": height_error,
            "gradient_vector_error_vs_1025": gradient_error,
            "mujoco_load_time": distribution_summary(load_times[resolution]),
            "mujoco_step_time": distribution_summary(step_times[resolution]),
            "timed_step_operation": "mujoco.mj_step after 10 warm-up steps per fresh block",
            "timing_mujoco_warning_counts": step_warning_counts[resolution],
            "faceting_diagnostic": faceting,
            "ant_smoke_passed": smoke["passed"],
            "initial_contact_count": smoke["initial_contacts"]["contact_count"],
            "initial_minimum_contact_distance_m": smoke["initial_contacts"][
                "minimum_contact_distance_m"
            ],
            "minimum_contact_distance_during_smoke_m": min(minimum_contacts)
            if minimum_contacts
            else None,
            "maximum_contact_count_during_smoke": maximum_contact_count,
            "initial_penetration_or_engine_warning_detected": (not smoke["passed"]),
            "validation_passed": validations[resolution].passed,
            "height_sha256": terrain.height_sha256,
        }
        rows.append(row)

    height_nonincreasing = (
        rows[1]["height_error_vs_1025"]["rmse"]
        <= rows[0]["height_error_vs_1025"]["rmse"]
        and rows[2]["height_error_vs_1025"]["rmse"]
        <= rows[1]["height_error_vs_1025"]["rmse"]
    )
    gradient_nonincreasing = (
        rows[1]["gradient_vector_error_vs_1025"]["rmse"]
        <= rows[0]["gradient_vector_error_vs_1025"]["rmse"]
        and rows[2]["gradient_vector_error_vs_1025"]["rmse"]
        <= rows[1]["gradient_vector_error_vs_1025"]["rmse"]
    )
    baseline_maximum_contacts = plane_baseline["maximum_contact_count_during_smoke"]
    for row in rows:
        multiplicity_difference = (
            row["maximum_contact_count_during_smoke"] > baseline_maximum_contacts
        )
        row["contact_multiplicity_exceeds_flat_plane_baseline"] = multiplicity_difference
        row["contact_anomaly_detected"] = (
            row["initial_penetration_or_engine_warning_detected"] or multiplicity_difference
        )
        row["contact_assessment"] = (
            "No initial penetration, termination, non-finite state or MuJoCo warning; "
            "however, the heightfield produced more simultaneous triangle contacts than "
            "the default plane and therefore requires a dedicated contact-dynamics check."
            if multiplicity_difference
            else "No bounded-smoke contact difference detected relative to the default plane."
        )
    report = {
        "scope": "development-only resolution and MuJoCo loading smoke check; no locomotion claim",
        "reference_resolution": 1025,
        "off_grid_probe_count": int(probe_x.size),
        "recipe_components_identical": recipe_components_identical,
        "constraint_scale_calibration": {
            "uncapped_native_scales": calibration_native_scales,
            "uncapped_applied_scales": calibration_uncapped_applied_scales,
            "initial_common_scale_cap": initial_common_scale_cap,
            "final_common_scale_cap": common_scale_cap,
            "cap_attempts": scale_cap_attempts,
        },
        "common_stochastic_residual_scale_cap": common_scale_cap,
        "constraint_scales_identical": constraint_scales_identical,
        "convergence_checks": {
            "height_rmse_nonincreasing": height_nonincreasing,
            "gradient_rmse_nonincreasing": gradient_nonincreasing,
            "common_constraint_scale_applied": constraint_scales_identical,
            "passed": (
                height_nonincreasing
                and gradient_nonincreasing
                and recipe_components_identical
                and constraint_scales_identical
            ),
            "note": (
                "384 deterministic off-grid probes after applying one calibrated common "
                "stochastic-residual scale; this is a development approximation sample, "
                "not a continuous-domain proof"
            ),
        },
        "load_schedule": [list(item) for item in schedule],
        "default_flat_plane_contact_baseline": plane_baseline,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "mujoco": importlib.metadata.version("mujoco"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "rows": rows,
    }
    output_json = ROOT / "outputs" / "manifests" / "resolution_check.json"
    output_csv = ROOT / "outputs" / "manifests" / "resolution_check.csv"
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "resolution",
                "dx_m",
                "dy_m",
                "applied_constraint_scale",
                "height_rmse_m_vs_1025",
                "height_max_abs_m_vs_1025",
                "gradient_rmse_vs_1025",
                "gradient_max_abs_vs_1025",
                "load_median_s",
                "step_median_s",
                "max_adjacent_normal_angle_deg",
                "contact_anomaly_detected",
                "validation_passed",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "resolution": row["resolution"],
                    "dx_m": row["dx_m"],
                    "dy_m": row["dy_m"],
                    "applied_constraint_scale": row["applied_constraint_scale"],
                    "height_rmse_m_vs_1025": row["height_error_vs_1025"]["rmse"],
                    "height_max_abs_m_vs_1025": row["height_error_vs_1025"][
                        "maximum_absolute"
                    ],
                    "gradient_rmse_vs_1025": row["gradient_vector_error_vs_1025"]["rmse"],
                    "gradient_max_abs_vs_1025": row["gradient_vector_error_vs_1025"][
                        "maximum_absolute"
                    ],
                    "load_median_s": row["mujoco_load_time"]["median_s"],
                    "step_median_s": row["mujoco_step_time"]["median_s"],
                    "max_adjacent_normal_angle_deg": row["faceting_diagnostic"][
                        "maximum_adjacent_normal_angle_deg"
                    ],
                    "contact_anomaly_detected": row["contact_anomaly_detected"],
                    "validation_passed": row["validation_passed"],
                }
            )
    print(json.dumps({"json": str(output_json.resolve()), "csv": str(output_csv.resolve())}, indent=2))


if __name__ == "__main__":
    main()
