"""Independent numerical checks for generated terrain fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from terrain_generator import TerrainData, differential_metrics
from terrain_queries import TerrainQueries


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: dict[str, bool]
    measurements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "measurements": self.measurements,
        }


def triangle_slope_metrics(height_m: np.ndarray, dx_m: float, dy_m: float) -> dict[str, float]:
    """Conservatively inspect both possible triangulations of every grid cell."""

    h00 = height_m[:-1, :-1]
    h01 = height_m[:-1, 1:]
    h10 = height_m[1:, :-1]
    h11 = height_m[1:, 1:]
    gx_bottom = (h01 - h00) / dx_m
    gx_top = (h11 - h10) / dx_m
    gy_left = (h10 - h00) / dy_m
    gy_right = (h11 - h01) / dy_m
    candidate_slopes = (
        np.hypot(gx_bottom, gy_left),
        np.hypot(gx_top, gy_right),
        np.hypot(gx_top, gy_left),
        np.hypot(gx_bottom, gy_right),
    )
    return {
        "maximum_triangle_slope": float(max(np.max(value) for value in candidate_slopes)),
        "maximum_neighbour_height_jump_m": float(
            max(np.max(np.abs(h01 - h00)), np.max(np.abs(h10 - h00)))
        ),
    }


def principal_curvature_diagnostic(height_m: np.ndarray, dx_m: float, dy_m: float) -> dict[str, float]:
    """Estimate exact graph-surface principal curvatures as a secondary diagnostic."""

    hy, hx = np.gradient(height_m, dy_m, dx_m, edge_order=2)
    hxy_a, hxx = np.gradient(hx, dy_m, dx_m, edge_order=2)
    hyy, hxy_b = np.gradient(hy, dy_m, dx_m, edge_order=2)
    hxy = 0.5 * (hxy_a + hxy_b)
    denominator = 1.0 + hx * hx + hy * hy
    mean = (
        (1.0 + hy * hy) * hxx
        - 2.0 * hx * hy * hxy
        + (1.0 + hx * hx) * hyy
    ) / (2.0 * denominator ** 1.5)
    gaussian = (hxx * hyy - hxy * hxy) / (denominator * denominator)
    root = np.sqrt(np.maximum(mean * mean - gaussian, 0.0))
    first = mean + root
    second = mean - root
    return {
        "maximum_absolute_principal_curvature_m_inverse": float(
            max(np.max(np.abs(first)), np.max(np.abs(second)))
        )
    }


def _safe_region_measurement(
    terrain: TerrainData,
    region_name: str,
    slope_field: np.ndarray,
) -> dict[str, float | bool]:
    region = getattr(terrain.config, region_name)
    x_mesh, y_mesh = np.meshgrid(
        terrain.x_coordinates_m, terrain.y_coordinates_m, indexing="xy"
    )
    mask = np.hypot(x_mesh - region.centre_x_m, y_mesh - region.centre_y_m) <= region.radius_m
    maximum_slope = float(np.max(slope_field[mask]))
    return {
        "maximum_absolute_slope": maximum_slope,
        "limit": region.maximum_absolute_slope,
        "grid_points": int(np.count_nonzero(mask)),
        "passed": maximum_slope <= region.maximum_absolute_slope + 1e-10,
    }


def validate_terrain(terrain: TerrainData) -> ValidationResult:
    """Verify physical bounds, recovery metadata, safe regions and continuity."""

    config = terrain.config
    dx = float(terrain.x_coordinates_m[1] - terrain.x_coordinates_m[0])
    dy = float(terrain.y_coordinates_m[1] - terrain.y_coordinates_m[0])
    metrics = differential_metrics(terrain.height_m, dx, dy)
    triangles = triangle_slope_metrics(terrain.height_m, dx, dy)
    curvature = principal_curvature_diagnostic(terrain.height_m, dx, dy)
    normalisation = terrain.metadata["normalisation"]
    recovered = (
        normalisation["physical_offset_m"]
        + terrain.normalised_height * normalisation["physical_scale_m"]
    )
    recovery_error = float(np.max(np.abs(recovered - terrain.height_m)))
    start = _safe_region_measurement(terrain, "start_safe_region", metrics["slope"])
    goal = _safe_region_measurement(terrain, "goal_safe_region", metrics["slope"])
    queries = TerrainQueries(terrain)
    start_centre_safe = queries.is_safe_spawn(
        config.start_safe_region.centre_x_m,
        config.start_safe_region.centre_y_m,
    )
    sampled_goal = queries.sample_safe_goal(seed=71_903)
    goal_gradient = queries.gradient(*sampled_goal)
    goal_slope = math.hypot(*goal_gradient)
    minimum_constructed = float(terrain.metadata["minimum_constructed_feature_width_m"])
    sampling_intervals = minimum_constructed / max(dx, dy)
    neighbour_bound = config.maximum_absolute_slope * math.hypot(dx, dy) + 1e-10

    checks = {
        "shape": terrain.height_m.shape == (config.nrow, config.ncol),
        "all_finite": bool(
            np.all(np.isfinite(terrain.height_m))
            and np.all(np.isfinite(terrain.normalised_height))
        ),
        "normalised_in_unit_interval": bool(
            np.min(terrain.normalised_height) >= 0.0
            and np.max(terrain.normalised_height) <= 1.0
        ),
        "physical_recovery": recovery_error <= 32.0 * np.finfo(np.float64).eps * max(
            1.0, config.maximum_height_m
        ),
        "height_bound": metrics["maximum_absolute_height_m"] <= config.maximum_height_m + 1e-10,
        "gradient_slope_bound": metrics["maximum_absolute_slope"] <= config.maximum_absolute_slope + 1e-10,
        "triangle_slope_bound": triangles["maximum_triangle_slope"] <= config.maximum_absolute_slope + 1e-10,
        "curvature_bound": metrics["maximum_curvature"] <= config.maximum_curvature + 1e-10,
        "minimum_feature_width_constructive": minimum_constructed + 1e-12
        >= config.minimum_feature_width_m,
        "minimum_feature_width_resolved": sampling_intervals >= 8.0 - 1e-10,
        "start_region_low_slope": bool(start["passed"]),
        "goal_region_low_slope": bool(goal["passed"]),
        "start_footprint_safe": bool(start_centre_safe),
        "sampled_goal_safe": goal_slope <= config.goal_safe_region.maximum_absolute_slope + 1e-10,
        "no_discrete_height_jump": triangles["maximum_neighbour_height_jump_m"] <= neighbour_bound,
        "friction_not_randomised": terrain.metadata.get("friction_randomised") is False,
    }
    measurements: dict[str, Any] = {
        "grid_spacing_m": {"dx": dx, "dy": dy},
        "maximum_absolute_height_m": metrics["maximum_absolute_height_m"],
        "maximum_gradient_slope": metrics["maximum_absolute_slope"],
        "maximum_triangle_slope": triangles["maximum_triangle_slope"],
        "maximum_hessian_spectral_norm_m_inverse": metrics["maximum_curvature"],
        **curvature,
        "maximum_neighbour_height_jump_m": triangles["maximum_neighbour_height_jump_m"],
        "physical_recovery_max_abs_error_m": recovery_error,
        "minimum_constructed_feature_width_m": minimum_constructed,
        "minimum_feature_sampling_intervals": sampling_intervals,
        "start_safe_region": start,
        "goal_safe_region": goal,
        "sampled_goal_m": list(sampled_goal),
        "sampled_goal_slope": goal_slope,
    }
    serialisable_checks = {name: bool(value) for name, value in checks.items()}
    return ValidationResult(bool(all(serialisable_checks.values())), serialisable_checks, measurements)


def assert_terrain_valid(terrain: TerrainData) -> ValidationResult:
    result = validate_terrain(terrain)
    if not result.passed:
        failed = [name for name, passed in result.checks.items() if not passed]
        raise AssertionError(f"terrain validation failed: {failed}")
    return result


def save_validation_result(result: ValidationResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
