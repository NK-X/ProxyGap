from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terrain_generator import differential_metrics, generate_terrain
from terrain_queries import TerrainQueries
from terrain_validation import assert_terrain_valid, validate_terrain


def test_composite_terrain_satisfies_all_geometry_bounds(development_config, tmp_path) -> None:
    from terrain_validation import save_validation_result

    terrain = generate_terrain(development_config)
    result = assert_terrain_valid(terrain)
    assert result.passed
    assert all(result.checks.values())
    save_validation_result(result, tmp_path / "validation.json")
    assert (tmp_path / "validation.json").is_file()


def test_flat_terrain_is_finite_and_exactly_flat(development_config) -> None:
    config = replace(
        development_config,
        hill_count=0,
        pit_count=0,
        random_fourier_terms=0,
        global_slope_x=0.0,
        global_slope_y=0.0,
    )
    terrain = generate_terrain(config)
    assert np.count_nonzero(terrain.height_m) == 0
    assert np.count_nonzero(terrain.normalised_height) == 0
    assert_terrain_valid(terrain)


def test_global_slope_component_is_not_silently_rescaled(development_config) -> None:
    config = replace(
        development_config,
        hill_count=0,
        pit_count=0,
        random_fourier_terms=0,
        global_slope_x=0.03,
        global_slope_y=-0.02,
        maximum_curvature=0.50,
    )
    terrain = generate_terrain(config)
    queries = TerrainQueries(terrain)
    gradient = queries.gradient(-5.25, 4.25)
    np.testing.assert_allclose(gradient, (0.03, -0.02), atol=2e-6, rtol=0.0)
    assert terrain.metadata["constraint_scale_scope"].startswith("stochastic residual only")


def test_slope_uses_vector_norm_not_component_maximum() -> None:
    x = np.linspace(-1.0, 1.0, 65)
    y = np.linspace(-1.0, 1.0, 65)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    limit = 0.20
    plane = 0.8 * limit * xx + 0.8 * limit * yy
    metrics = differential_metrics(plane, x[1] - x[0], y[1] - y[0])
    assert metrics["maximum_absolute_slope"] > limit
    assert np.isclose(metrics["maximum_absolute_slope"], np.sqrt(2.0) * 0.8 * limit)


def test_hessian_curvature_definition_matches_quadratic() -> None:
    x = np.linspace(-2.0, 2.0, 257)
    y = np.linspace(-2.0, 2.0, 257)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    expected = 0.17
    height = 0.5 * expected * xx * xx
    metrics = differential_metrics(height, x[1] - x[0], y[1] - y[0])
    assert np.isclose(metrics["maximum_curvature"], expected, atol=1e-10, rtol=0.0)


def test_safe_cores_and_constructed_feature_width_are_certified(development_config) -> None:
    terrain = generate_terrain(development_config)
    result = validate_terrain(terrain)
    assert result.checks["start_region_low_slope"]
    assert result.checks["goal_region_low_slope"]
    assert result.checks["no_discrete_height_jump"]
    assert result.checks["minimum_feature_width_constructive"]
    assert result.measurements["minimum_feature_sampling_intervals"] >= 8.0


def test_float32_recovery_and_common_scale_cap_are_certified(development_config) -> None:
    configurations = (
        development_config,
        replace(development_config, nrow=513, ncol=513),
    )
    uncapped = [generate_terrain(config) for config in configurations]
    common_cap = min(
        float(terrain.metadata["native_constraint_scale"])
        for terrain in uncapped
    )
    capped = [
        generate_terrain(config, stochastic_residual_scale_cap=common_cap)
        for config in configurations
    ]

    assert len(
        {float(terrain.metadata["applied_constraint_scale"]) for terrain in capped}
    ) == 1
    for terrain in capped:
        assert terrain.metadata["stochastic_residual_scale_cap"] == common_cap
        assert terrain.metadata["applied_constraint_scale"] == common_cap
        assert_terrain_valid(terrain)
        normalisation = terrain.metadata["normalisation"]
        recovered = (
            normalisation["physical_offset_m"]
            + terrain.normalised_height * normalisation["physical_scale_m"]
        )
        assert np.all(np.isfinite(recovered))
        np.testing.assert_array_equal(recovered, terrain.height_m)
        assert normalisation["physical_array_decoded_from_mujoco_values"] is True

    for invalid_cap in (False, -0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="stochastic_residual_scale_cap"):
            generate_terrain(
                development_config,
                stochastic_residual_scale_cap=invalid_cap,
            )
