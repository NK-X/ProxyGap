from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terrain_generator import TerrainData, generate_terrain
from terrain_queries import TerrainQueries


def synthetic_terrain(height_m: np.ndarray) -> TerrainData:
    from terrain_generator import TerrainConfig

    height = np.asarray(height_m, dtype=np.float64)
    minimum = float(np.min(height))
    scale = float(np.max(height) - minimum)
    normalised = (height - minimum) / scale if scale else np.zeros_like(height)
    return TerrainData(
        config=TerrainConfig(),
        x_coordinates_m=np.linspace(0.0, 1.0, height.shape[1]),
        y_coordinates_m=np.linspace(0.0, 1.0, height.shape[0]),
        height_m=height,
        normalised_height=normalised,
        metadata={
            "normalisation": {
                "physical_offset_m": minimum,
                "physical_scale_m": scale,
            }
        },
    )


def test_height_and_gradient_follow_mujoco_cell_triangulation() -> None:
    queries = TerrainQueries(synthetic_terrain(np.asarray([[0.0, 1.0], [2.0, 4.0]])))
    assert np.isclose(queries.height(0.75, 0.25), 1.5)
    np.testing.assert_allclose(queries.gradient(0.75, 0.25), (1.0, 3.0))
    assert np.isclose(queries.height(0.25, 0.75), 2.0)
    np.testing.assert_allclose(queries.gradient(0.25, 0.75), (2.0, 2.0))


def test_normal_is_unit_length_and_slope_sign_is_directional() -> None:
    queries = TerrainQueries(synthetic_terrain(np.asarray([[0.0, 0.2], [0.0, 0.2]])))
    normal = np.asarray(queries.normal(0.3, 0.4))
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert normal[2] > 0.0
    assert queries.slope_along(0.3, 0.4, 1.0, 0.0) > 0.0
    assert queries.slope_along(0.3, 0.4, -1.0, 0.0) < 0.0


def test_invalid_query_and_zero_direction_are_rejected() -> None:
    queries = TerrainQueries(synthetic_terrain(np.zeros((2, 2))))
    with pytest.raises(ValueError, match="outside"):
        queries.height(-0.01, 0.5)
    with pytest.raises(ValueError, match="non-zero"):
        queries.slope_along(0.5, 0.5, 0.0, 0.0)


def test_safe_spawn_and_goal_sampling_are_deterministic(development_config) -> None:
    flat_config = replace(
        development_config,
        hill_count=0,
        pit_count=0,
        random_fourier_terms=0,
        global_slope_x=0.0,
        global_slope_y=0.0,
    )
    terrain = generate_terrain(flat_config)
    queries = TerrainQueries(terrain)
    start = flat_config.start_safe_region
    assert queries.is_safe_spawn(start.centre_x_m, start.centre_y_m)
    assert not queries.is_safe_spawn(start.centre_x_m + start.radius_m, start.centre_y_m)
    first = queries.sample_safe_goal(12_345)
    second = queries.sample_safe_goal(12_345)
    third = queries.sample_safe_goal(12_346)
    assert first == second
    assert first != third
