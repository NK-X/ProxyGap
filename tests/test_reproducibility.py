from __future__ import annotations

from dataclasses import replace
import json

import numpy as np

from terrain_generator import generate_terrain, save_terrain_bundle, seed_for


def test_same_seed_is_elementwise_identical_and_hash_identical(development_config) -> None:
    first = generate_terrain(development_config)
    second = generate_terrain(development_config)
    assert np.array_equal(first.height_m, second.height_m)
    assert np.array_equal(first.normalised_height, second.normalised_height)
    assert first.height_sha256 == second.height_sha256
    assert first.normalised_sha256 == second.normalised_sha256


def test_different_seed_changes_the_field(development_config) -> None:
    first = generate_terrain(development_config)
    second = generate_terrain(
        replace(development_config, terrain_seed=seed_for("development", 43))
    )
    assert not np.array_equal(first.height_m, second.height_m)
    assert first.height_sha256 != second.height_sha256


def test_manifest_records_seed_hash_version_and_timestamp(development_config, tmp_path) -> None:
    terrain = generate_terrain(development_config)
    paths = save_terrain_bundle(terrain, tmp_path, "audit")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["terrain_seed"] == development_config.terrain_seed
    assert manifest["split"] == "development"
    assert manifest["height_array"]["canonical_array_sha256"] == terrain.height_sha256
    assert manifest["normalised_array"]["canonical_array_sha256"] == terrain.normalised_sha256
    assert manifest["generator_version"]
    assert manifest["generated_at_utc"].endswith("+00:00")


def test_goal_sampling_does_not_change_terrain_rng(development_config) -> None:
    from terrain_queries import TerrainQueries

    before = generate_terrain(development_config)
    queries = TerrainQueries(before)
    for goal_seed in range(20):
        queries.sample_safe_goal(goal_seed)
    after = generate_terrain(development_config)
    assert np.array_equal(before.height_m, after.height_m)
