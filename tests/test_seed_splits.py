from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terrain_generator import (
    SEED_NAMESPACES,
    TerrainConfig,
    generate_terrain,
    seed_for,
    split_for_seed,
)


def test_train_validation_test_namespaces_are_pairwise_disjoint() -> None:
    named = {name: set(range(lower, upper + 1)) for name, (lower, upper) in SEED_NAMESPACES.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert named[first].isdisjoint(named[second])
    for split in ("train", "validation", "test"):
        assert split_for_seed(seed_for(split, 0)) == split
        assert split_for_seed(seed_for(split, 999_999)) == split


def test_seed_cannot_be_labelled_as_the_wrong_split(development_config) -> None:
    with pytest.raises(ValueError, match="belongs to"):
        TerrainConfig.from_mapping(
            {
                **development_config.to_dict(),
                "split": "test",
                "terrain_seed": seed_for("train", 7),
            }
        )


def _dihedral_variants(array: np.ndarray) -> list[np.ndarray]:
    variants = [np.rot90(array, rotations) for rotations in range(4)]
    reflected = np.fliplr(array)
    variants.extend(np.rot90(reflected, rotations) for rotations in range(4))
    return variants


def test_splits_do_not_reuse_or_rotate_the_same_map(development_config) -> None:
    terrains = {
        split: generate_terrain(
            replace(development_config, split=split, terrain_seed=seed_for(split, 17))
        )
        for split in ("train", "validation", "test")
    }
    assert len({terrain.height_sha256 for terrain in terrains.values()}) == 3
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not any(
            np.array_equal(variant, terrains[second].height_m)
            for variant in _dihedral_variants(terrains[first].height_m)
        )
