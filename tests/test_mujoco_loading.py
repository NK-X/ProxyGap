from __future__ import annotations

from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mujoco_heightfield import (
    HFIELD_NAME,
    build_ant_heightfield_xml,
    load_mujoco_model,
    run_ant_smoke_test,
)
from terrain_generator import generate_terrain, seed_for


def flat_config(config):
    return replace(
        config,
        hill_count=0,
        pit_count=0,
        random_fourier_terms=0,
        global_slope_x=0.0,
        global_slope_y=0.0,
    )


def test_mujoco_loads_normalised_heightfield_and_fixed_friction(development_config, tmp_path) -> None:
    terrain = generate_terrain(development_config)
    xml_path = tmp_path / "ant_terrain.xml"
    model, data, field_id = load_mujoco_model(terrain, xml_path)
    address = int(model.hfield_adr[field_id])
    count = terrain.config.nrow * terrain.config.ncol
    np.testing.assert_array_equal(
        model.hfield_data[address : address + count],
        terrain.normalised_height.astype(np.float32).ravel(order="C"),
    )
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    np.testing.assert_allclose(model.geom_friction[floor_id], terrain.config.friction)
    assert np.all(np.isfinite(data.qpos))


def test_xml_contains_one_heightfield_and_no_plane(development_config, tmp_path) -> None:
    terrain = generate_terrain(development_config)
    xml_path = build_ant_heightfield_xml(terrain, tmp_path / "ant_terrain.xml")
    root = ET.parse(xml_path).getroot()
    fields = [item for item in root.find("asset").findall("hfield") if item.get("name") == HFIELD_NAME]
    floor = next(item for item in root.find("worldbody").findall("geom") if item.get("name") == "floor")
    assert len(fields) == 1
    assert floor.get("type") == "hfield"
    assert floor.get("hfield") == HFIELD_NAME
    assert "size" not in floor.attrib
    assert tuple(float(value) for value in floor.get("friction").split()) == terrain.config.friction


def test_positive_x_slope_is_not_transposed_or_flipped(development_config, tmp_path) -> None:
    config = replace(
        development_config,
        hill_count=0,
        pit_count=0,
        random_fourier_terms=0,
        global_slope_x=0.025,
        global_slope_y=0.0,
    )
    terrain = generate_terrain(config)
    model, _, field_id = load_mujoco_model(terrain, tmp_path / "positive_x.xml")
    address = int(model.hfield_adr[field_id])
    stored = np.asarray(
        model.hfield_data[address : address + config.nrow * config.ncol]
    ).reshape(config.nrow, config.ncol)
    assert float(np.mean(stored[:, -1])) > float(np.mean(stored[:, 0]))


def test_friction_is_identical_for_different_terrain_seeds(development_config, tmp_path) -> None:
    first = generate_terrain(development_config)
    second = generate_terrain(
        replace(development_config, terrain_seed=seed_for("development", 99))
    )
    first_xml = build_ant_heightfield_xml(first, tmp_path / "first.xml")
    second_xml = build_ant_heightfield_xml(second, tmp_path / "second.xml")
    first_floor = next(
        item for item in ET.parse(first_xml).getroot().find("worldbody").findall("geom") if item.get("name") == "floor"
    )
    second_floor = next(
        item for item in ET.parse(second_xml).getroot().find("worldbody").findall("geom") if item.get("name") == "floor"
    )
    assert first_floor.get("friction") == second_floor.get("friction")


def test_ant_v5_reset_and_ten_zero_action_steps(development_config, tmp_path) -> None:
    terrain = generate_terrain(flat_config(development_config))
    result = run_ant_smoke_test(terrain, tmp_path / "ant_smoke.xml", steps=10)
    assert result["passed"]
    assert result["steps"] == 10
    assert result["initial_contacts"]["contact_count"] == 0
    assert not any(result["initial_mujoco_warning_counts"])
    assert all(not any(record["mujoco_warning_counts"]) for record in result["records"])
