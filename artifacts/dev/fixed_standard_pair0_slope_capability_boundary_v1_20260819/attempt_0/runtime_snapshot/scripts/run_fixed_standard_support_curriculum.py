"""Diagnose and screen one support-timing intervention on standard terrains.

The four generated scenes differ only in analytic height.  Robot XML, friction,
controller interface, observation dimension, PPO settings and energy boundary
remain frozen.  A high-frequency audit expands each 0.05 s policy step into
five recorded 0.01 s MuJoCo steps without changing the held action.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import struct
import sys
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.ant_wrapper import quaternion_tilt_relative_to_normal  # noqa: E402
from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.fixed_goal_terrain import FixedGoalTerrainWrapper  # noqa: E402
from run_curved_gait_training import common_env_kwargs  # noqa: E402
from run_fixed_goal_support_priority_pilot import (  # noqa: E402
    _configure_continuation_model,
    recursive_json_differences,
)


DEFAULT_CONFIG = ROOT / "configs" / "fixed_standard_support_curriculum_v1_20260819.json"
FOOT_NAMES = (
    "left_ankle_geom",
    "right_ankle_geom",
    "third_ankle_geom",
    "fourth_ankle_geom",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed.lower() != str(expected_sha256).lower():
        raise ValueError(f"SHA-256 mismatch for {path}: {observed}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("status") != "frozen_bounded_standard_scene_pilot":
        raise ValueError("Standard-scene pilot configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("The bounded pilot must prohibit generalisation claims")
    sources = config["frozen_sources"]
    base_xml = ROOT / sources["base_scene_xml"]
    if sha256(base_xml) != sources["base_scene_xml_sha256"]:
        raise ValueError("Base Ant XML is missing or changed")
    preview = verified_json(
        ROOT / sources["local_preview_configuration"],
        sources["local_preview_configuration_sha256"],
    )
    reward = verified_json(
        ROOT / sources["reward_configuration"],
        sources["reward_configuration_sha256"],
    )
    checkpoint = ROOT / sources["source_checkpoint"]
    if sha256(checkpoint) != sources["source_checkpoint_sha256"]:
        raise ValueError("The 135D source checkpoint is missing or changed")
    if int(sources["observation_dimension"]) != 135 or int(sources["action_dimension"]) != 8:
        raise ValueError("This pilot is restricted to the frozen 135D/8D interface")
    comparator_config = verified_json(
        ROOT / sources["diagnostic_support_comparator_configuration"],
        sources["diagnostic_support_comparator_configuration_sha256"],
    )
    comparator_checkpoint = ROOT / sources["diagnostic_support_comparator_checkpoint"]
    if sha256(comparator_checkpoint) != sources[
        "diagnostic_support_comparator_checkpoint_sha256"
    ]:
        raise ValueError("The V20 diagnostic support comparator is missing or changed")
    if int(sources["diagnostic_support_comparator_observation_dimension"]) != 118:
        raise ValueError("The V20 diagnostic comparator must retain its 118D interface")
    if int(comparator_config["timesteps_per_policy"]) != int(
        sources["diagnostic_support_comparator_timesteps"]
    ):
        raise ValueError("V20 diagnostic comparator timestep metadata changed")
    if not bool(config["task_adapter"]["augment_local_terrain_observation"]):
        raise ValueError("The source checkpoint requires the local terrain preview")
    if bool(config["task_adapter"]["terrain_frame_shaping_enabled"]):
        raise ValueError("Terrain-frame shaping is not part of this intervention")
    if float(config["task_adapter"]["local_terrain_height_bound_m"]) != 6.0:
        raise ValueError("The frozen source checkpoint requires +/-6 m preview bounds")

    scene = config["standard_scenes"]
    if scene["scene_order"] != ["flat", "uphill_8deg", "downhill_8deg", "bowl_exit"]:
        raise ValueError("Standard-scene order changed")
    if list(scene["fixed_friction"]) != [1.0, 0.5, 0.5] or int(scene["condim"]) != 3:
        raise ValueError("Frozen contact settings changed")
    if int(scene["grid_rows"]) != int(scene["grid_cols"]):
        raise ValueError("Standard heightfields must use a square grid")
    if int(scene["grid_rows"]) < 129:
        raise ValueError("Standard heightfield resolution is too low")

    selection = config["intervention_selection_gate"]
    expected_path = "preserved_pre_pitch_reward.foot_contact_gap_shaping_weight"
    if selection.get("candidate") != "per_foot_contact_gap_weight_only":
        raise ValueError("Unexpected intervention candidate")
    if selection.get("permitted_reward_path") != expected_path:
        raise ValueError("Intervention reward path is not fail-closed")
    if float(selection["fixed_grace_seconds"]) != 0.5 or float(selection["fixed_scale_seconds"]) != 0.5:
        raise ValueError("Contact-gap timing constants must remain frozen")
    variants = config["training"]["variants"]
    if len(variants) != 2:
        raise ValueError("Exactly one control and one intervention are required")
    declared_weights = [float(item["foot_contact_gap_shaping_weight"]) for item in variants]
    if declared_weights != [float(selection["control_weight"]), float(selection["intervention_weight"])]:
        raise ValueError("Variant weights differ from the predeclared pair")
    rollout = int(config["training"]["parallel_environments"]) * int(config["ppo"]["n_steps"])
    if int(config["training"]["additional_target_timesteps_per_variant"]) % rollout:
        raise ValueError("Training budget is not divisible by the PPO rollout size")
    if int(config["training"]["parallel_environments"]) != len(scene["scene_order"]):
        raise ValueError("Each standard scene must have one training environment")
    if float(reward["preserved_pre_pitch_reward"]["ctrl_cost_weight"]) != 0.5:
        raise ValueError("The source reward no longer uses ctrl_cost_weight=0.5")
    energy = config["energy_boundary"]
    if float(energy["ctrl_cost_weight_unchanged"]) != 0.5:
        raise ValueError("Energy control weight changed")
    if energy["relative_mission_energy_v2_status"] != "measurement_only_not_implemented_as_reward":
        raise ValueError("Relative-energy V2 must remain measurement-only")
    if energy["energy_formula_changes"] != "prohibited":
        raise ValueError("The energy formula must remain outside this pilot")

    reward_control = reward_config_with_contact_gap_weight(
        reward, float(selection["control_weight"])
    )
    reward_intervention = reward_config_with_contact_gap_weight(
        reward, float(selection["intervention_weight"])
    )
    differences = recursive_json_differences(reward_control, reward_intervention)
    if len(differences) != 1 or differences[0][0] != expected_path:
        raise ValueError(f"Reward variants differ outside the permitted path: {differences}")
    if preview["base_policy"]["model_path"] != (
        "artifacts/dev/fixed_quad_terrain_v2_training_20260818/seed_62801/models/checkpoint_2465792.zip"
    ):
        raise ValueError("Unexpected preview lineage")
    return preview, reward


def reward_config_with_contact_gap_weight(
    base_reward_config: dict[str, Any], weight: float
) -> dict[str, Any]:
    result = copy.deepcopy(base_reward_config)
    preserved = result["preserved_pre_pitch_reward"]
    preserved["foot_contact_gap_shaping_weight"] = float(weight)
    preserved["foot_contact_gap_grace_seconds"] = 0.5
    preserved["foot_contact_gap_scale_seconds"] = 0.5
    return result


def build_standard_heights(scene_config: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = int(scene_config["grid_rows"])
    cols = int(scene_config["grid_cols"])
    extent = float(scene_config["map_half_extent_m"])
    x_axis = np.linspace(-extent, extent, cols, dtype=np.float64)
    y_axis = np.linspace(-extent, extent, rows, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    epsilon = float(scene_config["flat_numerical_relief_m"])
    flat = epsilon * (
        np.sin(math.pi * x_grid / extent) + 0.5 * np.sin(math.pi * y_grid / extent)
    )
    uphill = math.tan(math.radians(float(scene_config["uphill_slope_degrees"]))) * x_grid
    downhill = math.tan(math.radians(float(scene_config["downhill_slope_degrees"]))) * x_grid
    centre_x, centre_y = (float(value) for value in scene_config["bowl_centre_xy_m"])
    sigma_x = float(scene_config["bowl_sigma_x_m"])
    sigma_y = float(scene_config["bowl_sigma_y_m"])
    depth = float(scene_config["bowl_depth_m"])
    bowl = -depth * np.exp(
        -0.5
        * (
            np.square((x_grid - centre_x) / sigma_x)
            + np.square((y_grid - centre_y) / sigma_y)
        )
    )
    return {
        "flat": flat,
        "uphill_8deg": uphill,
        "downhill_8deg": downhill,
        "bowl_exit": bowl,
    }


def terrain_value(array: np.ndarray, x: float, y: float, extent: float) -> float:
    rows, cols = array.shape
    col_f = np.clip((x + extent) / (2.0 * extent) * (cols - 1), 0.0, cols - 1)
    row_f = np.clip((y + extent) / (2.0 * extent) * (rows - 1), 0.0, rows - 1)
    col0 = min(int(math.floor(col_f)), cols - 2)
    row0 = min(int(math.floor(row_f)), rows - 2)
    tx = float(col_f - col0)
    ty = float(row_f - row0)
    return float(
        (1.0 - ty) * ((1.0 - tx) * array[row0, col0] + tx * array[row0, col0 + 1])
        + ty * ((1.0 - tx) * array[row0 + 1, col0] + tx * array[row0 + 1, col0 + 1])
    )


def surface_pose(
    heights: np.ndarray, *, extent: float, xy: np.ndarray, heading_rad: float = 0.0
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spacing_y = 2.0 * extent / (heights.shape[0] - 1)
    spacing_x = 2.0 * extent / (heights.shape[1] - 1)
    dz_dy, dz_dx = np.gradient(heights, spacing_y, spacing_x)
    gx = terrain_value(dz_dx, float(xy[0]), float(xy[1]), extent)
    gy = terrain_value(dz_dy, float(xy[0]), float(xy[1]), extent)
    z = terrain_value(heights, float(xy[0]), float(xy[1]), extent)
    planar_forward = np.asarray(
        [math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float64
    )
    forward = np.asarray(
        [planar_forward[0], planar_forward[1], gx * planar_forward[0] + gy * planar_forward[1]],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    normal = np.asarray([-gx, -gy, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, normal)
    forward /= np.linalg.norm(forward)
    rotation = np.column_stack((forward, left, normal))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    position = np.asarray([xy[0], xy[1], z + 0.75], dtype=np.float64)
    return position, quaternion, {
        "terrain_height_m": z,
        "gradient_x": gx,
        "gradient_y": gy,
        "gradient_degrees": float(math.degrees(math.atan(math.hypot(gx, gy)))),
    }


def colourise_heightfield(heights: np.ndarray) -> Image.Image:
    span = max(float(np.ptp(heights)), 1e-12)
    value = np.clip((heights - float(np.min(heights))) / span, 0.0, 1.0)
    red = (45.0 + 165.0 * value).astype(np.uint8)
    green = (102.0 + 80.0 * (1.0 - np.abs(2.0 * value - 1.0))).astype(np.uint8)
    blue = (135.0 + 75.0 * (1.0 - value)).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")


def robot_signature(model: mujoco.MjModel) -> dict[str, Any]:
    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int64)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "body_mass": np.asarray(model.body_mass, dtype=np.float64).tolist(),
        "joint_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        ],
        "actuator_joint_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(index))
            for index in actuator_joint_ids
        ],
        "actuator_gear": np.asarray(model.actuator_gear, dtype=np.float64).tolist(),
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange, dtype=np.float64).tolist(),
        "robot_geom_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(1, model.ngeom)
        ],
        "robot_geom_type": np.asarray(model.geom_type[1:], dtype=np.int64).tolist(),
        "robot_geom_size": np.asarray(model.geom_size[1:], dtype=np.float64).tolist(),
        "robot_geom_friction": np.asarray(model.geom_friction[1:], dtype=np.float64).tolist(),
        "timestep_seconds": float(model.opt.timestep),
    }


def prepare_standard_scenes(
    config: dict[str, Any], output_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_xml = ROOT / config["frozen_sources"]["base_scene_xml"]
    scene_config = config["standard_scenes"]
    extent = float(scene_config["map_half_extent_m"])
    start = np.asarray(scene_config["start_xy_m"], dtype=np.float64)
    goal = np.asarray(scene_config["goal_xy_m"], dtype=np.float64)
    expected_friction = np.asarray(scene_config["fixed_friction"], dtype=np.float64)
    heights_by_name = build_standard_heights(scene_config)
    base_model = mujoco.MjModel.from_xml_path(str(source_xml.resolve()))
    expected_robot = robot_signature(base_model)
    records: dict[str, dict[str, Any]] = {}
    scene_root = output_root / "standard_scenes"
    for name in scene_config["scene_order"]:
        heights = np.asarray(heights_by_name[name], dtype=np.float64)
        scene_dir = scene_root / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        heights_path = scene_dir / "heights_m.npy"
        hfield_path = scene_dir / "terrain.hfield"
        texture_path = scene_dir / "terrain_contours.png"
        xml_path = scene_dir / "ant_standard_scene.xml"
        np.save(heights_path, heights, allow_pickle=False)
        hfield_path.write_bytes(
            struct.pack("<ii", heights.shape[0], heights.shape[1])
            + np.asarray(heights, dtype="<f4", order="C").tobytes(order="C")
        )
        colourise_heightfield(heights).save(texture_path)

        tree = ET.parse(source_xml)
        root = tree.getroot()
        hfield = root.find("./asset/hfield[@name='terrain']")
        texture = root.find("./asset/texture[@name='texplane']")
        floor = root.find("./worldbody/geom[@name='floor']")
        torso = root.find("./worldbody/body[@name='torso']")
        start_site = root.find("./worldbody/site[@name='fixed_start_marker']")
        goal_site = root.find("./worldbody/site[@name='fixed_goal_marker']")
        if None in (hfield, texture, floor, torso, start_site, goal_site):
            raise ValueError("Base XML is missing required standard-scene elements")
        height_range = float(np.ptp(heights))
        if height_range <= 0.0:
            raise ValueError("MuJoCo heightfield requires non-zero numerical range")
        assert hfield is not None and texture is not None and floor is not None
        assert torso is not None and start_site is not None and goal_site is not None
        hfield.set("file", hfield_path.name)
        hfield.set("size", f"{extent:.12g} {extent:.12g} {height_range:.12g} 1")
        texture.set("file", texture_path.name)
        floor.set("pos", f"0 0 {float(np.min(heights)):.12g}")
        floor.set("friction", "1 0.5 0.5")
        floor.set("condim", "3")
        position, quaternion, spawn = surface_pose(
            heights, extent=extent, xy=start, heading_rad=0.0
        )
        torso.set("pos", " ".join(f"{value:.12g}" for value in position))
        torso.set("quat", " ".join(f"{value:.12g}" for value in quaternion))
        start_z = terrain_value(heights, float(start[0]), float(start[1]), extent)
        goal_z = terrain_value(heights, float(goal[0]), float(goal[1]), extent)
        start_site.set("pos", f"{start[0]:.12g} {start[1]:.12g} {start_z + 0.04:.12g}")
        goal_site.set("pos", f"{goal[0]:.12g} {goal[1]:.12g} {goal_z + 0.04:.12g}")
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        compiled = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        observed_friction = np.asarray(compiled.geom_friction[floor_id], dtype=np.float64)
        if not np.allclose(observed_friction, expected_friction, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{name}: compiled friction changed: {observed_friction}")
        if int(compiled.geom_condim[floor_id]) != int(scene_config["condim"]):
            raise RuntimeError(f"{name}: compiled condim changed")
        if robot_signature(compiled) != expected_robot:
            raise RuntimeError(f"{name}: generated scene changed the frozen robot")
        spacing = 2.0 * extent / (heights.shape[0] - 1)
        dz_dy, dz_dx = np.gradient(heights, spacing, spacing)
        record = {
            "scene_name": name,
            "xml_path": str(xml_path),
            "xml_sha256": sha256(xml_path),
            "heights_path": str(heights_path),
            "heights_sha256": sha256(heights_path),
            "hfield_path": str(hfield_path),
            "hfield_sha256": sha256(hfield_path),
            "texture_path": str(texture_path),
            "texture_sha256": sha256(texture_path),
            "map_half_extent_m": extent,
            "start_xy_m": start.tolist(),
            "goal_xy_m": goal.tolist(),
            "minimum_height_m": float(np.min(heights)),
            "maximum_height_m": float(np.max(heights)),
            "height_range_m": height_range,
            "maximum_gradient_degrees": float(
                math.degrees(math.atan(float(np.max(np.hypot(dz_dx, dz_dy)))))
            ),
            "start_surface": spawn,
            "fixed_friction": observed_friction.tolist(),
            "condim": int(compiled.geom_condim[floor_id]),
            "robot_signature_matches_base": True,
        }
        write_json(scene_dir / "scene_manifest.json", record)
        record["scene_manifest_path"] = str(scene_dir / "scene_manifest.json")
        record["scene_manifest_sha256"] = sha256(scene_dir / "scene_manifest.json")
        records[name] = record
    generation = {
        "schema_version": "proxygap-standard-scene-manifest-v1",
        "base_xml": str(source_xml),
        "base_xml_sha256": sha256(source_xml),
        "robot_signature": expected_robot,
        "scenes": records,
    }
    write_json(output_root / "standard_scene_manifest.json", generation)
    return records, generation


def make_standard_env(
    config: dict[str, Any],
    reward_config: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    augment_local_terrain_observation: bool | None = None,
) -> FixedGoalTerrainWrapper:
    task = config["task_adapter"]
    curve_env = make_curved_gait_env(
        condition_id=condition_id,
        seed=seed,
        render_mode=None,
        xml_file=Path(scene["xml_path"]),
        max_episode_steps=max_episode_steps,
        terminate_when_unhealthy=False,
        profile="external",
        speed_min=cruise_speed,
        speed_max=cruise_speed,
        max_abs_curvature=float(task["maximum_abs_curvature_per_m"]),
        max_abs_lateral_speed=0.0,
        fixed_lateral_speed=0.0,
        heading_termination_enabled=False,
        terrain_frame_shaping_enabled=False,
        **common_env_kwargs(reward_config),
    )
    augment_preview = (
        bool(task["augment_local_terrain_observation"])
        if augment_local_terrain_observation is None
        else bool(augment_local_terrain_observation)
    )
    return FixedGoalTerrainWrapper(
        curve_env,
        heights_path=Path(scene["heights_path"]),
        expected_height_sha256=scene["heights_sha256"],
        map_half_extent_m=float(scene["map_half_extent_m"]),
        start_xy_m=scene["start_xy_m"],
        goal_xy_m=scene["goal_xy_m"],
        spawn_fraction=0.0,
        cruise_speed_m_per_s=cruise_speed,
        maximum_abs_curvature_per_m=float(task["maximum_abs_curvature_per_m"]),
        yaw_gain_per_second=float(task["yaw_gain_per_second"]),
        slow_radius_m=float(task["slow_radius_m"]),
        arrival_radius_m=float(task["arrival_radius_m"]),
        hold_radius_m=float(task["hold_radius_m"]),
        hold_seconds=float(task["hold_seconds"]),
        hold_speed_m_per_s=float(task["hold_speed_m_per_s"]),
        terminate_on_success=False,
        terrain_relative_healthy_clearance_m=tuple(
            float(value) for value in task["terrain_relative_healthy_clearance_m"]
        ),
        maximum_healthy_tilt_degrees=float(task["maximum_healthy_tilt_degrees"]),
        unhealthy_grace_steps=int(task["unhealthy_grace_steps"]),
        slip_speed_threshold_m_per_s=float(task["slip_speed_threshold_m_per_s"]),
        augment_local_terrain_observation=augment_preview,
        terrain_frame_shaping_enabled=False,
        terrain_preview_longitudinal_m=tuple(task["terrain_preview_longitudinal_m"]),
        terrain_preview_lateral_m=tuple(task["terrain_preview_lateral_m"]),
        local_terrain_height_bound_m=float(task["local_terrain_height_bound_m"]),
    )


def contact_masks_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: tuple[int, ...],
) -> tuple[np.ndarray, bool, bool]:
    foot_mask, nonfoot, torso, _ = contact_diagnostics_from_data(
        model, data, foot_geom_ids
    )
    return foot_mask, nonfoot, torso


def contact_diagnostics_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: tuple[int, ...],
) -> tuple[np.ndarray, bool, bool, np.ndarray]:
    """Return support state and per-foot maximum tangential contact speed."""
    foot_lookup = {int(geom_id): index for index, geom_id in enumerate(foot_geom_ids)}
    foot_mask = np.zeros(len(foot_geom_ids), dtype=bool)
    tangential_speeds = np.zeros(len(foot_geom_ids), dtype=np.float64)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
    nonfoot_robot_ground = False
    torso_ground = False
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom_1 = int(contact.geom1)
        geom_2 = int(contact.geom2)
        body_1 = int(model.geom_bodyid[geom_1])
        body_2 = int(model.geom_bodyid[geom_2])
        if body_1 == 0 and body_2 != 0:
            robot_geom = geom_2
        elif body_2 == 0 and body_1 != 0:
            robot_geom = geom_1
        else:
            continue
        if robot_geom in foot_lookup:
            foot_index = foot_lookup[robot_geom]
            foot_mask[foot_index] = True
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                np.asarray(contact.pos, dtype=np.float64),
                int(model.geom_bodyid[robot_geom]),
            )
            contact_velocity = jacobian_position @ np.asarray(
                data.qvel, dtype=np.float64
            )
            contact_normal = np.asarray(contact.frame[:3], dtype=np.float64)
            normal_norm = float(np.linalg.norm(contact_normal))
            if not np.isfinite(normal_norm) or normal_norm <= 1e-12:
                raise RuntimeError("MuJoCo returned an invalid contact normal")
            contact_normal /= normal_norm
            tangent_velocity = contact_velocity - float(
                np.dot(contact_velocity, contact_normal)
            ) * contact_normal
            tangential_speeds[foot_index] = max(
                tangential_speeds[foot_index],
                float(np.linalg.norm(tangent_velocity)),
            )
        else:
            nonfoot_robot_ground = True
            torso_ground = torso_ground or robot_geom == torso_id
    return foot_mask, nonfoot_robot_ground, torso_ground, tangential_speeds


def install_substep_contact_audit(env: FixedGoalTerrainWrapper) -> dict[str, Any]:
    ant = env.unwrapped
    if int(ant.frame_skip) != 5 or not math.isclose(float(ant.model.opt.timestep), 0.01):
        raise ValueError("High-frequency audit requires the frozen 5 x 0.01 s stepping contract")
    foot_ids = tuple(
        int(mujoco.mj_name2id(ant.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in FOOT_NAMES
    )
    if any(geom_id < 0 for geom_id in foot_ids):
        raise ValueError("High-frequency audit cannot find all four foot geometries")
    state: dict[str, Any] = {"last": None, "control_steps": 0}

    def audited_do_simulation(
        self: Any, ctrl: np.ndarray, n_frames: int
    ) -> None:
        if np.asarray(ctrl).shape != (self.model.nu,):
            raise ValueError("Action shape changed during substep contact audit")
        self.data.ctrl[:] = ctrl
        foot_masks: list[np.ndarray] = []
        nonfoot_masks: list[bool] = []
        torso_masks: list[bool] = []
        tangential_speed_rows: list[np.ndarray] = []
        for _ in range(int(n_frames)):
            mujoco.mj_step(self.model, self.data, nstep=1)
            feet, nonfoot, torso, tangential_speeds = contact_diagnostics_from_data(
                self.model, self.data, foot_ids
            )
            foot_masks.append(feet)
            nonfoot_masks.append(nonfoot)
            torso_masks.append(torso)
            tangential_speed_rows.append(tangential_speeds)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        state["last"] = {
            "foot_masks": np.asarray(foot_masks, dtype=bool),
            "nonfoot_robot_ground": np.asarray(nonfoot_masks, dtype=bool),
            "torso_ground": np.asarray(torso_masks, dtype=bool),
            "foot_tangential_speeds_m_per_s": np.asarray(
                tangential_speed_rows, dtype=np.float64
            ),
        }
        state["control_steps"] = int(state["control_steps"]) + 1

    ant.do_simulation = types.MethodType(audited_do_simulation, ant)
    return state


def summarise_substep_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No high-frequency contact rows were recorded")
    count = len(rows)
    endpoint_zero = np.asarray([bool(row["endpoint_zero_foot"]) for row in rows])
    full_interval_zero = np.asarray(
        [bool(row["full_interval_zero_foot"]) for row in rows]
    )
    earlier_contact = np.asarray(
        [bool(row["endpoint_zero_with_earlier_substep_contact"]) for row in rows]
    )
    endpoint_nonfoot = np.asarray(
        [bool(row["endpoint_nonfoot_robot_ground"]) for row in rows]
    )
    any_nonfoot = np.asarray(
        [bool(row["any_substep_nonfoot_robot_ground"]) for row in rows]
    )
    endpoint_torso = np.asarray([bool(row["endpoint_torso_ground"]) for row in rows])
    endpoint_count = max(1, int(np.sum(endpoint_zero)))
    slip_flags = np.asarray(
        [
            bool(value)
            for row in rows
            for value in json.loads(str(row.get("substep_any_slip_flags", "[]")))
        ],
        dtype=bool,
    )
    sustained_slip_minimum_seconds = float(
        rows[0].get("substep_sustained_slip_minimum_seconds", 0.2)
    )
    physics_dt = float(rows[0].get("physics_timestep_seconds", 0.01))
    minimum_slip_run = max(1, int(round(sustained_slip_minimum_seconds / physics_dt)))
    slip_runs: list[int] = []
    current_slip_run = 0
    for flag in slip_flags:
        if flag:
            current_slip_run += 1
        elif current_slip_run:
            slip_runs.append(current_slip_run)
            current_slip_run = 0
    if current_slip_run:
        slip_runs.append(current_slip_run)
    sustained_slip_runs = [run for run in slip_runs if run >= minimum_slip_run]
    return {
        "control_steps": count,
        "endpoint_zero_foot_fraction": float(np.mean(endpoint_zero)),
        "full_interval_zero_foot_fraction": float(np.mean(full_interval_zero)),
        "control_steps_with_any_substep_foot_contact_fraction": float(
            np.mean(~full_interval_zero)
        ),
        "mean_zero_foot_physics_substep_fraction": float(
            np.mean([float(row["zero_foot_substep_fraction"]) for row in rows])
        ),
        "endpoint_zero_with_earlier_substep_contact_fraction_of_all_steps": float(
            np.mean(earlier_contact)
        ),
        "endpoint_zero_with_earlier_substep_contact_fraction_conditional": float(
            np.sum(earlier_contact) / endpoint_count
        ),
        "endpoint_zero_with_endpoint_nonfoot_ground_fraction_conditional": float(
            np.sum(endpoint_zero & endpoint_nonfoot) / endpoint_count
        ),
        "endpoint_zero_with_any_substep_nonfoot_ground_fraction_conditional": float(
            np.sum(endpoint_zero & any_nonfoot) / endpoint_count
        ),
        "endpoint_zero_with_endpoint_torso_ground_fraction_conditional": float(
            np.sum(endpoint_zero & endpoint_torso) / endpoint_count
        ),
        "endpoint_vs_last_substep_foot_mask_mismatch_count": int(
            sum(int(row["endpoint_last_substep_mask_mismatch"]) for row in rows)
        ),
        "physics_substep_slip_speed_threshold_m_per_s": float(
            rows[0].get("substep_slip_speed_threshold_m_per_s", 0.2)
        ),
        "physics_substep_sustained_slip_minimum_seconds": sustained_slip_minimum_seconds,
        "physics_substep_sustained_slip_run_count": len(sustained_slip_runs),
        "physics_substep_sustained_slip_fraction": (
            float(sum(sustained_slip_runs)) / len(slip_flags)
            if slip_flags.size
            else 0.0
        ),
        "physics_substep_maximum_contact_tangential_speed_m_per_s": float(
            max(
                (
                    float(row.get("maximum_substep_contact_tangential_speed_m_per_s", 0.0))
                    for row in rows
                ),
                default=0.0,
            )
        ),
        "interpretation_boundary": (
            "full_interval_zero_foot means no named distal foot contacted world geometry "
            "during any of five sampled physics substeps; non-foot ground contact is reported separately"
        ),
    }


def vector_sum(summary: dict[str, Any], key: str) -> float:
    return float(np.sum(np.asarray(summary.get(key, []), dtype=np.float64)))


def evaluate_episode(
    model: PPO,
    config: dict[str, Any],
    reward_config: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    trace_path: Path | None = None,
    high_frequency_contact: bool = False,
    augment_local_terrain_observation: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    env = make_standard_env(
        config,
        reward_config,
        scene,
        condition_id=condition_id,
        seed=seed,
        max_episode_steps=max_episode_steps,
        cruise_speed=cruise_speed,
        augment_local_terrain_observation=augment_local_terrain_observation,
    )
    observation, _ = env.reset(seed=seed)
    substep_state = install_substep_contact_audit(env) if high_frequency_contact else None
    deterministic = True
    dt = float(env.unwrapped.dt)
    threshold = float(config["task_adapter"]["slip_speed_threshold_m_per_s"])
    sustained_seconds = float(config["diagnosis"]["slip_transient_minimum_seconds"])
    sustained_steps = int(round(sustained_seconds / dt))
    slip_runs: list[int] = []
    current_slip = 0
    support_sum = 0.0
    relative_tilt_squared_sum = 0.0
    relative_tilt_sum = 0.0
    relative_tilt_max = 0.0
    endpoint_nonfoot_count = 0
    endpoint_torso_count = 0
    trace_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        contact_mask = np.asarray(
            info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool
        )
        contact_speeds = np.asarray(
            info.get(
                "proxygap_foot_contact_tangential_speeds_m_per_s_step",
                np.zeros(4),
            ),
            dtype=np.float64,
        )
        active_speeds = contact_speeds[contact_mask]
        maximum_contact_speed = float(np.max(active_speeds)) if active_speeds.size else 0.0
        slip_flag = bool(maximum_contact_speed > threshold)
        if slip_flag:
            current_slip += 1
        elif current_slip:
            slip_runs.append(current_slip)
            current_slip = 0
        support_count = int(np.sum(contact_mask))
        support_sum += support_count
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        x, y = float(qpos[0]), float(qpos[1])
        normal = env._terrain_normal(x, y)
        relative_tilt = quaternion_tilt_relative_to_normal(qpos[3:7], normal)
        relative_tilt_sum += relative_tilt
        relative_tilt_squared_sum += relative_tilt**2
        relative_tilt_max = max(relative_tilt_max, relative_tilt)
        foot_ids = tuple(
            int(mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in FOOT_NAMES
        )
        endpoint_feet, endpoint_nonfoot, endpoint_torso = contact_masks_from_data(
            env.unwrapped.model, env.unwrapped.data, foot_ids
        )
        endpoint_nonfoot_count += int(endpoint_nonfoot)
        endpoint_torso_count += int(endpoint_torso)
        if not np.array_equal(endpoint_feet, contact_mask):
            raise RuntimeError("Independent endpoint contact mask disagrees with wrapper info")

        if substep_state is not None:
            last = substep_state.get("last")
            if last is None:
                raise RuntimeError("Substep audit did not record the current action")
            foot_masks = np.asarray(last["foot_masks"], dtype=bool)
            nonfoot = np.asarray(last["nonfoot_robot_ground"], dtype=bool)
            torso = np.asarray(last["torso_ground"], dtype=bool)
            substep_tangential_speeds = np.asarray(
                last["foot_tangential_speeds_m_per_s"], dtype=np.float64
            )
            if foot_masks.shape != (int(env.unwrapped.frame_skip), 4):
                raise RuntimeError("Unexpected substep contact matrix shape")
            if substep_tangential_speeds.shape != foot_masks.shape:
                raise RuntimeError("Unexpected substep tangential-speed matrix shape")
            zero_by_substep = ~np.any(foot_masks, axis=1)
            substep_slip_flags = np.any(
                foot_masks & (substep_tangential_speeds > threshold), axis=1
            )
            substep_rows.append(
                {
                    "condition_id": condition_id,
                    "scene_name": scene["scene_name"],
                    "evaluation_seed": seed,
                    "step": step,
                    "time_seconds": step * dt,
                    "endpoint_foot_mask": json.dumps(contact_mask.astype(int).tolist()),
                    "substep_foot_masks": json.dumps(foot_masks.astype(int).tolist()),
                    "per_foot_substep_duty_fraction": json.dumps(
                        np.mean(foot_masks, axis=0).tolist()
                    ),
                    "substep_foot_tangential_speeds_m_per_s": json.dumps(
                        substep_tangential_speeds.tolist()
                    ),
                    "substep_any_slip_flags": json.dumps(
                        substep_slip_flags.astype(int).tolist()
                    ),
                    "substep_slip_speed_threshold_m_per_s": threshold,
                    "substep_sustained_slip_minimum_seconds": sustained_seconds,
                    "physics_timestep_seconds": float(env.unwrapped.model.opt.timestep),
                    "maximum_substep_contact_tangential_speed_m_per_s": float(
                        np.max(substep_tangential_speeds)
                    ),
                    "endpoint_zero_foot": bool(not np.any(contact_mask)),
                    "full_interval_zero_foot": bool(np.all(zero_by_substep)),
                    "zero_foot_substep_fraction": float(np.mean(zero_by_substep)),
                    "endpoint_zero_with_earlier_substep_contact": bool(
                        not np.any(contact_mask) and np.any(foot_masks[:-1])
                    ),
                    "endpoint_nonfoot_robot_ground": bool(nonfoot[-1]),
                    "any_substep_nonfoot_robot_ground": bool(np.any(nonfoot)),
                    "endpoint_torso_ground": bool(torso[-1]),
                    "any_substep_torso_ground": bool(np.any(torso)),
                    "endpoint_last_substep_mask_mismatch": bool(
                        not np.array_equal(contact_mask, foot_masks[-1])
                    ),
                }
            )

        if trace_path is not None:
            goal = np.asarray(scene["goal_xy_m"], dtype=np.float64)
            distance = float(np.linalg.norm(goal - qpos[:2]))
            trace_rows.append(
                {
                    "condition_id": condition_id,
                    "scene_name": scene["scene_name"],
                    "evaluation_seed": seed,
                    "step": step,
                    "time_seconds": step * dt,
                    "x_m": x,
                    "y_m": y,
                    "terrain_height_m": float(env._terrain_height(x, y)),
                    "torso_z_m": float(qpos[2]),
                    "distance_to_goal_m": distance,
                    "support_count": support_count,
                    "foot_contact_mask": json.dumps(contact_mask.astype(int).tolist()),
                    "airborne_endpoint": bool(not np.any(contact_mask)),
                    "relative_torso_tilt_rad": relative_tilt,
                    "maximum_contact_tangential_speed_m_per_s": maximum_contact_speed,
                    "contact_speed_threshold_exceeded": slip_flag,
                    "endpoint_nonfoot_robot_ground": endpoint_nonfoot,
                    "endpoint_torso_ground": endpoint_torso,
                    "applied_action": json.dumps(
                        np.asarray(info.get("proxygap_applied_action", action)).tolist(),
                        separators=(",", ":"),
                    ),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    if current_slip:
        slip_runs.append(current_slip)
    summary = env.episode_summary()
    env.close()
    elapsed = max(1, step)
    sustained_runs = [run for run in slip_runs if run >= sustained_steps]
    longest_gaps = np.asarray(
        summary["longest_foot_no_contact_run_seconds_by_foot"], dtype=np.float64
    )
    row = {
        "condition_id": condition_id,
        "scene_name": scene["scene_name"],
        "evaluation_seed": seed,
        "checkpoint_timesteps": int(model.num_timesteps),
        **summary,
        "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"])
        - float(summary["fixed_goal_minimum_distance_m"]),
        "mean_support_count": support_sum / elapsed,
        "relative_torso_tilt_mean_rad": relative_tilt_sum / elapsed,
        "relative_torso_tilt_rms_rad": math.sqrt(relative_tilt_squared_sum / elapsed),
        "relative_torso_tilt_max_rad": relative_tilt_max,
        "longest_per_foot_no_contact_seconds_max": float(np.max(longest_gaps)),
        "corrected_sustained_slip_minimum_seconds": sustained_seconds,
        "corrected_sustained_slip_run_count": len(sustained_runs),
        "corrected_sustained_slip_step_count": int(sum(sustained_runs)),
        "corrected_sustained_slip_step_fraction": float(sum(sustained_runs)) / elapsed,
        "endpoint_nonfoot_robot_ground_fraction": endpoint_nonfoot_count / elapsed,
        "endpoint_torso_ground_fraction": endpoint_torso_count / elapsed,
        "actuator_abs_torque_time_integral_total_n_m_s": vector_sum(
            summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator"
        ),
        "actuator_positive_mechanical_work_total_j": vector_sum(
            summary, "actuator_positive_mechanical_work_j_by_actuator"
        ),
        "actuator_abs_mechanical_work_total_j": vector_sum(
            summary, "actuator_abs_mechanical_work_j_by_actuator"
        ),
    }
    if trace_path is not None:
        write_rows(trace_path, trace_rows)
    substep_summary = None
    if substep_rows:
        substep_summary = {
            "condition_id": condition_id,
            "scene_name": scene["scene_name"],
            "evaluation_seed": seed,
            **summarise_substep_rows(substep_rows),
        }
    return row, (
        {"summary": substep_summary, "rows": substep_rows}
        if substep_summary is not None
        else None
    )


def evaluate_matrix(
    model: PPO,
    config: dict[str, Any],
    reward_config: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    *,
    condition_id: str,
    seeds: list[int],
    max_episode_steps: int,
    cruise_speed: float,
    output_root: Path,
    trace_seed: int | None,
    augment_local_terrain_observation: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for scene_name in config["standard_scenes"]["scene_order"]:
        scene = scenes[scene_name]
        for seed in seeds:
            trace_path = (
                output_root
                / "traces"
                / f"{condition_id.lower()}_{scene_name}_seed_{seed}_trace.csv"
                if trace_seed is not None and int(seed) == int(trace_seed)
                else None
            )
            row, _ = evaluate_episode(
                model,
                config,
                reward_config,
                scene,
                condition_id=condition_id,
                seed=int(seed),
                max_episode_steps=max_episode_steps,
                cruise_speed=cruise_speed,
                trace_path=trace_path,
                augment_local_terrain_observation=(
                    augment_local_terrain_observation
                ),
            )
            rows.append(row)
            if trace_path is not None:
                trace_records.append(
                    {
                        "condition_id": condition_id,
                        "scene_name": scene_name,
                        "evaluation_seed": int(seed),
                        "path": str(trace_path),
                        "sha256": sha256(trace_path),
                        "rows": int(row["episode_length"]),
                    }
                )
    return rows, trace_records


def high_frequency_contact_matrix(
    model: PPO,
    config: dict[str, Any],
    reward_config: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    output_root: Path,
    augment_local_terrain_observation: bool | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene_name in config["diagnosis"]["high_frequency_contact_scenes"]:
        row, substep = evaluate_episode(
            model,
            config,
            reward_config,
            scenes[scene_name],
            condition_id=condition_id,
            seed=seed,
            max_episode_steps=max_episode_steps,
            cruise_speed=cruise_speed,
            high_frequency_contact=True,
            augment_local_terrain_observation=augment_local_terrain_observation,
        )
        assert substep is not None
        trace_path = (
            output_root
            / "high_frequency_contact"
            / f"{condition_id.lower()}_{scene_name}_seed_{seed}_substeps.csv"
        )
        write_rows(trace_path, substep["rows"])
        records.append(
            {
                **substep["summary"],
                "episode_length": int(row["episode_length"]),
                "trace_path": str(trace_path),
                "trace_sha256": sha256(trace_path),
            }
        )
    return records


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "fixed_goal_best_progress_m",
        "fixed_goal_net_progress_m",
        "task_airborne_step_fraction",
        "longest_airborne_run_seconds",
        "mean_support_count",
        "relative_torso_tilt_rms_rad",
        "corrected_sustained_slip_step_fraction",
        "endpoint_nonfoot_robot_ground_fraction",
        "cumulative_squared_action",
        "actuator_abs_torque_time_integral_total_n_m_s",
        "actuator_positive_mechanical_work_total_j",
        "actuator_abs_mechanical_work_total_j",
    )
    result: dict[str, Any] = {
        "episode_count": len(rows),
        "fall_count": int(sum(bool(row["fall"]) for row in rows)),
        "success_count": int(sum(bool(row["fixed_goal_success"]) for row in rows)),
        "success_rate": float(
            np.mean([bool(row["fixed_goal_success"]) for row in rows])
        ),
        "fixed_goal_final_distance_m_mean": float(
            np.mean([float(row["fixed_goal_final_distance_m"]) for row in rows])
        ),
        "termination_category_counts": {
            str(category): int(
                sum(str(row["termination_category"]) == str(category) for row in rows)
            )
            for category in sorted({str(row["termination_category"]) for row in rows})
        },
        "scene_aggregates": {},
    }
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_std_population"] = float(np.std(values))
    for scene_name in sorted({str(row["scene_name"]) for row in rows}):
        scene_rows = [row for row in rows if row["scene_name"] == scene_name]
        result["scene_aggregates"][scene_name] = {
            "episode_count": len(scene_rows),
            "fall_count": int(sum(bool(row["fall"]) for row in scene_rows)),
            **{
                f"{field}_mean": float(
                    np.mean([float(row[field]) for row in scene_rows])
                )
                for field in fields
            },
        }
    return result


def intervention_selection(
    config: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    substep_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = config["intervention_selection_gate"]
    mean_airborne = float(
        np.mean([float(row["task_airborne_step_fraction"]) for row in baseline_rows])
    )
    scene_gap_count = 0
    for scene_name in config["standard_scenes"]["scene_order"]:
        scene_values = [
            float(row["longest_per_foot_no_contact_seconds_max"])
            for row in baseline_rows
            if row["scene_name"] == scene_name
        ]
        if scene_values and float(np.mean(scene_values)) >= float(
            gate["longest_per_foot_gap_threshold_seconds"]
        ):
            scene_gap_count += 1
    maximum_full_interval_airborne = max(
        float(record["full_interval_zero_foot_fraction"])
        for record in substep_records
    )
    criteria = {
        "mean_endpoint_airborne_fraction": mean_airborne,
        "scenes_with_longest_per_foot_gap_above_threshold": scene_gap_count,
        "maximum_full_interval_airborne_fraction_in_audited_scenes": (
            maximum_full_interval_airborne
        ),
    }
    selected = bool(
        mean_airborne >= float(gate["minimum_mean_endpoint_airborne_fraction"])
        and scene_gap_count
        >= int(gate["minimum_scenes_with_longest_per_foot_gap_above_seconds"])
        and maximum_full_interval_airborne
        >= float(gate["minimum_full_interval_airborne_fraction_in_any_audited_scene"])
    )
    return {
        "schema_version": "proxygap-standard-support-intervention-selection-v1",
        "predeclared_gate": gate,
        "observed": criteria,
        "selected": selected,
        "selected_intervention": (
            "per_foot_contact_gap_weight_only" if selected else None
        ),
        "reason": (
            "Endpoint and full-interval audits both show substantial support gaps; "
            "screen the existing per-foot gap term without changing timing constants."
            if selected
            else "The predeclared evidence gate was not met; bounded training was not started."
        ),
    }


def make_vector_env(
    config: dict[str, Any],
    reward_config: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    monitor_path: Path,
    smoke: bool,
) -> VecMonitor:
    factories: list[Callable[[], gym.Env]] = []
    for rank, scene_name in enumerate(config["standard_scenes"]["scene_order"]):
        scene = scenes[scene_name]
        local_seed = int(seed) + 1000 * rank

        def factory(
            local_scene: dict[str, Any] = scene,
            env_seed: int = local_seed,
        ) -> gym.Env:
            return make_standard_env(
                config,
                reward_config,
                local_scene,
                condition_id=condition_id,
                seed=env_seed,
                max_episode_steps=max_episode_steps,
                cruise_speed=cruise_speed,
            )

        factories.append(factory)
    base = (
        DummyVecEnv(factories)
        if smoke
        else SubprocVecEnv(
            factories,
            start_method=str(config["execution"]["subprocess_start_method"]),
        )
    )
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return VecMonitor(base, filename=str(monitor_path))


def comparison_summary(
    config: dict[str, Any],
    paired_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in paired_rows:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    control_id = "MATCHED_CONTACT_GAP_W0_CONTROL"
    intervention_id = "CONTACT_GAP_W1_INTERVENTION"
    control = aggregate_rows(by_condition[control_id])
    intervention = aggregate_rows(by_condition[intervention_id])
    gate = config["paired_evaluation"]["retention_gate"]
    airborne_reduction = (
        control["task_airborne_step_fraction_mean"]
        - intervention["task_airborne_step_fraction_mean"]
    )
    support_increase = (
        intervention["mean_support_count_mean"] - control["mean_support_count_mean"]
    )
    progress_ratio = (
        intervention["fixed_goal_best_progress_m_mean"]
        / control["fixed_goal_best_progress_m_mean"]
        if control["fixed_goal_best_progress_m_mean"] > 1e-12
        else float("nan")
    )
    fall_delta = intervention["fall_count"] - control["fall_count"]
    slip_increase = (
        intervention["corrected_sustained_slip_step_fraction_mean"]
        - control["corrected_sustained_slip_step_fraction_mean"]
    )
    passed = bool(
        airborne_reduction
        >= float(gate["minimum_absolute_endpoint_airborne_fraction_reduction"])
        and support_increase >= float(gate["minimum_mean_support_count_increase"])
        and np.isfinite(progress_ratio)
        and progress_ratio >= float(gate["minimum_progress_ratio_to_matched_control"])
        and fall_delta <= int(gate["maximum_additional_falls"])
        and slip_increase <= float(gate["maximum_sustained_slip_fraction_increase"])
    )
    paired_deltas: list[dict[str, Any]] = []
    keys = {
        (str(row["scene_name"]), int(row["evaluation_seed"]))
        for row in by_condition[control_id]
    }
    control_lookup = {
        (str(row["scene_name"]), int(row["evaluation_seed"])): row
        for row in by_condition[control_id]
    }
    intervention_lookup = {
        (str(row["scene_name"]), int(row["evaluation_seed"])): row
        for row in by_condition[intervention_id]
    }
    delta_fields = (
        "fixed_goal_best_progress_m",
        "task_airborne_step_fraction",
        "mean_support_count",
        "relative_torso_tilt_rms_rad",
        "corrected_sustained_slip_step_fraction",
    )
    for key in sorted(keys):
        left = control_lookup[key]
        right = intervention_lookup[key]
        paired_deltas.append(
            {
                "scene_name": key[0],
                "evaluation_seed": key[1],
                **{
                    f"delta_{field}_intervention_minus_control": float(right[field])
                    - float(left[field])
                    for field in delta_fields
                },
                "delta_fall_intervention_minus_control": int(bool(right["fall"]))
                - int(bool(left["fall"])),
            }
        )
    result = {
        "schema_version": "proxygap-standard-support-comparison-v1",
        "condition_aggregates": {
            control_id: control,
            intervention_id: intervention,
        },
        "paired_deltas": paired_deltas,
        "retention_gate": {
            "predeclared_rule": gate,
            "observed_absolute_endpoint_airborne_fraction_reduction": airborne_reduction,
            "observed_mean_support_count_increase": support_increase,
            "observed_best_progress_ratio": progress_ratio,
            "observed_additional_falls": fall_delta,
            "observed_sustained_slip_fraction_increase": slip_increase,
            "passed": passed,
            "paired_control_condition": control_id,
            "promoted_condition": None,
            "incumbent_condition": "SOURCE_STAGE1_STANDARD_DIAGNOSIS",
        },
        "energy_boundary": (
            "ctrl_cost_weight remains 0.5; action, torque and work are diagnostics; "
            "relative-energy V2 remains measurement-only"
        ),
        "claim_boundary": config["paired_evaluation"]["claim_boundary"],
        "measurement_boundary": (
            "The legacy sustained-slip field is sampled at control-step endpoints; "
            "it is not a physics-substep-corrected slip integral. Promotion also requires "
            "a separately predeclared source-incumbent comparison."
        ),
    }
    if source_rows is not None:
        source = aggregate_rows(source_rows)
        result["source_incumbent_screen"] = {
            "SOURCE_STAGE1_STANDARD_DIAGNOSIS": source,
            "MATCHED_CONTACT_GAP_W0_CONTROL": control,
            "control_to_source_best_progress_ratio": float(
                control["fixed_goal_best_progress_m_mean"]
                / source["fixed_goal_best_progress_m_mean"]
            ),
            "control_minus_source_endpoint_airborne_fraction": float(
                control["task_airborne_step_fraction_mean"]
                - source["task_airborne_step_fraction_mean"]
            ),
            "control_minus_source_mean_support_count": float(
                control["mean_support_count_mean"] - source["mean_support_count_mean"]
            ),
            "promotion_decision": "no_candidate_promoted_source_remains_incumbent",
        }
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, base_reward = validate_config(config)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else ROOT
        / (
            config["execution"]["smoke_output_root"]
            if args.smoke
            else config["execution"]["output_root"]
        )
        / f"seed_{config['training']['training_seed']}"
    )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing support-pilot output root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    frozen_config = output_root / "frozen_run_config.json"
    frozen_config.write_bytes(config_path.read_bytes())
    scenes, scene_manifest = prepare_standard_scenes(config, output_root)
    execution = {
        "schema_version": "proxygap-standard-support-execution-v1",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "frozen_config": str(frozen_config),
        "frozen_config_sha256": sha256(frozen_config),
        "output_root": str(output_root),
        "smoke": bool(args.smoke),
        "diagnose_only": bool(args.diagnose_only),
        "source_checkpoint": str(ROOT / config["frozen_sources"]["source_checkpoint"]),
        "source_checkpoint_sha256": config["frozen_sources"]["source_checkpoint_sha256"],
        "standard_scene_manifest": str(output_root / "standard_scene_manifest.json"),
        "standard_scene_manifest_sha256": sha256(
            output_root / "standard_scene_manifest.json"
        ),
        "friction": config["standard_scenes"]["fixed_friction"],
        "condim": config["standard_scenes"]["condim"],
        "energy_formula_changed": False,
    }
    write_json(output_root / "execution_record.json", execution)
    if args.validate_only:
        print(json.dumps({"status": "validated", **execution}, indent=2))
        return

    source_path = ROOT / config["frozen_sources"]["source_checkpoint"]
    source_model = PPO.load(source_path, device=str(config["ppo"]["device"]))
    comparator_config_path = ROOT / config["frozen_sources"][
        "diagnostic_support_comparator_configuration"
    ]
    comparator_reward = verified_json(
        comparator_config_path,
        config["frozen_sources"][
            "diagnostic_support_comparator_configuration_sha256"
        ],
    )
    comparator_path = ROOT / config["frozen_sources"][
        "diagnostic_support_comparator_checkpoint"
    ]
    comparator_model = PPO.load(
        comparator_path, device=str(config["ppo"]["device"])
    )
    diagnosis = config["diagnosis"]
    diagnosis_steps = 120 if args.smoke else int(diagnosis["max_episode_steps"])
    diagnosis_seeds = [int(diagnosis["evaluation_seeds"][0])] if args.smoke else [
        int(value) for value in diagnosis["evaluation_seeds"]
    ]
    baseline_rows, baseline_traces = evaluate_matrix(
        source_model,
        config,
        base_reward,
        scenes,
        condition_id="SOURCE_STAGE1_STANDARD_DIAGNOSIS",
        seeds=diagnosis_seeds,
        max_episode_steps=diagnosis_steps,
        cruise_speed=float(diagnosis["cruise_speed_m_per_s"]),
        output_root=output_root,
        trace_seed=(
            diagnosis_seeds[0]
            if args.smoke
            else int(diagnosis["representative_trace_seed"])
        ),
    )
    write_rows(output_root / "logs" / "baseline_standard_diagnosis.csv", baseline_rows)
    high_frequency_steps = (
        40 if args.smoke else int(diagnosis["high_frequency_contact_steps"])
    )
    substep_records = high_frequency_contact_matrix(
        source_model,
        config,
        base_reward,
        scenes,
        condition_id="SOURCE_STAGE1_SUBSTEP_AUDIT",
        seed=int(diagnosis["high_frequency_contact_seed"]),
        max_episode_steps=high_frequency_steps,
        cruise_speed=float(diagnosis["cruise_speed_m_per_s"]),
        output_root=output_root,
    )
    comparator_rows, comparator_traces = evaluate_matrix(
        comparator_model,
        config,
        comparator_reward,
        scenes,
        condition_id="V20_SUPPORT_DIAGNOSTIC",
        seeds=diagnosis_seeds,
        max_episode_steps=diagnosis_steps,
        cruise_speed=float(diagnosis["cruise_speed_m_per_s"]),
        output_root=output_root,
        trace_seed=(
            diagnosis_seeds[0]
            if args.smoke
            else int(diagnosis["representative_trace_seed"])
        ),
        augment_local_terrain_observation=False,
    )
    write_rows(output_root / "logs" / "v20_standard_diagnosis.csv", comparator_rows)
    comparator_substeps = high_frequency_contact_matrix(
        comparator_model,
        config,
        comparator_reward,
        scenes,
        condition_id="V20_SUPPORT_SUBSTEP_AUDIT",
        seed=int(diagnosis["high_frequency_contact_seed"]),
        max_episode_steps=high_frequency_steps,
        cruise_speed=float(diagnosis["cruise_speed_m_per_s"]),
        output_root=output_root,
        augment_local_terrain_observation=False,
    )
    write_json(
        output_root / "high_frequency_contact" / "source_substep_summary.json",
        substep_records,
    )
    selection = intervention_selection(config, baseline_rows, substep_records)
    write_json(output_root / "intervention_selection.json", selection)
    diagnosis_summary = {
        "schema_version": "proxygap-standard-support-diagnosis-v1",
        "aggregate": aggregate_rows(baseline_rows),
        "high_frequency_contact": substep_records,
        "v20_support_comparator": {
            "checkpoint": str(comparator_path),
            "checkpoint_sha256": sha256(comparator_path),
            "observation_dimension": 118,
            "aggregate": aggregate_rows(comparator_rows),
            "high_frequency_contact": comparator_substeps,
            "traces": comparator_traces,
            "training_source": False,
        },
        "intervention_selection": selection,
        "traces": baseline_traces,
        "measurement_boundary": (
            "Endpoint contact is retained for compatibility. High-frequency rows distinguish "
            "an endpoint miss from an interval with no distal-foot contact and report non-foot ground contact."
        ),
    }
    write_json(output_root / "diagnosis_summary.json", diagnosis_summary)
    if args.diagnose_only or not bool(selection["selected"]):
        print(
            json.dumps(
                {
                    "status": "diagnosis_complete",
                    "output_root": str(output_root),
                    "selected": selection["selected"],
                    "diagnosis_summary_sha256": sha256(output_root / "diagnosis_summary.json"),
                },
                indent=2,
            )
        )
        return

    training = config["training"]
    training_steps = (
        int(config["ppo"]["n_steps"]) * int(training["parallel_environments"])
        if args.smoke
        else int(training["additional_target_timesteps_per_variant"])
    )
    training_max_steps = 120 if args.smoke else int(training["max_episode_steps"])
    runtime_rows: list[dict[str, Any]] = []
    models: dict[str, PPO] = {}
    checkpoint_records: list[dict[str, Any]] = []
    for variant in training["variants"]:
        condition_id = str(variant["condition_id"])
        reward_config = reward_config_with_contact_gap_weight(
            base_reward, float(variant["foot_contact_gap_shaping_weight"])
        )
        monitor_path = output_root / "logs" / f"{condition_id.lower()}_vecmonitor.csv"
        vector = make_vector_env(
            config,
            reward_config,
            scenes,
            condition_id=condition_id,
            seed=int(training["training_seed"]),
            max_episode_steps=training_max_steps,
            cruise_speed=float(training["cruise_speed_m_per_s"]),
            monitor_path=monitor_path,
            smoke=args.smoke,
        )
        model = _configure_continuation_model(
            source_path,
            vector,
            config["ppo"],
            training_seed=int(training["training_seed"]),
            smoke=args.smoke,
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=training_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        model_dir = output_root / "models" / condition_id.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = model_dir / f"checkpoint_{int(model.num_timesteps)}.zip"
        model.save(checkpoint)
        runtime_rows.append(
            {
                "condition_id": condition_id,
                "additional_training_timesteps": training_steps,
                "train_elapsed_seconds": elapsed,
                "steps_per_second": training_steps / max(elapsed, 1e-12),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        checkpoint_records.append(
            {
                "condition_id": condition_id,
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "num_timesteps": int(model.num_timesteps),
            }
        )
        models[condition_id] = model
        vector.close()
    write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)

    paired = config["paired_evaluation"]
    paired_steps = 120 if args.smoke else int(paired["max_episode_steps"])
    paired_seeds = [int(paired["seeds"][0])] if args.smoke else [
        int(value) for value in paired["seeds"]
    ]
    trace_seed = paired_seeds[0] if args.smoke else int(paired["representative_trace_seed"])
    paired_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    post_training_substeps: dict[str, list[dict[str, Any]]] = {}
    for variant in training["variants"]:
        condition_id = str(variant["condition_id"])
        reward_config = reward_config_with_contact_gap_weight(
            base_reward, float(variant["foot_contact_gap_shaping_weight"])
        )
        rows, traces = evaluate_matrix(
            models[condition_id],
            config,
            reward_config,
            scenes,
            condition_id=condition_id,
            seeds=paired_seeds,
            max_episode_steps=paired_steps,
            cruise_speed=float(paired["cruise_speed_m_per_s"]),
            output_root=output_root,
            trace_seed=trace_seed,
        )
        paired_rows.extend(rows)
        trace_records.extend(traces)
        post_training_substeps[condition_id] = high_frequency_contact_matrix(
            models[condition_id],
            config,
            reward_config,
            scenes,
            condition_id=condition_id,
            seed=int(diagnosis["high_frequency_contact_seed"]),
            max_episode_steps=high_frequency_steps,
            cruise_speed=float(paired["cruise_speed_m_per_s"]),
            output_root=output_root,
        )
    write_rows(output_root / "logs" / "paired_evaluation_episodes.csv", paired_rows)
    write_json(output_root / "high_frequency_contact" / "post_training_summary.json", post_training_substeps)
    comparison = comparison_summary(config, paired_rows, source_rows=baseline_rows)
    comparison["post_training_high_frequency_contact"] = post_training_substeps
    control_substep = float(
        np.mean(
            [
                row["full_interval_zero_foot_fraction"]
                for row in post_training_substeps["MATCHED_CONTACT_GAP_W0_CONTROL"]
            ]
        )
    )
    intervention_substep = float(
        np.mean(
            [
                row["full_interval_zero_foot_fraction"]
                for row in post_training_substeps["CONTACT_GAP_W1_INTERVENTION"]
            ]
        )
    )
    comparison["post_training_full_substep_screen"] = {
        "matched_w0_mean_full_interval_zero_foot_fraction": control_substep,
        "contact_gap_w1_mean_full_interval_zero_foot_fraction": intervention_substep,
        "observed_absolute_reduction": control_substep - intervention_substep,
        "promotion_use": False,
        "reason": "No full-substep threshold was predeclared in this v1 frozen configuration.",
    }
    write_json(output_root / "comparison_summary.json", comparison)

    manifest = {
        "schema_version": "proxygap-standard-support-pilot-manifest-v1",
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "frozen_config": {"path": str(frozen_config), "sha256": sha256(frozen_config)},
        "source_checkpoint": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "diagnostic_support_comparator": {
            "path": str(comparator_path),
            "sha256": sha256(comparator_path),
            "configuration": str(comparator_config_path),
            "configuration_sha256": sha256(comparator_config_path),
            "training_source": False,
        },
        "standard_scene_manifest": {
            "path": str(output_root / "standard_scene_manifest.json"),
            "sha256": sha256(output_root / "standard_scene_manifest.json"),
        },
        "diagnosis_summary": {
            "path": str(output_root / "diagnosis_summary.json"),
            "sha256": sha256(output_root / "diagnosis_summary.json"),
        },
        "intervention_selection": {
            "path": str(output_root / "intervention_selection.json"),
            "sha256": sha256(output_root / "intervention_selection.json"),
        },
        "checkpoints": checkpoint_records,
        "paired_episode_log": {
            "path": str(output_root / "logs" / "paired_evaluation_episodes.csv"),
            "sha256": sha256(output_root / "logs" / "paired_evaluation_episodes.csv"),
        },
        "comparison_summary": {
            "path": str(output_root / "comparison_summary.json"),
            "sha256": sha256(output_root / "comparison_summary.json"),
        },
        "representative_video_inputs": {
            "selection_rule": "predeclared paired seed across every standard scene; no best-looking selection",
            "trace_seed": trace_seed,
            "traces": trace_records,
            "video_rendered": False,
        },
        "failed_runs_retained": True,
        "existing_results_overwritten": False,
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "pilot_complete",
                "output_root": str(output_root),
                "retention_gate": comparison["retention_gate"],
                "manifest_sha256": sha256(output_root / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
