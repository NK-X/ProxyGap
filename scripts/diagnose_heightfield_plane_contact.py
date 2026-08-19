"""Compare plane and near-flat heightfield contact under matched inputs.

This is a read-only locomotion diagnostic.  It creates new scene files, loads
one frozen policy only to generate a reference action sequence, and then
replays the exact same initial state and controls in every MuJoCo model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
from PIL import Image
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for location in (SRC, SCRIPTS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from run_fixed_standard_support_curriculum import (  # noqa: E402
    FOOT_NAMES,
    make_standard_env,
    robot_signature,
    terrain_value,
)


DEFAULT_CONFIG = ROOT / "configs" / "heightfield_plane_contact_diagnostic_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
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
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_bounded_heightfield_contact_diagnostic":
        raise ValueError("Diagnostic configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("Diagnostic must prohibit a generalisation claim")
    frozen = config["frozen_sources"]
    for path_key, hash_key in (
        ("base_scene_xml", "base_scene_xml_sha256"),
        ("reward_configuration", "reward_configuration_sha256"),
        ("source_checkpoint", "source_checkpoint_sha256"),
    ):
        path = ROOT / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError(f"Frozen source missing or changed: {path}")
    if (int(frozen["observation_dimension"]), int(frozen["action_dimension"])) != (135, 8):
        raise ValueError("Only the frozen 135D/8D interface is permitted")
    scene = config["controlled_scene"]
    if list(scene["fixed_friction"]) != [1.0, 0.5, 0.5] or int(scene["condim"]) != 3:
        raise ValueError("Friction or contact dimension changed")
    if float(scene["physics_timestep_seconds"]) != 0.01 or int(scene["control_substeps"]) != 5:
        raise ValueError("Physics/control timing changed")
    ids = [item["id"] for item in scene["surface_variants"]]
    if ids != [
        "native_plane",
        "hfield_plateau_129",
        "hfield_plateau_257",
        "hfield_plateau_513",
        "hfield_microrelief_257",
    ]:
        raise ValueError("Surface comparison set changed")
    if config["energy_boundary"] != {
        "energy_formula_changes": "prohibited",
        "energy_reward_changes": "prohibited",
    }:
        raise ValueError("Energy boundary changed")


def surface_heights(kind: str, resolution: int, extent: float, config: dict[str, Any]) -> np.ndarray:
    if resolution < 3 or resolution % 2 == 0:
        raise ValueError("Resolution must be odd and at least three")
    if kind == "plane":
        return np.zeros((resolution, resolution), dtype=np.float64)
    if kind == "plateau":
        heights = np.zeros((resolution, resolution), dtype=np.float64)
        heights[-1, -1] = float(config["remote_sentinel_height_m"])
        return heights
    if kind == "microrelief":
        axis = np.linspace(-extent, extent, resolution, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(axis, axis)
        epsilon = float(config["microrelief_scale_m"])
        return epsilon * (
            np.sin(math.pi * x_grid / extent)
            + 0.5 * np.sin(math.pi * y_grid / extent)
        )
    raise ValueError(f"Unsupported surface kind: {kind}")


def build_scene(
    base_xml: Path,
    output_root: Path,
    variant: dict[str, Any],
    scene_config: dict[str, Any],
) -> dict[str, Any]:
    variant_id = str(variant["id"])
    kind = str(variant["kind"])
    resolution = int(variant["resolution"])
    extent = float(scene_config["map_half_extent_m"])
    heights = surface_heights(kind, resolution, extent, scene_config)
    start = np.asarray(scene_config["start_xy_m"], dtype=np.float64)
    # Hold absolute spawn height constant.  The micro-relief formula is not
    # generally zero at the chosen start, so translating every sampled height
    # by its start value removes that otherwise tiny initial-state confound.
    heights = heights - terrain_value(
        heights, float(start[0]), float(start[1]), extent
    )
    scene_dir = output_root / "scenes" / variant_id
    scene_dir.mkdir(parents=True, exist_ok=False)
    heights_path = scene_dir / "heights_m.npy"
    texture_path = scene_dir / "terrain.png"
    hfield_path = scene_dir / "terrain.hfield"
    xml_path = scene_dir / "ant_scene.xml"
    np.save(heights_path, heights, allow_pickle=False)
    Image.new("RGB", (64, 64), (96, 132, 116)).save(texture_path)

    height_range = float(np.ptp(heights))
    hfield_payload = heights
    if kind == "plane":
        # The unused hfield asset still must contain non-flat data to compile.
        hfield_payload = np.zeros_like(heights)
        hfield_payload[-1, -1] = float(scene_config["remote_sentinel_height_m"])
        height_range = float(np.ptp(hfield_payload))
    hfield_path.write_bytes(
        struct.pack("<ii", resolution, resolution)
        + np.asarray(hfield_payload, dtype="<f4", order="C").tobytes(order="C")
    )

    tree = ET.parse(base_xml)
    root = tree.getroot()
    hfield = root.find("./asset/hfield[@name='terrain']")
    texture = root.find("./asset/texture[@name='texplane']")
    floor = root.find("./worldbody/geom[@name='floor']")
    torso = root.find("./worldbody/body[@name='torso']")
    start_site = root.find("./worldbody/site[@name='fixed_start_marker']")
    goal_site = root.find("./worldbody/site[@name='fixed_goal_marker']")
    if any(item is None for item in (hfield, texture, floor, torso, start_site, goal_site)):
        raise ValueError("Base XML is missing a required element")
    assert hfield is not None and texture is not None and floor is not None
    assert torso is not None and start_site is not None and goal_site is not None
    hfield.set("file", hfield_path.name)
    hfield.set("size", f"{extent:.12g} {extent:.12g} {height_range:.12g} 1")
    texture.set("file", texture_path.name)
    floor.set("friction", "1 0.5 0.5")
    floor.set("condim", "3")
    floor.set("pos", "0 0 0")
    if kind == "plane":
        floor.set("type", "plane")
        floor.attrib.pop("hfield", None)
        floor.set("size", f"{extent:.12g} {extent:.12g} 0.1")
    else:
        floor.set("type", "hfield")
        floor.set("hfield", "terrain")
        floor.attrib.pop("size", None)
        floor.set("pos", f"0 0 {float(np.min(heights)):.12g}")
    goal = np.asarray(scene_config["goal_xy_m"], dtype=np.float64)
    start_z = terrain_value(heights, float(start[0]), float(start[1]), extent)
    goal_z = terrain_value(heights, float(goal[0]), float(goal[1]), extent)
    if abs(start_z) > 1e-15:
        raise RuntimeError(f"{variant_id}: start height was not normalised to zero")
    torso.set("pos", f"{start[0]:.12g} {start[1]:.12g} {float(scene_config['torso_clearance_m']):.12g}")
    torso.set("quat", "1 0 0 0")
    start_site.set("pos", f"{start[0]:.12g} {start[1]:.12g} {start_z + 0.04:.12g}")
    goal_site.set("pos", f"{goal[0]:.12g} {goal[1]:.12g} {goal_z + 0.04:.12g}")
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if not np.array_equal(model.geom_friction[floor_id], np.asarray([1.0, 0.5, 0.5])):
        raise RuntimeError(f"{variant_id}: compiled friction differs")
    if int(model.geom_condim[floor_id]) != 3:
        raise RuntimeError(f"{variant_id}: compiled condim differs")
    spacing = 2.0 * extent / (resolution - 1)
    row_axis = np.linspace(-extent, extent, resolution)
    col_axis = np.linspace(-extent, extent, resolution)
    local_mask = (
        (np.abs(row_axis[:, None] - start[1]) <= 2.0)
        & (np.abs(col_axis[None, :] - start[0]) <= 2.0)
    )
    return {
        "scene_name": variant_id,
        "kind": kind,
        "resolution": resolution,
        "xml_path": str(xml_path.resolve()),
        "xml_sha256": sha256(xml_path),
        "heights_path": str(heights_path.resolve()),
        "heights_sha256": sha256(heights_path),
        "hfield_path": str(hfield_path.resolve()),
        "hfield_sha256": sha256(hfield_path),
        "map_half_extent_m": extent,
        "start_xy_m": start.tolist(),
        "goal_xy_m": goal.tolist(),
        "height_min_m": float(np.min(heights)),
        "height_max_m": float(np.max(heights)),
        "height_range_m": float(np.ptp(heights)),
        "start_height_m": start_z,
        "grid_spacing_m": spacing,
        "maximum_abs_height_within_2m_of_start_m": float(
            np.max(np.abs(heights[local_mask]))
        ),
        "compiled_floor_type": int(model.geom_type[floor_id]),
        "compiled_floor_friction": model.geom_friction[floor_id].tolist(),
        "compiled_floor_condim": int(model.geom_condim[floor_id]),
        "compiled_floor_pos": model.geom_pos[floor_id].tolist(),
        "compiled_floor_margin_m": float(model.geom_margin[floor_id]),
        "compiled_floor_gap_m": float(model.geom_gap[floor_id]),
        "compiled_floor_solref": model.geom_solref[floor_id].tolist(),
        "compiled_floor_solimp": model.geom_solimp[floor_id].tolist(),
        "compiled_qpos0": model.qpos0.tolist(),
        "compiled_robot_signature": robot_signature(model),
        "compiled_hfield_data_min": float(np.min(model.hfield_data)),
        "compiled_hfield_data_max": float(np.max(model.hfield_data)),
    }


def contact_sample(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    foot_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_NAMES
    )
    lookup = {int(value): index for index, value in enumerate(foot_ids)}
    mask = np.zeros(4, dtype=bool)
    counts = np.zeros(4, dtype=np.int64)
    normal_force = np.zeros(4, dtype=np.float64)
    world_robot_contacts = 0
    minimum_distance = float("inf")
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if (body1 == 0) != (body2 == 0):
            world_robot_contacts += 1
            minimum_distance = min(minimum_distance, float(contact.dist))
        if geom1 in lookup and body2 == 0:
            foot_geom = geom1
        elif geom2 in lookup and body1 == 0:
            foot_geom = geom2
        else:
            continue
        index = lookup[foot_geom]
        mask[index] = True
        counts[index] += 1
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_force[index] += max(0.0, float(force[0]))
    return {
        "support_count": int(np.sum(mask)),
        "foot_contact_point_count": int(np.sum(counts)),
        "foot_contact_counts": counts.tolist(),
        "world_robot_contact_count": world_robot_contacts,
        "normal_force_sum_n": float(np.sum(normal_force)),
        "minimum_contact_distance_m": None if math.isinf(minimum_distance) else minimum_distance,
    }


def summarise_contact_rows(rows: list[dict[str, Any]], settled_window: int | None = None) -> dict[str, Any]:
    selected = rows[-settled_window:] if settled_window else rows
    supported_foot_instances = sum(int(row["support_count"]) for row in selected)
    foot_contact_points = sum(int(row["foot_contact_point_count"]) for row in selected)
    return {
        "sample_count": len(selected),
        "zero_foot_contact_fraction": float(np.mean([row["support_count"] == 0 for row in selected])),
        "mean_support_count": float(np.mean([row["support_count"] for row in selected])),
        "mean_foot_contact_point_count": float(np.mean([row["foot_contact_point_count"] for row in selected])),
        "mean_contacts_per_supported_foot": (
            float(foot_contact_points / supported_foot_instances)
            if supported_foot_instances else 0.0
        ),
        "mean_world_robot_contact_count": float(np.mean([row["world_robot_contact_count"] for row in selected])),
        "mean_normal_force_sum_n": float(np.mean([row["normal_force_sum_n"] for row in selected])),
        "maximum_normal_force_sum_n": float(np.max([row["normal_force_sum_n"] for row in selected])),
    }


def make_policy_env(
    config: dict[str, Any], reward: dict[str, Any], scene: dict[str, Any], seed: int, steps: int
):
    return make_standard_env(
        config,
        reward,
        scene,
        condition_id=f"HEIGHTFIELD_CONTACT_{scene['scene_name'].upper()}",
        seed=seed,
        max_episode_steps=steps,
        cruise_speed=float(config["policy_replay"]["cruise_speed_m_per_s"]),
    )


def initial_equivalence(
    model: PPO,
    config: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    policy = config["policy_replay"]
    seed = int(policy["seed"])
    steps = int(policy["control_steps"])
    reference_id = str(policy["reference_surface"])
    reference_env = make_policy_env(config, reward, scenes[reference_id], seed, steps)
    reference_obs, _ = reference_env.reset(seed=seed)
    reference_qpos = np.asarray(reference_env.unwrapped.data.qpos).copy()
    reference_qvel = np.asarray(reference_env.unwrapped.data.qvel).copy()
    reference_action, _ = model.predict(reference_obs, deterministic=True)
    records: dict[str, Any] = {}
    for variant_id, scene in scenes.items():
        env = make_policy_env(config, reward, scene, seed, steps)
        observation, _ = env.reset(seed=seed)
        action, _ = model.predict(observation, deterministic=True)
        record = {
            "qpos_max_abs_difference": float(np.max(np.abs(np.asarray(env.unwrapped.data.qpos) - reference_qpos))),
            "qvel_max_abs_difference": float(np.max(np.abs(np.asarray(env.unwrapped.data.qvel) - reference_qvel))),
            "observation_max_abs_difference": float(np.max(np.abs(np.asarray(observation) - reference_obs))),
            "initial_action_max_abs_difference": float(np.max(np.abs(np.asarray(action) - reference_action))),
        }
        records[variant_id] = record
        env.close()
    qtol = float(policy["initial_state_absolute_tolerance"])
    otol = float(policy["initial_observation_absolute_tolerance"])
    atol = float(policy["initial_action_absolute_tolerance"])
    for variant_id, record in records.items():
        if record["qpos_max_abs_difference"] > qtol or record["qvel_max_abs_difference"] > qtol:
            raise RuntimeError(f"{variant_id}: reset state is not matched")
        if record["observation_max_abs_difference"] > otol:
            raise RuntimeError(f"{variant_id}: reset observation exceeds tolerance")
        if record["initial_action_max_abs_difference"] > atol:
            raise RuntimeError(f"{variant_id}: initial policy action exceeds tolerance")

    actions: list[np.ndarray] = []
    observation = reference_obs
    for _ in range(steps):
        action, _ = model.predict(observation, deterministic=True)
        actions.append(np.asarray(action, dtype=np.float64).copy())
        observation, _, terminated, truncated, _ = reference_env.step(action)
        if terminated or truncated:
            if len(actions) != steps:
                raise RuntimeError("Reference action rollout ended before the frozen horizon")
    reference_env.close()
    return records, reference_qpos, reference_qvel, np.asarray(actions)


def replay_actions(
    scene: dict[str, Any], qpos: np.ndarray, qvel: np.ndarray, actions: np.ndarray, substeps: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    rows: list[dict[str, Any]] = []
    start_xy = np.asarray(qpos[:2], dtype=np.float64)
    for control_step, action in enumerate(actions, start=1):
        data.ctrl[:] = action
        for substep in range(1, substeps + 1):
            mujoco.mj_step(model, data)
            sample = contact_sample(model, data)
            rows.append(
                {
                    "surface_id": scene["scene_name"],
                    "control_step": control_step,
                    "physics_substep": substep,
                    "time_seconds": float(data.time),
                    "root_x_m": float(data.qpos[0]),
                    "root_y_m": float(data.qpos[1]),
                    "root_z_m": float(data.qpos[2]),
                    **sample,
                }
            )
    summary = summarise_contact_rows(rows)
    summary.update(
        {
            "surface_id": scene["scene_name"],
            "elapsed_seconds": float(data.time),
            "net_xy_displacement_m": float(np.linalg.norm(np.asarray(data.qpos[:2]) - start_xy)),
            "final_root_position": np.asarray(data.qpos[:3]).tolist(),
            "final_root_linear_speed_m_per_s": float(np.linalg.norm(np.asarray(data.qvel[:3]))),
        }
    )
    return summary, rows


def static_drop(
    scene: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = float(config["static_drop"]["root_height_m"])
    data.qvel[:] = 0.0
    data.ctrl[:] = np.asarray(config["static_drop"]["control"], dtype=np.float64)
    mujoco.mj_forward(model, data)
    rows: list[dict[str, Any]] = []
    for step in range(1, int(config["static_drop"]["physics_steps"]) + 1):
        mujoco.mj_step(model, data)
        rows.append(
            {
                "surface_id": scene["scene_name"],
                "physics_step": step,
                "time_seconds": float(data.time),
                "root_z_m": float(data.qpos[2]),
                "root_vertical_speed_m_per_s": float(data.qvel[2]),
                **contact_sample(model, data),
            }
        )
    settled = int(config["static_drop"]["settled_window_steps"])
    summary = {
        "surface_id": scene["scene_name"],
        "full": summarise_contact_rows(rows),
        "settled_window": summarise_contact_rows(rows, settled_window=settled),
        "final_root_z_m": float(data.qpos[2]),
        "final_abs_vertical_speed_m_per_s": abs(float(data.qvel[2])),
    }
    return summary, rows


def plane_settled_pose(scene: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    """Return one plane-derived pose for an instantaneous collision query."""
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = float(config["static_drop"]["root_height_m"])
    data.qvel[:] = 0.0
    data.ctrl[:] = np.asarray(config["static_drop"]["control"], dtype=np.float64)
    mujoco.mj_forward(model, data)
    for _ in range(int(config["static_drop"]["physics_steps"])):
        mujoco.mj_step(model, data)
    return np.asarray(data.qpos, dtype=np.float64).copy()


def identical_pose_contact_probe(
    scene: dict[str, Any], qpos: np.ndarray
) -> dict[str, Any]:
    """Query contacts at identical qpos and zero qvel without integrating."""
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    sample = contact_sample(model, data)
    foot_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): name
        for name in FOOT_NAMES
    }
    contacts: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in foot_ids and int(model.geom_bodyid[geom2]) == 0:
            foot_geom = geom1
        elif geom2 in foot_ids and int(model.geom_bodyid[geom1]) == 0:
            foot_geom = geom2
        else:
            continue
        contacts.append(
            {
                "foot": foot_ids[foot_geom],
                "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "distance_m": float(contact.dist),
                "normal": np.asarray(contact.frame[:3], dtype=np.float64).tolist(),
            }
        )
    normal_tilts = [
        math.degrees(
            math.acos(float(np.clip(contact["normal"][2], -1.0, 1.0)))
        )
        for contact in contacts
    ]
    return {
        "surface_id": scene["scene_name"],
        "identical_qpos": np.asarray(qpos, dtype=np.float64).tolist(),
        "zero_qvel": True,
        **sample,
        "mean_contact_normal_tilt_from_vertical_degrees": (
            float(np.mean(normal_tilts)) if normal_tilts else None
        ),
        "maximum_contact_normal_tilt_from_vertical_degrees": (
            float(np.max(normal_tilts)) if normal_tilts else None
        ),
        "foot_contacts": contacts,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config": str(config_path)}))
        return
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["execution"]["output_root"]).resolve()
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_root}")
    output_root.mkdir(parents=True)
    frozen_config = output_root / "frozen_config.json"
    write_json(frozen_config, config)
    base_xml = ROOT / config["frozen_sources"]["base_scene_xml"]
    scenes = {
        variant["id"]: build_scene(base_xml, output_root, variant, config["controlled_scene"])
        for variant in config["controlled_scene"]["surface_variants"]
    }
    signatures = [json.dumps(scene["compiled_robot_signature"], sort_keys=True) for scene in scenes.values()]
    if len(set(signatures)) != 1:
        raise RuntimeError("Generated surfaces changed the robot signature")
    qpos0 = [np.asarray(scene["compiled_qpos0"]) for scene in scenes.values()]
    if max(float(np.max(np.abs(value - qpos0[0]))) for value in qpos0) > 1e-12:
        raise RuntimeError("Generated surfaces changed qpos0")
    write_json(output_root / "scene_manifest.json", scenes)

    reward_path = ROOT / config["frozen_sources"]["reward_configuration"]
    reward = json.loads(reward_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / config["frozen_sources"]["source_checkpoint"]
    model = PPO.load(checkpoint, device="cpu")
    if tuple(model.observation_space.shape) != (135,) or tuple(model.action_space.shape) != (8,):
        raise RuntimeError("Checkpoint interface changed")
    equivalence, initial_qpos, initial_qvel, actions = initial_equivalence(
        model, config, reward, scenes
    )
    np.save(output_root / "reference_initial_qpos.npy", initial_qpos, allow_pickle=False)
    np.save(output_root / "reference_initial_qvel.npy", initial_qvel, allow_pickle=False)
    np.save(output_root / "reference_open_loop_actions.npy", actions, allow_pickle=False)
    write_json(output_root / "initial_equivalence.json", equivalence)

    replay_summaries: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    drop_summaries: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    substeps = int(config["controlled_scene"]["control_substeps"])
    for scene in scenes.values():
        replay_summary, rows = replay_actions(scene, initial_qpos, initial_qvel, actions, substeps)
        replay_summaries.append(replay_summary)
        replay_rows.extend(rows)
        drop_summary, rows = static_drop(scene, config)
        drop_summaries.append(drop_summary)
        drop_rows.extend(rows)
    write_rows(output_root / "logs" / "matched_open_loop_substeps.csv", replay_rows)
    write_rows(output_root / "logs" / "static_drop_substeps.csv", drop_rows)
    write_json(output_root / "matched_open_loop_summary.json", replay_summaries)
    write_json(output_root / "static_drop_summary.json", drop_summaries)
    probe_qpos = plane_settled_pose(scenes["native_plane"], config)
    np.save(output_root / "identical_pose_probe_qpos.npy", probe_qpos, allow_pickle=False)
    probe = [
        identical_pose_contact_probe(scene, probe_qpos) for scene in scenes.values()
    ]
    write_json(output_root / "identical_pose_contact_probe.json", probe)

    plane = next(item for item in replay_summaries if item["surface_id"] == "native_plane")
    effects = []
    for item in replay_summaries:
        effects.append(
            {
                "surface_id": item["surface_id"],
                "zero_foot_contact_fraction_difference_vs_plane": item["zero_foot_contact_fraction"] - plane["zero_foot_contact_fraction"],
                "mean_support_count_difference_vs_plane": item["mean_support_count"] - plane["mean_support_count"],
                "mean_contacts_per_supported_foot_ratio_vs_plane": (
                    item["mean_contacts_per_supported_foot"] / plane["mean_contacts_per_supported_foot"]
                    if plane["mean_contacts_per_supported_foot"] else None
                ),
                "net_displacement_ratio_vs_plane": (
                    item["net_xy_displacement_m"] / plane["net_xy_displacement_m"]
                    if plane["net_xy_displacement_m"] else None
                ),
            }
        )
    write_json(output_root / "effects_vs_native_plane.json", effects)
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }
    write_json(output_root / "software_versions.json", versions)
    manifest = {
        "schema_version": "proxygap-heightfield-plane-contact-diagnostic-v1",
        "status": "bounded_development_diagnostic",
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "frozen_config": {"path": str(frozen_config), "sha256": sha256(frozen_config)},
        "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "scene_manifest": {"path": str(output_root / "scene_manifest.json"), "sha256": sha256(output_root / "scene_manifest.json")},
        "initial_equivalence": {"path": str(output_root / "initial_equivalence.json"), "sha256": sha256(output_root / "initial_equivalence.json")},
        "matched_open_loop_summary": {"path": str(output_root / "matched_open_loop_summary.json"), "sha256": sha256(output_root / "matched_open_loop_summary.json")},
        "static_drop_summary": {"path": str(output_root / "static_drop_summary.json"), "sha256": sha256(output_root / "static_drop_summary.json")},
        "identical_pose_contact_probe": {"path": str(output_root / "identical_pose_contact_probe.json"), "sha256": sha256(output_root / "identical_pose_contact_probe.json")},
        "effects_vs_native_plane": {"path": str(output_root / "effects_vs_native_plane.json"), "sha256": sha256(output_root / "effects_vs_native_plane.json")},
        "diagnostic_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "reference_open_loop_actions": {"path": str(output_root / "reference_open_loop_actions.npy"), "sha256": sha256(output_root / "reference_open_loop_actions.npy")},
        "existing_results_overwritten": False,
        "training_performed": False,
        "map_friction_reward_energy_changed": False,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps({"status": "diagnostic_complete", "output_root": str(output_root), "manifest_sha256": sha256(output_root / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
