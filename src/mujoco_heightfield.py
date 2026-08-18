"""MuJoCo and Gymnasium integration for deterministic terrain arrays."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from terrain_generator import TerrainData
from terrain_queries import TerrainQueries


HFIELD_NAME = "ant_random_terrain"


def default_ant_xml_path() -> Path:
    """Resolve the Ant XML from the installed Gymnasium package."""

    import gymnasium.envs.mujoco.ant_v5 as ant_v5

    path = Path(ant_v5.__file__).resolve().parent / "assets" / "ant.xml"
    if not path.is_file():
        raise FileNotFoundError(f"installed Gymnasium Ant XML was not found: {path}")
    return path


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def build_ant_heightfield_xml(
    terrain: TerrainData,
    output_path: str | Path,
    base_xml_path: str | Path | None = None,
) -> Path:
    """Replace only the default plane with a heightfield in a copied Ant XML."""

    source = Path(base_xml_path) if base_xml_path is not None else default_ant_xml_path()
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Ant XML must contain asset and worldbody elements")
    for existing in list(asset.findall("hfield")):
        if existing.get("name") == HFIELD_NAME:
            asset.remove(existing)
    normalisation = terrain.metadata["normalisation"]
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": HFIELD_NAME,
            "nrow": str(terrain.config.nrow),
            "ncol": str(terrain.config.ncol),
            "size": " ".join(
                _format_float(value)
                for value in (
                    0.5 * terrain.config.terrain_length_m,
                    0.5 * terrain.config.terrain_width_m,
                    normalisation["mujoco_vertical_scale_m"],
                    terrain.config.heightfield_base_depth_m,
                )
            ),
        },
    )
    floor = next((geom for geom in worldbody.findall("geom") if geom.get("name") == "floor"), None)
    if floor is None:
        raise ValueError("Ant XML does not contain a worldbody floor geom")
    floor.set("type", "hfield")
    floor.set("hfield", HFIELD_NAME)
    floor.set("pos", f"0 0 {_format_float(normalisation['physical_offset_m'])}")
    floor.set("friction", " ".join(_format_float(value) for value in terrain.config.friction))
    floor.attrib.pop("size", None)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def install_heightfield_data(model: mujoco.MjModel, terrain: TerrainData) -> int:
    """Install C-order float32 values before reset or rendering."""

    field_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, HFIELD_NAME)
    if field_id < 0:
        raise ValueError(f"MuJoCo model does not contain hfield {HFIELD_NAME!r}")
    rows = int(model.hfield_nrow[field_id])
    columns = int(model.hfield_ncol[field_id])
    if (rows, columns) != terrain.normalised_height.shape:
        raise ValueError(
            f"MuJoCo hfield shape {(rows, columns)} != terrain shape {terrain.normalised_height.shape}"
        )
    address = int(model.hfield_adr[field_id])
    count = rows * columns
    values = np.asarray(terrain.normalised_height, dtype=np.float32).ravel(order="C")
    model.hfield_data[address : address + count] = values
    return field_id


def load_mujoco_model(
    terrain: TerrainData,
    xml_path: str | Path,
    base_xml_path: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Compile an Ant model, write height data and run forward kinematics."""

    destination = build_ant_heightfield_xml(terrain, xml_path, base_xml_path)
    model = mujoco.MjModel.from_xml_path(str(destination.resolve()))
    field_id = install_heightfield_data(model, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, field_id


class HeightfieldAntWrapper(gym.Wrapper):
    """Keep a terrain query object beside an otherwise unchanged Ant-v5."""

    def __init__(self, env: gym.Env, terrain: TerrainData):
        super().__init__(env)
        self.terrain = terrain
        self.terrain_queries = TerrainQueries(terrain)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        unwrapped = self.env.unwrapped
        surface_height = self.terrain_queries.height(
            float(unwrapped.data.qpos[0]), float(unwrapped.data.qpos[1])
        )
        info = dict(info)
        info.update(
            {
                "terrain_height_m": surface_height,
                "terrain_seed": self.terrain.config.terrain_seed,
                "terrain_height_sha256": self.terrain.height_sha256,
            }
        )
        return observation, info


def make_ant_terrain_env(
    terrain: TerrainData,
    xml_path: str | Path,
    render_mode: str | None = None,
) -> HeightfieldAntWrapper:
    """Create default Ant-v5 dynamics and replace only its ground geometry."""

    destination = build_ant_heightfield_xml(terrain, xml_path)
    env = gym.make("Ant-v5", xml_file=str(destination.resolve()), render_mode=render_mode)
    install_heightfield_data(env.unwrapped.model, terrain)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    return HeightfieldAntWrapper(env, terrain)


def _contact_summary(env: gym.Env) -> dict[str, float | int | None]:
    data = env.unwrapped.data
    distances = [float(data.contact[index].dist) for index in range(int(data.ncon))]
    return {
        "contact_count": int(data.ncon),
        "minimum_contact_distance_m": min(distances) if distances else None,
    }


def _warning_counts(env: gym.Env) -> list[int]:
    return [int(item.number) for item in env.unwrapped.data.warning]


def run_ant_smoke_test(
    terrain: TerrainData,
    xml_path: str | Path,
    reset_seed: int = 202_608_018,
    steps: int = 10,
) -> dict[str, Any]:
    """Reset Ant and execute zero-action integration steps without training."""

    if steps < 10:
        raise ValueError("the smoke test requires at least 10 steps")
    env = make_ant_terrain_env(terrain, xml_path)
    try:
        observation, info = env.reset(seed=reset_seed)
        if not np.all(np.isfinite(observation)):
            raise AssertionError("Ant reset returned a non-finite observation")
        initial_qpos = env.unwrapped.data.qpos.copy()
        surface = env.terrain_queries.height(float(initial_qpos[0]), float(initial_qpos[1]))
        initial_clearance = float(initial_qpos[2] - surface)
        initial_contacts = _contact_summary(env)
        initial_warnings = _warning_counts(env)
        if any(initial_warnings):
            raise AssertionError(f"MuJoCo warning at Ant reset: {initial_warnings}")
        minimum_distance = initial_contacts["minimum_contact_distance_m"]
        if minimum_distance is not None and minimum_distance < -0.001:
            raise AssertionError(f"initial contact penetration exceeded 0.001 m: {minimum_distance}")
        records: list[dict[str, Any]] = []
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        step_durations: list[float] = []
        for index in range(steps):
            started = time.perf_counter()
            observation, reward, terminated, truncated, step_info = env.step(action)
            step_durations.append(time.perf_counter() - started)
            finite = bool(
                np.all(np.isfinite(observation))
                and np.isfinite(reward)
                and np.all(np.isfinite(env.unwrapped.data.qpos))
                and np.all(np.isfinite(env.unwrapped.data.qvel))
            )
            if not finite:
                raise AssertionError(f"non-finite Ant state at smoke step {index + 1}")
            warnings = _warning_counts(env)
            if any(warnings):
                raise AssertionError(f"MuJoCo warning at smoke step {index + 1}: {warnings}")
            if terminated or truncated:
                raise AssertionError(
                    f"Ant terminated or truncated at smoke step {index + 1}: "
                    f"terminated={terminated}, truncated={truncated}"
                )
            records.append(
                {
                    "step": index + 1,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "torso_z_m": float(env.unwrapped.data.qpos[2]),
                    "mujoco_warning_counts": warnings,
                    **_contact_summary(env),
                }
            )
        return {
            "passed": True,
            "reset_seed": reset_seed,
            "steps": steps,
            "observation_shape": list(observation.shape),
            "initial_torso_clearance_m": initial_clearance,
            "initial_contacts": initial_contacts,
            "initial_mujoco_warning_counts": initial_warnings,
            "mean_step_time_s": float(np.mean(step_durations)),
            "median_step_time_s": float(np.median(step_durations)),
            "records": records,
            "reset_info": {
                key: value
                for key, value in info.items()
                if isinstance(value, (str, int, float, bool))
            },
        }
    finally:
        env.close()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
