"""Single-factor margin addendum for exact-flat heightfield contact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from diagnose_heightfield_plane_contact import (  # noqa: E402
    FOOT_NAMES,
    contact_sample,
    identical_pose_contact_probe,
    replay_actions,
    sha256,
    static_drop,
    write_json,
    write_rows,
)


DEFAULT_CONFIG = ROOT / "configs" / "heightfield_capsule_margin_diagnostic_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("status") != "frozen_bounded_heightfield_margin_addendum":
        raise ValueError("Margin addendum configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("Generalisation claim must remain prohibited")
    parent = config["frozen_parent"]
    parent_config = ROOT / parent["configuration"]
    artifact = ROOT / parent["artifact_root"]
    checks = {
        parent_config: parent["configuration_sha256"],
        artifact / "manifest.json": parent["manifest_sha256"],
        artifact / "reference_initial_qpos.npy": parent["reference_initial_qpos_sha256"],
        artifact / "reference_initial_qvel.npy": parent["reference_initial_qvel_sha256"],
        artifact / "reference_open_loop_actions.npy": parent["reference_open_loop_actions_sha256"],
        artifact / "identical_pose_probe_qpos.npy": parent["identical_pose_probe_qpos_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen parent input missing or changed: {path}")
    if config["source_surface_ids"] != [
        "native_plane",
        "hfield_plateau_129",
        "hfield_plateau_257",
    ]:
        raise ValueError("Surface comparison set changed")
    expected_conditions = [
        ("default_001_001", 0.01, 0.01),
        ("floor0_foot001", 0.0, 0.01),
        ("floor001_foot0", 0.01, 0.0),
        ("both0", 0.0, 0.0),
    ]
    observed = [
        (item["id"], float(item["floor_margin_m"]), float(item["foot_margin_m"]))
        for item in config["margin_conditions"]
    ]
    if observed != expected_conditions:
        raise ValueError("Margin matrix changed")
    controlled = config["controlled_parameters"]
    if controlled["fixed_friction"] != [1.0, 0.5, 0.5] or controlled["condim"] != 3:
        raise ValueError("Friction or condim changed")
    if controlled["solref"] != [0.02, 1.0] or controlled["solimp"] != [0.9, 0.95, 0.001, 0.5, 2.0]:
        raise ValueError("Solver parameters changed")
    if config["energy_boundary"]["energy_formula_changes"] != "prohibited":
        raise ValueError("Energy boundary changed")
    return json.loads((artifact / "scene_manifest.json").read_text(encoding="utf-8"))


def build_margin_scene(
    source: dict[str, Any],
    output_root: Path,
    source_surface_id: str,
    condition: dict[str, Any],
    controlled: dict[str, Any],
) -> dict[str, Any]:
    condition_id = str(condition["id"])
    scene_id = f"{source_surface_id}_{condition_id}"
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=False)
    source_xml = Path(source["xml_path"])
    source_dir = source_xml.parent
    shutil.copyfile(source_dir / "terrain.hfield", scene_dir / "terrain.hfield")
    shutil.copyfile(source_dir / "terrain.png", scene_dir / "terrain.png")
    xml_path = scene_dir / "ant_scene.xml"
    tree = ET.parse(source_xml)
    root = tree.getroot()
    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("Source scene lacks floor geom")
    floor.set("margin", f"{float(condition['floor_margin_m']):.12g}")
    for foot_name in FOOT_NAMES:
        geom = root.find(f".//geom[@name='{foot_name}']")
        if geom is None:
            raise ValueError(f"Source scene lacks {foot_name}")
        geom.set("margin", f"{float(condition['foot_margin_m']):.12g}")
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_NAMES
    ]
    if not np.array_equal(model.geom_friction[floor_id], np.asarray(controlled["fixed_friction"])):
        raise RuntimeError("Compiled friction changed")
    if int(model.geom_condim[floor_id]) != int(controlled["condim"]):
        raise RuntimeError("Compiled condim changed")
    if not np.array_equal(model.geom_solref[floor_id], np.asarray(controlled["solref"])):
        raise RuntimeError("Compiled solref changed")
    if not np.array_equal(model.geom_solimp[floor_id], np.asarray(controlled["solimp"])):
        raise RuntimeError("Compiled solimp changed")
    floor_margin = float(model.geom_margin[floor_id])
    foot_margins = np.asarray(model.geom_margin[foot_ids], dtype=np.float64)
    if floor_margin != float(condition["floor_margin_m"]):
        raise RuntimeError("Compiled floor margin changed")
    if not np.all(foot_margins == float(condition["foot_margin_m"])):
        raise RuntimeError("Compiled foot margin changed")
    return {
        "scene_name": scene_id,
        "source_surface_id": source_surface_id,
        "resolution": int(source["resolution"]),
        "margin_condition": condition_id,
        "floor_margin_m": floor_margin,
        "foot_margin_m": float(foot_margins[0]),
        "xml_path": str(xml_path.resolve()),
        "xml_sha256": sha256(xml_path),
        "hfield_sha256": sha256(scene_dir / "terrain.hfield"),
        "floor_friction": model.geom_friction[floor_id].tolist(),
        "floor_condim": int(model.geom_condim[floor_id]),
        "floor_solref": model.geom_solref[floor_id].tolist(),
        "floor_solimp": model.geom_solimp[floor_id].tolist(),
        "compiled_qpos0": model.qpos0.tolist(),
    }


def margin_probe(scene: dict[str, Any], qpos: np.ndarray) -> dict[str, Any]:
    result = identical_pose_contact_probe(scene, qpos)
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    foot_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_NAMES
    }
    margins: list[float] = []
    distances: list[float] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not ((geom1 in foot_ids and int(model.geom_bodyid[geom2]) == 0) or (geom2 in foot_ids and int(model.geom_bodyid[geom1]) == 0)):
            continue
        distances.append(float(contact.dist))
        margins.append(float(contact.includemargin))
    result.update(
        {
            "resolution": scene["resolution"],
            "margin_condition": scene["margin_condition"],
            "floor_margin_m": scene["floor_margin_m"],
            "foot_margin_m": scene["foot_margin_m"],
            "minimum_distance_m": min(distances) if distances else None,
            "maximum_distance_m": max(distances) if distances else None,
            "unique_include_margins_m": sorted(set(margins)),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_scenes = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config": str(config_path)}))
        return
    output_root = args.output_root.resolve() if args.output_root else (ROOT / config["execution"]["output_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_root}")
    output_root.mkdir(parents=True)
    frozen_config = output_root / "frozen_config.json"
    write_json(frozen_config, config)
    scenes: list[dict[str, Any]] = []
    for source_surface_id in config["source_surface_ids"]:
        source = parent_scenes[source_surface_id]
        for condition in config["margin_conditions"]:
            scenes.append(
                build_margin_scene(
                    source,
                    output_root,
                    source_surface_id,
                    condition,
                    config["controlled_parameters"],
                )
            )
    qpos0 = [np.asarray(scene["compiled_qpos0"]) for scene in scenes]
    if max(float(np.max(np.abs(value - qpos0[0]))) for value in qpos0) > 1e-12:
        raise RuntimeError("Margin variants changed qpos0")
    write_json(output_root / "scene_manifest.json", scenes)

    parent_root = ROOT / config["frozen_parent"]["artifact_root"]
    initial_qpos = np.load(parent_root / "reference_initial_qpos.npy", allow_pickle=False)
    initial_qvel = np.load(parent_root / "reference_initial_qvel.npy", allow_pickle=False)
    actions = np.load(parent_root / "reference_open_loop_actions.npy", allow_pickle=False)
    probe_qpos = np.load(parent_root / "identical_pose_probe_qpos.npy", allow_pickle=False)
    substeps = int(config["controlled_parameters"]["control_substeps"])
    drop_config = {
        "static_drop": {
            "root_height_m": config["controlled_parameters"]["static_drop_root_height_m"],
            "physics_steps": config["controlled_parameters"]["static_drop_physics_steps"],
            "settled_window_steps": config["controlled_parameters"]["static_drop_settled_window_steps"],
            "control": [0.0] * 8,
        }
    }

    probes = [margin_probe(scene, probe_qpos) for scene in scenes]
    replay_summaries: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    drop_summaries: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    for scene in scenes:
        summary, rows = replay_actions(scene, initial_qpos, initial_qvel, actions, substeps)
        summary.update({key: scene[key] for key in ("resolution", "margin_condition", "floor_margin_m", "foot_margin_m")})
        replay_summaries.append(summary)
        replay_rows.extend(rows)
        summary, rows = static_drop(scene, drop_config)
        summary.update({key: scene[key] for key in ("resolution", "margin_condition", "floor_margin_m", "foot_margin_m")})
        drop_summaries.append(summary)
        drop_rows.extend(rows)
    write_json(output_root / "identical_pose_margin_probe.json", probes)
    write_json(output_root / "matched_open_loop_margin_summary.json", replay_summaries)
    write_json(output_root / "static_drop_margin_summary.json", drop_summaries)
    write_rows(output_root / "logs" / "matched_open_loop_margin_substeps.csv", replay_rows)
    write_rows(output_root / "logs" / "static_drop_margin_substeps.csv", drop_rows)

    manifest = {
        "schema_version": "proxygap-heightfield-capsule-margin-diagnostic-v1",
        "status": "bounded_development_diagnostic",
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "frozen_config": {"path": str(frozen_config), "sha256": sha256(frozen_config)},
        "parent_manifest": {"path": str(parent_root / "manifest.json"), "sha256": sha256(parent_root / "manifest.json")},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "scene_manifest": {"path": str(output_root / "scene_manifest.json"), "sha256": sha256(output_root / "scene_manifest.json")},
        "identical_pose_margin_probe": {"path": str(output_root / "identical_pose_margin_probe.json"), "sha256": sha256(output_root / "identical_pose_margin_probe.json")},
        "matched_open_loop_margin_summary": {"path": str(output_root / "matched_open_loop_margin_summary.json"), "sha256": sha256(output_root / "matched_open_loop_margin_summary.json")},
        "static_drop_margin_summary": {"path": str(output_root / "static_drop_margin_summary.json"), "sha256": sha256(output_root / "static_drop_margin_summary.json")},
        "training_performed": False,
        "formal_map_modified": False,
        "friction_reward_energy_changed": False,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps({"status": "diagnostic_complete", "output_root": str(output_root), "manifest_sha256": sha256(output_root / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
