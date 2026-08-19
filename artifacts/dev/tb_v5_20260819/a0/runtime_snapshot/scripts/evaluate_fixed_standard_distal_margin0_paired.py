"""Read-only paired evaluation of the floor + four distal margin=0 candidate.

This diagnostic adds exactly four explicit floor-to-distal contact pairs with
margin=0 and gap=0 while retaining every geom margin at 0.01 m.  It never trains
a policy and conditionally evaluates the approved fixed map only after the
frozen standard-scene gate passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (SRC, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from proxygap.ant_wrapper import quaternion_tilt_relative_to_normal  # noqa: E402
from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.fixed_goal_terrain import FixedGoalTerrainWrapper  # noqa: E402
from run_curved_gait_training import common_env_kwargs  # noqa: E402
from run_fixed_standard_support_curriculum import (  # noqa: E402
    FOOT_NAMES,
    contact_masks_from_data,
    install_substep_contact_audit,
    sha256,
    validate_config as validate_standard_protocol,
    verified_json,
    write_json,
    write_rows,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "fixed_standard_distal_margin0_paired_diagnostic_v1_20260819.json"
)
CONTROL_ID = "DEFAULT_MARGIN_001_CONTROL"
CANDIDATE_ID = "EXPLICIT_FLOOR_DISTAL_PAIR_MARGIN0_CANDIDATE"
TARGET_NAMES = ("floor", *FOOT_NAMES)
ROBOT_GEOM_NAMES = (
    "torso_geom",
    "aux_1_geom",
    "left_leg_geom",
    "left_ankle_geom",
    "aux_2_geom",
    "right_leg_geom",
    "right_ankle_geom",
    "aux_3_geom",
    "back_leg_geom",
    "third_ankle_geom",
    "aux_4_geom",
    "rightback_leg_geom",
    "fourth_ankle_geom",
)
NON_DISTAL_ROBOT_GEOMS = tuple(
    name for name in ROBOT_GEOM_NAMES if name not in FOOT_NAMES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one seed and 20 standard control steps; fixed-map evaluation is skipped.",
    )
    return parser.parse_args()


def _exact_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA-256 changed: expected {expected}, observed {observed}")


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.get("status") != "frozen_read_only_paired_diagnostic":
        raise ValueError("The paired diagnostic configuration is not frozen")
    if config.get("formal_training") != "prohibited":
        raise ValueError("Training must remain prohibited")
    mechanism = config["mechanism_evidence"]
    for key, hash_key in (
        ("configuration", "configuration_sha256"),
        ("manifest", "manifest_sha256"),
        ("matched_open_loop_summary", "matched_open_loop_summary_sha256"),
    ):
        _exact_hash(ROOT / mechanism[key], mechanism[hash_key], f"mechanism {key}")
    if "_final_20260819" not in str(mechanism["manifest"]):
        raise ValueError("Mechanism evidence must point to the final audited artifact")
    mechanism_manifest = json.loads((ROOT / mechanism["manifest"]).read_text(encoding="utf-8"))
    if mechanism_manifest.get("training_performed") is not False:
        raise ValueError("Mechanism evidence unexpectedly includes training")
    mechanism_rows = json.loads(
        (ROOT / mechanism["matched_open_loop_summary"]).read_text(encoding="utf-8")
    )
    if not any(
        row.get("surface_id") == "hfield_plateau_257_both0"
        and float(row.get("floor_margin_m", -1)) == 0.0
        and float(row.get("foot_margin_m", -1)) == 0.0
        for row in mechanism_rows
    ):
        raise ValueError("Final mechanism evidence lacks the selected 257 both0 condition")

    frozen = config["frozen_standard_protocol"]
    standard_path = ROOT / frozen["configuration"]
    _exact_hash(standard_path, frozen["configuration_sha256"], "standard protocol")
    standard_config = json.loads(standard_path.read_text(encoding="utf-8"))
    _, reward_config = validate_standard_protocol(standard_config)
    scene_manifest = verified_json(
        ROOT / frozen["scene_manifest"], frozen["scene_manifest_sha256"]
    )
    _exact_hash(
        ROOT / frozen["source_checkpoint"],
        frozen["source_checkpoint_sha256"],
        "Source Stage1 checkpoint",
    )
    if int(frozen["observation_dimension"]) != 135 or int(frozen["action_dimension"]) != 8:
        raise ValueError("Only the frozen 135D/8D Source Stage1 policy is permitted")
    if scene_manifest.get("robot_signature", {}).get("robot_geom_names") != list(ROBOT_GEOM_NAMES):
        raise ValueError("Standard robot geom signature changed")

    conditions = config["margin_conditions"]
    observed_conditions = [
        (
            item["condition_id"],
            float(item["floor_geom_margin_m"]),
            float(item["distal_ankle_geom_margin_m"]),
            float(item["non_distal_robot_geom_margin_m"]),
            int(item["explicit_pair_count"]),
        )
        for item in conditions
    ]
    expected_conditions = [
        (CONTROL_ID, 0.01, 0.01, 0.01, 0),
        (CANDIDATE_ID, 0.01, 0.01, 0.01, 4),
    ]
    if observed_conditions != expected_conditions:
        raise ValueError("The fail-closed two-condition margin contract changed")
    permitted = config["permitted_xml_change"]
    if permitted["target_floor_geom"] != "floor":
        raise ValueError("Only the named floor geom may be changed")
    if tuple(permitted["target_distal_ankle_geoms"]) != FOOT_NAMES:
        raise ValueError("The four distal ankle targets changed")
    if permitted.get("change_type") != "add_four_explicit_contact_pairs_only":
        raise ValueError("Only four explicit floor-to-distal contact pairs may be added")
    if int(permitted["explicit_pair_count"]) != 4:
        raise ValueError("Exactly four explicit contact pairs are required")
    if any(
        not math.isclose(float(permitted[key]), 0.01, abs_tol=1e-12)
        for key in (
            "default_geom_margin_m_unchanged",
            "floor_geom_margin_m_unchanged",
            "distal_ankle_geom_margins_unchanged",
            "all_non_distal_robot_geom_margins_unchanged",
            "root_joint_margin_unchanged",
        )
    ):
        raise ValueError("All geom and root margins must remain 0.01")
    pair = permitted["explicit_pair_contract"]
    if (
        float(pair["margin_m"]) != 0.0
        or float(pair["gap_m"]) != 0.0
        or int(pair["condim"]) != 3
        or pair["friction"] != [1.0, 1.0, 0.5, 0.5, 0.5]
        or pair["solref"] != [0.02, 1.0]
        or pair["solreffriction"] != [0.0, 0.0]
        or pair["solimp"] != [0.9, 0.95, 0.001, 0.5, 2.0]
        or float(pair["adhesion"]) != 0.0
    ):
        raise ValueError("Explicit contact-pair contract changed")
    equivalence = permitted["engineering_equivalence_evidence"]
    _exact_hash(ROOT / equivalence["configuration"], equivalence["configuration_sha256"], "pair-equivalence config")
    equivalence_root = ROOT / equivalence["artifact_root"]
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("compiled_contract_checks.json", "compiled_contract_checks_sha256"),
        ("equivalence_summary.json", "equivalence_summary_sha256"),
    ):
        _exact_hash(equivalence_root / name, equivalence[key], f"pair-equivalence {name}")
    equivalence_summary = json.loads((equivalence_root / "equivalence_summary.json").read_text(encoding="utf-8"))
    if equivalence_summary.get("passed") is not True:
        raise ValueError("Explicit contact-pair engineering equivalence smoke did not pass")
    contact = config["controlled_contact_contract"]
    if contact["fixed_friction"] != [1.0, 0.5, 0.5] or int(contact["condim"]) != 3:
        raise ValueError("Friction or condim changed")
    if contact["solref"] != [0.02, 1.0] or contact["solimp"] != [0.9, 0.95, 0.001, 0.5, 2.0]:
        raise ValueError("Contact solver parameters changed")
    for flag in (
        "terrain_height_assets_unchanged",
        "reward_unchanged",
        "observation_unchanged",
        "checkpoint_unchanged",
        "friction_unchanged",
        "energy_formula_unchanged",
    ):
        if contact.get(flag) is not True:
            raise ValueError(f"Frozen invariant changed: {flag}")

    fixed = config["conditional_fixed_map_evaluation"]
    if fixed.get("enabled_only_if_standard_gate_passes") is not True:
        raise ValueError("Fixed-map evaluation must remain conditional")
    for key, hash_key in (
        ("fixed_map_configuration", "fixed_map_configuration_sha256"),
        ("reference_evaluation_contract", "reference_evaluation_contract_sha256"),
        ("source_xml", "source_xml_sha256"),
        ("source_heights", "source_heights_sha256"),
        ("source_hfield", "source_hfield_sha256"),
        ("source_texture", "source_texture_sha256"),
    ):
        _exact_hash(ROOT / fixed[key], fixed[hash_key], f"fixed-map {key}")
    fixed_config = json.loads((ROOT / fixed["fixed_map_configuration"]).read_text(encoding="utf-8"))
    reference = json.loads((ROOT / fixed["reference_evaluation_contract"]).read_text(encoding="utf-8"))
    if reference["evaluation_seeds"] != fixed["seeds"] or int(reference["horizon_steps"]) != int(fixed["horizon_steps"]):
        raise ValueError("Fixed-map paired seeds or horizon differ from the reference contract")
    if reference["controller"] != fixed["controller"]:
        raise ValueError("Fixed-map controller differs from the reference contract")
    approved = fixed_config["approved_map"]
    if approved["fixed_friction"] != contact["fixed_friction"] or int(approved["condim"]) != int(contact["condim"]):
        raise ValueError("Fixed-map contact contract changed")
    return standard_config, reward_config, scene_manifest


def explicit_pair_block(pair: dict[str, Any]) -> str:
    def values(key: str) -> str:
        return " ".join(f"{float(value):.12g}" for value in pair[key])

    lines = ["  <contact>"]
    for distal in FOOT_NAMES:
        lines.append(
            "    <pair "
            f'name="floor_{distal}_margin0" geom1="floor" geom2="{distal}" '
            f'margin="{float(pair["margin_m"]):.12g}" '
            f'gap="{float(pair["gap_m"]):.12g}" '
            f'condim="{int(pair["condim"])}" '
            f'friction="{values("friction")}" '
            f'solref="{values("solref")}" '
            f'solreffriction="{values("solreffriction")}" '
            f'solimp="{values("solimp")}" '
            f'adhesion="{float(pair["adhesion"]):.12g}" />'
        )
    lines.append("  </contact>")
    return "\n".join(lines)


def inject_explicit_pairs(source_text: str, pair: dict[str, Any]) -> str:
    """Add exactly four explicit floor-to-distal pairs without changing geoms."""
    if "<contact" in source_text:
        raise ValueError("Source XML unexpectedly already has a contact section")
    marker = "  </worldbody>\n"
    if source_text.count(marker) != 1:
        raise ValueError("Cannot locate unique worldbody closing marker")
    block = explicit_pair_block(pair)
    return source_text.replace(marker, f"{marker}{block}\n", 1)


def reverse_explicit_pairs(candidate_text: str, pair: dict[str, Any]) -> str:
    block = explicit_pair_block(pair) + "\n"
    if candidate_text.count(block) != 1:
        raise ValueError("Candidate does not contain one exact explicit-pair block")
    return candidate_text.replace(block, "", 1)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    value = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    if value < 0:
        raise ValueError(f"Compiled model lacks geom {name}")
    return value


def audit_compiled_pair(
    source_xml: Path, candidate_xml: Path, pair: dict[str, Any]
) -> dict[str, Any]:
    source_text = source_xml.read_text(encoding="utf-8")
    candidate_text = candidate_xml.read_text(encoding="utf-8")
    if reverse_explicit_pairs(candidate_text, pair) != source_text:
        raise ValueError("Candidate XML differs beyond four explicit contact pairs")
    source = mujoco.MjModel.from_xml_path(str(source_xml.resolve()))
    candidate = mujoco.MjModel.from_xml_path(str(candidate_xml.resolve()))
    if (source.nq, source.nv, source.nu) != (candidate.nq, candidate.nv, candidate.nu):
        raise ValueError("Compiled dimensions changed")
    array_fields = (
        "body_mass",
        "geom_type",
        "geom_size",
        "geom_bodyid",
        "geom_friction",
        "geom_condim",
        "geom_solref",
        "geom_solimp",
        "geom_gap",
        "actuator_gear",
        "actuator_ctrlrange",
        "qpos0",
    )
    for field in array_fields:
        left = np.asarray(getattr(source, field))
        right = np.asarray(getattr(candidate, field))
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"Compiled non-margin field changed: {field}")
    if not math.isclose(float(source.opt.timestep), float(candidate.opt.timestep), abs_tol=0.0):
        raise ValueError("Physics timestep changed")
    source_floor = _geom_id(source, "floor")
    candidate_floor = _geom_id(candidate, "floor")
    if not math.isclose(float(source.geom_margin[source_floor]), 0.01, abs_tol=1e-12):
        raise ValueError("Source floor margin is not 0.01")
    if not math.isclose(float(candidate.geom_margin[candidate_floor]), 0.01, abs_tol=1e-12):
        raise ValueError("Candidate floor geom margin changed")
    for name in FOOT_NAMES:
        if not math.isclose(float(source.geom_margin[_geom_id(source, name)]), 0.01, abs_tol=1e-12):
            raise ValueError(f"Source distal margin changed: {name}")
        if not math.isclose(float(candidate.geom_margin[_geom_id(candidate, name)]), 0.01, abs_tol=1e-12):
            raise ValueError(f"Candidate distal geom margin changed: {name}")
    for name in NON_DISTAL_ROBOT_GEOMS:
        if not math.isclose(float(candidate.geom_margin[_geom_id(candidate, name)]), 0.01, abs_tol=1e-12):
            raise ValueError(f"Candidate non-distal margin changed: {name}")
    default_match = re.search(r"<default>.*?<geom\b[^>]*\bmargin=\"([^\"]+)\"", candidate_text, re.S)
    root_match = re.search(
        r'<joint\b(?=[^>]*\bname="root")(?=[^>]*\bmargin="([^"]+)")[^>]*>',
        candidate_text,
    )
    if default_match is None or not math.isclose(float(default_match.group(1)), 0.01, abs_tol=1e-12):
        raise ValueError("Candidate default geom margin changed")
    if root_match is None or not math.isclose(float(root_match.group(1)), 0.01, abs_tol=1e-12):
        raise ValueError("Candidate root joint margin changed")
    if int(source.npair) != 0 or int(candidate.npair) != 4:
        raise ValueError("Expected zero source pairs and exactly four candidate pairs")
    expected_targets = {frozenset(("floor", name)) for name in FOOT_NAMES}
    observed_targets: set[frozenset[str]] = set()
    compiled_pairs: list[dict[str, Any]] = []
    for index in range(int(candidate.npair)):
        geom1 = mujoco.mj_id2name(
            candidate,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(candidate.pair_geom1[index]),
        )
        geom2 = mujoco.mj_id2name(
            candidate,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(candidate.pair_geom2[index]),
        )
        observed_targets.add(frozenset((str(geom1), str(geom2))))
        record = {
            "geom1": geom1,
            "geom2": geom2,
            "margin": float(candidate.pair_margin[index]),
            "gap": float(candidate.pair_gap[index]),
            "condim": int(candidate.pair_dim[index]),
            "friction": np.asarray(candidate.pair_friction[index]).tolist(),
            "solref": np.asarray(candidate.pair_solref[index]).tolist(),
            "solreffriction": np.asarray(candidate.pair_solreffriction[index]).tolist(),
            "solimp": np.asarray(candidate.pair_solimp[index]).tolist(),
            "adhesion": float(candidate.pair_adhesion[index]),
        }
        if (
            record["margin"] != float(pair["margin_m"])
            or record["gap"] != float(pair["gap_m"])
            or record["condim"] != int(pair["condim"])
            or not np.array_equal(record["friction"], pair["friction"])
            or not np.array_equal(record["solref"], pair["solref"])
            or not np.array_equal(record["solreffriction"], pair["solreffriction"])
            or not np.array_equal(record["solimp"], pair["solimp"])
            or record["adhesion"] != float(pair["adhesion"])
        ):
            raise ValueError("Compiled explicit pair contract changed")
        compiled_pairs.append(record)
    if observed_targets != expected_targets:
        raise ValueError("Explicit contact pair target set changed")
    return {
        "only_four_permitted_explicit_pairs_added": True,
        "explicit_pair_count": 4,
        "source_floor_margin_m": float(source.geom_margin[source_floor]),
        "candidate_floor_geom_margin_m": float(candidate.geom_margin[candidate_floor]),
        "source_distal_margins_m": {
            name: float(source.geom_margin[_geom_id(source, name)]) for name in FOOT_NAMES
        },
        "candidate_distal_geom_margins_m": {
            name: float(candidate.geom_margin[_geom_id(candidate, name)]) for name in FOOT_NAMES
        },
        "candidate_non_distal_margins_m": {
            name: float(candidate.geom_margin[_geom_id(candidate, name)])
            for name in NON_DISTAL_ROBOT_GEOMS
        },
        "default_geom_margin_m": float(default_match.group(1)),
        "root_joint_margin_m": float(root_match.group(1)),
        "compiled_explicit_pairs": compiled_pairs,
        "friction": candidate.geom_friction[candidate_floor].tolist(),
        "condim": int(candidate.geom_condim[candidate_floor]),
        "solref": candidate.geom_solref[candidate_floor].tolist(),
        "solimp": candidate.geom_solimp[candidate_floor].tolist(),
        "physics_timestep_seconds": float(candidate.opt.timestep),
    }


def prepare_pair(
    source: dict[str, Any],
    output_root: Path,
    suite_name: str,
    pair_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_xml = Path(source["xml_path"]).resolve()
    source_heights = Path(source["heights_path"]).resolve()
    source_hfield = Path(source["hfield_path"]).resolve()
    source_texture = Path(source["texture_path"]).resolve()
    for path, expected_key in (
        (source_xml, "xml_sha256"),
        (source_heights, "heights_sha256"),
        (source_hfield, "hfield_sha256"),
        (source_texture, "texture_sha256"),
    ):
        if expected_key in source:
            _exact_hash(path, source[expected_key], f"{suite_name} {expected_key}")
    records: dict[str, dict[str, Any]] = {}
    for condition_id in (CONTROL_ID, CANDIDATE_ID):
        scene_dir = output_root / "scenes" / suite_name / condition_id.lower()
        scene_dir.mkdir(parents=True, exist_ok=False)
        xml_path = scene_dir / source_xml.name
        heights_path = scene_dir / "heights_m.npy"
        hfield_path = scene_dir / "terrain.hfield"
        texture_path = scene_dir / "terrain_contours.png"
        shutil.copyfile(source_heights, heights_path)
        shutil.copyfile(source_hfield, hfield_path)
        shutil.copyfile(source_texture, texture_path)
        if condition_id == CONTROL_ID:
            shutil.copyfile(source_xml, xml_path)
            if sha256(xml_path) != sha256(source_xml):
                raise RuntimeError("Control XML is not byte-identical to its source")
        else:
            xml_path.write_text(
                inject_explicit_pairs(
                    source_xml.read_text(encoding="utf-8"), pair_contract
                ),
                encoding="utf-8",
                newline="",
            )
        record = copy.deepcopy(source)
        record.update(
            {
                "suite_name": suite_name,
                "condition_id": condition_id,
                "xml_path": str(xml_path.resolve()),
                "xml_sha256": sha256(xml_path),
                "heights_path": str(heights_path.resolve()),
                "heights_sha256": sha256(heights_path),
                "hfield_path": str(hfield_path.resolve()),
                "hfield_sha256": sha256(hfield_path),
                "texture_path": str(texture_path.resolve()),
                "texture_sha256": sha256(texture_path),
            }
        )
        if record["heights_sha256"] != sha256(source_heights) or record["hfield_sha256"] != sha256(source_hfield) or record["texture_sha256"] != sha256(source_texture):
            raise RuntimeError("Terrain assets changed while preparing a margin pair")
        records[condition_id] = record
    audit = audit_compiled_pair(
        Path(records[CONTROL_ID]["xml_path"]),
        Path(records[CANDIDATE_ID]["xml_path"]),
        pair_contract,
    )
    return records, audit


def make_eval_env(
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    fixed_contract: dict[str, Any] | None,
) -> FixedGoalTerrainWrapper:
    task = copy.deepcopy(protocol["task_adapter"])
    if fixed_contract is not None:
        task.update(fixed_contract["controller"])
        task.update(
            {
                "arrival_radius_m": 1.5,
                "hold_radius_m": 2.0,
                "hold_seconds": 2.0,
                "hold_speed_m_per_s": 0.05,
            }
        )
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
        **common_env_kwargs(reward),
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
        yaw_deadband_degrees=float(task.get("yaw_deadband_degrees", 0.0)),
        curvature_speed_reduction_gain=float(task.get("curvature_speed_reduction_gain", 0.0)),
        minimum_turn_speed_fraction=float(task.get("minimum_turn_speed_fraction", 1.0)),
        slow_radius_m=float(task["slow_radius_m"]),
        arrival_radius_m=float(task["arrival_radius_m"]),
        hold_radius_m=float(task["hold_radius_m"]),
        hold_seconds=float(task["hold_seconds"]),
        hold_speed_m_per_s=float(task["hold_speed_m_per_s"]),
        terminate_on_success=fixed_contract is not None,
        terrain_relative_healthy_clearance_m=tuple(task["terrain_relative_healthy_clearance_m"]),
        maximum_healthy_tilt_degrees=float(task["maximum_healthy_tilt_degrees"]),
        unhealthy_grace_steps=int(task["unhealthy_grace_steps"]),
        slip_speed_threshold_m_per_s=float(task["slip_speed_threshold_m_per_s"]),
        augment_local_terrain_observation=True,
        terrain_frame_shaping_enabled=False,
        terrain_preview_longitudinal_m=tuple(task["terrain_preview_longitudinal_m"]),
        terrain_preview_lateral_m=tuple(task["terrain_preview_lateral_m"]),
        local_terrain_height_bound_m=float(task["local_terrain_height_bound_m"]),
    )


def _summarise_substeps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No physics-substep contact rows were recorded")
    return {
        "control_steps": len(rows),
        "full_interval_zero_foot_fraction": float(
            np.mean([bool(row["full_interval_zero_foot"]) for row in rows])
        ),
        "zero_foot_physics_substep_fraction": float(
            np.mean([float(row["zero_foot_physics_substep_fraction"]) for row in rows])
        ),
        "any_substep_nonfoot_ground_fraction": float(
            np.mean([bool(row["any_substep_nonfoot_ground"]) for row in rows])
        ),
        "any_substep_torso_ground_fraction": float(
            np.mean([bool(row["any_substep_torso_ground"]) for row in rows])
        ),
        "endpoint_last_substep_mismatch_count": int(
            sum(bool(row["endpoint_last_substep_mismatch"]) for row in rows)
        ),
    }


def evaluate_episode(
    model: PPO,
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    high_frequency: bool,
    retain_trace: bool,
    fixed_contract: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = make_eval_env(
        protocol,
        reward,
        scene,
        condition_id=condition_id,
        seed=seed,
        max_episode_steps=max_episode_steps,
        cruise_speed=cruise_speed,
        fixed_contract=fixed_contract,
    )
    observation, _ = env.reset(seed=seed)
    if tuple(observation.shape) != (135,):
        env.close()
        raise RuntimeError(f"Observation interface changed: {observation.shape}")
    substep_state = install_substep_contact_audit(env) if high_frequency else None
    dt = float(env.unwrapped.dt)
    slip_threshold = float(protocol["task_adapter"]["slip_speed_threshold_m_per_s"])
    minimum_slip_steps = int(round(0.2 / dt))
    slip_run = 0
    sustained_slip_steps = 0
    support_sum = 0.0
    endpoint_airborne_count = 0
    endpoint_nonfoot_count = 0
    endpoint_torso_count = 0
    tilt_squared_sum = 0.0
    substep_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward_value, terminated, truncated, info = env.step(action)
        step += 1
        contact_mask = np.asarray(info["proxygap_foot_contact_mask_step"], dtype=bool)
        contact_speeds = np.asarray(
            info["proxygap_foot_contact_tangential_speeds_m_per_s_step"], dtype=np.float64
        )
        active = contact_speeds[contact_mask]
        slip_now = bool(active.size and float(np.max(active)) > slip_threshold)
        if slip_now:
            slip_run += 1
        else:
            if slip_run >= minimum_slip_steps:
                sustained_slip_steps += slip_run
            slip_run = 0
        support = int(np.sum(contact_mask))
        support_sum += support
        endpoint_airborne_count += int(support == 0)
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        terrain_normal = env._terrain_normal(float(qpos[0]), float(qpos[1]))
        tilt = quaternion_tilt_relative_to_normal(qpos[3:7], terrain_normal)
        tilt_squared_sum += tilt * tilt
        foot_ids = tuple(_geom_id(env.unwrapped.model, name) for name in FOOT_NAMES)
        endpoint_mask, endpoint_nonfoot, endpoint_torso = contact_masks_from_data(
            env.unwrapped.model, env.unwrapped.data, foot_ids
        )
        if not np.array_equal(endpoint_mask, contact_mask):
            env.close()
            raise RuntimeError("Independent endpoint contact mask disagrees with wrapper")
        endpoint_nonfoot_count += int(endpoint_nonfoot)
        endpoint_torso_count += int(endpoint_torso)
        if substep_state is not None:
            last = substep_state.get("last")
            if last is None:
                env.close()
                raise RuntimeError("Physics-substep contact audit failed to record")
            masks = np.asarray(last["foot_masks"], dtype=bool)
            nonfoot = np.asarray(last["nonfoot_robot_ground"], dtype=bool)
            torso = np.asarray(last["torso_ground"], dtype=bool)
            if masks.shape != (5, 4):
                env.close()
                raise RuntimeError(f"Unexpected substep contact shape {masks.shape}")
            zero = ~np.any(masks, axis=1)
            substep_rows.append(
                {
                    "condition_id": condition_id,
                    "scene_name": scene["scene_name"],
                    "evaluation_seed": seed,
                    "step": step,
                    "substep_foot_masks": json.dumps(masks.astype(int).tolist(), separators=(",", ":")),
                    "full_interval_zero_foot": bool(np.all(zero)),
                    "zero_foot_physics_substep_fraction": float(np.mean(zero)),
                    "any_substep_nonfoot_ground": bool(np.any(nonfoot)),
                    "any_substep_torso_ground": bool(np.any(torso)),
                    "endpoint_last_substep_mismatch": bool(not np.array_equal(contact_mask, masks[-1])),
                }
            )
        if retain_trace:
            trace_rows.append(
                {
                    "condition_id": condition_id,
                    "scene_name": scene["scene_name"],
                    "evaluation_seed": seed,
                    "step": step,
                    "time_seconds": step * dt,
                    "x_m": float(qpos[0]),
                    "y_m": float(qpos[1]),
                    "torso_z_m": float(qpos[2]),
                    "terrain_height_m": float(env._terrain_height(float(qpos[0]), float(qpos[1]))),
                    "distance_to_goal_m": float(info["proxygap_fixed_goal_distance_m"]),
                    "support_count": support,
                    "endpoint_airborne": support == 0,
                    "foot_contact_mask": json.dumps(contact_mask.astype(int).tolist()),
                    "maximum_contact_tangential_speed_m_per_s": float(np.max(active)) if active.size else 0.0,
                    "endpoint_nonfoot_ground": endpoint_nonfoot,
                    "endpoint_torso_ground": endpoint_torso,
                    "terrain_relative_tilt_rad": tilt,
                    "applied_action": json.dumps(np.asarray(info.get("proxygap_applied_action", action)).tolist(), separators=(",", ":")),
                    "reward": float(reward_value),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    if slip_run >= minimum_slip_steps:
        sustained_slip_steps += slip_run
    summary = env.episode_summary()
    env.close()
    count = max(1, step)
    row = {
        "condition_id": condition_id,
        "scene_name": scene["scene_name"],
        "evaluation_seed": seed,
        "episode_length": step,
        "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"]) - float(summary["fixed_goal_minimum_distance_m"]),
        "fixed_goal_net_progress_m": float(summary["fixed_goal_initial_distance_m"]) - float(summary["fixed_goal_final_distance_m"]),
        "fixed_goal_initial_distance_m": float(summary["fixed_goal_initial_distance_m"]),
        "fixed_goal_final_distance_m": float(summary["fixed_goal_final_distance_m"]),
        "fixed_goal_success": bool(summary["fixed_goal_success"]),
        "fall": bool(summary["fall"]),
        "termination_category": str(summary["termination_category"]),
        "endpoint_airborne_fraction": endpoint_airborne_count / count,
        "mean_support_count": support_sum / count,
        "endpoint_sampled_sustained_slip_fraction": sustained_slip_steps / count,
        "endpoint_nonfoot_ground_fraction": endpoint_nonfoot_count / count,
        "endpoint_torso_ground_fraction": endpoint_torso_count / count,
        "terrain_relative_tilt_rms_rad": math.sqrt(tilt_squared_sum / count),
        "cumulative_squared_action": float(summary["cumulative_squared_action"]),
        "actuator_abs_torque_time_integral_total_n_m_s": float(np.sum(summary["actuator_abs_torque_time_integral_n_m_s_by_actuator"])),
        "actuator_positive_mechanical_work_total_j": float(np.sum(summary["actuator_positive_mechanical_work_j_by_actuator"])),
        "actuator_abs_mechanical_work_total_j": float(np.sum(summary["actuator_abs_mechanical_work_j_by_actuator"])),
        "full_interval_zero_foot_fraction": None,
    }
    if high_frequency:
        sub = _summarise_substeps(substep_rows)
        row.update(sub)
    return row, trace_rows, substep_rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    numeric = (
        "fixed_goal_best_progress_m",
        "fixed_goal_net_progress_m",
        "endpoint_airborne_fraction",
        "mean_support_count",
        "endpoint_sampled_sustained_slip_fraction",
        "endpoint_nonfoot_ground_fraction",
        "endpoint_torso_ground_fraction",
        "terrain_relative_tilt_rms_rad",
        "cumulative_squared_action",
        "actuator_abs_torque_time_integral_total_n_m_s",
        "actuator_positive_mechanical_work_total_j",
        "actuator_abs_mechanical_work_total_j",
    )
    result: dict[str, Any] = {
        "episode_count": len(rows),
        "fall_count": int(sum(bool(row["fall"]) for row in rows)),
        "success_count": int(sum(bool(row["fixed_goal_success"]) for row in rows)),
    }
    for field in numeric:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_std_population"] = float(np.std(values))
    full = [float(row["full_interval_zero_foot_fraction"]) for row in rows if row.get("full_interval_zero_foot_fraction") is not None]
    result["pooled_full_interval_zero_foot_fraction"] = float(np.mean(full)) if full else None
    return result


def standard_gate(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate = config["standard_gate"]
    control_rows = [row for row in rows if row["condition_id"] == CONTROL_ID]
    candidate_rows = [row for row in rows if row["condition_id"] == CANDIDATE_ID]
    control = aggregate(control_rows)
    candidate = aggregate(candidate_rows)
    reduction = float(control["pooled_full_interval_zero_foot_fraction"] - candidate["pooled_full_interval_zero_foot_fraction"])
    support_increase = candidate["mean_support_count_mean"] - control["mean_support_count_mean"]
    progress_ratio = candidate["fixed_goal_best_progress_m_mean"] / max(control["fixed_goal_best_progress_m_mean"], 1e-12)
    fall_delta = candidate["fall_count"] - control["fall_count"]
    slip_increase = candidate["endpoint_sampled_sustained_slip_fraction_mean"] - control["endpoint_sampled_sustained_slip_fraction_mean"]
    success_delta = candidate["success_count"] - control["success_count"]
    per_scene: dict[str, Any] = {}
    scenes_passing = 0
    for scene_name in config["standard_evaluation"]["scene_order"]:
        left = aggregate([row for row in control_rows if row["scene_name"] == scene_name])
        right = aggregate([row for row in candidate_rows if row["scene_name"] == scene_name])
        scene_reduction = float(left["pooled_full_interval_zero_foot_fraction"] - right["pooled_full_interval_zero_foot_fraction"])
        scene_pass = scene_reduction >= float(gate["per_scene_full_interval_reduction_threshold"])
        scenes_passing += int(scene_pass)
        per_scene[scene_name] = {
            "control": left,
            "candidate": right,
            "full_interval_zero_foot_fraction_reduction": scene_reduction,
            "meets_reduction_threshold": scene_pass,
        }
    checks = {
        "pooled_full_interval_reduction": reduction >= float(gate["minimum_pooled_full_interval_zero_foot_fraction_reduction"]),
        "scene_count": scenes_passing >= int(gate["minimum_scenes_with_full_interval_reduction_at_least_threshold"]),
        "support": support_increase >= float(gate["minimum_mean_support_count_increase"]),
        "progress": progress_ratio >= float(gate["minimum_best_progress_ratio"]),
        "falls": fall_delta <= int(gate["maximum_additional_falls"]),
        "slip": slip_increase <= float(gate["maximum_endpoint_sampled_sustained_slip_fraction_increase"]),
        "success": success_delta >= int(gate["minimum_task_success_count_difference"]),
    }
    return {
        "schema_version": "proxygap-distal-margin0-standard-gate-v1",
        "predeclared_gate": gate,
        "control": control,
        "candidate": candidate,
        "per_scene": per_scene,
        "observed": {
            "pooled_full_interval_zero_foot_fraction_reduction": reduction,
            "scenes_meeting_reduction_threshold": scenes_passing,
            "mean_support_count_increase": support_increase,
            "best_progress_ratio": progress_ratio,
            "additional_falls": fall_delta,
            "endpoint_sampled_sustained_slip_fraction_increase": slip_increase,
            "task_success_count_difference": success_delta,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def fixed_gate(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = config["conditional_fixed_map_evaluation"]
    gate = fixed["comparison_gate"]
    control_rows = [row for row in rows if row["condition_id"] == CONTROL_ID]
    candidate_rows = [row for row in rows if row["condition_id"] == CANDIDATE_ID]
    control = aggregate(control_rows)
    candidate = aggregate(candidate_rows)
    representative_seed = int(fixed["representative_seed"])
    rep_control = next(row for row in control_rows if int(row["evaluation_seed"]) == representative_seed)
    rep_candidate = next(row for row in candidate_rows if int(row["evaluation_seed"]) == representative_seed)
    rep_reduction = float(rep_control["full_interval_zero_foot_fraction"] - rep_candidate["full_interval_zero_foot_fraction"])
    endpoint_reduction = control["endpoint_airborne_fraction_mean"] - candidate["endpoint_airborne_fraction_mean"]
    support_increase = candidate["mean_support_count_mean"] - control["mean_support_count_mean"]
    progress_ratio = candidate["fixed_goal_best_progress_m_mean"] / max(control["fixed_goal_best_progress_m_mean"], 1e-12)
    fall_delta = candidate["fall_count"] - control["fall_count"]
    slip_increase = candidate["endpoint_sampled_sustained_slip_fraction_mean"] - control["endpoint_sampled_sustained_slip_fraction_mean"]
    success_delta = candidate["success_count"] - control["success_count"]
    checks = {
        "representative_full_interval_reduction": rep_reduction >= float(gate["minimum_representative_full_interval_zero_foot_fraction_reduction"]),
        "endpoint_airborne_reduction": endpoint_reduction >= float(gate["minimum_endpoint_airborne_fraction_reduction"]),
        "support": support_increase >= float(gate["minimum_mean_support_count_increase"]),
        "progress": progress_ratio >= float(gate["minimum_best_progress_ratio"]),
        "falls": fall_delta <= int(gate["maximum_additional_falls"]),
        "slip": slip_increase <= float(gate["maximum_endpoint_sampled_sustained_slip_fraction_increase"]),
        "success": success_delta >= int(gate["minimum_task_success_count_difference"]),
    }
    return {
        "schema_version": "proxygap-distal-margin0-fixed-map-gate-v1",
        "predeclared_gate": gate,
        "control": control,
        "candidate": candidate,
        "representative_seed": representative_seed,
        "observed": {
            "representative_full_interval_zero_foot_fraction_reduction": rep_reduction,
            "endpoint_airborne_fraction_reduction": endpoint_reduction,
            "mean_support_count_increase": support_increase,
            "best_progress_ratio": progress_ratio,
            "additional_falls": fall_delta,
            "endpoint_sampled_sustained_slip_fraction_increase": slip_increase,
            "task_success_count_difference": success_delta,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _run_matrix(
    model: PPO,
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, dict[str, Any]]],
    *,
    scene_order: list[str],
    seeds: list[int],
    representative_seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    high_frequency_all: bool,
    high_frequency_representative: bool,
    fixed_contract: dict[str, Any] | None,
    output_root: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_id in (CONTROL_ID, CANDIDATE_ID):
        for scene_name in scene_order:
            scene = scenes[scene_name][condition_id]
            for seed in seeds:
                audit = high_frequency_all or (
                    high_frequency_representative and int(seed) == int(representative_seed)
                )
                retain = int(seed) == int(representative_seed)
                row, trace, substeps = evaluate_episode(
                    model,
                    protocol,
                    reward,
                    scene,
                    condition_id=condition_id,
                    seed=int(seed),
                    max_episode_steps=max_episode_steps,
                    cruise_speed=cruise_speed,
                    high_frequency=audit,
                    retain_trace=retain,
                    fixed_contract=fixed_contract,
                )
                rows.append(row)
                if retain:
                    stem = f"{prefix}_{condition_id.lower()}_{scene_name}_seed_{seed}"
                    trace_path = output_root / "traces" / f"{stem}_trace.csv"
                    write_rows(trace_path, trace)
                    if audit:
                        write_rows(output_root / "substeps" / f"{stem}_substeps.csv", substeps)
                print(json.dumps({
                    "suite": prefix,
                    "condition": condition_id,
                    "scene": scene_name,
                    "seed": int(seed),
                    "progress_m": row["fixed_goal_best_progress_m"],
                    "endpoint_airborne": row["endpoint_airborne_fraction"],
                    "full_interval_zero": row.get("full_interval_zero_foot_fraction"),
                    "fall": row["fall"],
                }))
    return rows


def _fixed_source(config: dict[str, Any]) -> dict[str, Any]:
    fixed = config["conditional_fixed_map_evaluation"]
    fixed_config = json.loads((ROOT / fixed["fixed_map_configuration"]).read_text(encoding="utf-8"))
    approved = fixed_config["approved_map"]
    return {
        "scene_name": "approved_fixed_map",
        "xml_path": str((ROOT / fixed["source_xml"]).resolve()),
        "xml_sha256": fixed["source_xml_sha256"],
        "heights_path": str((ROOT / fixed["source_heights"]).resolve()),
        "heights_sha256": fixed["source_heights_sha256"],
        "hfield_path": str((ROOT / fixed["source_hfield"]).resolve()),
        "hfield_sha256": fixed["source_hfield_sha256"],
        "texture_path": str((ROOT / fixed["source_texture"]).resolve()),
        "texture_sha256": fixed["source_texture_sha256"],
        "map_half_extent_m": float(approved["map_half_extent_m"]),
        "start_xy_m": approved["start_xy_m"],
        "goal_xy_m": approved["goal_xy_m"],
    }


def _artifact_hashes(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward, source_manifest = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config": str(config_path), "sha256": sha256(config_path)}, indent=2))
        return
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / config["execution"]["output_root"]).resolve()
    )
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "frozen_config.json").write_bytes(config_path.read_bytes())
    torch.set_num_threads(int(config["execution"]["torch_num_threads"]))

    standard_pairs: dict[str, dict[str, dict[str, Any]]] = {}
    scene_audits: dict[str, Any] = {}
    for scene_name in config["standard_evaluation"]["scene_order"]:
        source = source_manifest["scenes"][scene_name]
        pair, audit = prepare_pair(
            source,
            output_root,
            f"standard_{scene_name}",
            config["permitted_xml_change"]["explicit_pair_contract"],
        )
        standard_pairs[scene_name] = pair
        scene_audits[scene_name] = audit
    write_json(output_root / "standard_scene_margin_audit.json", scene_audits)

    model = PPO.load(
        ROOT / config["frozen_standard_protocol"]["source_checkpoint"],
        device=str(config["execution"]["device"]),
    )
    if tuple(model.observation_space.shape) != (135,) or tuple(model.action_space.shape) != (8,):
        raise RuntimeError("Loaded checkpoint interface is not the frozen 135D/8D contract")
    standard_eval = config["standard_evaluation"]
    standard_seeds = [int(standard_eval["seeds"][0])] if args.smoke else [int(value) for value in standard_eval["seeds"]]
    standard_steps = 20 if args.smoke else int(standard_eval["max_episode_steps"])
    standard_rows = _run_matrix(
        model,
        protocol,
        reward,
        standard_pairs,
        scene_order=list(standard_eval["scene_order"]),
        seeds=standard_seeds,
        representative_seed=(standard_seeds[0] if args.smoke else int(standard_eval["representative_seed"])),
        max_episode_steps=standard_steps,
        cruise_speed=float(standard_eval["cruise_speed_m_per_s"]),
        high_frequency_all=True,
        high_frequency_representative=False,
        fixed_contract=None,
        output_root=output_root,
        prefix="standard",
    )
    write_rows(output_root / "standard_paired_episode_metrics.csv", standard_rows)
    standard_result = standard_gate(config, standard_rows)
    if args.smoke:
        standard_result["passed"] = False
        standard_result["smoke_override"] = "Fixed-map evaluation is intentionally skipped in smoke mode."
    write_json(output_root / "standard_gate.json", standard_result)

    fixed_status: dict[str, Any]
    fixed_rows: list[dict[str, Any]] = []
    if standard_result["passed"] and not args.smoke:
        fixed_source = _fixed_source(config)
        fixed_pair, fixed_audit = prepare_pair(
            fixed_source,
            output_root,
            "fixed_approved_map",
            config["permitted_xml_change"]["explicit_pair_contract"],
        )
        write_json(output_root / "fixed_map_margin_audit.json", fixed_audit)
        fixed_scene = {"approved_fixed_map": fixed_pair}
        fixed_config = config["conditional_fixed_map_evaluation"]
        fixed_rows = _run_matrix(
            model,
            protocol,
            reward,
            fixed_scene,
            scene_order=["approved_fixed_map"],
            seeds=[int(value) for value in fixed_config["seeds"]],
            representative_seed=int(fixed_config["representative_seed"]),
            max_episode_steps=int(fixed_config["horizon_steps"]),
            cruise_speed=float(fixed_config["controller"]["cruise_speed_m_per_s"]),
            high_frequency_all=False,
            high_frequency_representative=True,
            fixed_contract=fixed_config,
            output_root=output_root,
            prefix="fixed_map",
        )
        write_rows(output_root / "fixed_map_paired_episode_metrics.csv", fixed_rows)
        fixed_status = fixed_gate(config, fixed_rows)
        write_json(output_root / "fixed_map_gate.json", fixed_status)
    else:
        fixed_status = {
            "status": "skipped_fail_closed",
            "reason": "standard_gate_failed" if not args.smoke else "smoke_mode",
            "fixed_map_assets_copied": False,
            "fixed_map_episodes_run": 0,
        }
        write_json(output_root / "fixed_map_gate.json", fixed_status)

    contract_worth = bool(standard_result["passed"] and fixed_status.get("passed", False))
    decision = {
        "schema_version": "proxygap-distal-margin0-contract-decision-v1",
        "standard_gate_passed": bool(standard_result["passed"]),
        "fixed_map_gate_passed": bool(fixed_status.get("passed", False)),
        "margin0_worth_next_training_contract": contract_worth,
        "training_performed": False,
        "checkpoint_modified": False,
        "map_modified": False,
        "source_checkpoint_remains_incumbent": True,
        "interpretation": (
            "Both predeclared gates passed; margin0 is eligible only as a separately trained candidate contract."
            if contract_worth
            else "At least one predeclared gate did not pass; do not promote margin0 into the next training contract from this diagnostic."
        ),
    }
    write_json(output_root / "contract_decision.json", decision)
    manifest = {
        "schema_version": "proxygap-fixed-standard-distal-margin0-paired-manifest-v1",
        "status": "complete_read_only_development_diagnostic",
        "configuration": {"path": str(config_path), "sha256": sha256(config_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "source_checkpoint": {
            "path": str((ROOT / config["frozen_standard_protocol"]["source_checkpoint"]).resolve()),
            "sha256": config["frozen_standard_protocol"]["source_checkpoint_sha256"],
            "observation_dimension": 135,
            "action_dimension": 8,
        },
        "candidate_scope": "four explicit floor-to-distal contact pairs use margin=0 and gap=0; every geom margin remains 0.01",
        "all_geom_margins_m": 0.01,
        "standard_gate_passed": bool(standard_result["passed"]),
        "fixed_map_evaluation_status": "completed" if fixed_rows else "skipped_fail_closed",
        "fixed_map_gate_passed": bool(fixed_status.get("passed", False)),
        "margin0_worth_next_training_contract": contract_worth,
        "training_performed": False,
        "friction_reward_observation_energy_checkpoint_changed": False,
        "artifacts": _artifact_hashes(output_root),
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete",
        "output_root": str(output_root),
        "standard_gate_passed": standard_result["passed"],
        "fixed_map_status": fixed_status.get("status", "completed"),
        "fixed_map_gate_passed": fixed_status.get("passed", False),
        "margin0_worth_next_training_contract": contract_worth,
        "manifest_sha256": sha256(output_root / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
