"""Run the frozen L2 exploratory DEFAULT_CONTINUE versus PAIR0_ADAPT pilot.

The script deliberately excludes fixed-map evaluation, video rendering and
promotion.  Each condition starts from the same checkpoint and training seed.
Every worker validates its compiled MuJoCo contact contract before PPO is
allowed to collect a transition.  Evaluation samples all five 0.01 s physics
substeps and applies the force-gated, landing-grace, duration-corrected slip
definition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (SRC, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_fixed_standard_distal_margin0_paired import (  # noqa: E402
    CANDIDATE_ID as SOURCE_PAIR0_ID,
    CONTROL_ID as SOURCE_DEFAULT_ID,
    ROBOT_GEOM_NAMES,
    prepare_pair,
    validate_config as validate_pair_diagnostic,
)
from evaluate_local_preview_final_paired_direct_goal import (  # noqa: E402
    DurationCorrectedSlipTracker,
)
from run_fixed_goal_support_priority_pilot import (  # noqa: E402
    _configure_continuation_model,
)
from run_fixed_standard_support_curriculum import (  # noqa: E402
    FOOT_NAMES,
    make_standard_env,
    prepare_standard_scenes,
    sha256,
    validate_config as validate_standard_protocol,
    verified_json,
    write_json,
    write_rows,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "fixed_standard_pair0_adaptation_l2_pilot_v1_20260819.json"
)
DEFAULT_ID = "DEFAULT_CONTINUE"
PAIR0_ID = "PAIR0_ADAPT"
CONDITION_IDS = (DEFAULT_ID, PAIR0_ID)
EXPECTED_SCENES = ("flat", "uphill_8deg", "downhill_8deg", "bowl_exit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser.parse_args()


def _exact(observed: Any, expected: Any) -> bool:
    return bool(
        np.array_equal(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
        )
    )


def _required_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen field changed: {label}")


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate every L2 boundary before constructing an output directory."""

    _required_equal(
        config.get("schema_version"),
        "proxygap-fixed-standard-pair0-adaptation-l2-pilot-v1",
        "schema_version",
    )
    _required_equal(
        config.get("status"),
        "frozen_l2_exploratory_single_training_seed",
        "status",
    )
    source = config["source"]
    protocol = verified_json(
        ROOT / source["standard_protocol"], source["standard_protocol_sha256"]
    )
    _, reward = validate_standard_protocol(protocol)
    _required_equal(
        source["reward_configuration"],
        protocol["frozen_sources"]["reward_configuration"],
        "reward configuration path",
    )
    _required_equal(
        source["reward_configuration_sha256"],
        protocol["frozen_sources"]["reward_configuration_sha256"],
        "reward configuration hash",
    )
    _required_equal(
        sha256(ROOT / source["checkpoint"]),
        source["checkpoint_sha256"],
        "source checkpoint hash",
    )
    if "artifacts/smoke" in source["checkpoint"].replace("\\", "/"):
        raise ValueError("An old smoke checkpoint cannot be a source")
    if bool(source["old_smoke_checkpoint_as_source"]):
        raise ValueError("Old smoke checkpoint use must remain false")
    for key, expected in (
        ("checkpoint_timesteps", 2596864),
        ("observation_dimension", 135),
        ("action_dimension", 8),
    ):
        _required_equal(int(source[key]), expected, f"source.{key}")

    evidence = config["authorising_l2_evidence"]
    pair_config = verified_json(
        ROOT / evidence["pair_diagnostic_configuration"],
        evidence["pair_diagnostic_configuration_sha256"],
    )
    pair_protocol, pair_reward, _ = validate_pair_diagnostic(pair_config)
    if pair_protocol != protocol or pair_reward != reward:
        raise ValueError("Pair diagnostic did not use the frozen protocol and reward")
    verified_json(
        ROOT / evidence["corrected_slip_configuration"],
        evidence["corrected_slip_configuration_sha256"],
    )
    corrected = verified_json(
        ROOT / evidence["corrected_slip_summary"],
        evidence["corrected_slip_summary_sha256"],
    )
    corrected_manifest = verified_json(
        ROOT / evidence["corrected_slip_manifest"],
        evidence["corrected_slip_manifest_sha256"],
    )
    interpretation = corrected["exploratory_training_interpretation"]
    if not bool(interpretation["supports_new_small_bounded_exploratory_training"]):
        raise ValueError("Corrected-slip audit does not support the bounded L2 pilot")
    if bool(interpretation["formal_training_or_promotion_authorised"]):
        raise ValueError("The L2 evidence must not authorise formal training")
    if (
        corrected_manifest.get("status")
        != "complete_read_only_supplementary_audit"
        or bool(corrected_manifest.get("training_performed", True))
        or bool(corrected_manifest.get("parent_gate_modified", True))
        or bool(corrected_manifest.get("fixed_map_evaluated", True))
        or not bool(
            corrected_manifest.get(
                "supports_new_small_bounded_exploratory_training", False
            )
        )
    ):
        raise ValueError("Corrected-slip manifest crossed the L2 boundary")

    _required_equal(
        [row["condition_id"] for row in config["conditions"]],
        list(CONDITION_IDS),
        "condition order",
    )
    _required_equal(
        [int(row["explicit_pair_count"]) for row in config["conditions"]],
        [0, 4],
        "condition pair counts",
    )
    pair = config["contact_contract"]
    source_pair = pair_config["permitted_xml_change"]["explicit_pair_contract"]
    _required_equal(pair["terrain_geom"], "floor", "terrain geom")
    _required_equal(pair["distal_geoms"], list(FOOT_NAMES), "distal geoms")
    _required_equal(float(pair["all_geom_margins_m"]), 0.01, "geom margin")
    for key, local_key in (
        ("margin_m", "explicit_pair_margin_m"),
        ("gap_m", "explicit_pair_gap_m"),
        ("condim", "condim"),
        ("adhesion", "adhesion"),
    ):
        _required_equal(float(pair[local_key]), float(source_pair[key]), local_key)
    for key, local_key in (
        ("friction", "explicit_pair_friction"),
        ("solref", "solref"),
        ("solreffriction", "solreffriction"),
        ("solimp", "solimp"),
    ):
        if not _exact(pair[local_key], source_pair[key]):
            raise ValueError(f"Explicit-pair field changed: {local_key}")
    if bool(pair["all_ground_pairs_included"]):
        raise ValueError("All-ground pairs are outside this pilot")

    training = config["training"]
    _required_equal(training["scene_order"], list(EXPECTED_SCENES), "training scenes")
    _required_equal(training["condition_run_order"], list(CONDITION_IDS), "run order")
    for key, expected in (
        ("training_seed", 62805),
        ("parallel_environments", 4),
        ("max_episode_steps", 600),
        ("additional_timesteps_per_condition", 65536),
        ("checkpoint_interval_timesteps", 16384),
    ):
        _required_equal(int(training[key]), expected, f"training.{key}")
    _required_equal(
        training["checkpoint_additional_timesteps"],
        [16384, 32768, 49152, 65536],
        "checkpoint schedule",
    )
    _required_equal(float(training["cruise_speed_m_per_s"]), 0.55, "training speed")
    if not (
        bool(training["same_source_checkpoint_for_both_conditions"])
        and bool(training["same_training_seed_for_both_conditions"])
    ):
        raise ValueError("Conditions must share the source and seed")

    evaluation = config["evaluation"]
    _required_equal(evaluation["seeds"], [82801, 82802, 82803], "evaluation seeds")
    _required_equal(evaluation["scene_order"], list(EXPECTED_SCENES), "evaluation scenes")
    for key, expected in (
        ("episodes_per_checkpoint", 12),
        ("max_episode_steps", 600),
        ("physics_substeps_per_control_step", 5),
    ):
        _required_equal(int(evaluation[key]), expected, f"evaluation.{key}")
    _required_equal(float(evaluation["cruise_speed_m_per_s"]), 0.55, "evaluation speed")
    _required_equal(float(evaluation["physics_timestep_seconds"]), 0.01, "physics dt")
    if not bool(evaluation["all_five_physics_substeps_required"]):
        raise ValueError("Five-substep evaluation cannot be released")
    slip = evaluation["corrected_slip"]
    for key, expected in (
        ("tangential_speed_threshold_m_per_s", 0.2),
        ("minimum_normal_force_n", 1.0),
        ("landing_grace_seconds", 0.1),
        ("minimum_sustained_seconds", 0.2),
    ):
        _required_equal(float(slip[key]), expected, f"corrected_slip.{key}")

    if config["ppo"] != protocol["ppo"]:
        raise ValueError("PPO hyperparameters differ from the frozen standard protocol")
    invariants = config["invariants"]
    required_true = (
        "reward_unchanged",
        "observation_135d_unchanged",
        "friction_unchanged",
        "energy_formula_unchanged",
        "control_frequency_20_hz_unchanged",
        "robot_xml_change_limited_to_four_explicit_pairs_for_pair0",
    )
    if not all(bool(invariants[key]) for key in required_true):
        raise ValueError("A frozen invariant was released")
    if invariants["energy_status"] != "measurement_only_not_reward":
        raise ValueError("Energy must remain measurement-only")
    if float(reward["preserved_pre_pitch_reward"]["ctrl_cost_weight"]) != 0.5:
        raise ValueError("Reward changed")
    execution = config["execution"]
    if any(bool(execution[key]) for key in ("fixed_map_evaluation", "video_rendering", "promotion")):
        raise ValueError("Fixed-map, video and promotion must remain disabled")
    if config["turning_gate"] != {
        "status": "not_run_in_this_pilot",
        "fixed_map_entry_authorised_if_not_run": False,
    }:
        raise ValueError("Turning boundary changed")
    _validate_predeclared_thresholds(config)
    return protocol, reward, pair_config, corrected


def _validate_predeclared_thresholds(config: dict[str, Any]) -> None:
    final = config["prospective_final_gate"]
    expected_final = {
        "minimum_pair0_minus_default_pooled_full_interval_zero_foot_reduction": 0.10,
        "minimum_pair0_minus_default_mean_support_count_increase": 0.20,
        "minimum_pair0_to_default_mean_best_progress_ratio": 0.90,
        "minimum_pair0_to_default_uphill_best_progress_ratio": 0.85,
        "minimum_pair0_to_default_downhill_best_progress_ratio": 0.85,
        "maximum_pair0_minus_default_corrected_sustained_slip_per_supported_fraction": 0.02,
        "maximum_pair0_minus_default_corrected_slip_events_per_100_supported_substeps": 0.20,
        "frozen_unadapted_pair0_pooled_full_interval_zero_foot_fraction": 0.028055555555555556,
        "maximum_pair0_full_interval_worsening_vs_frozen_unadapted_pair0": 0.03,
    }
    for key, expected in expected_final.items():
        _required_equal(float(final[key]), expected, f"final gate {key}")
    early = config["checkpoint_early_stopping"]
    expected_early = {
        "nonfoot_contact_minimum_sustained_seconds": 0.20,
        "maximum_pair0_pooled_full_interval_zero_foot_fraction": 0.08,
        "maximum_corrected_sustained_slip_per_supported_fraction_each_condition": 0.02,
        "maximum_corrected_slip_events_per_100_supported_substeps_each_condition": 0.20,
        "qualified_transient_per_supported_warning_threshold": 0.10742616033755274,
    }
    for key, expected in expected_early.items():
        _required_equal(float(early[key]), expected, f"early stop {key}")
    special = early["pair0_checkpoint_32768"]
    _required_equal(float(special["frozen_pair0_mean_best_progress_m"]), 6.082593159719103, "frozen progress")
    _required_equal(float(special["minimum_fraction_of_frozen_pair0_progress"]), 0.80, "progress fraction")
    _required_equal(float(special["absolute_progress_floor_m"]), 4.866074527775282, "progress floor")
    _required_equal(float(special["maximum_improvement_from_checkpoint_16384_m"]), 0.10, "progress improvement")


def _pair_contract(config: dict[str, Any]) -> dict[str, Any]:
    pair = config["contact_contract"]
    return {
        "margin_m": pair["explicit_pair_margin_m"],
        "gap_m": pair["explicit_pair_gap_m"],
        "condim": pair["condim"],
        "friction": pair["explicit_pair_friction"],
        "solref": pair["solref"],
        "solreffriction": pair["solreffriction"],
        "solimp": pair["solimp"],
        "adhesion": pair["adhesion"],
    }


def prepare_condition_scenes(
    config: dict[str, Any], protocol: dict[str, Any], output_root: Path
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    controls, generation = prepare_standard_scenes(protocol, output_root / "generated")
    scenes: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in CONDITION_IDS}
    audits: dict[str, Any] = {}
    for scene_name in EXPECTED_SCENES:
        pair, audit = prepare_pair(
            controls[scene_name],
            output_root / "condition_assets",
            f"standard_{scene_name}",
            _pair_contract(config),
        )
        scenes[DEFAULT_ID][scene_name] = dict(pair[SOURCE_DEFAULT_ID])
        scenes[PAIR0_ID][scene_name] = dict(pair[SOURCE_PAIR0_ID])
        scenes[DEFAULT_ID][scene_name]["condition_id"] = DEFAULT_ID
        scenes[PAIR0_ID][scene_name]["condition_id"] = PAIR0_ID
        audits[scene_name] = audit
    write_json(output_root / "scene_generation.json", generation)
    write_json(output_root / "explicit_pair_audits.json", audits)
    return scenes, audits


def compiled_contract_audit(
    model: mujoco.MjModel,
    scene: dict[str, Any],
    condition_id: str,
    config: dict[str, Any],
    *,
    worker_seed: int,
) -> dict[str, Any]:
    """Fail closed on the exact compiled model used by one worker."""

    if sha256(scene["xml_path"]) != scene["xml_sha256"]:
        raise RuntimeError("Worker XML hash changed")
    if (int(model.nq), int(model.nv), int(model.nu)) != (15, 14, 8):
        raise RuntimeError("Worker robot dimensions changed")
    pair_count = 0 if condition_id == DEFAULT_ID else 4
    if int(model.npair) != pair_count:
        raise RuntimeError("Worker explicit-pair count changed")
    pair = config["contact_contract"]
    geom_records: list[dict[str, Any]] = []
    for name in ("floor", *ROBOT_GEOM_NAMES):
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            raise RuntimeError(f"Worker lacks geometry {name}")
        record = {
            "name": name,
            "margin": float(model.geom_margin[geom_id]),
            "friction": np.asarray(model.geom_friction[geom_id]).tolist(),
            "condim": int(model.geom_condim[geom_id]),
            "solref": np.asarray(model.geom_solref[geom_id]).tolist(),
            "solimp": np.asarray(model.geom_solimp[geom_id]).tolist(),
        }
        if (
            record["margin"] != float(pair["all_geom_margins_m"])
            or not _exact(record["friction"], pair["geom_friction"])
            or record["condim"] != int(pair["condim"])
            or not _exact(record["solref"], pair["solref"])
            or not _exact(record["solimp"], pair["solimp"])
        ):
            raise RuntimeError(f"Worker geom contract changed: {name}")
        geom_records.append(record)
    explicit_records: list[dict[str, Any]] = []
    expected_targets = {
        frozenset((pair["terrain_geom"], name)) for name in pair["distal_geoms"]
    }
    observed_targets: set[frozenset[str]] = set()
    for index in range(int(model.npair)):
        geom1 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[index])))
        geom2 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[index])))
        observed_targets.add(frozenset((geom1, geom2)))
        record = {
            "geom1": geom1,
            "geom2": geom2,
            "margin": float(model.pair_margin[index]),
            "gap": float(model.pair_gap[index]),
            "condim": int(model.pair_dim[index]),
            "friction": np.asarray(model.pair_friction[index]).tolist(),
            "solref": np.asarray(model.pair_solref[index]).tolist(),
            "solreffriction": np.asarray(model.pair_solreffriction[index]).tolist(),
            "solimp": np.asarray(model.pair_solimp[index]).tolist(),
            "adhesion": float(model.pair_adhesion[index]),
        }
        if (
            record["margin"] != float(pair["explicit_pair_margin_m"])
            or record["gap"] != float(pair["explicit_pair_gap_m"])
            or record["condim"] != int(pair["condim"])
            or not _exact(record["friction"], pair["explicit_pair_friction"])
            or not _exact(record["solref"], pair["solref"])
            or not _exact(record["solreffriction"], pair["solreffriction"])
            or not _exact(record["solimp"], pair["solimp"])
            or record["adhesion"] != float(pair["adhesion"])
        ):
            raise RuntimeError("Worker explicit-pair solver contract changed")
        explicit_records.append(record)
    if condition_id == PAIR0_ID and observed_targets != expected_targets:
        raise RuntimeError("Worker explicit-pair targets changed")
    if condition_id == DEFAULT_ID and observed_targets:
        raise RuntimeError("Default worker unexpectedly compiled explicit pairs")
    if not math.isclose(float(model.opt.timestep), 0.01, abs_tol=0.0):
        raise RuntimeError("Worker physics timestep changed")
    return {
        "condition_id": condition_id,
        "scene_name": scene["scene_name"],
        "worker_seed": int(worker_seed),
        "xml_path": str(Path(scene["xml_path"]).resolve()),
        "xml_sha256": scene["xml_sha256"],
        "npair": int(model.npair),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "physics_timestep_seconds": float(model.opt.timestep),
        "geom_contracts": geom_records,
        "explicit_pairs": explicit_records,
        "passed": True,
    }


def make_training_vec_env(
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    condition_id: str,
    monitor_path: Path,
) -> tuple[VecMonitor, list[dict[str, Any]]]:
    factories: list[Callable[[], gym.Env]] = []
    base_seed = int(config["training"]["training_seed"])
    for rank, scene_name in enumerate(EXPECTED_SCENES):
        scene = scenes[scene_name]
        local_seed = base_seed + 1000 * rank

        def factory(
            local_scene: dict[str, Any] = scene,
            env_seed: int = local_seed,
        ) -> gym.Env:
            env = make_standard_env(
                protocol,
                reward,
                local_scene,
                condition_id=condition_id,
                seed=env_seed,
                max_episode_steps=int(config["training"]["max_episode_steps"]),
                cruise_speed=float(config["training"]["cruise_speed_m_per_s"]),
            )
            audit = compiled_contract_audit(
                env.unwrapped.model,
                local_scene,
                condition_id,
                config,
                worker_seed=env_seed,
            )
            if tuple(env.observation_space.shape) != (135,) or tuple(env.action_space.shape) != (8,):
                env.close()
                raise RuntimeError("Worker observation or action contract changed")
            setattr(env, "_pair0_worker_contract_audit", audit)
            return env

        factories.append(factory)
    base = SubprocVecEnv(
        factories,
        start_method=str(config["execution"]["subprocess_start_method"]),
    )
    audits = list(base.get_attr("_pair0_worker_contract_audit"))
    if len(audits) != 4 or not all(bool(row["passed"]) for row in audits):
        base.close()
        raise RuntimeError("Not all four training workers passed fail-closed audit")
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return VecMonitor(base, filename=str(monitor_path)), audits


def _physics_contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    lookup = {geom_id: index for index, geom_id in enumerate(foot_ids)}
    contacts = np.zeros(4, dtype=bool)
    speeds = np.zeros(4, dtype=np.float64)
    normal_forces = np.zeros(4, dtype=np.float64)
    nonfoot = False
    torso = False
    torso_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom"))
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if body1 == 0 and body2 != 0:
            robot_geom = geom2
        elif body2 == 0 and body1 != 0:
            robot_geom = geom1
        else:
            continue
        if robot_geom not in lookup:
            nonfoot = True
            torso = torso or robot_geom == torso_id
            continue
        foot_index = lookup[robot_geom]
        contacts[foot_index] = True
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_forces[foot_index] += max(0.0, float(force[0]))
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacp,
            jacr,
            np.asarray(contact.pos, dtype=np.float64),
            int(model.geom_bodyid[robot_geom]),
        )
        velocity = jacp @ np.asarray(data.qvel, dtype=np.float64)
        normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("Invalid MuJoCo contact normal")
        normal /= norm
        tangent = velocity - float(np.dot(velocity, normal)) * normal
        speeds[foot_index] = max(speeds[foot_index], float(np.linalg.norm(tangent)))
    return contacts, speeds, normal_forces, nonfoot, torso


def install_five_substep_audit(env: gym.Env) -> dict[str, Any]:
    ant = env.unwrapped
    if int(ant.frame_skip) != 5 or not math.isclose(float(ant.model.opt.timestep), 0.01):
        raise RuntimeError("Evaluation requires exactly 5 x 0.01 s physics substeps")
    foot_ids = tuple(
        int(mujoco.mj_name2id(ant.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in FOOT_NAMES
    )
    if any(value < 0 for value in foot_ids):
        raise RuntimeError("Evaluation cannot find all distal feet")
    state: dict[str, Any] = {"last": None}

    def audited_do_simulation(self: Any, ctrl: np.ndarray, n_frames: int) -> None:
        if int(n_frames) != 5 or np.asarray(ctrl).shape != (self.model.nu,):
            raise RuntimeError("Control stepping contract changed")
        self.data.ctrl[:] = ctrl
        rows = []
        for _ in range(int(n_frames)):
            mujoco.mj_step(self.model, self.data, nstep=1)
            contacts, speeds, forces, nonfoot, torso = _physics_contact_diagnostics(
                self.model, self.data, foot_ids
            )
            rows.append(
                {
                    "contacts": contacts,
                    "speeds": speeds,
                    "forces": forces,
                    "nonfoot": nonfoot,
                    "torso": torso,
                }
            )
        mujoco.mj_rnePostConstraint(self.model, self.data)
        state["last"] = rows

    ant.do_simulation = types.MethodType(audited_do_simulation, ant)
    return state


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _vector_sum(summary: dict[str, Any], key: str) -> float:
    return float(np.sum(np.asarray(summary.get(key, []), dtype=np.float64)))


def evaluate_episode(
    model: PPO,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    checkpoint_additional_timesteps: int,
    max_episode_steps: int,
    retain_substeps: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = make_standard_env(
        protocol,
        reward,
        scene,
        condition_id=condition_id,
        seed=seed,
        max_episode_steps=max_episode_steps,
        cruise_speed=float(config["evaluation"]["cruise_speed_m_per_s"]),
    )
    compiled_contract_audit(env.unwrapped.model, scene, condition_id, config, worker_seed=seed)
    observation, _ = env.reset(seed=seed)
    audit_state = install_five_substep_audit(env)
    slip = config["evaluation"]["corrected_slip"]
    tracker = DurationCorrectedSlipTracker(
        dt=float(config["evaluation"]["physics_timestep_seconds"]),
        speed_threshold=float(slip["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip["minimum_sustained_seconds"]),
    )
    contact_rows: list[np.ndarray] = []
    nonfoot_rows: list[bool] = []
    torso_rows: list[bool] = []
    control_fullzero: list[bool] = []
    traces: list[dict[str, Any]] = []
    finite = True
    terminated = truncated = False
    control_step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward_value, terminated, truncated, _ = env.step(action)
        control_step += 1
        rows = audit_state.get("last")
        if not isinstance(rows, list) or len(rows) != 5:
            env.close()
            raise RuntimeError("Five-substep audit did not return five rows")
        control_contacts = []
        for substep_index, substep in enumerate(rows, start=1):
            contacts = np.asarray(substep["contacts"], dtype=bool)
            speeds = np.asarray(substep["speeds"], dtype=np.float64)
            forces = np.asarray(substep["forces"], dtype=np.float64)
            if contacts.shape != (4,) or speeds.shape != (4,) or forces.shape != (4,):
                env.close()
                raise RuntimeError("Substep foot vector shape changed")
            raw, qualified = tracker.update(
                contact_mask=contacts,
                tangential_speeds=speeds,
                normal_forces=forces,
            )
            contact_rows.append(contacts.copy())
            control_contacts.append(contacts)
            nonfoot_rows.append(bool(substep["nonfoot"]))
            torso_rows.append(bool(substep["torso"]))
            if retain_substeps:
                traces.append(
                    {
                        "condition_id": condition_id,
                        "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
                        "scene_name": scene["scene_name"],
                        "evaluation_seed": seed,
                        "control_step": control_step,
                        "physics_substep": substep_index,
                        "physics_time_seconds": ((control_step - 1) * 5 + substep_index) * 0.01,
                        "contact_mask": json.dumps(contacts.astype(int).tolist()),
                        "tangential_speeds_m_per_s": json.dumps(speeds.tolist(), separators=(",", ":")),
                        "normal_forces_n": json.dumps(forces.tolist(), separators=(",", ":")),
                        "supported": int(np.any(contacts)),
                        "nonfoot_ground": int(bool(substep["nonfoot"])),
                        "torso_ground": int(bool(substep["torso"])),
                        "raw_slip_any": int(np.any(raw)),
                        "qualified_slip_any": int(np.any(qualified)),
                        "corrected_sustained_slip_any": 0,
                    }
                )
        control_fullzero.append(not np.any(np.asarray(control_contacts, dtype=bool)))
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
        finite = finite and bool(
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
            and np.isfinite(reward_value)
            and np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
        )
        if control_step > max_episode_steps:
            env.close()
            raise RuntimeError("Evaluation exceeded its frozen horizon")
    corrected = tracker.finalise()
    contacts = np.asarray(contact_rows, dtype=bool)
    candidate = np.asarray(corrected["candidate"], dtype=bool)
    sustained = np.asarray(corrected["sustained"], dtype=bool)
    if contacts.shape != candidate.shape or contacts.shape != sustained.shape:
        env.close()
        raise RuntimeError("Corrected-slip output shape changed")
    if retain_substeps:
        for index, row in enumerate(traces):
            row["corrected_sustained_slip_any"] = int(np.any(sustained[index]))
    summary = env.episode_summary()
    env.close()
    supported = np.any(contacts, axis=1)
    qualified_any = np.any(candidate, axis=1)
    sustained_any = np.any(sustained, axis=1)
    supported_count = int(np.sum(supported))
    nonfoot_longest = _longest_true_run(nonfoot_rows)
    dt = float(config["evaluation"]["physics_timestep_seconds"])
    events = [
        {
            "condition_id": condition_id,
            "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
            "scene_name": scene["scene_name"],
            "evaluation_seed": seed,
            **event,
        }
        for event in corrected["events"]
    ]
    row = {
        "condition_id": condition_id,
        "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
        "checkpoint_timesteps": int(model.num_timesteps),
        "scene_name": scene["scene_name"],
        "evaluation_seed": seed,
        "control_steps": control_step,
        "physics_substeps": int(contacts.shape[0]),
        "finite": finite,
        "fall": bool(summary.get("fall", False) or summary.get("inner_absolute_z_fall", False)),
        "outer_terrain_fall": bool(summary.get("fall", False)),
        "inner_absolute_z_fall": bool(summary.get("inner_absolute_z_fall", False)),
        "fixed_goal_success": bool(summary["fixed_goal_success"]),
        "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"]) - float(summary["fixed_goal_minimum_distance_m"]),
        "fixed_goal_net_progress_m": float(summary["fixed_goal_net_progress_m"]),
        "full_interval_zero_foot_count": int(np.sum(control_fullzero)),
        "support_count_sum_physics_substeps": int(np.sum(contacts)),
        "supported_physics_substep_count": supported_count,
        "qualified_slip_physics_substep_count": int(np.sum(qualified_any)),
        "corrected_sustained_slip_physics_substep_count": int(np.sum(sustained_any)),
        "corrected_slip_event_count": len(events),
        "torso_ground_any": bool(np.any(torso_rows)),
        "nonfoot_ground_longest_run_seconds": nonfoot_longest * dt,
        "sustained_nonfoot_contact": bool(
            nonfoot_longest * dt
            >= float(config["checkpoint_early_stopping"]["nonfoot_contact_minimum_sustained_seconds"])
        ),
        "cumulative_squared_action": float(summary.get("cumulative_squared_action", 0.0)),
        "actuator_abs_torque_time_integral_total_n_m_s": _vector_sum(summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator"),
        "actuator_positive_mechanical_work_total_j": _vector_sum(summary, "actuator_positive_mechanical_work_j_by_actuator"),
        "actuator_abs_mechanical_work_total_j": _vector_sum(summary, "actuator_abs_mechanical_work_j_by_actuator"),
    }
    return row, traces, events


def aggregate_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation rows")
    total_control = sum(int(row["control_steps"]) for row in rows)
    total_substeps = sum(int(row["physics_substeps"]) for row in rows)
    supported = sum(int(row["supported_physics_substep_count"]) for row in rows)
    result = {
        "episode_count": len(rows),
        "nonfinite_episode_count": sum(not bool(row["finite"]) for row in rows),
        "fall_count": sum(bool(row["fall"]) for row in rows),
        "success_count": sum(bool(row["fixed_goal_success"]) for row in rows),
        "torso_ground_episode_count": sum(bool(row["torso_ground_any"]) for row in rows),
        "sustained_nonfoot_contact_episode_count": sum(bool(row["sustained_nonfoot_contact"]) for row in rows),
        "mean_best_progress_m": float(np.mean([float(row["fixed_goal_best_progress_m"]) for row in rows])),
        "pooled_full_interval_zero_foot_fraction": sum(int(row["full_interval_zero_foot_count"]) for row in rows) / max(1, total_control),
        "mean_support_count": sum(int(row["support_count_sum_physics_substeps"]) for row in rows) / max(1, total_substeps),
        "qualified_slip_per_supported_fraction": sum(int(row["qualified_slip_physics_substep_count"]) for row in rows) / max(1, supported),
        "corrected_sustained_slip_per_supported_fraction": sum(int(row["corrected_sustained_slip_physics_substep_count"]) for row in rows) / max(1, supported),
        "corrected_slip_event_count": sum(int(row["corrected_slip_event_count"]) for row in rows),
        "supported_physics_substep_count": supported,
        "total_control_steps": total_control,
        "total_physics_substeps": total_substeps,
    }
    result["corrected_slip_events_per_100_supported_substeps"] = (
        100.0 * result["corrected_slip_event_count"] / max(1, supported)
    )
    return result


def evaluate_checkpoint(
    model: PPO,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    *,
    condition_id: str,
    checkpoint_additional_timesteps: int,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    evaluation_root = output_root / condition_id.lower() / "evaluations" / f"additional_{checkpoint_additional_timesteps}"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    seeds = (
        [int(value) for value in config["smoke"]["evaluation_seeds"]]
        if smoke
        else [int(value) for value in config["evaluation"]["seeds"]]
    )
    horizon = (
        int(config["smoke"]["evaluation_max_episode_steps"])
        if smoke
        else int(config["evaluation"]["max_episode_steps"])
    )
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for scene_name in EXPECTED_SCENES:
        for seed in seeds:
            row, local_trace, local_events = evaluate_episode(
                model,
                config,
                protocol,
                reward,
                scenes[scene_name],
                condition_id=condition_id,
                seed=seed,
                checkpoint_additional_timesteps=checkpoint_additional_timesteps,
                max_episode_steps=horizon,
                retain_substeps=seed == int(config["evaluation"]["representative_trace_seed"]),
            )
            rows.append(row)
            traces.extend(local_trace)
            events.extend(local_events)
    expected = len(EXPECTED_SCENES) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError("Evaluation episode count changed")
    write_rows(evaluation_root / "episode_metrics.csv", rows)
    if traces:
        write_rows(evaluation_root / "representative_substep_trace.csv", traces)
    write_event_rows(evaluation_root / "corrected_slip_events.csv", events)
    aggregate = aggregate_episode_rows(rows)
    aggregate["per_scene"] = {
        scene_name: aggregate_episode_rows(
            [row for row in rows if row["scene_name"] == scene_name]
        )
        for scene_name in EXPECTED_SCENES
    }
    aggregate["condition_id"] = condition_id
    aggregate["checkpoint_additional_timesteps"] = checkpoint_additional_timesteps
    aggregate["evaluation_seeds"] = seeds
    aggregate["max_episode_steps"] = horizon
    write_json(evaluation_root / "aggregate.json", aggregate)
    return aggregate


EVENT_COLUMNS = (
    "condition_id",
    "checkpoint_additional_timesteps",
    "scene_name",
    "evaluation_seed",
    "foot_index",
    "start_step",
    "end_step",
    "duration_steps",
    "duration_seconds",
    "maximum_tangential_speed_m_per_s",
    "mean_tangential_speed_m_per_s",
    "minimum_normal_force_n",
    "mean_normal_force_n",
    "slip_distance_proxy_m",
)


def write_event_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVENT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_stop_decision(
    config: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    condition_id: str,
    checkpoint_additional_timesteps: int,
    pair0_checkpoint_16384_progress_m: float | None,
) -> dict[str, Any]:
    early = config["checkpoint_early_stopping"]
    failures = {
        "nonfinite": aggregate["nonfinite_episode_count"] > 0,
        "fall": aggregate["fall_count"] > int(early["maximum_fall_count_each_condition"]),
        "torso_ground": aggregate["torso_ground_episode_count"] > int(early["maximum_torso_ground_episode_count_each_condition"]),
        "sustained_nonfoot_contact": aggregate["sustained_nonfoot_contact_episode_count"] > int(early["maximum_sustained_nonfoot_contact_episode_count_each_condition"]),
        "corrected_sustained_slip": aggregate["corrected_sustained_slip_per_supported_fraction"] > float(early["maximum_corrected_sustained_slip_per_supported_fraction_each_condition"]),
        "corrected_slip_event_rate": aggregate["corrected_slip_events_per_100_supported_substeps"] > float(early["maximum_corrected_slip_events_per_100_supported_substeps_each_condition"]),
        "pair0_full_interval_zero_foot": bool(
            condition_id == PAIR0_ID
            and aggregate["pooled_full_interval_zero_foot_fraction"]
            > float(early["maximum_pair0_pooled_full_interval_zero_foot_fraction"])
        ),
    }
    special_triggered = False
    special_observed: dict[str, Any] | None = None
    if condition_id == PAIR0_ID and checkpoint_additional_timesteps == 32768:
        if pair0_checkpoint_16384_progress_m is None:
            raise RuntimeError("PAIR0 32,768 early stop lacks the 16,384 checkpoint")
        special = early["pair0_checkpoint_32768"]
        progress = float(aggregate["mean_best_progress_m"])
        improvement = progress - float(pair0_checkpoint_16384_progress_m)
        below_floor = progress < float(special["absolute_progress_floor_m"])
        insufficient_improvement = improvement <= float(
            special["maximum_improvement_from_checkpoint_16384_m"]
        )
        special_triggered = below_floor and insufficient_improvement
        special_observed = {
            "mean_best_progress_m": progress,
            "checkpoint_16384_mean_best_progress_m": pair0_checkpoint_16384_progress_m,
            "improvement_m": improvement,
            "below_absolute_floor": below_floor,
            "insufficient_improvement": insufficient_improvement,
            "triggered": special_triggered,
        }
    qualified_warning = bool(
        aggregate["qualified_slip_per_supported_fraction"]
        > float(early["qualified_transient_per_supported_warning_threshold"])
    )
    return {
        "condition_id": condition_id,
        "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
        "catastrophe_checks": failures,
        "pair0_checkpoint_32768_rule": special_observed,
        "qualified_transient_warning": qualified_warning,
        "qualified_transient_is_warning_only": True,
        "early_stop_triggered": bool(any(failures.values()) or special_triggered),
    }


def final_gate(
    config: dict[str, Any],
    final_aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate = config["prospective_final_gate"]
    if set(final_aggregates) != set(CONDITION_IDS):
        return {
            "evaluable": False,
            "passed": False,
            "reason": "Both conditions did not reach the frozen 65,536-step budget",
            "fixed_map_entry_authorised": False,
            "candidate_promoted": False,
        }
    control, pair0 = final_aggregates[DEFAULT_ID], final_aggregates[PAIR0_ID]
    progress_ratio = pair0["mean_best_progress_m"] / max(control["mean_best_progress_m"], 1e-12)
    uphill_ratio = pair0["per_scene"]["uphill_8deg"]["mean_best_progress_m"] / max(
        control["per_scene"]["uphill_8deg"]["mean_best_progress_m"], 1e-12
    )
    downhill_ratio = pair0["per_scene"]["downhill_8deg"]["mean_best_progress_m"] / max(
        control["per_scene"]["downhill_8deg"]["mean_best_progress_m"], 1e-12
    )
    observed = {
        "pooled_full_interval_zero_foot_reduction": control["pooled_full_interval_zero_foot_fraction"] - pair0["pooled_full_interval_zero_foot_fraction"],
        "mean_support_count_increase": pair0["mean_support_count"] - control["mean_support_count"],
        "mean_best_progress_ratio": progress_ratio,
        "uphill_best_progress_ratio": uphill_ratio,
        "downhill_best_progress_ratio": downhill_ratio,
        "success_count_difference": pair0["success_count"] - control["success_count"],
        "pair0_fall_count": pair0["fall_count"],
        "pair0_torso_ground_episode_count": pair0["torso_ground_episode_count"],
        "pair0_sustained_nonfoot_contact_episode_count": pair0["sustained_nonfoot_contact_episode_count"],
        "corrected_sustained_slip_fraction_delta": pair0["corrected_sustained_slip_per_supported_fraction"] - control["corrected_sustained_slip_per_supported_fraction"],
        "corrected_slip_events_per_100_delta": pair0["corrected_slip_events_per_100_supported_substeps"] - control["corrected_slip_events_per_100_supported_substeps"],
        "pair0_full_interval_worsening_vs_frozen_unadapted_pair0": pair0["pooled_full_interval_zero_foot_fraction"] - float(gate["frozen_unadapted_pair0_pooled_full_interval_zero_foot_fraction"]),
    }
    checks = {
        "full_interval_reduction": observed["pooled_full_interval_zero_foot_reduction"] >= float(gate["minimum_pair0_minus_default_pooled_full_interval_zero_foot_reduction"]),
        "support_increase": observed["mean_support_count_increase"] >= float(gate["minimum_pair0_minus_default_mean_support_count_increase"]),
        "progress_retention": progress_ratio >= float(gate["minimum_pair0_to_default_mean_best_progress_ratio"]),
        "uphill_progress_retention": uphill_ratio >= float(gate["minimum_pair0_to_default_uphill_best_progress_ratio"]),
        "downhill_progress_retention": downhill_ratio >= float(gate["minimum_pair0_to_default_downhill_best_progress_ratio"]),
        "success_nondecrease": observed["success_count_difference"] >= int(gate["minimum_pair0_minus_default_success_count"]),
        "no_pair0_falls": pair0["fall_count"] <= int(gate["maximum_pair0_fall_count"]),
        "no_pair0_torso_ground": pair0["torso_ground_episode_count"] <= int(gate["maximum_pair0_torso_ground_episode_count"]),
        "no_pair0_sustained_nonfoot": pair0["sustained_nonfoot_contact_episode_count"] <= int(gate["maximum_pair0_sustained_nonfoot_contact_episode_count"]),
        "corrected_slip_fraction_not_worse": observed["corrected_sustained_slip_fraction_delta"] <= float(gate["maximum_pair0_minus_default_corrected_sustained_slip_per_supported_fraction"]),
        "corrected_slip_event_rate_not_worse": observed["corrected_slip_events_per_100_delta"] <= float(gate["maximum_pair0_minus_default_corrected_slip_events_per_100_supported_substeps"]),
        "frozen_pair0_contact_retained": observed["pair0_full_interval_worsening_vs_frozen_unadapted_pair0"] <= float(gate["maximum_pair0_full_interval_worsening_vs_frozen_unadapted_pair0"]),
    }
    passed = bool(all(checks.values()))
    return {
        "evaluable": True,
        "observed": observed,
        "checks": checks,
        "passed": passed,
        "turning_regression_test_status": "not_run_in_this_pilot",
        "fixed_map_entry_authorised": False,
        "candidate_promoted": False,
        "decision": (
            "standard_scene_gate_passed_turning_test_still_required"
            if passed
            else "pair0_adaptation_rejected_by_predeclared_standard_scene_gate"
        ),
    }


def _save_checkpoint(model: PPO, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    model.save(path_without_suffix)
    path = path_without_suffix.with_suffix(".zip")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            return f"unavailable:{type(error).__name__}"

    status = run("status", "--short")
    return {"head": run("rev-parse", "HEAD"), "status_short": status, "dirty": bool(status)}


def run_pilot(
    config_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    output_root: Path,
    *,
    smoke: bool,
    attempt: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite attempt root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
        (output_root / "frozen_config.json").write_bytes(config_path.read_bytes())
        scenes, audits = prepare_condition_scenes(config, protocol, output_root)
        training_records: dict[str, Any] = {}
        final_aggregates: dict[str, dict[str, Any]] = {}
        budget = (
            int(config["smoke"]["additional_timesteps_per_condition"])
            if smoke
            else int(config["training"]["additional_timesteps_per_condition"])
        )
        interval = (
            int(config["smoke"]["checkpoint_interval_timesteps"])
            if smoke
            else int(config["training"]["checkpoint_interval_timesteps"])
        )
        schedule = list(range(interval, budget + 1, interval))
        for condition_id in config["training"]["condition_run_order"]:
            condition_root = output_root / condition_id.lower()
            vec_env, worker_audits = make_training_vec_env(
                config,
                protocol,
                reward,
                scenes[condition_id],
                condition_id,
                condition_root / "logs" / "train_monitor.csv",
            )
            write_json(condition_root / "worker_contract_audits.json", worker_audits)
            checkpoint_records: list[dict[str, Any]] = []
            early_stopped = False
            pair0_16384_progress: float | None = None
            model: PPO | None = None
            try:
                model = _configure_continuation_model(
                    ROOT / config["source"]["checkpoint"],
                    vec_env,
                    config["ppo"],
                    training_seed=int(config["training"]["training_seed"]),
                    smoke=False,
                )
                if int(model.num_timesteps) != int(config["source"]["checkpoint_timesteps"]):
                    raise RuntimeError("Source checkpoint timestep metadata changed")
                previous_additional = 0
                for additional in schedule:
                    chunk = additional - previous_additional
                    model.learn(
                        total_timesteps=chunk,
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
                    expected_absolute = int(config["source"]["checkpoint_timesteps"]) + additional
                    if int(model.num_timesteps) != expected_absolute:
                        raise RuntimeError("PPO checkpoint schedule changed")
                    checkpoint = _save_checkpoint(
                        model,
                        condition_root / "models" / f"checkpoint_{expected_absolute}",
                    )
                    aggregate = evaluate_checkpoint(
                        model,
                        config,
                        protocol,
                        reward,
                        scenes[condition_id],
                        condition_id=condition_id,
                        checkpoint_additional_timesteps=additional,
                        output_root=output_root,
                        smoke=smoke,
                    )
                    if condition_id == PAIR0_ID and additional == 16384:
                        pair0_16384_progress = float(aggregate["mean_best_progress_m"])
                    stop = (
                        {
                            "condition_id": condition_id,
                            "checkpoint_additional_timesteps": additional,
                            "early_stop_triggered": False,
                            "smoke_gate_not_applied": True,
                        }
                        if smoke
                        else checkpoint_stop_decision(
                            config,
                            aggregate,
                            condition_id=condition_id,
                            checkpoint_additional_timesteps=additional,
                            pair0_checkpoint_16384_progress_m=pair0_16384_progress,
                        )
                    )
                    checkpoint_dir = condition_root / "evaluations" / f"additional_{additional}"
                    write_json(checkpoint_dir / "checkpoint_stop_decision.json", stop)
                    record = {
                        "additional_timesteps": additional,
                        "absolute_timesteps": expected_absolute,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": sha256(checkpoint),
                        "aggregate": aggregate,
                        "stop_decision": stop,
                    }
                    checkpoint_records.append(record)
                    previous_additional = additional
                    if bool(stop["early_stop_triggered"]):
                        early_stopped = True
                        break
            finally:
                vec_env.close()
            if model is None:
                raise RuntimeError("PPO model was not created")
            record = {
                "condition_id": condition_id,
                "same_source_checkpoint": config["source"]["checkpoint"],
                "same_training_seed": config["training"]["training_seed"],
                "worker_contract_audits": worker_audits,
                "checkpoint_records": checkpoint_records,
                "early_stopped": early_stopped,
                "completed_additional_timesteps": checkpoint_records[-1]["additional_timesteps"],
                "requested_additional_timesteps": budget,
            }
            training_records[condition_id] = record
            write_json(condition_root / "training_record.json", record)
            if not early_stopped and checkpoint_records[-1]["additional_timesteps"] == budget:
                final_aggregates[condition_id] = checkpoint_records[-1]["aggregate"]

        gate = (
            {
                "evaluable": False,
                "passed": False,
                "reason": "Engineering smoke does not apply scientific gates",
                "fixed_map_entry_authorised": False,
                "candidate_promoted": False,
            }
            if smoke
            else final_gate(config, final_aggregates)
        )
        write_json(output_root / "prospective_final_gate.json", gate)
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "device": config["ppo"]["device"],
        }
        write_json(output_root / "environment.json", environment)
        manifest_path = output_root / "manifest.json"
        manifest = {
            "schema_version": "proxygap-pair0-adaptation-l2-manifest-v1",
            "status": "engineering_smoke_passed" if smoke else "l2_exploratory_pilot_complete",
            "stage": "L1_engineering_smoke" if smoke else "L2_exploratory_single_training_seed",
            "attempt": attempt,
            "configuration": {"path": str(config_path), "sha256": sha256(config_path)},
            "source_checkpoint": {"path": str(ROOT / config["source"]["checkpoint"]), "sha256": config["source"]["checkpoint_sha256"]},
            "condition_training": training_records,
            "prospective_final_gate": gate,
            "fixed_map_evaluated": False,
            "video_rendered": False,
            "candidate_promoted": False,
            "turning_regression_test": "not_run",
            "energy_formula_changed": False,
            "friction_changed": False,
            "old_artifacts_overwritten": False,
            "elapsed_seconds": float(time.perf_counter() - started),
            "environment": environment,
            "git": _git_record(),
            "claim_boundary": config["claim_boundary"],
        }
        write_json(manifest_path, manifest)
        return {
            "status": manifest["status"],
            "output_root": str(output_root),
            "manifest_sha256": sha256(manifest_path),
            "gate": gate,
        }
    except Exception as error:
        write_json(
            output_root / "FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-pair0-adaptation-l2-failure-v1",
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_run_retained": True,
                "old_artifacts_overwritten": False,
                "fixed_map_evaluated": False,
                "video_rendered": False,
                "candidate_promoted": False,
            },
        )
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward, _, _ = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated_l2_exploratory", "config_sha256": sha256(config_path)}, indent=2))
        return
    maximum_attempt = int(config["execution"]["maximum_protocol_retry_index"])
    if args.attempt < 0 or args.attempt > maximum_attempt:
        raise ValueError(f"Attempt must be in [0, {maximum_attempt}]")
    base = (
        args.output_root.resolve()
        if args.output_root is not None
        else ROOT
        / (
            config["execution"]["smoke_output_root"]
            if args.smoke
            else config["execution"]["development_output_root"]
        )
    )
    output_root = base / config["execution"]["attempt_subdirectory_template"].format(attempt=args.attempt)
    result = run_pilot(
        config_path,
        config,
        protocol,
        reward,
        output_root,
        smoke=bool(args.smoke),
        attempt=args.attempt,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
