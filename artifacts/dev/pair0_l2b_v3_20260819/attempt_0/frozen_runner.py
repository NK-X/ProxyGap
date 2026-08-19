"""Run the once-only L2b matched-budget PAIR0 continuation.

The mechanics, contact instrumentation and aggregation are imported from the
audited L2 runner.  This wrapper only adds the two different L2 endpoint
sources, exact worker-seed provenance, process isolation, dual final gate and
the no-checkpoint-selection/hard-stop contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
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

import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "fixed_standard_pair0_adaptation_l2b_extension_v3_20260819.json"
)
DEFAULT_ID = "DEFAULT_CONTINUE"
PAIR0_ID = "PAIR0_ADAPT"
CONDITION_IDS = (DEFAULT_ID, PAIR0_ID)
SCENES = ("flat", "uphill_8deg", "downhill_8deg", "bowl_exit")
RUNTIME_DEPENDENCY_PATHS = (
    "scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py",
    "scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py",
    "scripts/evaluate_fixed_standard_distal_margin0_paired.py",
    "scripts/evaluate_local_preview_final_paired_direct_goal.py",
    "scripts/run_fixed_goal_support_priority_pilot.py",
    "scripts/run_fixed_standard_support_curriculum.py",
    "scripts/run_fixed_goal_terrain_training.py",
    "scripts/run_curved_gait_training.py",
    "src/proxygap/__init__.py",
    "src/proxygap/ant_wrapper.py",
    "src/proxygap/curved_gait.py",
    "src/proxygap/fixed_goal_terrain.py",
    "src/proxygap/metrics.py",
    "src/proxygap/planar_transition.py",
    "src/proxygap/experiment.py",
    "src/proxygap/divergence.py",
    "src/proxygap/protocol.py",
    "src/proxygap/selection.py",
    "src/proxygap/two_experiment_protocol.py",
)
FROZEN_V3_TOP_LEVEL_KEYS = (
    "schema_version",
    "config_id",
    "status",
    "purpose",
    "source",
    "authorisation",
    "conditions",
    "contact_contract",
    "training",
    "evaluation",
    "prospective_final_gate",
    "checkpoint_early_stopping",
    "ppo",
    "energy_boundary",
    "invariants",
    "smoke",
    "execution",
    "claim_boundary",
    "runtime_dependency_contract",
)
FROZEN_V3_SECTION_SHA256 = {
    "source": "f1f0222d41c02d69e9419ed5fbd6a1fc3a5fc697063ba1cd1b7d4cfa4272ec59",
    "authorisation": "cb2226b1ed48deec0d2c5db979bb493caf1a6396a23aa6f3b8d519b792219276",
    "conditions": "df3d860708f84cf0e21a36ae38f5beb02479e94aa5743e5419027c586eb0d27b",
    "contact_contract": "1d0f8624cd8874469e397d476f74d1a4a490a31fc9d7cc144222e630aeac2afe",
    "training": "ed53b0b87716f662b81ec0902ca2a14060f5444fb8f04d12505fba692203111e",
    "evaluation": "bd86523a8bd9124f8be41db0b9a6c7227c77d683222a913cbb6bddef43f1ccb3",
    "prospective_final_gate": "f33d98e66dd3a81291a648d6a6f458f26a589c1f3dbdb4cdbb7f4c19385b8a85",
    "checkpoint_early_stopping": "342d4944efc5436998a23624b119be77b06b63aad6d05f6c26217f45dcc7888e",
    "ppo": "59f0b4889cb599e57e901f6c2bf88ec3e2b6d75f737e66bbb95b02823655e395",
    "energy_boundary": "cb6d7d5ff2578d312f9c939d43a9528c71c8d72d582dbfda848f56025856dd30",
    "invariants": "ba105776c50f8115752177ef9f8268212ab0a8f077f94352c9e2866b32f88432",
    "smoke": "f7f9c7e3bbbb03848d421f3c806ddc88f3d1d11a428639328d2b028bd9253db1",
    "execution": "44a8e5a26ea7a71d05fa20c02d3ad889b04f2023fd26601d936efc9416681add",
    "claim_boundary": "7c83df6b883de1ceea874aec0e1ce01a196c6deaabb36c6cb9ba39f88bebdecf",
}
RUNTIME_SELF_RELATIVE_PATH = (
    "scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py"
)
FROZEN_V3_NORMALISED_RUNTIME_CONTRACT_SHA256 = (
    "ed499890cc12a63016c48f0ab1abf71fce8315b44b8f47a0b8252d2c24cf490a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--condition-worker", choices=CONDITION_IDS)
    parser.add_argument("--worker-mode", choices=("smoke", "run"))
    return parser.parse_args()


def _equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen field changed: {label}")


def _array_equal(observed: Any, expected: Any, label: str) -> None:
    if not np.array_equal(
        np.asarray(observed, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
    ):
        raise ValueError(f"Frozen array changed: {label}")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_frozen_v3_sections(config: dict[str, Any]) -> None:
    """Freeze every non-runtime-map field used by the V3 scientific contract."""

    _equal(tuple(config), FROZEN_V3_TOP_LEVEL_KEYS, "V3 top-level keys and order")
    _equal(
        config.get("config_id"),
        "fixed_standard_pair0_adaptation_l2b_extension_v3_20260819",
        "config_id",
    )
    _equal(
        config.get("purpose"),
        "Gate-only audit repair of the one and only matched-budget L2b extension. V3 supersedes V2 after its formal attempt was interrupted for a protocol mismatch; no V1 or V2 weights or metrics are reused.",
        "purpose",
    )
    for section, expected_digest in FROZEN_V3_SECTION_SHA256.items():
        if section not in config:
            raise ValueError(f"Frozen V3 section missing: {section}")
        observed_digest = _canonical_json_sha256(config[section])
        if observed_digest != expected_digest:
            raise ValueError(f"Frozen V3 section changed: {section}")


def _verify_checkpoint(path: Path, expected_hash: str, expected_steps: int) -> None:
    _equal(l2.sha256(path), expected_hash, f"checkpoint hash {path}")
    model = PPO.load(path, device="cpu")
    if int(model.num_timesteps) != expected_steps:
        raise ValueError(f"Checkpoint timestep metadata changed: {path}")
    if tuple(model.observation_space.shape) != (135,):
        raise ValueError(f"Checkpoint observation shape changed: {path}")
    if tuple(model.action_space.shape) != (8,):
        raise ValueError(f"Checkpoint action shape changed: {path}")
    if not model.policy.optimizer.state_dict().get("state"):
        raise ValueError(f"Checkpoint lacks loaded optimiser state: {path}")


def validate_runtime_dependency_map(config: dict[str, Any]) -> dict[str, str]:
    """Verify the exact live transitive dependency closure used by L2b."""

    contract = config["runtime_dependency_contract"]
    _equal(
        tuple(contract),
        (
            "snapshot_root_name",
            "exact_relative_path_sha256",
            "verify_live_before_each_worker",
            "verify_snapshot_before_each_worker",
            "verify_live_and_snapshot_after_each_worker",
            "copy_preserving_relative_paths",
        ),
        "runtime dependency contract keys and order",
    )
    _equal(contract.get("snapshot_root_name"), "runtime_snapshot", "runtime snapshot root")
    normalised_contract = {
        **contract,
        "exact_relative_path_sha256": {
            **contract["exact_relative_path_sha256"],
            RUNTIME_SELF_RELATIVE_PATH: "<RUNNER_SELF_SHA256>",
        },
    }
    _equal(
        _canonical_json_sha256(normalised_contract),
        FROZEN_V3_NORMALISED_RUNTIME_CONTRACT_SHA256,
        "normalised runtime dependency contract",
    )
    expected = {
        str(path): str(digest)
        for path, digest in contract["exact_relative_path_sha256"].items()
    }
    if tuple(expected) != RUNTIME_DEPENDENCY_PATHS:
        raise ValueError("Frozen runtime dependency path order or membership changed")
    if not all(
        bool(contract[key])
        for key in (
            "verify_live_before_each_worker",
            "verify_snapshot_before_each_worker",
            "verify_live_and_snapshot_after_each_worker",
            "copy_preserving_relative_paths",
        )
    ):
        raise ValueError("Runtime dependency verification boundary was released")
    observed: dict[str, str] = {}
    for relative_path in RUNTIME_DEPENDENCY_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = l2.sha256(path)
        if digest != expected[relative_path]:
            raise ValueError(f"Runtime dependency changed: {relative_path}")
        observed[relative_path] = digest
    return observed


def snapshot_runtime_dependencies(
    config: dict[str, Any], output_root: Path
) -> tuple[Path, dict[str, str]]:
    """Copy the verified closure under an attempt-local path-preserving root."""

    observed = validate_runtime_dependency_map(config)
    snapshot_root = output_root / config["runtime_dependency_contract"][
        "snapshot_root_name"
    ]
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for relative_path, digest in observed.items():
        destination = snapshot_root / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
        if l2.sha256(destination) != digest:
            raise RuntimeError(f"Runtime snapshot copy changed: {relative_path}")
    return snapshot_root, observed


def validate_runtime_snapshot(
    config: dict[str, Any], snapshot_root: Path
) -> dict[str, str]:
    actual_relative_paths = tuple(
        sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file()
        )
    )
    expected_relative_paths = tuple(sorted(RUNTIME_DEPENDENCY_PATHS))
    if actual_relative_paths != expected_relative_paths:
        raise RuntimeError("Runtime snapshot path membership changed")
    expected = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    observed: dict[str, str] = {}
    for relative_path in RUNTIME_DEPENDENCY_PATHS:
        path = snapshot_root / Path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = l2.sha256(path)
        if digest != expected[relative_path]:
            raise RuntimeError(f"Runtime snapshot changed: {relative_path}")
        observed[relative_path] = digest
    return observed


def validate_attempt_semantics(
    config: dict[str, Any],
    *,
    attempt: int,
    base: Path,
    smoke: bool,
    custom_output_root_used: bool,
) -> None:
    """Prevent retry or custom-root bypass of the V3 once-only formal run."""

    maximum = int(config["execution"]["maximum_protocol_retry_index"])
    if maximum != 0 or attempt != 0:
        raise ValueError("V3 permits only canonical attempt_0")
    expected = ROOT / (
        config["execution"]["smoke_output_root"]
        if smoke
        else config["execution"]["development_output_root"]
    )
    if not smoke and custom_output_root_used:
        raise ValueError("Formal V3 custom output roots are forbidden")
    if not smoke and base.resolve() != expected.resolve():
        raise ValueError("Formal V3 must use the canonical development root")


def validate_parent_config_path(config_path: Path) -> None:
    """Forbid an alternate configuration from consuming canonical attempt_0."""

    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("Executable V3 parent runs require the canonical configuration path")


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on every frozen L2b source, budget and claim boundary."""

    _validate_frozen_v3_sections(config)
    _equal(
        config.get("schema_version"),
        "proxygap-fixed-standard-pair0-adaptation-l2b-extension-v3",
        "schema_version",
    )
    _equal(
        config.get("status"),
        "frozen_l2b_v3_gate_only_audit_repair_once_only_matched_budget_extension",
        "status",
    )
    source = config["source"]
    protocol = l2.verified_json(
        ROOT / source["standard_protocol"], source["standard_protocol_sha256"]
    )
    _, reward = l2.validate_standard_protocol(protocol)
    _equal(
        source["reward_configuration"],
        protocol["frozen_sources"]["reward_configuration"],
        "reward path",
    )
    _equal(
        source["reward_configuration_sha256"],
        protocol["frozen_sources"]["reward_configuration_sha256"],
        "reward hash",
    )
    for path_key, hash_key in (
        ("l2_frozen_configuration", "l2_frozen_configuration_sha256"),
        ("l2_manifest", "l2_manifest_sha256"),
        ("l2_final_gate", "l2_final_gate_sha256"),
        ("l2_runtime_dependency", "l2_runtime_dependency_sha256"),
    ):
        _equal(l2.sha256(ROOT / source[path_key]), source[hash_key], path_key)
    _equal(
        l2.sha256(ROOT / source["superseded_v1_configuration"]),
        source["superseded_v1_configuration_sha256"],
        "superseded V1 configuration",
    )
    _equal(
        l2.sha256(ROOT / source["superseded_v2_configuration"]),
        source["superseded_v2_configuration_sha256"],
        "superseded V2 configuration",
    )
    authorisation = config["authorisation"]
    for path_key, hash_key in (
        ("v1_smoke_supersession_record", "v1_smoke_supersession_record_sha256"),
        (
            "v1_development_supersession_record",
            "v1_development_supersession_record_sha256",
        ),
        ("v1_interrupted_attempt_record", "v1_interrupted_attempt_record_sha256"),
        ("v2_invalid_attempt_record", "v2_invalid_attempt_record_sha256"),
    ):
        _equal(
            l2.sha256(ROOT / authorisation[path_key]),
            authorisation[hash_key],
            path_key,
        )
    _equal(
        int(authorisation["maximum_l2b_protocol_retry_index"]),
        0,
        "maximum L2b retry index",
    )
    _equal(
        int(authorisation["maximum_contact_budget_extensions_after_l2"]),
        1,
        "maximum contact-budget extensions",
    )
    _equal(int(source["checkpoint_timesteps"]), 2662400, "source timesteps")
    _equal(int(source["observation_dimension"]), 135, "observation dimension")
    _equal(int(source["action_dimension"]), 8, "action dimension")
    expected_sources = {
        DEFAULT_ID: (
            "artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/default_continue/models/checkpoint_2662400.zip",
            "6549c279ca5795636d3b1d6f61c36782f4f843a32107276adf0630c39871cb6f",
            0,
        ),
        PAIR0_ID: (
            "artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/pair0_adapt/models/checkpoint_2662400.zip",
            "9eb1268352aeb90024f681b70ca3b42cb036f6e5ea882e56dbb85262bd8c500e",
            4,
        ),
    }
    for condition_id, (path, digest, pair_count) in expected_sources.items():
        record = source["conditions"][condition_id]
        _equal(record["checkpoint"].replace("\\", "/"), path, f"{condition_id} path")
        _equal(record["checkpoint_sha256"], digest, f"{condition_id} hash")
        _equal(int(record["explicit_pair_count"]), pair_count, f"{condition_id} pair count")
        _verify_checkpoint(ROOT / path, digest, 2662400)

    _equal([row["condition_id"] for row in config["conditions"]], list(CONDITION_IDS), "condition order")
    _equal([int(row["explicit_pair_count"]) for row in config["conditions"]], [0, 4], "condition pair counts")
    contact = config["contact_contract"]
    _equal(contact["terrain_geom"], "floor", "terrain geom")
    _equal(contact["distal_geoms"], list(l2.FOOT_NAMES), "distal geoms")
    _equal(float(contact["all_geom_margins_m"]), 0.01, "geom margin")
    _equal(float(contact["explicit_pair_margin_m"]), 0.0, "pair margin")
    _equal(float(contact["explicit_pair_gap_m"]), 0.0, "pair gap")
    _equal(int(contact["condim"]), 3, "condim")
    _array_equal(contact["geom_friction"], [1.0, 0.5, 0.5], "geom friction")
    _array_equal(contact["explicit_pair_friction"], [1.0, 1.0, 0.5, 0.5, 0.5], "pair friction")
    if bool(contact["all_ground_pairs_included"]):
        raise ValueError("All-ground contact pairs remain outside L2b")

    training = config["training"]
    _equal(int(training["master_seed"]), 62806, "master seed")
    _equal(
        training["worker_effective_seeds_by_scene"],
        {"flat": 62806, "uphill_8deg": 62807, "downhill_8deg": 62808, "bowl_exit": 62809},
        "worker seeds",
    )
    _equal(training["scene_order"], list(SCENES), "training scenes")
    _equal(training["condition_run_order"], list(CONDITION_IDS), "condition order")
    _equal(int(training["parallel_environments"]), 4, "parallel envs")
    _equal(int(training["max_episode_steps"]), 600, "training horizon")
    _equal(int(training["additional_timesteps_per_condition"]), 65536, "budget")
    _equal(int(training["checkpoint_interval_timesteps"]), 16384, "interval")
    _equal(training["checkpoint_additional_timesteps"], [16384, 32768, 49152, 65536], "relative checkpoints")
    _equal(training["checkpoint_absolute_timesteps"], [2678784, 2695168, 2711552, 2727936], "absolute checkpoints")
    _equal(float(training["cruise_speed_m_per_s"]), 0.55, "cruise speed")
    if not all(
        bool(training[key])
        for key in (
            "same_master_seed_for_both_conditions",
            "same_worker_effective_seeds_for_both_conditions",
            "independent_clean_process_required_for_each_condition",
            "load_optimizer_state_from_each_condition_checkpoint",
        )
    ):
        raise ValueError("Matched continuation or process isolation was released")
    if bool(training["reset_num_timesteps"]):
        raise ValueError("Continuation must retain absolute timesteps")

    evaluation = config["evaluation"]
    _equal(evaluation["intermediate_safety_audit"]["seeds"], [82801, 82802, 82803], "intermediate seeds")
    _equal(evaluation["final_heldout"]["seeds"], [83801, 83802, 83803, 83804, 83805], "held-out seeds")
    _equal(evaluation["final_continuity"]["seeds"], [82801, 82802, 82803], "continuity seeds")
    if bool(evaluation["final_heldout"]["used_at_intermediate_checkpoints"]):
        raise ValueError("Held-out seeds cannot be used for intermediate checks")
    _equal(evaluation["scene_order"], list(SCENES), "evaluation scenes")
    _equal(int(evaluation["max_episode_steps"]), 600, "evaluation horizon")
    _equal(int(evaluation["physics_substeps_per_control_step"]), 5, "physics substeps")
    if not bool(evaluation["all_five_physics_substeps_required"]):
        raise ValueError("Five-substep audit cannot be disabled")
    slip = evaluation["corrected_slip"]
    _equal(slip["primary_denominator"], "physics_substeps_with_at_least_one_distal_foot_in_contact_and_normal_force_at_least_1N", "primary denominator")
    _equal(slip["secondary_denominator"], "physics_substeps_with_at_least_one_distal_foot_contact_of_any_force", "secondary denominator")
    _equal(slip["zero_primary_denominator"], "non_evaluable", "zero denominator")

    if config["ppo"] != protocol["ppo"]:
        raise ValueError("PPO changed from the frozen standard protocol")
    invariants = config["invariants"]
    for key in (
        "reward_unchanged",
        "observation_135d_unchanged",
        "friction_unchanged",
        "energy_formula_unchanged",
        "control_frequency_20_hz_unchanged",
        "robot_xml_change_limited_to_four_explicit_pairs_for_pair0",
        "old_l2_artifacts_overwritten",
        "intermediate_checkpoint_selection_forbidden",
    ):
        expected = False if key == "old_l2_artifacts_overwritten" else True
        _equal(bool(invariants[key]), expected, f"invariant {key}")
    energy = config["energy_boundary"]
    _equal(energy["status"], "measurement_only_not_reward_or_gate", "energy status")
    _equal(float(energy["reward_weight"]), 0.0, "energy reward weight")
    _equal(
        energy["raw_components_required"],
        [
            "cumulative_squared_action",
            "actuator_abs_torque_time_integral_total_n_m_s",
            "actuator_positive_mechanical_work_total_j",
            "actuator_abs_mechanical_work_total_j",
        ],
        "energy raw components",
    )
    if not bool(energy["formula_unchanged"]):
        raise ValueError("Energy formula changed")
    if not bool(energy["nonfinite_energy_component_is_run_failure"]):
        raise ValueError("Non-finite energy must fail closed")
    if bool(energy["electrical_battery_energy_claim_permitted"]):
        raise ValueError("Electrical battery-energy claims remain forbidden")
    execution = config["execution"]
    if any(bool(execution[key]) for key in ("fixed_map_evaluation", "video_rendering", "promotion")):
        raise ValueError("Fixed-map, video and promotion are forbidden")
    _equal(
        int(execution["maximum_protocol_retry_index"]),
        0,
        "execution maximum retry index",
    )
    if not bool(execution["fail_if_attempt_root_exists"]):
        raise ValueError("Attempt-root overwrite refusal cannot be disabled")
    if not bool(execution["hard_stop_after_this_extension"]):
        raise ValueError("The once-only hard stop was released")
    _equal(execution["smoke_output_root"], "artifacts/smoke/pair0_l2b_v3_20260819", "smoke root")
    _equal(execution["development_output_root"], "artifacts/dev/pair0_l2b_v3_20260819", "development root")
    _equal(execution["attempt_subdirectory_template"], "attempt_{attempt}", "attempt template")
    _equal(execution["subprocess_start_method"], "spawn", "subprocess start method")
    _equal(
        execution["next_intervention_if_final_gate_fails"],
        "retain_the_existing_13d_local_terrain_preview_and_redesign_or_strengthen_terrain_feature_utilisation_and_the_terrain_normal_downhill_controller_in_a_separately_predeclared_architecture_experiment_with_reward_held_fixed_initially",
        "next intervention",
    )
    smoke = config["smoke"]
    _equal(int(smoke["additional_timesteps_per_condition"]), 2048, "smoke budget")
    _equal(int(smoke["checkpoint_interval_timesteps"]), 2048, "smoke interval")
    _equal(smoke["evaluation_seeds"], [82801], "smoke evaluation seeds")
    _equal(int(smoke["evaluation_max_episode_steps"]), 80, "smoke horizon")
    if bool(smoke["scientific_gate_applied"]):
        raise ValueError("Engineering smoke cannot apply the scientific gate")
    _validate_gate_contract(config)
    validate_runtime_dependency_map(config)
    return protocol, reward


def _validate_gate_contract(config: dict[str, Any]) -> None:
    gate = config["prospective_final_gate"]
    if not bool(gate["both_heldout_and_continuity_required"]):
        raise ValueError("Both final gates are mandatory")
    if not bool(gate["zero_primary_denominator_makes_gate_non_evaluable"]):
        raise ValueError("Zero primary denominator must make the final gate non-evaluable")
    if not bool(gate["required_all_checks"]):
        raise ValueError("Every prospective final-gate check must remain mandatory")
    heldout = gate["heldout"]
    expected_heldout = {
        "minimum_pair0_minus_default_pooled_full_interval_zero_foot_reduction": 0.10,
        "minimum_pair0_minus_default_mean_support_count_increase": 0.20,
        "minimum_pair0_to_default_mean_best_progress_ratio": 0.90,
        "minimum_pair0_to_default_uphill_best_progress_ratio": 0.85,
        "minimum_pair0_to_default_downhill_best_progress_ratio": 0.85,
        "maximum_pair0_minus_default_corrected_sustained_slip_per_force_qualified_supported_fraction": 0.02,
        "maximum_pair0_minus_default_corrected_slip_events_per_100_force_qualified_supported_substeps": 0.20,
        "frozen_unadapted_pair0_pooled_full_interval_zero_foot_fraction": 0.028055555555555556,
        "maximum_pair0_full_interval_worsening_vs_frozen_unadapted_pair0": 0.03,
    }
    for key, expected in expected_heldout.items():
        _equal(float(heldout[key]), expected, f"heldout gate {key}")
    continuity = gate["continuity"]
    for key, expected in {
        "minimum_pair0_mean_best_progress_m": 7.19339076746881,
        "minimum_pair0_uphill_mean_best_progress_m": 6.18579923623122,
        "minimum_pair0_downhill_mean_best_progress_m": 8.81135708033187,
        "maximum_pair0_pooled_full_interval_zero_foot_fraction": 0.058055555555555555,
    }.items():
        _equal(float(continuity[key]), expected, f"continuity gate {key}")
    _equal(int(gate["only_absolute_final_checkpoint_may_be_evaluated_for_promotion_decision"]), 2727936, "final checkpoint")
    early = config["checkpoint_early_stopping"]
    if bool(early["performance_futility_stopping_enabled"]) or bool(early["intermediate_promotion_enabled"]):
        raise ValueError("Performance stopping or intermediate promotion is forbidden")
    _equal(float(early["maximum_pair0_pooled_full_interval_zero_foot_fraction"]), 0.08, "early zero-foot")
    _equal(float(early["maximum_corrected_sustained_slip_per_force_qualified_supported_fraction_each_condition"]), 0.02, "early slip")
    _equal(float(early["maximum_corrected_slip_events_per_100_force_qualified_supported_substeps_each_condition"]), 0.20, "early events")
    _equal(float(early["nonfoot_contact_minimum_sustained_seconds"]), 0.20, "nonfoot duration")
    for key in (
        "maximum_fall_count_each_condition",
        "maximum_torso_ground_episode_count_each_condition",
        "maximum_sustained_nonfoot_contact_episode_count_each_condition",
    ):
        _equal(int(early[key]), 0, f"early gate {key}")
    if not bool(early["fail_on_any_nonfinite_or_contract_mismatch"]):
        raise ValueError("Non-finite/contract fail-closed rule was released")
    if not bool(early["zero_force_qualified_supported_denominator_is_failure"]):
        raise ValueError("Zero primary denominator must fail closed")
    if not bool(early["qualified_transient_warning_only"]):
        raise ValueError("Qualified transient must remain warning-only")
    energy = config["energy_boundary"]
    if not bool(energy["formula_unchanged"]):
        raise ValueError("Energy formula must remain frozen")
    if not bool(energy["nonfinite_energy_component_is_run_failure"]):
        raise ValueError("Non-finite energy must fail closed")
    if bool(energy["electrical_battery_energy_claim_permitted"]):
        raise ValueError("Electrical battery-energy claims remain forbidden")


def make_training_vec_env(
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    condition_id: str,
    monitor_path: Path,
) -> tuple[VecMonitor, list[dict[str, Any]]]:
    """Construct all workers with the same exact seeds later consumed at reset."""

    seed_map = config["training"]["worker_effective_seeds_by_scene"]
    factories: list[Callable[[], gym.Env]] = []
    for scene_name in SCENES:
        scene = scenes[scene_name]
        local_seed = int(seed_map[scene_name])

        def factory(
            local_scene: dict[str, Any] = scene,
            env_seed: int = local_seed,
        ) -> gym.Env:
            env = l2.make_standard_env(
                protocol,
                reward,
                local_scene,
                condition_id=condition_id,
                seed=env_seed,
                max_episode_steps=int(config["training"]["max_episode_steps"]),
                cruise_speed=float(config["training"]["cruise_speed_m_per_s"]),
            )
            audit = l2.compiled_contract_audit(
                env.unwrapped.model,
                local_scene,
                condition_id,
                config,
                construction_seed=env_seed,
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
        raise RuntimeError("Not all four workers passed contact-contract audit")
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return VecMonitor(base, filename=str(monitor_path)), audits


def _evaluate_set(
    model: PPO,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    *,
    condition_id: str,
    checkpoint_additional_timesteps: int,
    output_root: Path,
    label: str,
    seeds: list[int],
    horizon: int,
) -> dict[str, Any]:
    evaluation_root = output_root / condition_id.lower() / "evaluations" / label
    evaluation_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trace_seed = int(config["evaluation"]["representative_trace_seed"])
    for scene_name in SCENES:
        for seed in seeds:
            row, local_traces, local_events = l2.evaluate_episode(
                model,
                config,
                protocol,
                reward,
                scenes[scene_name],
                condition_id=condition_id,
                seed=int(seed),
                checkpoint_additional_timesteps=checkpoint_additional_timesteps,
                max_episode_steps=horizon,
                retain_substeps=int(seed) == trace_seed and label == "final_continuity",
            )
            rows.append(row)
            traces.extend(local_traces)
            events.extend(local_events)
    l2.write_rows(evaluation_root / "episode_metrics.csv", rows)
    if traces:
        l2.write_rows(evaluation_root / "representative_substep_trace.csv", traces)
    l2.write_event_rows(evaluation_root / "corrected_slip_events.csv", events)
    aggregate = l2.aggregate_episode_rows(rows)
    aggregate["per_scene"] = {
        scene_name: l2.aggregate_episode_rows(
            [row for row in rows if row["scene_name"] == scene_name]
        )
        for scene_name in SCENES
    }
    energy_keys = list(config["energy_boundary"]["raw_components_required"])
    aggregate["energy_components_finite"] = all(
        math.isfinite(float(row[key])) for row in rows for key in energy_keys
    )
    aggregate.update(
        {
            "condition_id": condition_id,
            "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
            "checkpoint_absolute_timesteps": int(model.num_timesteps),
            "evaluation_label": label,
            "evaluation_seeds": seeds,
            "max_episode_steps": horizon,
        }
    )
    l2.write_json(evaluation_root / "aggregate.json", aggregate)
    return aggregate


def _safe_nonnegative_integer(value: Any, *, failure_default: int = 1) -> int:
    if isinstance(value, bool):
        return failure_default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return failure_default
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return failure_default
    return int(numeric)


def _strict_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def checkpoint_stop_decision(
    config: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    condition_id: str,
    checkpoint_additional_timesteps: int,
) -> dict[str, Any]:
    early = config["checkpoint_early_stopping"]
    evaluable = _strict_true(aggregate.get("force_qualified_slip_evaluable"))
    failures = {
        "nonfinite": _safe_nonnegative_integer(
            aggregate.get("nonfinite_episode_count")
        )
        > 0,
        "nonfinite_energy": not _strict_true(
            aggregate.get("energy_components_finite")
        ),
        "force_qualified_slip_non_evaluable": not evaluable,
        "fall": int(aggregate["fall_count"]) > int(early["maximum_fall_count_each_condition"]),
        "torso_ground": int(aggregate["torso_ground_episode_count"]) > int(early["maximum_torso_ground_episode_count_each_condition"]),
        "sustained_nonfoot_contact": int(aggregate["sustained_nonfoot_contact_episode_count"]) > int(early["maximum_sustained_nonfoot_contact_episode_count_each_condition"]),
        "corrected_sustained_slip": bool(
            evaluable
            and float(aggregate["corrected_sustained_slip_per_force_qualified_supported_fraction"])
            > float(early["maximum_corrected_sustained_slip_per_force_qualified_supported_fraction_each_condition"])
        ),
        "corrected_slip_event_rate": bool(
            evaluable
            and float(aggregate["corrected_slip_events_per_100_force_qualified_supported_substeps"])
            > float(early["maximum_corrected_slip_events_per_100_force_qualified_supported_substeps_each_condition"])
        ),
        "pair0_full_interval_zero_foot": bool(
            condition_id == PAIR0_ID
            and float(aggregate["pooled_full_interval_zero_foot_fraction"])
            > float(early["maximum_pair0_pooled_full_interval_zero_foot_fraction"])
        ),
    }
    return {
        "condition_id": condition_id,
        "checkpoint_additional_timesteps": checkpoint_additional_timesteps,
        "catastrophe_checks": failures,
        "performance_futility_checked": False,
        "intermediate_promotion_permitted": False,
        "early_stop_triggered": bool(any(failures.values())),
    }


def _final_finiteness_failure(
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    evidence = {
        condition_id: {
            "nonfinite_episode_count": _safe_nonnegative_integer(
                aggregates[condition_id].get("nonfinite_episode_count")
            ),
            "energy_components_finite": _strict_true(
                aggregates[condition_id].get("energy_components_finite")
            ),
        }
        for condition_id in CONDITION_IDS
    }
    failed = [
        condition_id
        for condition_id, row in evidence.items()
        if row["nonfinite_episode_count"] > 0
        or not row["energy_components_finite"]
    ]
    if not failed:
        return None
    return {
        "evaluable": False,
        "passed": False,
        "reason": "Final evaluation contains non-finite episode or energy data",
        "failed_conditions": failed,
        "finiteness_evidence": evidence,
    }


def _paired_gate(config: dict[str, Any], aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate = config["prospective_final_gate"]["heldout"]
    if set(aggregates) != set(CONDITION_IDS):
        return {"evaluable": False, "passed": False, "reason": "Both held-out conditions are required"}
    finiteness_failure = _final_finiteness_failure(aggregates)
    if finiteness_failure is not None:
        return finiteness_failure
    control, pair0 = aggregates[DEFAULT_ID], aggregates[PAIR0_ID]
    if not all(
        _strict_true(row.get("force_qualified_slip_evaluable"))
        for row in (control, pair0)
    ):
        return {"evaluable": False, "passed": False, "reason": "Zero force-qualified denominator"}
    observed = {
        "pooled_full_interval_zero_foot_reduction": float(control["pooled_full_interval_zero_foot_fraction"]) - float(pair0["pooled_full_interval_zero_foot_fraction"]),
        "mean_support_count_increase": float(pair0["mean_support_count"]) - float(control["mean_support_count"]),
        "mean_best_progress_ratio": float(pair0["mean_best_progress_m"]) / max(float(control["mean_best_progress_m"]), 1e-12),
        "uphill_best_progress_ratio": float(pair0["per_scene"]["uphill_8deg"]["mean_best_progress_m"]) / max(float(control["per_scene"]["uphill_8deg"]["mean_best_progress_m"]), 1e-12),
        "downhill_best_progress_ratio": float(pair0["per_scene"]["downhill_8deg"]["mean_best_progress_m"]) / max(float(control["per_scene"]["downhill_8deg"]["mean_best_progress_m"]), 1e-12),
        "success_count_difference": int(pair0["success_count"]) - int(control["success_count"]),
        "pair0_fall_count": int(pair0["fall_count"]),
        "pair0_torso_ground_episode_count": int(pair0["torso_ground_episode_count"]),
        "pair0_sustained_nonfoot_contact_episode_count": int(pair0["sustained_nonfoot_contact_episode_count"]),
        "corrected_sustained_slip_fraction_delta": float(pair0["corrected_sustained_slip_per_force_qualified_supported_fraction"]) - float(control["corrected_sustained_slip_per_force_qualified_supported_fraction"]),
        "corrected_slip_events_per_100_delta": float(pair0["corrected_slip_events_per_100_force_qualified_supported_substeps"]) - float(control["corrected_slip_events_per_100_force_qualified_supported_substeps"]),
        "pair0_full_interval_worsening_vs_frozen_unadapted_pair0": float(pair0["pooled_full_interval_zero_foot_fraction"]) - float(gate["frozen_unadapted_pair0_pooled_full_interval_zero_foot_fraction"]),
    }
    checks = {
        "full_interval_reduction": observed["pooled_full_interval_zero_foot_reduction"] >= float(gate["minimum_pair0_minus_default_pooled_full_interval_zero_foot_reduction"]),
        "support_increase": observed["mean_support_count_increase"] >= float(gate["minimum_pair0_minus_default_mean_support_count_increase"]),
        "progress_retention": observed["mean_best_progress_ratio"] >= float(gate["minimum_pair0_to_default_mean_best_progress_ratio"]),
        "uphill_progress_retention": observed["uphill_best_progress_ratio"] >= float(gate["minimum_pair0_to_default_uphill_best_progress_ratio"]),
        "downhill_progress_retention": observed["downhill_best_progress_ratio"] >= float(gate["minimum_pair0_to_default_downhill_best_progress_ratio"]),
        "success_nondecrease": observed["success_count_difference"] >= int(gate["minimum_pair0_minus_default_success_count"]),
        "no_pair0_falls": observed["pair0_fall_count"] <= int(gate["maximum_pair0_fall_count"]),
        "no_pair0_torso_ground": observed["pair0_torso_ground_episode_count"] <= int(gate["maximum_pair0_torso_ground_episode_count"]),
        "no_pair0_sustained_nonfoot": observed["pair0_sustained_nonfoot_contact_episode_count"] <= int(gate["maximum_pair0_sustained_nonfoot_contact_episode_count"]),
        "corrected_slip_fraction_not_worse": observed["corrected_sustained_slip_fraction_delta"] <= float(gate["maximum_pair0_minus_default_corrected_sustained_slip_per_force_qualified_supported_fraction"]),
        "corrected_slip_event_rate_not_worse": observed["corrected_slip_events_per_100_delta"] <= float(gate["maximum_pair0_minus_default_corrected_slip_events_per_100_force_qualified_supported_substeps"]),
        "frozen_pair0_contact_retained": observed["pair0_full_interval_worsening_vs_frozen_unadapted_pair0"] <= float(gate["maximum_pair0_full_interval_worsening_vs_frozen_unadapted_pair0"]),
    }
    return {"evaluable": True, "observed": observed, "checks": checks, "passed": bool(all(checks.values()))}


def _continuity_gate(config: dict[str, Any], aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate = config["prospective_final_gate"]["continuity"]
    if set(aggregates) != set(CONDITION_IDS):
        return {"evaluable": False, "passed": False, "reason": "Both continuity conditions are required"}
    finiteness_failure = _final_finiteness_failure(aggregates)
    if finiteness_failure is not None:
        return finiteness_failure
    if not all(
        _strict_true(aggregates[condition_id].get("force_qualified_slip_evaluable"))
        for condition_id in CONDITION_IDS
    ):
        return {"evaluable": False, "passed": False, "reason": "Zero force-qualified denominator"}
    pair0 = aggregates[PAIR0_ID]
    observed = {
        "pair0_mean_best_progress_m": float(pair0["mean_best_progress_m"]),
        "pair0_uphill_mean_best_progress_m": float(pair0["per_scene"]["uphill_8deg"]["mean_best_progress_m"]),
        "pair0_downhill_mean_best_progress_m": float(pair0["per_scene"]["downhill_8deg"]["mean_best_progress_m"]),
        "pair0_fall_count": int(pair0["fall_count"]),
        "pair0_torso_ground_episode_count": int(pair0["torso_ground_episode_count"]),
        "pair0_sustained_nonfoot_contact_episode_count": int(pair0["sustained_nonfoot_contact_episode_count"]),
        "pair0_pooled_full_interval_zero_foot_fraction": float(pair0["pooled_full_interval_zero_foot_fraction"]),
        "pair0_corrected_sustained_slip_per_force_qualified_supported_fraction": float(pair0["corrected_sustained_slip_per_force_qualified_supported_fraction"]),
        "pair0_corrected_slip_events_per_100_force_qualified_supported_substeps": float(pair0["corrected_slip_events_per_100_force_qualified_supported_substeps"]),
    }
    checks = {
        "overall_absolute_floor": observed["pair0_mean_best_progress_m"] >= float(gate["minimum_pair0_mean_best_progress_m"]),
        "uphill_absolute_floor": observed["pair0_uphill_mean_best_progress_m"] >= float(gate["minimum_pair0_uphill_mean_best_progress_m"]),
        "downhill_absolute_floor": observed["pair0_downhill_mean_best_progress_m"] >= float(gate["minimum_pair0_downhill_mean_best_progress_m"]),
        "no_pair0_falls": observed["pair0_fall_count"] <= int(gate["maximum_pair0_fall_count"]),
        "no_pair0_torso_ground": observed["pair0_torso_ground_episode_count"] <= int(gate["maximum_pair0_torso_ground_episode_count"]),
        "no_pair0_sustained_nonfoot": observed["pair0_sustained_nonfoot_contact_episode_count"] <= int(gate["maximum_pair0_sustained_nonfoot_contact_episode_count"]),
        "contact_retained": observed["pair0_pooled_full_interval_zero_foot_fraction"] <= float(gate["maximum_pair0_pooled_full_interval_zero_foot_fraction"]),
        "corrected_slip_safe": observed["pair0_corrected_sustained_slip_per_force_qualified_supported_fraction"] <= float(gate["maximum_pair0_corrected_sustained_slip_per_force_qualified_supported_fraction"]),
        "corrected_slip_event_rate_safe": observed["pair0_corrected_slip_events_per_100_force_qualified_supported_substeps"] <= float(gate["maximum_pair0_corrected_slip_events_per_100_force_qualified_supported_substeps"]),
    }
    return {"evaluable": True, "observed": observed, "checks": checks, "passed": bool(all(checks.values()))}


def final_gate(
    config: dict[str, Any],
    heldout_aggregates: dict[str, dict[str, Any]],
    continuity_aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    heldout = _paired_gate(config, heldout_aggregates)
    continuity = _continuity_gate(config, continuity_aggregates)
    evaluable = bool(heldout.get("evaluable", False) and continuity.get("evaluable", False))
    passed = bool(evaluable and heldout.get("passed", False) and continuity.get("passed", False))
    return {
        "evaluable": evaluable,
        "heldout": heldout,
        "continuity": continuity,
        "passed": passed,
        "absolute_final_checkpoint": 2727936,
        "intermediate_checkpoint_selected": False,
        "fixed_map_entry_authorised": False,
        "video_rendering_authorised": False,
        "candidate_promoted": False,
        "hard_stop_further_contact_budget_extension": True,
        "decision": (
            "l2b_final_gate_passed_diagnostic_only"
            if passed
            else "l2b_final_gate_failed_stop_contact_budget_extension"
        ),
    }


def _train_condition_worker(
    config_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    output_root: Path,
    condition_id: str,
    *,
    smoke: bool,
) -> dict[str, Any]:
    snapshot_root = output_root / config["runtime_dependency_contract"][
        "snapshot_root_name"
    ]
    runtime_live_before = validate_runtime_dependency_map(config)
    runtime_snapshot_before = validate_runtime_snapshot(config, snapshot_root)
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    if int(torch.get_num_threads()) != int(config["ppo"]["torch_num_threads"]):
        raise RuntimeError("Condition worker torch thread contract changed")
    scenes_all = json.loads((output_root / "prepared_scenes.json").read_text(encoding="utf-8"))
    scenes = scenes_all[condition_id]
    condition_root = output_root / condition_id.lower()
    if condition_root.exists():
        raise FileExistsError(f"Refusing to overwrite condition root: {condition_root}")
    vec_env, audits = make_training_vec_env(
        config,
        protocol,
        reward,
        scenes,
        condition_id,
        condition_root / "logs" / "train_monitor.csv",
    )
    records: list[dict[str, Any]] = []
    early_stopped = False
    source = config["source"]["conditions"][condition_id]
    try:
        model = l2._configure_continuation_model(
            ROOT / source["checkpoint"],
            vec_env,
            config["ppo"],
            training_seed=int(config["training"]["master_seed"]),
            smoke=smoke,
        )
        audits = l2.record_effective_reset_seeds(
            vec_env,
            audits,
            training_seed=int(config["training"]["master_seed"]),
        )
        expected_effective = list(config["training"]["worker_effective_seeds_by_scene"].values())
        observed_construction = [int(row["construction_seed"]) for row in audits]
        observed_effective = [int(row["effective_first_reset_seed"]) for row in audits]
        if observed_construction != expected_effective or observed_effective != expected_effective:
            raise RuntimeError("Construction/effective worker seed provenance mismatch")
        l2.write_json(condition_root / "worker_contract_audits.json", audits)
        if int(model.num_timesteps) != int(config["source"]["checkpoint_timesteps"]):
            raise RuntimeError("Source checkpoint timestep metadata changed")
        budget = int(config["smoke"]["additional_timesteps_per_condition"] if smoke else config["training"]["additional_timesteps_per_condition"])
        interval = int(config["smoke"]["checkpoint_interval_timesteps"] if smoke else config["training"]["checkpoint_interval_timesteps"])
        previous = 0
        for additional in range(interval, budget + 1, interval):
            model.learn(total_timesteps=additional - previous, reset_num_timesteps=False, progress_bar=False)
            absolute = int(config["source"]["checkpoint_timesteps"]) + additional
            if int(model.num_timesteps) != absolute:
                raise RuntimeError("Absolute checkpoint schedule changed")
            checkpoint = l2._save_checkpoint(
                model, condition_root / "models" / f"checkpoint_{absolute}"
            )
            seeds = (
                [int(value) for value in config["smoke"]["evaluation_seeds"]]
                if smoke
                else [int(value) for value in config["evaluation"]["intermediate_safety_audit"]["seeds"]]
            )
            horizon = int(config["smoke"]["evaluation_max_episode_steps"] if smoke else config["evaluation"]["max_episode_steps"])
            aggregate = _evaluate_set(
                model,
                config,
                protocol,
                reward,
                scenes,
                condition_id=condition_id,
                checkpoint_additional_timesteps=additional,
                output_root=output_root,
                label=f"additional_{additional}",
                seeds=seeds,
                horizon=horizon,
            )
            stop = checkpoint_stop_decision(
                config,
                aggregate,
                condition_id=condition_id,
                checkpoint_additional_timesteps=additional,
            )
            l2.write_json(
                condition_root / "evaluations" / f"additional_{additional}" / "checkpoint_stop_decision.json",
                stop,
            )
            record = {
                "additional_timesteps": additional,
                "absolute_timesteps": absolute,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": l2.sha256(checkpoint),
                "aggregate": aggregate,
                "stop_decision": stop,
            }
            records.append(record)
            previous = additional
            if bool(stop["early_stop_triggered"]):
                early_stopped = True
                break
        final_heldout = None
        final_continuity = None
        if not smoke and not early_stopped and records[-1]["absolute_timesteps"] == 2727936:
            final_heldout = _evaluate_set(
                model,
                config,
                protocol,
                reward,
                scenes,
                condition_id=condition_id,
                checkpoint_additional_timesteps=65536,
                output_root=output_root,
                label="final_heldout",
                seeds=[int(value) for value in config["evaluation"]["final_heldout"]["seeds"]],
                horizon=int(config["evaluation"]["max_episode_steps"]),
            )
            final_continuity = _evaluate_set(
                model,
                config,
                protocol,
                reward,
                scenes,
                condition_id=condition_id,
                checkpoint_additional_timesteps=65536,
                output_root=output_root,
                label="final_continuity",
                seeds=[int(value) for value in config["evaluation"]["final_continuity"]["seeds"]],
                horizon=int(config["evaluation"]["max_episode_steps"]),
            )
        runtime_live_after = validate_runtime_dependency_map(config)
        runtime_snapshot_after = validate_runtime_snapshot(config, snapshot_root)
        result = {
            "condition_id": condition_id,
            "process_id": int(__import__("os").getpid()),
            "torch_num_threads": int(torch.get_num_threads()),
            "runtime_dependency_verification": {
                "live_before_env_model": runtime_live_before,
                "snapshot_before_env_model": runtime_snapshot_before,
                "live_after_training": runtime_live_after,
                "snapshot_after_training": runtime_snapshot_after,
            },
            "source_checkpoint": source,
            "master_seed": int(config["training"]["master_seed"]),
            "worker_contract_audits": audits,
            "checkpoint_records": records,
            "early_stopped": early_stopped,
            "completed_additional_timesteps": records[-1]["additional_timesteps"],
            "requested_additional_timesteps": budget,
            "final_heldout": final_heldout,
            "final_continuity": final_continuity,
        }
        l2.write_json(condition_root / "training_record.json", result)
        return result
    finally:
        vec_env.close()


def _run_condition_subprocess(
    config_path: Path,
    output_root: Path,
    condition_id: str,
    *,
    smoke: bool,
) -> dict[str, Any]:
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--condition-worker",
        condition_id,
        "--worker-mode",
        "smoke" if smoke else "run",
    ]
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    log_root = output_root / "process_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / f"{condition_id.lower()}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{condition_id.lower()}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{condition_id} clean worker failed with exit {completed.returncode}")
    result_path = output_root / condition_id.lower() / "training_record.json"
    if not result_path.is_file():
        raise RuntimeError(f"{condition_id} worker did not write a training record")
    return json.loads(result_path.read_text(encoding="utf-8"))


def gate_from_condition_records(
    config: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    smoke: bool,
) -> tuple[str, dict[str, Any]]:
    """Convert worker records to a fail-closed run status without dereferencing None."""

    early_stopped_conditions = [
        key for key in CONDITION_IDS if bool(records[key]["early_stopped"])
    ]
    if smoke:
        if early_stopped_conditions:
            raise RuntimeError(
                "Engineering smoke failed closed after an early stop: "
                + ", ".join(early_stopped_conditions)
            )
        return "engineering_smoke_passed", {
            "evaluable": False,
            "passed": False,
            "reason": "Engineering smoke does not apply the scientific gate",
            "fixed_map_entry_authorised": False,
            "candidate_promoted": False,
        }
    missing_final = [
        key
        for key in CONDITION_IDS
        if records[key]["final_heldout"] is None
        or records[key]["final_continuity"] is None
    ]
    if early_stopped_conditions or missing_final:
        return "l2b_once_only_extension_complete", {
            "evaluable": False,
            "passed": False,
            "reason": "One or more conditions stopped before the frozen final checkpoint",
            "early_stopped_conditions": early_stopped_conditions,
            "missing_final_evaluations": missing_final,
            "absolute_final_checkpoint": 2727936,
            "intermediate_checkpoint_selected": False,
            "fixed_map_entry_authorised": False,
            "video_rendering_authorised": False,
            "candidate_promoted": False,
            "hard_stop_further_contact_budget_extension": True,
            "decision": "l2b_early_stop_non_evaluable_stop_contact_budget_extension",
        }
    heldout = {key: records[key]["final_heldout"] for key in CONDITION_IDS}
    continuity = {key: records[key]["final_continuity"] for key in CONDITION_IDS}
    return "l2b_once_only_extension_complete", final_gate(
        config, heldout, continuity
    )


def run_extension(
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
        (output_root / "frozen_config.json").write_bytes(config_path.read_bytes())
        frozen_config_sha256 = l2.sha256(output_root / "frozen_config.json")
        if frozen_config_sha256 != l2.sha256(config_path):
            raise RuntimeError("Frozen configuration copy does not match the validated source")
        live_runner_path = Path(__file__).resolve()
        live_runner_sha256_before_workers = l2.sha256(live_runner_path)
        frozen_runner_path = output_root / "frozen_runner.py"
        shutil.copy2(live_runner_path, frozen_runner_path)
        if l2.sha256(frozen_runner_path) != live_runner_sha256_before_workers:
            raise RuntimeError("Frozen L2b runner differs from the live runner")
        l2_runtime_path = ROOT / config["source"]["l2_runtime_dependency"]
        l2_runtime_sha256_before_workers = l2.sha256(l2_runtime_path)
        if (
            l2_runtime_sha256_before_workers
            != config["source"]["l2_runtime_dependency_sha256"]
        ):
            raise RuntimeError("Imported L2 runtime dependency changed before execution")
        frozen_l2_runtime_path = output_root / "frozen_l2_runtime.py"
        shutil.copy2(l2_runtime_path, frozen_l2_runtime_path)
        if l2.sha256(frozen_l2_runtime_path) != l2_runtime_sha256_before_workers:
            raise RuntimeError("Frozen L2 runtime copy differs from imported dependency")
        snapshot_root, runtime_live_before_workers = snapshot_runtime_dependencies(
            config, output_root
        )
        runtime_snapshot_before_workers = validate_runtime_snapshot(config, snapshot_root)
        scenes, _ = l2.prepare_condition_scenes(config, protocol, output_root)
        l2.write_json(output_root / "prepared_scenes.json", scenes)
        frozen_config_path = output_root / "frozen_config.json"
        records: dict[str, dict[str, Any]] = {}
        per_condition_parent_checks: dict[str, dict[str, dict[str, str]]] = {}
        for condition_id in CONDITION_IDS:
            live_before_worker = validate_runtime_dependency_map(config)
            snapshot_before_worker = validate_runtime_snapshot(config, snapshot_root)
            records[condition_id] = _run_condition_subprocess(
                frozen_config_path,
                output_root,
                condition_id,
                smoke=smoke,
            )
            live_after_worker = validate_runtime_dependency_map(config)
            snapshot_after_worker = validate_runtime_snapshot(config, snapshot_root)
            per_condition_parent_checks[condition_id] = {
                "live_before_worker": live_before_worker,
                "snapshot_before_worker": snapshot_before_worker,
                "live_after_worker": live_after_worker,
                "snapshot_after_worker": snapshot_after_worker,
            }
        process_ids = [int(records[key]["process_id"]) for key in CONDITION_IDS]
        if len(set(process_ids)) != 2:
            raise RuntimeError("Conditions did not execute in independent clean processes")
        if l2.sha256(output_root / "frozen_config.json") != frozen_config_sha256:
            raise RuntimeError("Frozen configuration changed while workers were running")
        live_runner_sha256_after_workers = l2.sha256(live_runner_path)
        if live_runner_sha256_after_workers != live_runner_sha256_before_workers:
            raise RuntimeError("Live L2b runner changed while workers ran")
        if l2.sha256(frozen_runner_path) != live_runner_sha256_before_workers:
            raise RuntimeError("Frozen L2b runner changed while workers ran")
        l2_runtime_sha256_after_workers = l2.sha256(l2_runtime_path)
        if l2_runtime_sha256_after_workers != l2_runtime_sha256_before_workers:
            raise RuntimeError("Imported L2 runtime dependency changed while workers ran")
        if l2.sha256(frozen_l2_runtime_path) != l2_runtime_sha256_before_workers:
            raise RuntimeError("Frozen L2 runtime copy changed while workers ran")
        runtime_live_after_workers = validate_runtime_dependency_map(config)
        runtime_snapshot_after_workers = validate_runtime_snapshot(config, snapshot_root)
        status, gate = gate_from_condition_records(config, records, smoke=smoke)
        l2.write_json(output_root / "prospective_final_gate.json", gate)
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "device": config["ppo"]["device"],
        }
        l2.write_json(output_root / "environment.json", environment)
        manifest_path = output_root / "manifest.json"
        manifest = {
            "schema_version": "proxygap-pair0-adaptation-l2b-extension-manifest-v3",
            "status": status,
            "stage": "L1_engineering_smoke" if smoke else "L2b_once_only_single_training_seed_extension",
            "attempt": attempt,
            "configuration": {"path": str(config_path), "sha256": l2.sha256(config_path)},
            "frozen_configuration": {
                "path": str((output_root / "frozen_config.json").resolve()),
                "sha256_before_and_after_workers": frozen_config_sha256,
            },
            "runtime_code": {
                "runner_path": str(live_runner_path),
                "runner_sha256_before_workers": live_runner_sha256_before_workers,
                "runner_sha256_after_workers": live_runner_sha256_after_workers,
                "frozen_runner_sha256": l2.sha256(frozen_runner_path),
                "imported_l2_runtime_path": str(l2_runtime_path.resolve()),
                "imported_l2_runtime_sha256_before_workers": l2_runtime_sha256_before_workers,
                "imported_l2_runtime_sha256_after_workers": l2_runtime_sha256_after_workers,
                "frozen_l2_runtime_path": str(frozen_l2_runtime_path.resolve()),
                "frozen_l2_runtime_sha256": l2.sha256(frozen_l2_runtime_path),
            },
            "runtime_dependency_closure": {
                "snapshot_root": str(snapshot_root.resolve()),
                "expected_relative_path_sha256": config[
                    "runtime_dependency_contract"
                ]["exact_relative_path_sha256"],
                "parent_live_before_workers": runtime_live_before_workers,
                "parent_snapshot_before_workers": runtime_snapshot_before_workers,
                "per_condition_parent_checks": per_condition_parent_checks,
                "parent_live_after_workers": runtime_live_after_workers,
                "parent_snapshot_after_workers": runtime_snapshot_after_workers,
            },
            "condition_process_ids": dict(zip(CONDITION_IDS, process_ids)),
            "condition_training": records,
            "prospective_final_gate": gate,
            "fixed_map_evaluated": False,
            "video_rendered": False,
            "candidate_promoted": False,
            "energy_formula_changed": False,
            "energy_status": "measurement_only_not_reward_or_gate",
            "friction_changed": False,
            "old_l2_artifacts_overwritten": False,
            "hard_stop_further_contact_budget_extension": not smoke,
            "elapsed_seconds": float(time.perf_counter() - started),
            "environment": environment,
            "git": l2._git_record(),
            "claim_boundary": config["claim_boundary"],
            "artifact_inventory_excludes_manifest_itself": l2.artifact_inventory(
                output_root
            ),
        }
        l2.write_json(manifest_path, manifest)
        return {
            "status": status,
            "output_root": str(output_root),
            "manifest_sha256": l2.sha256(manifest_path),
            "gate": gate,
        }
    except Exception as error:
        l2.write_json(
            output_root / "FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-pair0-adaptation-l2b-extension-failure-v3",
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_run_retained": True,
                "old_l2_artifacts_overwritten": False,
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
    protocol, reward = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated_l2b_v3_once_only", "config_sha256": l2.sha256(config_path)}, indent=2))
        return
    if not args.condition_worker:
        validate_parent_config_path(config_path)
    if args.condition_worker:
        if args.output_root is None or args.worker_mode is None:
            raise ValueError("Condition workers require --output-root and --worker-mode")
        worker_smoke = args.worker_mode == "smoke"
        worker_output_root = args.output_root.resolve()
        if not worker_smoke:
            expected_worker_root = (
                ROOT
                / config["execution"]["development_output_root"]
                / config["execution"]["attempt_subdirectory_template"].format(attempt=0)
            ).resolve()
            if worker_output_root != expected_worker_root:
                raise ValueError("Formal V3 condition workers require the canonical attempt_0 root")
        result = _train_condition_worker(
            config_path,
            config,
            protocol,
            reward,
            worker_output_root,
            args.condition_worker,
            smoke=worker_smoke,
        )
        print(json.dumps({"status": "condition_complete", "condition": result["condition_id"]}, indent=2))
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
    validate_attempt_semantics(
        config,
        attempt=args.attempt,
        base=base,
        smoke=bool(args.smoke),
        custom_output_root_used=args.output_root is not None,
    )
    output_root = base / config["execution"]["attempt_subdirectory_template"].format(attempt=args.attempt)
    result = run_extension(
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
