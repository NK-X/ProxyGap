"""V5 once-only repair for the PAIR0 compiled-contact audit gate.

V5 changes one engineering predicate only.  The frozen ``audit_compiled_pair``
function is fail-by-exception and returns a detailed contract record without a
``passed`` key.  V2 incorrectly interpreted that missing key as failure.  This
wrapper validates every returned field against the exact frozen compiled
contract and delegates the unchanged formal training/evaluation to V2.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_fixed_standard_distal_margin0_paired as pair_tools  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v2 as v2  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair as v3  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v4_short_root as v4  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair_20260819.json"
V2_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json"
V3_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json"
V4_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v4_short_root_20260819.json"
RUNTIME_SELF = "scripts/run_fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair.py"
CONFIG_SELF = "configs/fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair_20260819.json"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF,
    "configs/fixed_standard_pair0_turn_balance_continuation_v4_short_root_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v4_short_root.py",
    "configs/fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair.py",
    "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v2.py",
    "scripts/evaluate_fixed_standard_distal_margin0_paired.py",
)
EXPECTED_AUDIT_FIELDS = (
    "only_four_permitted_explicit_pairs_added",
    "explicit_pair_count",
    "source_floor_margin_m",
    "candidate_floor_geom_margin_m",
    "source_distal_margins_m",
    "candidate_distal_geom_margins_m",
    "candidate_non_distal_margins_m",
    "default_geom_margin_m",
    "root_joint_margin_m",
    "compiled_explicit_pairs",
    "friction",
    "condim",
    "solref",
    "solimp",
    "physics_timestep_seconds",
)
EXPECTED_COMPILED_PAIR_FIELDS = (
    "geom1",
    "geom2",
    "margin",
    "gap",
    "condim",
    "friction",
    "solref",
    "solreffriction",
    "solimp",
    "adhesion",
)
EXPECTED_V2_FAILURE_SHA = "21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23"
EXPECTED_V4_FAILURE_SHA = "9695bd3b5d628907053a2f785ec874efa18b2fc47a317c452a88566c0d624812"
MAX_PATH_CHARS = 239


def equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} changed: {observed!r} != {expected!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--engineering-smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args()


def validate_v5_runtime(config: dict[str, Any]) -> dict[str, str]:
    expected = config["V5_runtime_contract"]["exact_relative_path_sha256"]
    equal(tuple(expected), EXPECTED_RUNTIME_PATHS, "V5 runtime membership/order")
    observed: dict[str, str] = {}
    for relative in EXPECTED_RUNTIME_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = v3.sha256(path)
    equal(observed, expected, "V5 runtime hashes")
    return observed


def snapshot_v5_runtime(config: dict[str, Any], attempt_root: Path) -> dict[str, str]:
    live = validate_v5_runtime(config)
    snapshot_root = attempt_root / "v5_runtime_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    snapshot_paths = (CONFIG_SELF, *EXPECTED_RUNTIME_PATHS)
    observed: dict[str, str] = {}
    for relative in snapshot_paths:
        source = ROOT / relative
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = v3.sha256(source)
        equal(v3.sha256(target), digest, f"V5 snapshot {relative}")
        observed[relative] = digest
    equal(
        {relative: observed[relative] for relative in EXPECTED_RUNTIME_PATHS},
        live,
        "V5 snapshot/live runtime",
    )
    return observed


def validate_v5_runtime_snapshot(config: dict[str, Any], attempt_root: Path) -> dict[str, str]:
    snapshot_root = attempt_root / "v5_runtime_snapshot"
    expected_paths = (CONFIG_SELF, *EXPECTED_RUNTIME_PATHS)
    actual_paths = tuple(
        sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file()
        )
    )
    equal(actual_paths, tuple(sorted(expected_paths)), "V5 snapshot membership")
    observed = {
        relative: v3.sha256(snapshot_root / relative) for relative in expected_paths
    }
    equal(observed[CONFIG_SELF], v3.sha256(DEFAULT_CONFIG), "V5 snapshot config")
    expected_runtime = config["V5_runtime_contract"]["exact_relative_path_sha256"]
    equal(
        {relative: observed[relative] for relative in EXPECTED_RUNTIME_PATHS},
        expected_runtime,
        "V5 snapshot runtime hashes",
    )
    return observed


def _validate_failure_root(
    root: Path,
    record_path: Path,
    *,
    expected_sha: str,
    expected_exception: str,
) -> dict[str, Any]:
    if not root.is_dir() or not record_path.is_file():
        raise FileNotFoundError(record_path)
    equal(record_path.resolve(), (root / "FAILURE_RECORD.json").resolve(), "failure path")
    equal(v3.sha256(record_path), expected_sha, "failure record SHA")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    equal(record["failed_stage"], "prepare_standard_slope_scenes", "failed stage")
    equal(record["exception_type"], expected_exception, "failure exception")
    equal(bool(record["scientifically_evaluable"]), False, "failure evaluability")
    equal(bool(record["all_decisions_withheld"]), True, "failure decision boundary")
    equal(bool(record["retry_permitted"]), False, "failure retry boundary")
    equal((root / "manifest.json").exists(), False, "failure success manifest absence")
    equal(
        (root / "FINAL_OPTIMISATION_HARD_STOP.json").exists(),
        False,
        "failure hard-stop marker absence",
    )
    zip_paths = sorted(root.rglob("*.zip"))
    records = sorted(root.rglob("training_record.json"))
    equal(len(zip_paths), 0, "failure checkpoint count")
    equal(len(records), 0, "failure training record count")
    equal(record["runtime_live_before"], record["runtime_snapshot_before"], "failure runtime maps")
    equal(len(record["runtime_live_before"]), len(v2.EXPECTED_RUNTIME_PATHS), "failure runtime count")
    return {
        "root": str(root.resolve()),
        "failure_record": str(record_path.resolve()),
        "failure_record_sha256": expected_sha,
        "exception_type": expected_exception,
        "training_started": False,
        "checkpoint_count": 0,
        "training_record_count": 0,
        "scientifically_evaluable": False,
        "same_root_retry_permitted": False,
    }


def validate_frozen_failures(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["immutable_pretraining_failures"]
    v2_row = frozen["V2_path_length_failure"]
    v4_row = frozen["V4_false_audit_failure"]
    equal(v2_row["failure_record_sha256"], EXPECTED_V2_FAILURE_SHA, "V2 frozen failure SHA")
    equal(v4_row["failure_record_sha256"], EXPECTED_V4_FAILURE_SHA, "V4 frozen failure SHA")
    v2_evidence = _validate_failure_root(
        ROOT / v2_row["root"],
        ROOT / v2_row["failure_record"],
        expected_sha=EXPECTED_V2_FAILURE_SHA,
        expected_exception="FileNotFoundError",
    )
    v4_evidence = _validate_failure_root(
        ROOT / v4_row["root"],
        ROOT / v4_row["failure_record"],
        expected_sha=EXPECTED_V4_FAILURE_SHA,
        expected_exception="RuntimeError",
    )
    v4_record = json.loads(
        (ROOT / v4_row["failure_record"]).read_text(encoding="utf-8")
    )
    equal(
        v4_record["exception_message"],
        "one or more standard PAIR0 scene/contact audits failed",
        "V4 false-failure message",
    )
    return {"V2": v2_evidence, "V4": v4_evidence}


def validate_compiled_audit_exact(
    audit: dict[str, Any],
    pair_contract: dict[str, Any],
    contact_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete success record returned by ``audit_compiled_pair``.

    The function deliberately rejects a synthetic ``passed`` key.  Successful
    validation is the conjunction of the exact returned contract fields.
    """

    equal(tuple(audit), EXPECTED_AUDIT_FIELDS, "compiled audit field set/order")
    equal(bool(audit["only_four_permitted_explicit_pairs_added"]), True, "pair-only XML delta")
    equal(int(audit["explicit_pair_count"]), 4, "compiled pair count")
    margin = float(contact_contract["all_geom_margins_m"])
    equal(float(audit["source_floor_margin_m"]), margin, "source floor margin")
    equal(float(audit["candidate_floor_geom_margin_m"]), margin, "candidate floor margin")
    expected_distal = {name: margin for name in pair_tools.FOOT_NAMES}
    expected_non_distal = {name: margin for name in pair_tools.NON_DISTAL_ROBOT_GEOMS}
    equal(audit["source_distal_margins_m"], expected_distal, "source distal margins")
    equal(audit["candidate_distal_geom_margins_m"], expected_distal, "candidate distal margins")
    equal(audit["candidate_non_distal_margins_m"], expected_non_distal, "non-distal margins")
    equal(float(audit["default_geom_margin_m"]), margin, "default geom margin")
    equal(float(audit["root_joint_margin_m"]), margin, "root joint margin")
    equal(audit["friction"], contact_contract["geom_friction"], "floor friction")
    equal(int(audit["condim"]), int(contact_contract["condim"]), "floor condim")
    equal(audit["solref"], contact_contract["solref"], "floor solref")
    equal(audit["solimp"], contact_contract["solimp"], "floor solimp")
    equal(float(audit["physics_timestep_seconds"]), 0.01, "physics timestep")

    compiled_pairs = audit["compiled_explicit_pairs"]
    equal(len(compiled_pairs), 4, "compiled explicit-pair rows")
    observed_targets: set[frozenset[str]] = set()
    for index, row in enumerate(compiled_pairs):
        equal(tuple(row), EXPECTED_COMPILED_PAIR_FIELDS, f"compiled pair {index} fields")
        observed_targets.add(frozenset((str(row["geom1"]), str(row["geom2"]))))
        equal(float(row["margin"]), float(pair_contract["margin_m"]), f"pair {index} margin")
        equal(float(row["gap"]), float(pair_contract["gap_m"]), f"pair {index} gap")
        equal(int(row["condim"]), int(pair_contract["condim"]), f"pair {index} condim")
        equal(row["friction"], pair_contract["friction"], f"pair {index} friction")
        equal(row["solref"], pair_contract["solref"], f"pair {index} solref")
        equal(row["solreffriction"], pair_contract["solreffriction"], f"pair {index} solreffriction")
        equal(row["solimp"], pair_contract["solimp"], f"pair {index} solimp")
        equal(float(row["adhesion"]), float(pair_contract["adhesion"]), f"pair {index} adhesion")
    expected_targets = {
        frozenset(("floor", name)) for name in pair_tools.FOOT_NAMES
    }
    equal(observed_targets, expected_targets, "compiled pair target set")
    return {
        "compiled_contract_exact": True,
        "success_derived_from_all_exact_fields": True,
        "synthetic_passed_field_used": False,
        "explicit_pair_count": 4,
        "target_set": sorted("floor+" + name for name in pair_tools.FOOT_NAMES),
    }


def prepare_standard_pair0_scenes_v5(
    v1: dict[str, Any], protocol: dict[str, Any], output_root: Path
) -> dict[str, dict[str, Any]]:
    """Prepare four standard scenes and validate the exact compiled contract."""

    controls, generation = v2.slope.prepare_standard_scenes(
        protocol, output_root / "generated"
    )
    pair_contract = v2.l2._pair_contract(v1)
    scenes: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for scene_name in v2.l2.EXPECTED_SCENES:
        pair, audit = v2.slope.prepare_pair(
            controls[scene_name],
            output_root / "condition_assets",
            f"turn_balance_{scene_name}",
            pair_contract,
        )
        validations[scene_name] = validate_compiled_audit_exact(
            audit, pair_contract, v1["contact_contract"]
        )
        scene = dict(pair[v2.slope.CANDIDATE_ID])
        scene["condition_id"] = v2.l2.PAIR0_ID
        scene["scene_name"] = scene_name
        scenes[scene_name] = scene
        audits[scene_name] = audit
    expected_scenes = tuple(v2.l2.EXPECTED_SCENES)
    equal(tuple(scenes), expected_scenes, "standard scene order")
    equal(tuple(audits), expected_scenes, "standard audit order")
    equal(tuple(validations), expected_scenes, "standard validation order")
    if not all(row["compiled_contract_exact"] is True for row in validations.values()):
        raise RuntimeError("one or more exact compiled PAIR0 contracts failed")
    v2.write_json(output_root / "scene_generation.json", generation)
    v2.write_json(output_root / "explicit_pair_audits.json", audits)
    v2.write_json(output_root / "compiled_contract_validations.json", validations)
    v2.write_json(output_root / "prepared_scenes.json", scenes)
    return scenes


def planned_relative_paths() -> tuple[str, ...]:
    paths = set(v4.planned_relative_paths())
    paths.add("standard_slope_assets/compiled_contract_validations.json")
    for relative in (CONFIG_SELF, *EXPECTED_RUNTIME_PATHS):
        paths.add(f"v5_runtime_snapshot/{relative}")
    return tuple(sorted(paths))


def validate_path_budget(output_root: Path) -> dict[str, Any]:
    rows = [
        {
            "relative_path": relative,
            "absolute_path": str(output_root / relative),
            "characters": len(str(output_root / relative)),
        }
        for relative in planned_relative_paths()
    ]
    maximum = max(rows, key=lambda row: int(row["characters"]))
    if any(int(row["characters"]) > MAX_PATH_CHARS for row in rows):
        raise ValueError(f"planned path exceeds {MAX_PATH_CHARS}: {maximum}")
    return {
        "limit_characters_inclusive": MAX_PATH_CHARS,
        "planned_path_count": len(rows),
        "maximum": maximum,
        "all_within_limit": True,
    }


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    equal(
        config["status"],
        "frozen_preformal_compiled_audit_repair_stage_a_authorised",
        "V5 status",
    )
    frozen = config["frozen_scientific_protocol"]
    path_hash_pairs = (
        ("V2_configuration", "V2_configuration_sha256"),
        ("V2_runner", "V2_runner_sha256"),
        ("V3_configuration", "V3_configuration_sha256"),
        ("V3_runner", "V3_runner_sha256"),
        ("V4_configuration", "V4_configuration_sha256"),
        ("V4_runner", "V4_runner_sha256"),
    )
    for path_key, hash_key in path_hash_pairs:
        equal(v3.sha256(ROOT / frozen[path_key]), frozen[hash_key], hash_key)
    for key in (
        "scientific_protocol_changed",
        "seeds_changed",
        "budgets_changed",
        "commands_changed",
        "reward_changed",
        "friction_changed",
        "energy_changed",
        "gates_changed",
        "intermediate_selection_changed",
    ):
        equal(bool(frozen[key]), False, key)

    repair = config["compiled_audit_repair"]
    equal(
        repair["V4_false_predicate"],
        "bool(audit.get('passed'))",
        "V4 false predicate",
    )
    equal(bool(repair["audit_function_returns_passed_field"]), False, "audit passed-field absence")
    equal(
        repair["V5_success_predicate"],
        "all_exact_compiled_contract_fields_validate_or_raise",
        "V5 success predicate",
    )
    equal(bool(repair["add_synthetic_passed_field"]), False, "synthetic passed field")

    execution = config["execution"]
    equal(execution["canonical_engineering_smoke_root"], "artifacts/smoke/tb_v5_20260819/a0", "V5 smoke root")
    equal(execution["canonical_formal_root"], "artifacts/dev/tb_v5_20260819/a0", "V5 formal root")
    equal(bool(execution["stage_A_user_authorised"]), True, "Stage A authorisation")
    equal(bool(execution["formal_root_must_not_exist_before_launch"]), True, "fresh formal root")
    equal(bool(execution["engineering_smoke_root_must_not_exist_before_launch"]), True, "fresh smoke root")
    equal(bool(execution["smoke_trains_policy"]), False, "smoke training boundary")
    equal(bool(execution["reuse_partial_weights"]), False, "partial weights")
    equal(bool(execution["retry_in_same_root"]), False, "same-root retry")
    for key in ("fixed_map", "video_during_training_or_numeric_gate", "promotion"):
        equal(bool(execution[key]), False, key)

    boundary = config["post_formal_boundary"]
    equal(bool(boundary["hard_stop_after_pass_fail_or_non_evaluable"]), True, "hard stop")
    equal(bool(boundary["further_optimisation_authorised"]), False, "no further optimisation")
    equal(int(boundary["video_seed"]), 96131, "video seed")
    equal(boundary["video_turn_conditions"], ["curve_left_020", "curve_right_020"], "video conditions")

    validate_v5_runtime(config)
    failures = validate_frozen_failures(config)
    v4_config = json.loads(V4_CONFIG.read_text(encoding="utf-8"))
    v2_config, v1, protocol, reward, checkpoint, inherited = v4.validate_config(v4_config)
    equal(v3.sha256(V2_CONFIG), frozen["V2_configuration_sha256"], "V2 config SHA")
    equal(v3.sha256(checkpoint), frozen["source_checkpoint_sha256"], "source checkpoint")
    equal(inherited["V2_smoke_manifest_sha256"], frozen["V2_smoke_manifest_sha256"], "V2 smoke manifest")

    equal(int(v2_config["formal"]["additional_timesteps_per_condition"]), 65_536, "formal branch budget")
    equal(int(v2_config["formal"]["absolute_final_checkpoint_timesteps"]), 2_793_472, "formal final timestep")
    equal(bool(v2_config["formal"]["save_intermediate_checkpoints"]), False, "intermediate save")
    equal(bool(v2_config["formal"]["evaluate_intermediate_checkpoints"]), False, "intermediate evaluation")
    equal(bool(v2_config["formal"]["select_intermediate_checkpoint"]), False, "intermediate selection")
    equal(int(v1["training"]["master_seed"]), 63_806, "training seed")
    equal(int(v1["training"]["training_seed_count"]), 1, "training seed count")
    equal(v1["final_evaluation"]["heldout_seeds"], [96131, 96137, 96149, 96153, 96177], "held-out seeds")
    equal(v3.sha256(checkpoint), "5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c", "frozen checkpoint")
    evidence = {
        "repair_version": "V5_exact_compiled_contract_no_passed_field",
        "V5_configuration": str(DEFAULT_CONFIG.resolve()),
        "V5_configuration_sha256": v3.sha256(DEFAULT_CONFIG),
        "V5_runner": str((ROOT / RUNTIME_SELF).resolve()),
        "V5_runner_sha256": v3.sha256(ROOT / RUNTIME_SELF),
        "V5_runtime": validate_v5_runtime(config),
        "frozen_failures": failures,
        "inherited_V4_validation": inherited,
        "scientific_protocol_changed": False,
    }
    return v2_config, v1, protocol, reward, checkpoint, evidence


def engineering_smoke_root(config: dict[str, Any]) -> Path:
    return (ROOT / config["execution"]["canonical_engineering_smoke_root"]).resolve()


def formal_root(config: dict[str, Any]) -> Path:
    return (ROOT / config["execution"]["canonical_formal_root"]).resolve()


def run_engineering_smoke(
    config: dict[str, Any],
    v1: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    output_root = engineering_smoke_root(config)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite the unique V5 smoke root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    stage = "smoke_root_created"
    try:
        stage = "freeze_configuration_and_runtime"
        frozen_config = output_root / "frozen_v5_config.json"
        shutil.copy2(DEFAULT_CONFIG, frozen_config)
        equal(v3.sha256(frozen_config), v3.sha256(DEFAULT_CONFIG), "frozen V5 config")
        runtime_before = validate_v5_runtime(config)
        snapshot = snapshot_v5_runtime(config, output_root)
        equal(
            {relative: snapshot[relative] for relative in EXPECTED_RUNTIME_PATHS},
            runtime_before,
            "smoke runtime before/snapshot",
        )

        stage = "prepare_and_validate_four_standard_pair0_scenes"
        scenes = prepare_standard_pair0_scenes_v5(
            v1, protocol, output_root / "standard_slope_assets"
        )
        equal(tuple(scenes), tuple(v2.l2.EXPECTED_SCENES), "smoke scene set")

        stage = "verify_no_training_or_selection"
        equal(list(output_root.rglob("*.zip")), [], "smoke checkpoint absence")
        equal(list(output_root.rglob("training_record.json")), [], "smoke training record absence")
        equal((output_root / "FINAL_OPTIMISATION_HARD_STOP.json").exists(), False, "smoke hard stop absence")
        equal(v3.sha256(checkpoint), "5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c", "source checkpoint after smoke")

        stage = "post_smoke_provenance"
        runtime_after = validate_v5_runtime(config)
        equal(runtime_after, runtime_before, "V5 live runtime before/after")
        snapshot_after = validate_v5_runtime_snapshot(config, output_root)
        equal(snapshot_after, snapshot, "V5 snapshot before/after")
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gymnasium": gym.__version__,
            "mujoco": mujoco.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
        }
        v2.write_json(output_root / "environment.json", environment)
        summary = {
            "schema_version": "proxygap-turn-balance-v5-compiled-audit-smoke-v1",
            "status": "engineering_smoke_passed_no_training",
            "scene_names": list(scenes),
            "compiled_contract_validations": json.loads(
                (output_root / "standard_slope_assets/compiled_contract_validations.json").read_text(encoding="utf-8")
            ),
            "training_performed": False,
            "checkpoint_written_or_selected": False,
            "scientifically_evaluable": False,
            "formal_training_started": False,
        }
        v2.write_json(output_root / "summary.json", summary)
        manifest = {
            "schema_version": "proxygap-turn-balance-v5-engineering-smoke-manifest-v1",
            "status": "engineering_smoke_passed_no_training",
            "mode": "engineering_smoke",
            "configuration": str(DEFAULT_CONFIG.resolve()),
            "configuration_sha256": v3.sha256(DEFAULT_CONFIG),
            "frozen_configuration": str(frozen_config.resolve()),
            "frozen_configuration_sha256": v3.sha256(frozen_config),
            "runner": str((ROOT / RUNTIME_SELF).resolve()),
            "runner_sha256": v3.sha256(ROOT / RUNTIME_SELF),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_sha256": v3.sha256(checkpoint),
            "runtime_live_before": runtime_before,
            "runtime_snapshot": snapshot,
            "runtime_live_after": runtime_after,
            "summary": summary,
            "training_performed": False,
            "checkpoint_written_or_selected": False,
            "formal_training_started": False,
            "scientifically_evaluable": False,
            "formal_pre_run_decision": "GO",
            "formal_pre_run_scope": "engineering_gate_only_not_formal_result",
            "artifact_inventory_excludes_manifest_itself": v2.artifact_inventory(output_root),
            "environment": environment,
            "git": v2.git_record(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        v2.write_json(output_root / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        v2.write_json(
            output_root / "FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-turn-balance-v5-engineering-smoke-failure-v1",
                "mode": "engineering_smoke",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "training_performed": False,
                "formal_training_started": False,
                "scientifically_evaluable": False,
                "formal_pre_run_decision": "HOLD",
                "retry_permitted": False,
                "partial_root_permanently_reserved": True,
                "configuration_sha256": v3.sha256(DEFAULT_CONFIG),
                "runner_sha256": v3.sha256(ROOT / RUNTIME_SELF),
                "source_checkpoint_sha256": (
                    v3.sha256(checkpoint) if checkpoint.is_file() else None
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def validate_completed_engineering_smoke(
    config: dict[str, Any], v1: dict[str, Any], checkpoint: Path
) -> dict[str, Any]:
    root = engineering_smoke_root(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("formal run requires the successful canonical V5 engineering smoke")
    if (root / "FAILURE_RECORD.json").exists():
        raise RuntimeError("canonical V5 smoke root contains a failure record")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    equal(manifest["schema_version"], "proxygap-turn-balance-v5-engineering-smoke-manifest-v1", "smoke schema")
    equal(manifest["status"], "engineering_smoke_passed_no_training", "smoke status")
    equal(manifest["mode"], "engineering_smoke", "smoke mode")
    equal(manifest["configuration_sha256"], v3.sha256(DEFAULT_CONFIG), "smoke config SHA")
    equal(manifest["runner_sha256"], v3.sha256(ROOT / RUNTIME_SELF), "smoke runner SHA")
    equal(manifest["source_checkpoint_sha256"], v3.sha256(checkpoint), "smoke checkpoint SHA")
    equal(bool(manifest["training_performed"]), False, "smoke training")
    equal(bool(manifest["checkpoint_written_or_selected"]), False, "smoke selection")
    equal(bool(manifest["formal_training_started"]), False, "smoke formal start")
    equal(bool(manifest["scientifically_evaluable"]), False, "smoke science")
    equal(manifest["formal_pre_run_decision"], "GO", "smoke GO gate")
    equal(manifest["runtime_live_before"], validate_v5_runtime(config), "smoke live before")
    equal(manifest["runtime_live_after"], validate_v5_runtime(config), "smoke live after")
    equal(manifest["runtime_snapshot"], validate_v5_runtime_snapshot(config, root), "smoke snapshot")
    equal(
        manifest["artifact_inventory_excludes_manifest_itself"],
        v2.artifact_inventory(root),
        "smoke inventory",
    )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() == ".zip"
            or path.name == "training_record.json"
            or path.name == "FINAL_OPTIMISATION_HARD_STOP.json"
        )
    ]
    equal(forbidden, [], "smoke forbidden training/selection artifacts")
    audits = json.loads(
        (root / "standard_slope_assets/explicit_pair_audits.json").read_text(encoding="utf-8")
    )
    validations = json.loads(
        (root / "standard_slope_assets/compiled_contract_validations.json").read_text(encoding="utf-8")
    )
    pair_contract = v2.l2._pair_contract(v1)
    equal(tuple(audits), tuple(v2.l2.EXPECTED_SCENES), "smoke audit scenes")
    equal(tuple(validations), tuple(v2.l2.EXPECTED_SCENES), "smoke validation scenes")
    for scene_name in v2.l2.EXPECTED_SCENES:
        observed = validate_compiled_audit_exact(
            audits[scene_name], pair_contract, v1["contact_contract"]
        )
        equal(observed, validations[scene_name], f"smoke {scene_name} validation")
    return {
        "V5_engineering_smoke_manifest": str(manifest_path.resolve()),
        "V5_engineering_smoke_manifest_sha256": v3.sha256(manifest_path),
        "V5_engineering_smoke_artifact_count_excluding_manifest": len(
            manifest["artifact_inventory_excludes_manifest_itself"]
        ),
        "formal_pre_run_decision": "GO",
        "engineering_only_not_scientific_result": True,
        "training_performed": False,
        "checkpoint_written_or_selected": False,
        "exact_compiled_contract_scene_count": len(validations),
    }


@contextmanager
def install_v5_parent_repairs(
    config: dict[str, Any],
) -> Iterator[None]:
    """Install only the parent-side audit predicate and V5 provenance snapshot."""

    original_prepare = v2.prepare_standard_pair0_scenes
    original_validate_runtime = v2.validate_runtime_dependencies
    original_snapshot_runtime = v2.snapshot_runtime_dependencies

    def validate_runtime_with_v5(v2_config: dict[str, Any]) -> dict[str, str]:
        observed = original_validate_runtime(v2_config)
        validate_v5_runtime(config)
        return observed

    def snapshot_runtime_with_v5(
        v2_config: dict[str, Any], attempt_root: Path
    ) -> tuple[Path, dict[str, str]]:
        snapshot, live = original_snapshot_runtime(v2_config, attempt_root)
        snapshot_v5_runtime(config, attempt_root)
        return snapshot, live

    v2.prepare_standard_pair0_scenes = prepare_standard_pair0_scenes_v5
    v2.validate_runtime_dependencies = validate_runtime_with_v5
    v2.snapshot_runtime_dependencies = snapshot_runtime_with_v5
    try:
        yield
    finally:
        v2.prepare_standard_pair0_scenes = original_prepare
        v2.validate_runtime_dependencies = original_validate_runtime
        v2.snapshot_runtime_dependencies = original_snapshot_runtime


def main() -> None:
    args = parse_args()
    if args.config.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical V5 configuration is permitted")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    v2_config, v1, protocol, reward, checkpoint, evidence = validate_config(config)
    path_evidence = validate_path_budget(formal_root(config))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "V5_VALIDATION_OK_FORMAL_NOT_STARTED",
                    "formal_pre_run_decision": (
                        "GO"
                        if (engineering_smoke_root(config) / "manifest.json").is_file()
                        and not (engineering_smoke_root(config) / "FAILURE_RECORD.json").exists()
                        else "HOLD_PENDING_CANONICAL_ENGINEERING_SMOKE"
                    ),
                    "planned_path_preflight": path_evidence,
                    "V5_configuration_sha256": v3.sha256(DEFAULT_CONFIG),
                    "V5_runner_sha256": v3.sha256(ROOT / RUNTIME_SELF),
                },
                sort_keys=True,
            )
        )
        return
    if args.engineering_smoke:
        manifest = run_engineering_smoke(config, v1, protocol, checkpoint)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "formal_pre_run_decision": manifest["formal_pre_run_decision"],
                    "output_root": str(engineering_smoke_root(config)),
                    "manifest_sha256": v3.sha256(
                        engineering_smoke_root(config) / "manifest.json"
                    ),
                    "formal_training_started": False,
                },
                sort_keys=True,
            )
        )
        return

    output_root = formal_root(config)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite the unique V5 formal root: {output_root}")
    smoke_evidence = validate_completed_engineering_smoke(config, v1, checkpoint)
    evidence["V5_engineering_smoke"] = smoke_evidence
    evidence["V5_formal_path_preflight"] = path_evidence
    with install_v5_parent_repairs(config):
        result = v2.run_parent_attempt(
            V2_CONFIG,
            v2_config,
            v1,
            protocol,
            reward,
            checkpoint,
            output_root,
            smoke=False,
            smoke_prerequisite=evidence,
        )
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    equal(manifest["validated_smoke_prerequisite"], evidence, "formal V5 evidence")
    validate_v5_runtime_snapshot(config, output_root)
    print(json.dumps({"status": result["status"], "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
