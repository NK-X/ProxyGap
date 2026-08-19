"""One-time short-root wrapper for the unchanged final turn-balance protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fixed_standard_pair0_turn_balance_continuation_v2 as v2  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair as v3  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v4_short_root_20260819.json"
V2_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json"
V3_CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json"
RUNTIME_SELF = "scripts/run_fixed_standard_pair0_turn_balance_continuation_v4_short_root.py"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF,
    "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v2.py",
    "configs/fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair.py",
)
EXPECTED_OLD_FAILURE_SHA = "21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23"
MAX_PATH_CHARS = 239


def equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} changed: {observed!r} != {expected!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args()


def validate_v4_runtime(config: dict[str, Any]) -> dict[str, str]:
    expected = config["V4_runtime_contract"]["exact_relative_path_sha256"]
    equal(tuple(expected), EXPECTED_RUNTIME_PATHS, "V4 runtime membership/order")
    observed = {relative: v3.sha256(ROOT / relative) for relative in EXPECTED_RUNTIME_PATHS}
    equal(observed, expected, "V4 runtime hashes")
    return observed


def validate_old_failure(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["immutable_V2_formal_failure"]
    root = (ROOT / frozen["root"]).resolve()
    record_path = (ROOT / frozen["failure_record"]).resolve()
    equal(record_path, root / "FAILURE_RECORD.json", "old failure record path")
    equal(v3.sha256(record_path), EXPECTED_OLD_FAILURE_SHA, "old failure SHA")
    equal(EXPECTED_OLD_FAILURE_SHA, frozen["failure_record_sha256"], "frozen old failure SHA")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    equal(record["failed_stage"], "prepare_standard_slope_scenes", "old failed stage")
    equal(record["exception_type"], "FileNotFoundError", "old exception type")
    equal(record["configuration_sha256"], v3.sha256(V2_CONFIG), "old V2 config SHA")
    equal(record["source_checkpoint_observed_sha256"], config["frozen_scientific_protocol"]["source_checkpoint_sha256"], "old source checkpoint SHA")
    equal(bool(record["scientifically_evaluable"]), False, "old failure evaluability")
    equal(bool(record["all_decisions_withheld"]), True, "old failure decisions")
    equal(bool(record["retry_permitted"]), False, "old failure retry")
    equal((root / "manifest.json").exists(), False, "old success manifest absence")
    equal((root / "FINAL_OPTIMISATION_HARD_STOP.json").exists(), False, "old hard-stop marker absence")
    zip_paths = sorted(root.rglob("*.zip"))
    records = sorted(root.rglob("training_record.json"))
    equal(len(zip_paths), int(frozen["zip_count"]), "old zip count")
    equal(len(records), int(frozen["training_record_count"]), "old training record count")
    equal(bool(frozen["training_started"]), False, "old training-start boundary")
    equal(record["runtime_live_before"], record["runtime_snapshot_before"], "old runtime before maps")
    equal(len(record["runtime_live_before"]), 25, "old runtime count")
    return {
        "root": str(root),
        "failure_record": str(record_path),
        "failure_record_sha256": v3.sha256(record_path),
        "failed_stage": record["failed_stage"],
        "exception_type": record["exception_type"],
        "training_started": False,
        "zip_count": len(zip_paths),
        "training_record_count": len(records),
        "scientifically_evaluable": False,
        "retry_same_root": False,
    }


def planned_relative_paths() -> tuple[str, ...]:
    paths: set[str] = {
        "frozen_config.json", "training_scene.json", "summary.json", "final_gate.json",
        "REPORT.md", "FINAL_OPTIMISATION_HARD_STOP.json", "environment.json", "manifest.json",
        "FAILURE_RECORD.json", "standard_slope_assets/scene_generation.json",
        "standard_slope_assets/explicit_pair_audits.json", "standard_slope_assets/prepared_scenes.json",
    }
    paths.update(f"runtime_snapshot/{relative}" for relative in v2.EXPECTED_RUNTIME_PATHS)
    asset_names = ("ant_standard_scene.xml", "heights_m.npy", "terrain_contours.png", "terrain.hfield")
    variants = ("default_margin_001_control", "explicit_floor_distal_pair_margin0_candidate")
    for prefix, scenes in (("training_scene_assets", ("flat",)), ("standard_slope_assets", ("flat", "uphill_8deg", "downhill_8deg", "bowl_exit"))):
        generated_root = "scene_source" if prefix == "training_scene_assets" else "generated"
        paths.add(f"{prefix}/{generated_root}/standard_scene_manifest.json")
        for scene in scenes:
            for name in (*asset_names, "scene_manifest.json"):
                paths.add(f"{prefix}/{generated_root}/standard_scenes/{scene}/{name}")
            for variant in variants:
                for name in asset_names:
                    paths.add(f"{prefix}/condition_assets/scenes/turn_balance_{scene}/{variant}/{name}")
    for condition in ("c0_straight_continue", "c1_balanced_turn"):
        paths.update({
            f"{condition}/monitor.csv.monitor.csv", f"{condition}/training_record.json",
            f"{condition}/WORKER_FAILURE_RECORD.json", f"{condition}/models/checkpoint_2793472.zip",
            f"{condition}/final_evaluation/turn_episode_metrics.csv",
            f"{condition}/final_evaluation/turn_corrected_slip_events.csv",
            f"{condition}/final_evaluation/turn_aggregates.json",
            f"{condition}/final_evaluation/slope_episode_metrics.csv",
            f"{condition}/final_evaluation/slope_representative_substep_traces.csv",
            f"{condition}/final_evaluation/slope_corrected_slip_events.csv",
            f"{condition}/final_evaluation/slope_aggregate.json",
            f"{condition}/final_evaluation/branch_gate.json",
            f"condition_process_logs/{condition.upper()}.stdout.txt",
            f"condition_process_logs/{condition.upper()}.stderr.txt",
        })
    return tuple(sorted(paths))


def validate_path_budget(output_root: Path) -> dict[str, Any]:
    rows = [{"relative_path": relative, "absolute_path": str(output_root / relative), "characters": len(str(output_root / relative))} for relative in planned_relative_paths()]
    maximum = max(rows, key=lambda row: row["characters"])
    if any(row["characters"] > MAX_PATH_CHARS for row in rows):
        raise ValueError(f"planned path exceeds {MAX_PATH_CHARS}: {maximum}")
    return {"limit_characters_inclusive": MAX_PATH_CHARS, "planned_path_count": len(rows), "maximum": maximum, "all_within_limit": True}


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    equal(config["status"], "frozen_engineering_path_repair_pending_independent_go", "V4 status")
    frozen = config["frozen_scientific_protocol"]
    for path_key, sha_key in (("V2_configuration", "V2_configuration_sha256"), ("V2_runner", "V2_runner_sha256"), ("V3_configuration", "V3_configuration_sha256"), ("V3_runner", "V3_runner_sha256")):
        equal(v3.sha256(ROOT / frozen[path_key]), frozen[sha_key], sha_key)
    for key in ("scientific_protocol_changed", "seeds_changed", "budgets_changed", "commands_changed", "reward_changed", "friction_changed", "energy_changed", "gates_changed"):
        equal(bool(frozen[key]), False, key)
    execution = config["execution"]
    equal(execution["canonical_formal_root"], "artifacts/dev/tb_v4_20260819/a0", "short root")
    equal(execution["only_change_from_V3_formal_execution"], "canonical_output_root_shortening", "only repair")
    equal(bool(execution["smoke_rerun"]), False, "smoke rerun")
    equal(bool(execution["reuse_partial_weights"]), False, "partial weights")
    equal(bool(execution["retry_after_V4_failure"]), False, "V4 retry")
    for key in ("fixed_map", "video_during_training_or_numeric_gate", "promotion"):
        equal(bool(execution[key]), False, key)
    equal(bool(config["post_formal_boundary"]["hard_stop_after_pass_fail_or_non_evaluable"]), True, "hard stop")
    equal(bool(config["post_formal_boundary"]["further_optimisation_authorised"]), False, "no further optimisation")
    equal(int(config["post_formal_boundary"]["video_seed"]), 96131, "video seed")
    validate_v4_runtime(config)
    old_failure = validate_old_failure(config)
    v3_config = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    v2_config, v1, protocol, reward, checkpoint = v3.validate_config(v3_config)
    equal(v3.sha256(checkpoint), frozen["source_checkpoint_sha256"], "source checkpoint")
    evidence = v3.validate_completed_v2_smoke(v3_config, v2_config)
    equal(evidence["V2_smoke_manifest_sha256"], frozen["V2_smoke_manifest_sha256"], "smoke manifest")
    evidence["V4_short_root_repair"] = {"V4_configuration": str(DEFAULT_CONFIG.resolve()), "V4_configuration_sha256": v3.sha256(DEFAULT_CONFIG), "V4_runner": str((ROOT / RUNTIME_SELF).resolve()), "V4_runner_sha256": v3.sha256(ROOT / RUNTIME_SELF), "old_V2_failure": old_failure}
    return v2_config, v1, protocol, reward, checkpoint, evidence


def formal_root(config: dict[str, Any]) -> Path:
    return (ROOT / config["execution"]["canonical_formal_root"]).resolve()


def main() -> None:
    args = parse_args()
    if args.config.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical V4 configuration is permitted")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    v2_config, v1, protocol, reward, checkpoint, evidence = validate_config(config)
    output_root = formal_root(config)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite the unique V4 root: {output_root}")
    path_evidence = validate_path_budget(output_root)
    evidence["V4_short_root_repair"]["planned_path_preflight"] = path_evidence
    if args.validate_only:
        print(json.dumps({"status": "V4_SHORT_ROOT_VALIDATION_OK_NO_FORMAL_RUN", "planned_path_preflight": path_evidence}))
        return
    result = v2.run_parent_attempt(V2_CONFIG, v2_config, v1, protocol, reward, checkpoint, output_root, smoke=False, smoke_prerequisite=evidence)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    equal(manifest["validated_smoke_prerequisite"], evidence, "formal retained V4 evidence")
    print(json.dumps({"status": result["status"], "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
