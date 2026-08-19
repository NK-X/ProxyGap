"""Gate-only V3 repair for the immutable completed V2 engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fixed_standard_pair0_turn_balance_continuation_v2 as v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json"
)
V2_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v2_20260819.json"
)
RUNTIME_SELF = "scripts/run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair.py"
EXPECTED_GATE_RUNTIME_PATHS = (
    RUNTIME_SELF,
    "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation_v2.py",
)
EXPECTED_ALLOWLIST = (
    (
        "training_scene_assets/scene_source/standard_scene_manifest.json",
        7224,
        "a68ae4276365cf818d1f98d2c52dc810c59215ba9346e7834292b51802e79efd",
    ),
    (
        "training_scene_assets/scene_source/standard_scenes/flat/scene_manifest.json",
        1876,
        "da17bbebb58f9d18ed3b2dc3d97e26a9b53028dcf05228fd5cebaba29cb4af89",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} changed: {observed!r} != {expected!r}")


def validate_gate_runtime(config: dict[str, Any]) -> dict[str, str]:
    expected = config["gate_runtime_contract"]["exact_relative_path_sha256"]
    equal(tuple(expected), EXPECTED_GATE_RUNTIME_PATHS, "V3 gate runtime membership/order")
    observed: dict[str, str] = {}
    for relative_path in EXPECTED_GATE_RUNTIME_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        equal(digest, expected[relative_path], f"V3 gate runtime {relative_path}")
        observed[relative_path] = digest
    return observed


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    equal(
        config.get("status"),
        "frozen_gate_only_repair_pending_independent_go",
        "V3 status",
    )
    frozen = config["frozen_v2_protocol"]
    equal(Path(frozen["configuration"]), V2_CONFIG.relative_to(ROOT), "V2 config path")
    equal(sha256(V2_CONFIG), frozen["configuration_sha256"], "V2 config SHA")
    equal(
        sha256(ROOT / frozen["runner"]),
        frozen["runner_sha256"],
        "V2 runner SHA",
    )
    for key in (
        "scientific_protocol_changed",
        "seeds_changed",
        "budgets_changed",
        "commands_changed",
        "reward_changed",
        "friction_changed",
        "energy_changed",
        "gates_changed",
    ):
        equal(bool(frozen[key]), False, f"V3 scientific boundary {key}")
    allowlist = tuple(
        (str(row["relative_path"]), int(row["size_bytes"]), str(row["sha256"]))
        for row in config["exact_scene_manifest_allowlist"]
    )
    equal(allowlist, EXPECTED_ALLOWLIST, "V3 exact scene-manifest allowlist")
    execution = config["execution"]
    equal(
        execution["permitted_modes"],
        ["validate_only", "formal_after_independent_go"],
        "V3 modes",
    )
    equal(bool(execution["smoke_rerun"]), False, "V3 smoke rerun boundary")
    equal(int(execution["formal_attempt_index"]), 0, "V3 formal attempt")
    equal(bool(execution["retry_after_partial_root"]), False, "V3 retry boundary")
    equal(bool(execution["fixed_map"]), False, "V3 fixed-map boundary")
    equal(
        bool(execution["video_during_training_or_numeric_gate"]),
        False,
        "V3 video boundary",
    )
    equal(bool(execution["promotion"]), False, "V3 promotion boundary")
    post = config["post_formal_boundary"]
    equal(bool(post["hard_stop_after_pass_fail_or_non_evaluable"]), True, "V3 hard stop")
    equal(bool(post["further_optimisation_authorised"]), False, "V3 no more optimisation")
    equal(int(post["video_seed"]), 96131, "V3 video seed")
    equal(post["video_turn_conditions"], ["curve_left_020", "curve_right_020"], "V3 video conditions")
    equal(bool(post["video_participates_in_gate"]), False, "V3 video gate boundary")
    validate_gate_runtime(config)

    v2_config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    v1, protocol, reward, checkpoint = v2.validate_config(v2_config)
    equal(
        sha256(ROOT / v2_config["design_source"]["configuration"]),
        frozen["V1_configuration_sha256"],
        "V1 config SHA through V2",
    )
    equal(sha256(checkpoint), frozen["source_checkpoint_sha256"], "source checkpoint SHA")
    return v2_config, v1, protocol, reward, checkpoint


def validate_completed_v2_smoke(
    config: dict[str, Any], v2_config: dict[str, Any]
) -> dict[str, Any]:
    smoke = config["immutable_completed_smoke"]
    smoke_root = (ROOT / smoke["root"]).resolve()
    manifest_path = (ROOT / smoke["manifest"]).resolve()
    equal(manifest_path, smoke_root / "manifest.json", "V3 smoke manifest path")
    equal(int(manifest_path.stat().st_size), int(smoke["manifest_size_bytes"]), "V3 smoke manifest size")
    equal(sha256(manifest_path), smoke["manifest_sha256"], "V3 smoke manifest SHA")
    equal(v2.sha256(V2_CONFIG), smoke["V2_configuration_sha256"], "V3 smoke V2 config SHA")
    equal(bool(smoke["rerun_permitted"]), False, "V3 smoke rerun")
    equal(bool(smoke["mutation_permitted"]), False, "V3 smoke mutation")

    allow_paths = [row[0] for row in EXPECTED_ALLOWLIST]
    expected_error = (
        "smoke contains forbidden scientific/checkpoint/media artifacts: "
        f"{allow_paths!r}"
    )
    try:
        v2.validate_smoke_manifest_and_inventory(
            v2_config, str(smoke["V2_configuration_sha256"]), smoke_root
        )
    except RuntimeError as error:
        equal(str(error), expected_error, "only suppressed frozen V2 smoke error")
    else:
        raise RuntimeError("frozen V2 validator no longer reached its exact allowlist-only rejection")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["artifact_inventory_excludes_manifest_itself"]
    inventory_by_path = {str(row["relative_path"]): row for row in inventory}
    equal(len(inventory_by_path), len(inventory), "V3 smoke inventory uniqueness")
    nested_manifests = sorted(
        path for path in inventory_by_path if path.lower().endswith("manifest.json")
    )
    equal(nested_manifests, sorted(allow_paths), "V3 exact nested-manifest membership")
    allowlist_evidence: list[dict[str, Any]] = []
    for relative_path, size_bytes, digest in EXPECTED_ALLOWLIST:
        path = smoke_root / relative_path
        row = inventory_by_path[relative_path]
        equal(int(path.stat().st_size), size_bytes, f"V3 allowlist file size {relative_path}")
        equal(sha256(path), digest, f"V3 allowlist file SHA {relative_path}")
        equal(int(row["size_bytes"]), size_bytes, f"V3 allowlist inventory size {relative_path}")
        equal(str(row["sha256"]), digest, f"V3 allowlist inventory SHA {relative_path}")
        json.loads(path.read_text(encoding="utf-8"))
        allowlist_evidence.append(
            {"relative_path": relative_path, "size_bytes": size_bytes, "sha256": digest}
        )
    other_forbidden = [
        path
        for path in inventory_by_path
        if path not in allow_paths
        and (
            path.lower().endswith(".zip")
            or "/models/" in f"/{path.lower()}/"
            or "final_evaluation" in path.lower()
            or path.lower().endswith((".mp4", ".avi", ".mov"))
            or "fixed_map" in path.lower()
            or "promotion" in path.lower()
            or path.lower().endswith("manifest.json")
        )
    ]
    equal(other_forbidden, [], "V3 non-allowlisted forbidden artifacts")
    equal(len(inventory), 51, "V3 frozen smoke artifact count")
    return {
        "gate_version": "V3_exact_two_scene_manifest_allowlist",
        "V3_configuration": str(DEFAULT_CONFIG.resolve()),
        "V3_configuration_sha256": sha256(DEFAULT_CONFIG),
        "V3_runner": str((ROOT / RUNTIME_SELF).resolve()),
        "V3_runner_sha256": sha256(ROOT / RUNTIME_SELF),
        "V3_gate_runtime_before": validate_gate_runtime(config),
        "V2_configuration": str(V2_CONFIG.resolve()),
        "V2_configuration_sha256": sha256(V2_CONFIG),
        "V2_smoke_manifest": str(manifest_path),
        "V2_smoke_manifest_sha256": sha256(manifest_path),
        "V2_smoke_artifact_count_excluding_manifest": len(inventory),
        "V2_original_strict_validation_reached_only_allowlist_rejection": True,
        "exact_scene_manifest_allowlist": allowlist_evidence,
        "scientific_protocol_changed": False,
        "smoke_rerun": False,
        "scientifically_evaluable": False,
        "checkpoint_written": False,
        "full_V2_smoke_semantics_runtime_and_inventory_validated": True,
    }


def formal_root(config: dict[str, Any]) -> Path:
    return (ROOT / config["execution"]["formal_output_root"]).resolve()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical V3 gate-repair configuration is permitted")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    v2_config, v1, protocol, reward, checkpoint = validate_config(config)
    evidence = validate_completed_v2_smoke(config, v2_config)
    output_root = formal_root(config)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite the unique formal root: {output_root}")
    if args.validate_only:
        print("V3_GATE_REPAIR_VALIDATION_OK_NO_FORMAL_RUN")
        return
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
    equal(
        manifest["validated_smoke_prerequisite"],
        evidence,
        "formal manifest retained V3 smoke-gate evidence",
    )
    print(json.dumps({"status": result["status"], "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
