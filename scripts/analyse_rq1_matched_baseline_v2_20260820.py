from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "formal" / "rq1_matched_baseline_v1_20260820" / "attempt_0"
EXPECTED_CONDITIONS = ("D0_DEFAULT_REWARD", "S1_STAGE1_SHAPED")
EXPECTED_TRAINING_SEEDS = (62401, 62402, 62403)
EXPECTED_EVALUATION_SEEDS = tuple(range(72401, 72411))
EXPECTED_TARGETS = (250000, 500000, 750000, 1000000)
PRIMARY_TARGET = 1000000
REQUIRED_FINITE_FIELDS = (
    "fixed_horizon_mean_forward_velocity",
    "net_displacement_direction_error_degrees",
    "forward_path_efficiency",
    "normalised_action_roughness",
    "action_saturation_rate",
    "torso_tilt_rms",
    "airborne_step_fraction",
    "common_rescored_return",
    "condition_objective_return",
)
REQUIRED_BOOLEAN_FIELDS = (
    "unhealthy_termination",
    "full_horizon_completed",
)
REQUIRED_VECTOR_FIELDS = {
    "support_count_step_fractions_0_to_4": 5,
    "foot_contact_duty_fraction_by_foot": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def finite_mean(rows: list[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = as_float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def parse_vector(value: str, length: int) -> list[float]:
    vector = [float(item) for item in json.loads(value)]
    if len(vector) != length:
        raise ValueError(f"Expected {length} values, received {len(vector)}")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("Vector contains a non-finite value")
    return vector


def validate_manifest_inventory(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed_rows = manifest.get("files")
    if not isinstance(listed_rows, list):
        raise ValueError("Source manifest has no file inventory")
    listed = {str(row["path"]): row for row in listed_rows}
    if len(listed) != len(listed_rows):
        raise ValueError("Source manifest contains duplicate paths")
    if int(manifest.get("file_count", -1)) != len(listed_rows):
        raise ValueError("Source manifest file_count differs from its file list")
    current_paths = {
        path.relative_to(run_root).as_posix(): path
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(listed) != set(current_paths):
        added = sorted(set(current_paths) - set(listed))
        missing = sorted(set(listed) - set(current_paths))
        raise ValueError(
            f"Source inventory differs from the sealed manifest; added={added}, missing={missing}"
        )
    mismatches = []
    for relative, path in current_paths.items():
        row = listed[relative]
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if int(row["bytes"]) != actual_size or str(row["sha256"]).lower() != actual_hash:
            mismatches.append(
                {
                    "path": relative,
                    "listed_bytes": int(row["bytes"]),
                    "actual_bytes": actual_size,
                    "listed_sha256": str(row["sha256"]).lower(),
                    "actual_sha256": actual_hash,
                }
            )
    if mismatches:
        raise ValueError(f"Source manifest verification failed: {mismatches}")
    return {
        "manifest_file_count": int(manifest["file_count"]),
        "verified_file_count": len(current_paths),
        "inventory_exact": True,
        "all_sizes_and_hashes_match": True,
    }


def validate_inputs(run_root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    execution_path = run_root / "execution_record.json"
    config_path = run_root / "frozen_run_config.json"
    manifest_path = run_root / "manifest.json"
    required = (execution_path, config_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed-run evidence: {missing}")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if execution.get("status") != "complete" or execution.get("failures") != []:
        raise ValueError("The source run is not complete and failure-free")
    if execution.get("scientific_status") != "formally_evaluated_resource_limited":
        raise ValueError("The source run has the wrong scientific status")
    if sha256(config_path) != execution["config_sha256"]:
        raise ValueError("Frozen config hash does not match the execution record")
    snapshot_root = run_root.parent / "source_snapshot_v1_preinterpretation"
    snapshot_manifest_path = snapshot_root / "manifest.json"
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError("The post-launch, pre-interpretation source snapshot is missing")
    snapshot_validation = validate_manifest_inventory(snapshot_root, snapshot_manifest_path)
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    snapshot_hashes = {str(row["path"]): str(row["sha256"]).lower() for row in snapshot_manifest["files"]}
    for relative, expected_hash in execution["source_hashes"].items():
        normalised = str(relative).replace("\\", "/")
        if snapshot_hashes.get(normalised) != str(expected_hash).lower():
            raise ValueError(f"Source snapshot does not recover executed source: {normalised}")
    manifest_validation = validate_manifest_inventory(run_root, manifest_path)
    raw_relative = "logs/evaluation_metrics_full.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_manifest_rows = [row for row in manifest["files"] if row["path"] == raw_relative]
    if len(raw_manifest_rows) != 1:
        raise ValueError("Source manifest does not bind exactly one raw evaluation table")
    raw_actual_hash = sha256(run_root / raw_relative)
    if str(raw_manifest_rows[0]["sha256"]).lower() != raw_actual_hash:
        raise ValueError("Raw evaluation hash differs from the source manifest")
    if len(rows) != 240:
        raise ValueError(f"Expected 240 raw evaluation rows, received {len(rows)}")
    conditions = {row["condition_id"] for row in rows}
    training_seeds = {int(row["training_seed"]) for row in rows}
    targets = {int(row["target_timesteps"]) for row in rows}
    if conditions != set(EXPECTED_CONDITIONS):
        raise ValueError(f"Unexpected conditions: {conditions}")
    if training_seeds != set(EXPECTED_TRAINING_SEEDS):
        raise ValueError(f"Unexpected training seeds: {training_seeds}")
    if targets != set(EXPECTED_TARGETS):
        raise ValueError(f"Unexpected checkpoint targets: {targets}")
    schema = set(rows[0]) if rows else set()
    required_schema = (
        set(REQUIRED_FINITE_FIELDS)
        | set(REQUIRED_BOOLEAN_FIELDS)
        | set(REQUIRED_VECTOR_FIELDS)
        | {"condition_id", "training_seed", "target_timesteps", "seed"}
    )
    missing_schema = sorted(required_schema - schema)
    if missing_schema:
        raise ValueError(f"Raw evaluation schema is missing required fields: {missing_schema}")
    invalid_values = []
    for row_index, row in enumerate(rows, start=2):
        identity = {
            "row": row_index,
            "condition_id": row.get("condition_id"),
            "training_seed": row.get("training_seed"),
            "target_timesteps": row.get("target_timesteps"),
            "evaluation_seed": row.get("seed"),
        }
        for field in REQUIRED_FINITE_FIELDS:
            try:
                value = as_float(row[field])
            except (KeyError, TypeError, ValueError):
                invalid_values.append({**identity, "field": field, "value": row.get(field), "reason": "not_numeric"})
                continue
            if not math.isfinite(value):
                invalid_values.append({**identity, "field": field, "value": row.get(field), "reason": "not_finite"})
        for field in REQUIRED_BOOLEAN_FIELDS:
            if str(row.get(field)).strip().lower() not in {"true", "false"}:
                invalid_values.append({**identity, "field": field, "value": row.get(field), "reason": "not_boolean"})
        for field, length in REQUIRED_VECTOR_FIELDS.items():
            try:
                parse_vector(row[field], length)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                invalid_values.append({**identity, "field": field, "value": row.get(field), "reason": repr(error)})
    if invalid_values:
        raise ValueError(
            "Required episode metrics are missing, invalid or non-finite; "
            f"fail-closed examples={invalid_values[:10]}, total={len(invalid_values)}"
        )
    keys = set()
    for row in rows:
        key = (
            row["condition_id"],
            int(row["training_seed"]),
            int(row["target_timesteps"]),
            int(row["seed"]),
        )
        if key in keys:
            raise ValueError(f"Duplicate episode key: {key}")
        keys.add(key)
    expected_keys = {
        (condition, training_seed, target, evaluation_seed)
        for condition in EXPECTED_CONDITIONS
        for training_seed in EXPECTED_TRAINING_SEEDS
        for target in EXPECTED_TARGETS
        for evaluation_seed in EXPECTED_EVALUATION_SEEDS
    }
    if keys != expected_keys:
        raise ValueError("Raw evaluation key set differs from the frozen design")
    primary_counts = {}
    for condition in EXPECTED_CONDITIONS:
        for seed in EXPECTED_TRAINING_SEEDS:
            count = sum(
                row["condition_id"] == condition
                and int(row["training_seed"]) == seed
                and int(row["target_timesteps"]) == PRIMARY_TARGET
                for row in rows
            )
            primary_counts[f"{condition}:{seed}"] = count
            if count != 10:
                raise ValueError(f"Primary policy {condition}:{seed} has {count} episodes")
    return {
        "execution_record_sha256": sha256(execution_path),
        "frozen_run_config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(manifest_path),
        "raw_evaluation_sha256": sha256(run_root / "logs" / "evaluation_metrics_full.csv"),
        "raw_evaluation_manifest_sha256": str(raw_manifest_rows[0]["sha256"]).lower(),
        "source_manifest_validation": manifest_validation,
        "source_snapshot_manifest_sha256": sha256(snapshot_manifest_path),
        "source_snapshot_validation": snapshot_validation,
        "rows": len(rows),
        "primary_episode_counts": primary_counts,
        "source_execution_status": execution["status"],
        "source_scientific_status": execution["scientific_status"],
        "source_config_id": config["config_id"],
    }


def build_policy_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        if int(row["target_timesteps"]) == PRIMARY_TARGET:
            groups.setdefault((row["condition_id"], int(row["training_seed"])), []).append(row)
    result = []
    for (condition, seed), group in sorted(groups.items()):
        velocities = [as_float(row["fixed_horizon_mean_forward_velocity"]) for row in group]
        support_counts = []
        duties = []
        for row in group:
            fractions = parse_vector(row["support_count_step_fractions_0_to_4"], 5)
            support_counts.append(sum(index * fraction for index, fraction in enumerate(fractions)))
            duties.append(parse_vector(row["foot_contact_duty_fraction_by_foot"], 4))
        duty = np.asarray(duties, dtype=np.float64)
        result.append(
            {
                "condition_id": condition,
                "training_seed": seed,
                "evaluation_episodes_nested": len(group),
                "valid_episode_count_all_required_metrics": len(group),
                "valid_n__target_speed_abs_error_m_per_s": len(group),
                "missing_n__target_speed_abs_error_m_per_s": 0,
                "valid_n__direction_error_degrees": len(group),
                "missing_n__direction_error_degrees": 0,
                "valid_n__forward_path_efficiency": len(group),
                "missing_n__forward_path_efficiency": 0,
                "valid_n__normalised_action_roughness": len(group),
                "missing_n__normalised_action_roughness": 0,
                "valid_n__unhealthy_termination_rate": len(group),
                "missing_n__unhealthy_termination_rate": 0,
                "target_speed_abs_error_m_per_s": float(np.mean([abs(value - 1.0) for value in velocities])),
                "fixed_horizon_mean_forward_velocity_m_per_s": float(np.mean(velocities)),
                "direction_error_degrees": finite_mean(group, "net_displacement_direction_error_degrees"),
                "forward_path_efficiency": finite_mean(group, "forward_path_efficiency"),
                "normalised_action_roughness": finite_mean(group, "normalised_action_roughness"),
                "action_saturation_rate": finite_mean(group, "action_saturation_rate"),
                "unhealthy_termination_rate": finite_mean(group, "unhealthy_termination"),
                "full_horizon_completion_rate": finite_mean(group, "full_horizon_completed"),
                "torso_tilt_rms_degrees": math.degrees(finite_mean(group, "torso_tilt_rms")),
                "airborne_step_fraction": finite_mean(group, "airborne_step_fraction"),
                "mean_distal_support_count": float(np.mean(support_counts)),
                "foot_1_duty_fraction": float(np.mean(duty[:, 0])),
                "foot_2_duty_fraction": float(np.mean(duty[:, 1])),
                "foot_3_duty_fraction": float(np.mean(duty[:, 2])),
                "foot_4_duty_fraction": float(np.mean(duty[:, 3])),
                "common_default_reward_rescore": finite_mean(group, "common_rescored_return"),
                "condition_objective_return_not_cross_condition_comparable": finite_mean(group, "condition_objective_return"),
            }
        )
    if len(result) != 6:
        raise ValueError(f"Expected six policy rows, received {len(result)}")
    return result


def build_pairs(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["condition_id"], row["training_seed"]): row for row in policy_rows}
    metrics = [
        key
        for key in policy_rows[0]
        if key not in {
            "condition_id",
            "training_seed",
            "evaluation_episodes_nested",
            "valid_episode_count_all_required_metrics",
        }
        and not key.startswith("valid_n__")
        and not key.startswith("missing_n__")
        and not key.startswith("foot_")
        and key != "condition_objective_return_not_cross_condition_comparable"
    ]
    pairs = []
    for seed in EXPECTED_TRAINING_SEEDS:
        default = indexed[("D0_DEFAULT_REWARD", seed)]
        shaped = indexed[("S1_STAGE1_SHAPED", seed)]
        pair: dict[str, Any] = {"training_seed": seed}
        for metric in metrics:
            pair[f"default__{metric}"] = default[metric]
            pair[f"shaped__{metric}"] = shaped[metric]
            pair[f"shaped_minus_default__{metric}"] = shaped[metric] - default[metric]
        pairs.append(pair)
    return pairs


def build_condition_summary(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        key
        for key in policy_rows[0]
        if key not in {
            "condition_id",
            "training_seed",
            "evaluation_episodes_nested",
            "valid_episode_count_all_required_metrics",
        }
        and not key.startswith("valid_n__")
        and not key.startswith("missing_n__")
    ]
    result = []
    for condition in EXPECTED_CONDITIONS:
        group = [row for row in policy_rows if row["condition_id"] == condition]
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result.append(
                {
                    "condition_id": condition,
                    "metric": metric,
                    "training_seed_n": len(group),
                    "mean_across_training_seeds": float(np.mean(values)),
                    "median_across_training_seeds": float(np.median(values)),
                    "minimum_across_training_seeds": float(np.min(values)),
                    "maximum_across_training_seeds": float(np.max(values)),
                    "uncertainty_note": "raw range only; n=3 is insufficient for stable interval inference",
                }
            )
    return result


def build_decision(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    directions = {
        "target_speed_abs_error_m_per_s": "lower",
        "direction_error_degrees": "lower",
        "forward_path_efficiency": "higher",
        "normalised_action_roughness": "lower",
    }
    counts = {}
    for metric, preferred in directions.items():
        values = [float(pair[f"shaped_minus_default__{metric}"]) for pair in pairs]
        counts[metric] = sum(value < 0 if preferred == "lower" else value > 0 for value in values)
    safety_values = [float(pair["shaped_minus_default__unhealthy_termination_rate"]) for pair in pairs]
    safety_non_worse = sum(value <= 0 for value in safety_values)
    quality_metrics_passing = sum(count >= 2 for count in counts.values())
    safety_gate = safety_non_worse >= 2
    quality_gate = quality_metrics_passing >= 3
    return {
        "analysis_unit": "independent training seed",
        "training_pairs_n": 3,
        "evaluation_episodes_per_policy_nested": 10,
        "quality_improved_pair_counts_out_of_3": counts,
        "safety_non_worse_pair_count_out_of_3": safety_non_worse,
        "quality_metrics_passing_2_of_3_rule_out_of_4": quality_metrics_passing,
        "safety_gate_pass": safety_gate,
        "quality_gate_pass": quality_gate,
        "joint_descriptive_gate_pass": safety_gate and quality_gate,
        "statistical_boundary": "descriptive only; minimum exact two-sided sign-test p-value at n=3 is 0.25",
        "natural_gait_claim": "not tested and not supported by this gate",
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    raw_path = run_root / "logs" / "evaluation_metrics_full.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    rows = read_csv(raw_path)
    validation = validate_inputs(run_root, rows)
    if args.validate_only:
        print(json.dumps({"status": "validated", "input": validation}, indent=2))
        return
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else run_root.parent / f"{run_root.name}_analysis_v2_boolean_repair"
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty analysis root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    policy_rows = build_policy_rows(rows)
    pairs = build_pairs(policy_rows)
    condition_summary = build_condition_summary(policy_rows)
    decision = build_decision(pairs)
    write_csv(output_root / "policy_level_metrics.csv", policy_rows)
    write_csv(output_root / "paired_training_seed_effects.csv", pairs)
    write_csv(output_root / "condition_summary_across_training_seeds.csv", condition_summary)
    result = {
        "status": "versioned_analysis_repair_complete",
        "reason": "The in-run V1 analysis did not parse CSV True/False values as numeric rates. Raw episode rows and training were unaffected; this directory preserves a non-destructive corrected analysis.",
        "input_validation": validation,
        "decision": decision,
        "claim_boundary": "Resource-limited descriptive evidence conditional on three paired training seeds, flat-ground Ant-v5, this PPO implementation and the frozen reward package.",
        "source_code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest_files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        manifest_files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (output_root / "manifest.json").write_text(
        json.dumps({"file_count": len(manifest_files), "files": manifest_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
