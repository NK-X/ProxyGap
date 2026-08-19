"""Supplement the frozen margin-pair gate with duration-corrected slip evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_standard_distal_margin0_paired as parent  # noqa: E402
from evaluate_local_preview_final_paired_direct_goal import (  # noqa: E402
    DurationCorrectedSlipTracker,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_explicit_pair_corrected_slip_audit_v1_20260819.json"
)
EVENT_COLUMNS = (
    "condition_id",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def exact_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = parent.sha256(path)
    if observed.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA-256 changed: {observed}")


def write_event_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write parseable evidence even when the corrected tracker finds no events."""
    if rows:
        parent.write_rows(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(EVENT_COLUMNS) + "\n", encoding="utf-8")


def validate_config(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    if config.get("status") != "frozen_read_only_supplementary_audit":
        raise ValueError("Supplementary slip configuration is not frozen")
    if config.get("formal_training") != "prohibited":
        raise ValueError("Training must remain prohibited")
    frozen = config["frozen_parent"]
    parent_config_path = ROOT / frozen["configuration"]
    exact_hash(parent_config_path, frozen["configuration_sha256"], "parent config")
    artifact_root = ROOT / frozen["artifact_root"]
    exact_hash(artifact_root / "manifest.json", frozen["manifest_sha256"], "parent manifest")
    exact_hash(artifact_root / "standard_gate.json", frozen["standard_gate_sha256"], "parent standard gate")
    exact_hash(
        artifact_root / "standard_paired_episode_metrics.csv",
        frozen["standard_episode_metrics_sha256"],
        "parent standard metrics",
    )
    parent_gate = json.loads((artifact_root / "standard_gate.json").read_text(encoding="utf-8"))
    if parent_gate.get("passed") is not False or frozen.get("original_standard_gate_passed") is not False:
        raise ValueError("This audit requires the frozen failed parent gate")
    if frozen.get("original_gate_conclusion_must_remain_unchanged") is not True:
        raise ValueError("Parent gate conclusion must remain immutable")
    tracker = config["tracker_source"]
    exact_hash(ROOT / tracker["script"], tracker["script_sha256"], "tracker source")
    if tracker.get("class") != "DurationCorrectedSlipTracker":
        raise ValueError("Unexpected slip tracker class")
    evaluation = config["evaluation"]
    if evaluation["conditions"] != [parent.CONTROL_ID, parent.CANDIDATE_ID]:
        raise ValueError("Paired conditions changed")
    if evaluation["scene_order"] != ["flat", "uphill_8deg", "downhill_8deg", "bowl_exit"]:
        raise ValueError("Standard scene order changed")
    if evaluation["seeds"] != [79801, 79802, 79803]:
        raise ValueError("Evaluation seeds changed")
    if int(evaluation["max_episode_steps"]) != 600 or float(evaluation["cruise_speed_m_per_s"]) != 0.55:
        raise ValueError("Horizon or cruise speed changed")
    if int(evaluation["expected_episode_count"]) != 24:
        raise ValueError("Expected episode count changed")
    slip = config["duration_corrected_slip"]
    if (
        float(slip["tangential_speed_threshold_m_per_s"]) != 0.2
        or float(slip["minimum_normal_force_n"]) != 1.0
        or float(slip["landing_grace_seconds"]) != 0.1
        or float(slip["minimum_sustained_seconds"]) != 0.2
    ):
        raise ValueError("Corrected-slip thresholds changed")
    if config["exploratory_training_interpretation"].get("formal_promotion_authorised") is not False:
        raise ValueError("Supplementary audit cannot authorise formal promotion")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    protocol, reward, source_manifest = parent.validate_config(parent_config)
    import csv

    with (artifact_root / "standard_paired_episode_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        parent_rows = list(csv.DictReader(handle))
    if len(parent_rows) != 24:
        raise ValueError("Parent paired result no longer contains 24 episodes")
    return parent_config, protocol, reward, source_manifest, parent_rows


def evaluate_episode(
    model: PPO,
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    config: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = config["evaluation"]
    env = parent.make_eval_env(
        protocol,
        reward,
        scene,
        condition_id=condition_id,
        seed=seed,
        max_episode_steps=int(evaluation["max_episode_steps"]),
        cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
        fixed_contract=None,
    )
    observation, _ = env.reset(seed=seed)
    if tuple(observation.shape) != (135,):
        env.close()
        raise RuntimeError("Observation contract changed")
    dt = float(env.unwrapped.dt)
    slip_config = config["duration_corrected_slip"]
    tracker = DurationCorrectedSlipTracker(
        dt=dt,
        speed_threshold=float(slip_config["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip_config["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip_config["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip_config["minimum_sustained_seconds"]),
    )
    trace: list[dict[str, Any]] = []
    contact_history: list[np.ndarray] = []
    support_sum = 0
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        step += 1
        contacts = np.asarray(info["proxygap_foot_contact_mask_step"], dtype=bool)
        speeds = np.asarray(
            info["proxygap_foot_contact_tangential_speeds_m_per_s_step"],
            dtype=np.float64,
        )
        forces = np.asarray(info["proxygap_foot_normal_forces_n_step"], dtype=np.float64)
        if contacts.shape != (4,) or speeds.shape != (4,) or forces.shape != (4,):
            env.close()
            raise RuntimeError("Foot slip vectors changed shape")
        raw, qualified = tracker.update(
            contact_mask=contacts,
            tangential_speeds=speeds,
            normal_forces=forces,
        )
        support_count = int(np.sum(contacts))
        support_sum += support_count
        contact_history.append(contacts.copy())
        trace.append(
            {
                "condition_id": condition_id,
                "scene_name": scene["scene_name"],
                "evaluation_seed": seed,
                "step": step,
                "time_seconds": step * dt,
                "support_count": support_count,
                "supported_step": int(support_count > 0),
                "contact_mask": json.dumps(contacts.astype(int).tolist()),
                "tangential_speeds_m_per_s": json.dumps(speeds.tolist(), separators=(",", ":")),
                "normal_forces_n": json.dumps(forces.tolist(), separators=(",", ":")),
                "raw_speed_exceedance_any": int(np.any(raw)),
                "qualified_post_grace_force_gated_any": int(np.any(qualified)),
                "duration_corrected_sustained_any": 0,
            }
        )
    result = tracker.finalise()
    raw = np.asarray(result["raw"], dtype=bool)
    qualified = np.asarray(result["candidate"], dtype=bool)
    sustained = np.asarray(result["sustained"], dtype=bool)
    contacts = np.asarray(contact_history, dtype=bool)
    if raw.shape != (step, 4) or qualified.shape != (step, 4) or sustained.shape != (step, 4):
        env.close()
        raise RuntimeError("Corrected-slip tracker returned an unexpected shape")
    for index, row in enumerate(trace):
        row["duration_corrected_sustained_any"] = int(np.any(sustained[index]))
    summary = env.episode_summary()
    env.close()
    supported = np.any(contacts, axis=1)
    raw_any = np.any(raw, axis=1)
    qualified_any = np.any(qualified, axis=1)
    sustained_any = np.any(sustained, axis=1)
    supported_steps = int(np.sum(supported))
    contact_samples = int(np.sum(contacts))

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    events = [
        {
            "condition_id": condition_id,
            "scene_name": scene["scene_name"],
            "evaluation_seed": seed,
            **event,
        }
        for event in result["events"]
    ]
    row = {
        "condition_id": condition_id,
        "scene_name": scene["scene_name"],
        "evaluation_seed": seed,
        "episode_steps": step,
        "fall": bool(summary["fall"]),
        "fixed_goal_success": bool(summary["fixed_goal_success"]),
        "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"]) - float(summary["fixed_goal_minimum_distance_m"]),
        "mean_support_count": support_sum / step,
        "supported_step_count": supported_steps,
        "distal_foot_contact_sample_count": contact_samples,
        "raw_step_count": int(np.sum(raw_any)),
        "qualified_step_count": int(np.sum(qualified_any)),
        "duration_corrected_sustained_step_count": int(np.sum(sustained_any)),
        "raw_all_step_fraction": float(np.mean(raw_any)),
        "qualified_all_step_fraction": float(np.mean(qualified_any)),
        "duration_corrected_sustained_all_step_fraction": float(np.mean(sustained_any)),
        "raw_per_supported_step_fraction": fraction(int(np.sum(raw_any)), supported_steps),
        "qualified_per_supported_step_fraction": fraction(int(np.sum(qualified_any)), supported_steps),
        "duration_corrected_sustained_per_supported_step_fraction": fraction(int(np.sum(sustained_any)), supported_steps),
        "raw_foot_sample_count": int(np.sum(raw)),
        "qualified_foot_sample_count": int(np.sum(qualified)),
        "duration_corrected_sustained_foot_sample_count": int(np.sum(sustained)),
        "raw_per_distal_foot_contact_sample_fraction": fraction(int(np.sum(raw)), contact_samples),
        "qualified_per_distal_foot_contact_sample_fraction": fraction(int(np.sum(qualified)), contact_samples),
        "duration_corrected_sustained_per_distal_foot_contact_sample_fraction": fraction(int(np.sum(sustained)), contact_samples),
        "duration_corrected_slip_event_count": len(events),
        "duration_corrected_slip_longest_event_seconds": max(
            (float(event["duration_seconds"]) for event in events), default=0.0
        ),
    }
    return row, trace, events


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty corrected-slip rows")
    total_steps = sum(int(row["episode_steps"]) for row in rows)
    supported_steps = sum(int(row["supported_step_count"]) for row in rows)
    contact_samples = sum(int(row["distal_foot_contact_sample_count"]) for row in rows)

    def pooled(count_key: str, denominator: int) -> float:
        return (
            sum(int(row[count_key]) for row in rows) / denominator
            if denominator
            else 0.0
        )

    result = {
        "episode_count": len(rows),
        "fall_count": sum(bool(row["fall"]) for row in rows),
        "success_count": sum(bool(row["fixed_goal_success"]) for row in rows),
        "total_control_steps": total_steps,
        "supported_step_count": supported_steps,
        "distal_foot_contact_sample_count": contact_samples,
        "mean_support_count": float(np.mean([float(row["mean_support_count"]) for row in rows])),
        "mean_best_progress_m": float(np.mean([float(row["fixed_goal_best_progress_m"]) for row in rows])),
        "pooled_raw_all_step_fraction": pooled("raw_step_count", total_steps),
        "pooled_qualified_all_step_fraction": pooled("qualified_step_count", total_steps),
        "pooled_duration_corrected_sustained_all_step_fraction": pooled("duration_corrected_sustained_step_count", total_steps),
        "pooled_raw_per_supported_step_fraction": pooled("raw_step_count", supported_steps),
        "pooled_qualified_per_supported_step_fraction": pooled("qualified_step_count", supported_steps),
        "pooled_duration_corrected_sustained_per_supported_step_fraction": pooled("duration_corrected_sustained_step_count", supported_steps),
        "pooled_raw_per_distal_foot_contact_sample_fraction": pooled("raw_foot_sample_count", contact_samples),
        "pooled_qualified_per_distal_foot_contact_sample_fraction": pooled("qualified_foot_sample_count", contact_samples),
        "pooled_duration_corrected_sustained_per_distal_foot_contact_sample_fraction": pooled("duration_corrected_sustained_foot_sample_count", contact_samples),
        "duration_corrected_slip_event_count": sum(int(row["duration_corrected_slip_event_count"]) for row in rows),
    }
    result["events_per_100_supported_steps"] = (
        100.0 * result["duration_corrected_slip_event_count"] / supported_steps
        if supported_steps
        else 0.0
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_config, protocol, reward, source_manifest, parent_rows = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config_sha256": parent.sha256(config_path)}, indent=2))
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

    scene_pairs: dict[str, dict[str, dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    pair_contract = parent_config["permitted_xml_change"]["explicit_pair_contract"]
    for scene_name in config["evaluation"]["scene_order"]:
        pair, audit = parent.prepare_pair(
            source_manifest["scenes"][scene_name],
            output_root,
            f"standard_{scene_name}",
            pair_contract,
        )
        scene_pairs[scene_name] = pair
        audits[scene_name] = audit
    parent.write_json(output_root / "scene_pair_audits.json", audits)
    model = PPO.load(
        ROOT / parent_config["frozen_standard_protocol"]["source_checkpoint"],
        device=str(config["execution"]["device"]),
    )
    all_rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for condition_id in config["evaluation"]["conditions"]:
        for scene_name in config["evaluation"]["scene_order"]:
            for seed in config["evaluation"]["seeds"]:
                row, trace, events = evaluate_episode(
                    model,
                    protocol,
                    reward,
                    scene_pairs[scene_name][condition_id],
                    config,
                    condition_id=condition_id,
                    seed=int(seed),
                )
                all_rows.append(row)
                all_trace.extend(trace)
                all_events.extend(events)
                print(json.dumps({
                    "condition": condition_id,
                    "scene": scene_name,
                    "seed": seed,
                    "supported_steps": row["supported_step_count"],
                    "raw_per_supported": row["raw_per_supported_step_fraction"],
                    "qualified_per_supported": row["qualified_per_supported_step_fraction"],
                    "sustained_per_supported": row["duration_corrected_sustained_per_supported_step_fraction"],
                    "events": row["duration_corrected_slip_event_count"],
                }))
    if len(all_rows) != int(config["evaluation"]["expected_episode_count"]):
        raise RuntimeError("Corrected-slip audit did not produce exactly 24 episodes")

    parent_lookup = {
        (row["condition_id"], row["scene_name"], int(row["evaluation_seed"])): row
        for row in parent_rows
    }
    maximum_progress_difference = 0.0
    maximum_support_difference = 0.0
    for row in all_rows:
        key = (row["condition_id"], row["scene_name"], int(row["evaluation_seed"]))
        frozen = parent_lookup[key]
        maximum_progress_difference = max(
            maximum_progress_difference,
            abs(float(row["fixed_goal_best_progress_m"]) - float(frozen["fixed_goal_best_progress_m"])),
        )
        maximum_support_difference = max(
            maximum_support_difference,
            abs(float(row["mean_support_count"]) - float(frozen["mean_support_count"])),
        )
        if bool(row["fall"]) != (str(frozen["fall"]).lower() == "true"):
            raise RuntimeError("Corrected-slip rerun fall outcome differs from the frozen parent")
    if maximum_progress_difference > 1e-9 or maximum_support_difference > 1e-12:
        raise RuntimeError("Corrected-slip rerun did not reproduce parent kinematics")

    parent.write_rows(output_root / "corrected_slip_episode_metrics.csv", all_rows)
    parent.write_rows(output_root / "corrected_slip_step_trace.csv", all_trace)
    write_event_rows(output_root / "corrected_slip_events.csv", all_events)
    by_condition = {
        condition: aggregate([row for row in all_rows if row["condition_id"] == condition])
        for condition in config["evaluation"]["conditions"]
    }
    control = by_condition[parent.CONTROL_ID]
    candidate = by_condition[parent.CANDIDATE_ID]
    interpretation = config["exploratory_training_interpretation"]
    observed = {
        "duration_corrected_sustained_per_supported_step_fraction_delta": candidate["pooled_duration_corrected_sustained_per_supported_step_fraction"] - control["pooled_duration_corrected_sustained_per_supported_step_fraction"],
        "events_per_100_supported_steps_delta": candidate["events_per_100_supported_steps"] - control["events_per_100_supported_steps"],
        "mean_support_count_increase": candidate["mean_support_count"] - control["mean_support_count"],
        "additional_falls": candidate["fall_count"] - control["fall_count"],
        "best_progress_ratio": candidate["mean_best_progress_m"] / max(control["mean_best_progress_m"], 1e-12),
    }
    checks = {
        "corrected_sustained_fraction_not_materially_worse": observed["duration_corrected_sustained_per_supported_step_fraction_delta"] <= float(interpretation["material_fraction_tolerance"]),
        "event_rate_not_materially_worse": observed["events_per_100_supported_steps_delta"] <= float(interpretation["material_event_rate_per_100_supported_steps_tolerance"]),
        "support_gain_reproduced": observed["mean_support_count_increase"] >= float(interpretation["minimum_reproduced_support_count_increase"]),
        "no_additional_falls": observed["additional_falls"] <= 0,
    }
    recommendation = bool(all(checks.values()))
    summary = {
        "schema_version": "proxygap-fixed-standard-corrected-slip-summary-v1",
        "condition_aggregates": by_condition,
        "parent_rerun_equivalence": {
            "maximum_absolute_best_progress_difference_m": maximum_progress_difference,
            "maximum_absolute_mean_support_count_difference": maximum_support_difference,
            "fall_outcomes_identical": True,
        },
        "exploratory_training_interpretation": {
            "status": "post_parent_gate_supporting_audit_not_a_replacement_gate",
            "observed": observed,
            "checks": checks,
            "supports_new_small_bounded_exploratory_training": recommendation,
            "formal_training_or_promotion_authorised": False,
            "parent_standard_gate_remains_failed": True,
            "fixed_map_remains_skipped": True,
        },
        "measurement_boundary": config["duration_corrected_slip"]["sampling_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    parent.write_json(output_root / "corrected_slip_summary.json", summary)
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({
                "relative_path": str(path.relative_to(output_root)).replace("\\", "/"),
                "path": str(path.resolve()),
                "sha256": parent.sha256(path),
                "bytes": path.stat().st_size,
            })
    manifest = {
        "schema_version": "proxygap-fixed-standard-corrected-slip-manifest-v1",
        "status": "complete_read_only_supplementary_audit",
        "configuration": {"path": str(config_path), "sha256": parent.sha256(config_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": parent.sha256(Path(__file__))},
        "episode_count": len(all_rows),
        "training_performed": False,
        "parent_gate_modified": False,
        "fixed_map_evaluated": False,
        "supports_new_small_bounded_exploratory_training": recommendation,
        "artifacts": artifacts,
        "claim_boundary": config["claim_boundary"],
    }
    parent.write_json(output_root / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete",
        "output_root": str(output_root),
        "supports_new_small_bounded_exploratory_training": recommendation,
        "manifest_sha256": parent.sha256(output_root / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
