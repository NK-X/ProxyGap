from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_standard_support_curriculum import (  # noqa: E402
    aggregate_rows,
    evaluate_episode,
    evaluate_matrix,
    high_frequency_contact_matrix,
    prepare_standard_scenes,
    robot_signature,
    sha256,
    validate_config as validate_support_protocol,
    verified_json,
    write_json,
    write_rows,
)


DEFAULT_CONFIG = ROOT / "configs" / "fixed_standard_speed_ablation_v1_20260819.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only paired speed diagnosis on fixed standard terrains."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _require_exact_keys(records: list[dict[str, Any]], expected: list[str]) -> None:
    observed = [str(record.get("model_id")) for record in records]
    if observed != expected:
        raise ValueError(f"Model identities changed: {observed!r}")


def validate_ablation_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != "proxygap-fixed-standard-speed-ablation-v1":
        raise ValueError("Unexpected speed-ablation schema")
    protocol_ref = config["protocol_source"]
    protocol = verified_json(
        ROOT / protocol_ref["configuration"],
        protocol_ref["configuration_sha256"],
    )
    _, reward = validate_support_protocol(protocol)
    _require_exact_keys(
        config["models"], ["SOURCE_STAGE1", "MATCHED_CONTACT_GAP_W0"]
    )
    if config["models"][0]["checkpoint"] != protocol["frozen_sources"][
        "source_checkpoint"
    ]:
        raise ValueError("SOURCE_STAGE1 path differs from the frozen protocol source")
    if config["models"][0]["checkpoint_sha256"] != protocol["frozen_sources"][
        "source_checkpoint_sha256"
    ]:
        raise ValueError("SOURCE_STAGE1 SHA differs from the frozen protocol source")
    for record in config["models"]:
        checkpoint = ROOT / record["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256(checkpoint) != str(record["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint SHA mismatch for {record['model_id']}")
        if bool(record["training_source"]):
            raise ValueError("This is a read-only diagnosis; no model is a training source")
    evaluation = config["evaluation"]
    if [float(value) for value in evaluation["speeds_m_per_s"]] != [
        0.2,
        0.3,
        0.4,
        0.55,
    ]:
        raise ValueError("Predeclared speed matrix changed")
    if float(evaluation["baseline_speed_m_per_s"]) != 0.55:
        raise ValueError("The paired baseline speed must be 0.55 m/s")
    if evaluation["scene_order"] != ["flat", "uphill_8deg", "downhill_8deg"]:
        raise ValueError("Speed diagnosis scene order changed")
    if [int(seed) for seed in evaluation["seeds"]] != [77801, 77802, 77803]:
        raise ValueError("Speed diagnosis seeds changed")
    if int(evaluation["max_episode_steps"]) != 600:
        raise ValueError("Speed diagnosis horizon changed")
    high_frequency = config["high_frequency_contact"]
    if high_frequency["scenes"] != ["flat", "uphill_8deg"]:
        raise ValueError("High-frequency audit scenes changed")
    plane = config["flat_plane_comparator"]
    if not bool(plane["enabled"]) or bool(plane["training_use"]):
        raise ValueError("Native-plane comparator must remain read-only and enabled")
    if plane["model_ids"] != ["SOURCE_STAGE1", "MATCHED_CONTACT_GAP_W0"]:
        raise ValueError("Native-plane comparator model identities changed")
    if float(plane["speed_m_per_s"]) != 0.55:
        raise ValueError("Native-plane comparator speed changed")
    if [int(seed) for seed in plane["seeds"]] != [77801, 77802, 77803]:
        raise ValueError("Native-plane comparator seeds changed")
    execution = config["execution"]
    if not bool(execution["fail_if_output_root_exists"]):
        raise ValueError("Formal output must fail closed if its root exists")
    for name in ("training_timesteps",):
        if int(execution[name]) != 0:
            raise ValueError("Speed diagnosis must not train")
    for name in ("energy_formula_changed", "reward_changed", "friction_changed"):
        if bool(execution[name]):
            raise ValueError(f"Read-only contract violated: {name}")
    if list(protocol["standard_scenes"]["fixed_friction"]) != [1.0, 0.5, 0.5]:
        raise ValueError("Frozen friction changed")
    return protocol, reward


def _condition_id(model_id: str, speed: float) -> str:
    return f"{model_id}_SPEED_{int(round(100.0 * speed)):03d}"


def _progress_denominator(config: dict[str, Any], speed: float) -> float:
    evaluation = config["evaluation"]
    commanded = (
        float(speed)
        * int(evaluation["max_episode_steps"])
        * float(evaluation["control_dt_seconds"])
    )
    return min(commanded, float(evaluation["start_to_goal_distance_m"]))


def _with_task_outcomes(
    aggregate: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    result = dict(aggregate)
    success_count = int(sum(bool(row["fixed_goal_success"]) for row in rows))
    result.update(
        {
            "success_count": success_count,
            "success_rate": success_count / len(rows),
            "fixed_goal_final_distance_m_mean": float(
                np.mean([float(row["fixed_goal_final_distance_m"]) for row in rows])
            ),
            "termination_category_counts": {
                str(category): int(
                    sum(str(row["termination_category"]) == str(category) for row in rows)
                )
                for category in sorted({str(row["termination_category"]) for row in rows})
            },
        }
    )
    return result


def prepare_native_plane_comparator(
    flat_scene: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    """Change only the ground collision backend; preserve the compiled robot."""
    scene_dir = output_root / "flat_plane_comparator" / "native_plane"
    scene_dir.mkdir(parents=True, exist_ok=False)
    xml_path = scene_dir / "ant_native_plane.xml"
    texture_path = scene_dir / Path(flat_scene["texture_path"]).name
    shutil.copy2(Path(flat_scene["texture_path"]), texture_path)
    tree = ET.parse(Path(flat_scene["xml_path"]))
    root = tree.getroot()
    asset = root.find("./asset")
    hfield = root.find("./asset/hfield[@name='terrain']")
    floor = root.find("./worldbody/geom[@name='floor']")
    if asset is None or hfield is None or floor is None:
        raise ValueError("Flat heightfield XML lacks the expected terrain elements")
    asset.remove(hfield)
    floor.attrib.pop("hfield", None)
    floor.set("type", "plane")
    floor.set("size", "20 20 1")
    floor.set("pos", "0 0 0")
    floor.set("friction", "1 0.5 0.5")
    floor.set("condim", "3")
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    flat_model = mujoco.MjModel.from_xml_path(str(Path(flat_scene["xml_path"])))
    plane_model = mujoco.MjModel.from_xml_path(str(xml_path))
    if robot_signature(plane_model) != robot_signature(flat_model):
        raise RuntimeError("Native-plane comparator changed the frozen robot")
    floor_id = mujoco.mj_name2id(
        plane_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if not np.array_equal(
        plane_model.geom_friction[floor_id], np.asarray([1.0, 0.5, 0.5])
    ):
        raise RuntimeError("Native-plane comparator friction changed")
    if int(plane_model.geom_condim[floor_id]) != 3:
        raise RuntimeError("Native-plane comparator condim changed")
    record = {
        **flat_scene,
        "scene_name": "native_plane",
        "xml_path": str(xml_path),
        "xml_sha256": sha256(xml_path),
        "collision_backend": "native_plane",
        "wrapper_height_query_source": "same numerical-flat height array",
        "robot_signature_matches_flat_heightfield": True,
        "fixed_friction": [1.0, 0.5, 0.5],
        "condim": 3,
    }
    write_json(scene_dir / "scene_manifest.json", record)
    return record


def summarise_flat_plane_comparator(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for model_id in config["flat_plane_comparator"]["model_ids"]:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        heightfield_rows = [
            row for row in model_rows if row["surface_backend"] == "flat_heightfield"
        ]
        plane_rows = [
            row for row in model_rows if row["surface_backend"] == "native_plane"
        ]
        heightfield = _with_task_outcomes(
            aggregate_rows(heightfield_rows), heightfield_rows
        )
        plane = _with_task_outcomes(aggregate_rows(plane_rows), plane_rows)
        records[model_id] = {
            "flat_heightfield": heightfield,
            "native_plane": plane,
            "plane_minus_heightfield": {
                "best_progress_m": float(
                    plane["fixed_goal_best_progress_m_mean"]
                    - heightfield["fixed_goal_best_progress_m_mean"]
                ),
                "endpoint_airborne_fraction": float(
                    plane["task_airborne_step_fraction_mean"]
                    - heightfield["task_airborne_step_fraction_mean"]
                ),
                "mean_support_count": float(
                    plane["mean_support_count_mean"]
                    - heightfield["mean_support_count_mean"]
                ),
                "endpoint_sampled_sustained_slip_fraction": float(
                    plane["corrected_sustained_slip_step_fraction_mean"]
                    - heightfield["corrected_sustained_slip_step_fraction_mean"]
                ),
                "falls": int(plane["fall_count"] - heightfield["fall_count"]),
            },
        }
    pooled_heightfield_rows = [
        row for row in rows if row["surface_backend"] == "flat_heightfield"
    ]
    pooled_plane_rows = [
        row for row in rows if row["surface_backend"] == "native_plane"
    ]
    pooled_heightfield = _with_task_outcomes(
        aggregate_rows(pooled_heightfield_rows), pooled_heightfield_rows
    )
    pooled_plane = _with_task_outcomes(
        aggregate_rows(pooled_plane_rows), pooled_plane_rows
    )
    pooled_delta = {
        "best_progress_m": float(
            pooled_plane["fixed_goal_best_progress_m_mean"]
            - pooled_heightfield["fixed_goal_best_progress_m_mean"]
        ),
        "endpoint_airborne_fraction": float(
            pooled_plane["task_airborne_step_fraction_mean"]
            - pooled_heightfield["task_airborne_step_fraction_mean"]
        ),
        "mean_support_count": float(
            pooled_plane["mean_support_count_mean"]
            - pooled_heightfield["mean_support_count_mean"]
        ),
        "endpoint_sampled_sustained_slip_fraction": float(
            pooled_plane["corrected_sustained_slip_step_fraction_mean"]
            - pooled_heightfield["corrected_sustained_slip_step_fraction_mean"]
        ),
        "falls": int(pooled_plane["fall_count"] - pooled_heightfield["fall_count"]),
    }
    thresholds = config["flat_plane_comparator"]["large_difference_thresholds"]
    large = bool(
        abs(pooled_delta["best_progress_m"])
        >= float(thresholds["absolute_best_progress_difference_m"])
        or abs(pooled_delta["endpoint_airborne_fraction"])
        >= float(thresholds["absolute_endpoint_airborne_fraction_difference"])
        or abs(pooled_delta["mean_support_count"])
        >= float(thresholds["absolute_mean_support_count_difference"])
        or pooled_delta["falls"] >= int(thresholds["additional_falls"])
    )
    return {
        "schema_version": "proxygap-flat-plane-heightfield-comparator-v1",
        "model_aggregates": records,
        "pooled": {
            "flat_heightfield": pooled_heightfield,
            "native_plane": pooled_plane,
            "plane_minus_heightfield": pooled_delta,
        },
        "predeclared_thresholds": thresholds,
        "large_backend_difference": large,
        "decision": (
            "A predeclared difference threshold was met; diagnose heightfield contact transfer before another reward scan."
            if large
            else "No pooled predeclared backend-difference threshold was met in this bounded comparator."
        ),
        "training_use": False,
    }


def summarise_speed_matrix(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    model_records: dict[str, dict[str, Any]] = {}
    pooled_records: dict[str, dict[str, Any]] = {}
    speeds = [float(value) for value in config["evaluation"]["speeds_m_per_s"]]
    baseline_speed = float(config["evaluation"]["baseline_speed_m_per_s"])
    for model in config["models"]:
        model_id = str(model["model_id"])
        model_records[model_id] = {}
        for speed in speeds:
            subset = [
                row
                for row in rows
                if row["model_id"] == model_id
                and np.isclose(float(row["speed_m_per_s"]), speed)
            ]
            aggregate = _with_task_outcomes(aggregate_rows(subset), subset)
            aggregate["progress_to_commanded_distance_ratio"] = float(
                aggregate["fixed_goal_best_progress_m_mean"]
                / _progress_denominator(config, speed)
            )
            model_records[model_id][f"{speed:.2f}"] = aggregate
    for speed in speeds:
        subset = [
            row
            for row in rows
            if np.isclose(float(row["speed_m_per_s"]), speed)
        ]
        aggregate = _with_task_outcomes(aggregate_rows(subset), subset)
        aggregate["progress_to_commanded_distance_ratio"] = float(
            aggregate["fixed_goal_best_progress_m_mean"]
            / _progress_denominator(config, speed)
        )
        pooled_records[f"{speed:.2f}"] = aggregate

    baseline = pooled_records[f"{baseline_speed:.2f}"]
    rule = config["selection_rule"]
    candidates: list[dict[str, Any]] = []
    for speed in speeds:
        if np.isclose(speed, baseline_speed):
            continue
        aggregate = pooled_records[f"{speed:.2f}"]
        airborne_reduction = float(
            baseline["task_airborne_step_fraction_mean"]
            - aggregate["task_airborne_step_fraction_mean"]
        )
        support_increase = float(
            aggregate["mean_support_count_mean"]
            - baseline["mean_support_count_mean"]
        )
        fall_delta = int(aggregate["fall_count"] - baseline["fall_count"])
        slip_delta = float(
            aggregate["corrected_sustained_slip_step_fraction_mean"]
            - baseline["corrected_sustained_slip_step_fraction_mean"]
        )
        nonstalled = bool(
            aggregate["fixed_goal_best_progress_m_mean"]
            >= float(rule["minimum_mean_best_progress_m"])
            and aggregate["progress_to_commanded_distance_ratio"]
            >= float(rule["minimum_progress_to_commanded_distance_ratio"])
        )
        safe = bool(
            fall_delta <= int(rule["maximum_additional_falls_vs_baseline"])
            and slip_delta
            <= float(
                rule["maximum_sustained_slip_fraction_increase_vs_baseline"]
            )
        )
        meaningful = bool(
            airborne_reduction
            >= float(
                rule[
                    "minimum_endpoint_airborne_fraction_reduction_vs_baseline"
                ]
            )
            or support_increase
            >= float(rule["minimum_mean_support_count_increase_vs_baseline"])
        )
        candidates.append(
            {
                "speed_m_per_s": speed,
                "mean_best_progress_m": float(
                    aggregate["fixed_goal_best_progress_m_mean"]
                ),
                "progress_to_commanded_distance_ratio": float(
                    aggregate["progress_to_commanded_distance_ratio"]
                ),
                "endpoint_airborne_fraction": float(
                    aggregate["task_airborne_step_fraction_mean"]
                ),
                "mean_support_count": float(aggregate["mean_support_count_mean"]),
                "endpoint_sampled_sustained_slip_fraction": float(
                    aggregate["corrected_sustained_slip_step_fraction_mean"]
                ),
                "endpoint_airborne_fraction_reduction_vs_055": airborne_reduction,
                "mean_support_count_increase_vs_055": support_increase,
                "additional_falls_vs_055": fall_delta,
                "endpoint_sampled_sustained_slip_fraction_increase_vs_055": slip_delta,
                "nonstalled": nonstalled,
                "safe_by_endpoint_metrics": safe,
                "meaningful_support_change": meaningful,
                "eligible": bool(nonstalled and safe and meaningful),
            }
        )
    eligible = [record for record in candidates if record["eligible"]]
    viable = [
        record
        for record in candidates
        if record["nonstalled"] and record["safe_by_endpoint_metrics"]
    ]
    audit_pool = eligible or viable or candidates
    selected = min(
        audit_pool,
        key=lambda record: (
            record["endpoint_airborne_fraction"],
            -record["mean_support_count"],
            -record["progress_to_commanded_distance_ratio"],
            -record["speed_m_per_s"],
        ),
    )
    return {
        "schema_version": "proxygap-fixed-standard-speed-ablation-summary-v1",
        "model_speed_aggregates": model_records,
        "pooled_speed_aggregates": pooled_records,
        "selection_rule": rule,
        "candidate_records": candidates,
        "selected_high_frequency_audit_speed_m_per_s": selected["speed_m_per_s"],
        "endpoint_selection_gate_passed": bool(eligible),
        "speed_reduction_recommended": False,
        "decision": (
            "A nonbaseline speed met the endpoint gate; final recommendation awaits the predeclared full-substep gate."
            if eligible
            else "No nonbaseline speed met the endpoint gate; the deterministic best viable speed is audited only and cannot be recommended."
        ),
        "measurement_boundary": (
            "The legacy field corrected_sustained_slip_step_fraction is sampled only at control-step endpoints. "
            "This report therefore labels it endpoint-sampled sustained slip; substep slip speed is not available."
        ),
    }


def apply_full_substep_gate(
    config: dict[str, Any],
    summary: dict[str, Any],
    high_frequency: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(summary)
    candidate = float(result["selected_high_frequency_audit_speed_m_per_s"])
    baseline = float(config["high_frequency_contact"]["paired_baseline_speed_m_per_s"])

    def pooled(speed: float, field: str) -> float:
        values = [
            float(record[field])
            for model_records in high_frequency.values()
            for record in model_records[f"{speed:.2f}"]
        ]
        return float(np.mean(values))

    baseline_full = pooled(baseline, "full_interval_zero_foot_fraction")
    candidate_full = pooled(candidate, "full_interval_zero_foot_fraction")
    reduction = baseline_full - candidate_full
    threshold = float(
        config["selection_rule"][
            "minimum_full_interval_zero_foot_fraction_reduction_vs_baseline"
        ]
    )
    substep_passed = bool(reduction >= threshold)
    recommended = bool(result["endpoint_selection_gate_passed"] and substep_passed)
    result["full_substep_gate"] = {
        "candidate_speed_m_per_s": candidate,
        "baseline_speed_m_per_s": baseline,
        "candidate_full_interval_zero_foot_fraction_mean": candidate_full,
        "baseline_full_interval_zero_foot_fraction_mean": baseline_full,
        "observed_reduction": reduction,
        "minimum_required_reduction": threshold,
        "passed": substep_passed,
    }
    result["speed_reduction_recommended"] = recommended
    if recommended:
        result["decision"] = (
            "The candidate met both the endpoint/non-stall gate and the paired full-substep support gate."
        )
    elif not result["endpoint_selection_gate_passed"]:
        result["decision"] = (
            "No nonbaseline speed met the endpoint/non-stall gate; substep evidence cannot promote it."
        )
    else:
        result["decision"] = (
            "The endpoint candidate failed the paired full-substep support gate and is not recommended."
        )
    return result


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "status_short_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _software_record() -> dict[str, Any]:
    packages = {}
    for name in ("gymnasium", "mujoco", "numpy", "stable-baselines3", "torch"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward = validate_ablation_config(config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                },
                indent=2,
            )
        )
        return

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (ROOT / config["execution"]["output_root"]).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(config["execution"]["torch_num_threads"]))
    frozen_config = output_root / "frozen_speed_ablation_config.json"
    frozen_config.write_bytes(config_path.read_bytes())
    frozen_protocol = output_root / "frozen_support_protocol.json"
    frozen_protocol.write_bytes(
        (ROOT / config["protocol_source"]["configuration"]).read_bytes()
    )
    scenes, _ = prepare_standard_scenes(protocol, output_root)
    native_plane = prepare_native_plane_comparator(scenes["flat"], output_root)
    eval_protocol = copy.deepcopy(protocol)
    eval_protocol["standard_scenes"]["scene_order"] = list(
        config["evaluation"]["scene_order"]
    )
    eval_protocol["diagnosis"]["high_frequency_contact_scenes"] = list(
        config["high_frequency_contact"]["scenes"]
    )

    all_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    models: dict[str, PPO] = {}
    for model_record in config["models"]:
        model_id = str(model_record["model_id"])
        model = PPO.load(
            ROOT / model_record["checkpoint"],
            device=str(config["execution"]["device"]),
        )
        models[model_id] = model
        for speed in config["evaluation"]["speeds_m_per_s"]:
            speed = float(speed)
            rows, traces = evaluate_matrix(
                model,
                eval_protocol,
                reward,
                scenes,
                condition_id=_condition_id(model_id, speed),
                seeds=[int(seed) for seed in config["evaluation"]["seeds"]],
                max_episode_steps=int(config["evaluation"]["max_episode_steps"]),
                cruise_speed=speed,
                output_root=output_root,
                trace_seed=int(config["evaluation"]["representative_trace_seed"]),
            )
            for row in rows:
                row["model_id"] = model_id
                row["speed_m_per_s"] = speed
                row["endpoint_sampled_sustained_slip_step_fraction"] = row[
                    "corrected_sustained_slip_step_fraction"
                ]
            all_rows.extend(rows)
            trace_records.extend(traces)
    write_rows(output_root / "logs" / "speed_matrix_episodes.csv", all_rows)
    summary = summarise_speed_matrix(config, all_rows)

    candidate_speed = float(summary["selected_high_frequency_audit_speed_m_per_s"])
    baseline_speed = float(config["high_frequency_contact"]["paired_baseline_speed_m_per_s"])
    high_frequency_records: dict[str, Any] = {}
    for model_id, model in models.items():
        high_frequency_records[model_id] = {}
        for speed in dict.fromkeys((candidate_speed, baseline_speed)):
            records = high_frequency_contact_matrix(
                model,
                eval_protocol,
                reward,
                scenes,
                condition_id=f"{_condition_id(model_id, speed)}_SUBSTEP",
                seed=int(config["high_frequency_contact"]["seed"]),
                max_episode_steps=int(
                    config["high_frequency_contact"]["control_steps"]
                ),
                cruise_speed=speed,
                output_root=output_root,
            )
            high_frequency_records[model_id][f"{speed:.2f}"] = records
    write_json(output_root / "high_frequency_contact" / "paired_summary.json", high_frequency_records)
    summary = apply_full_substep_gate(config, summary, high_frequency_records)
    write_json(output_root / "speed_ablation_summary.json", summary)

    plane_rows: list[dict[str, Any]] = []
    plane_trace_seed = int(config["evaluation"]["representative_trace_seed"])
    plane_speed = float(config["flat_plane_comparator"]["speed_m_per_s"])
    for model_id, model in models.items():
        for seed in config["flat_plane_comparator"]["seeds"]:
            trace_path = (
                output_root
                / "traces"
                / f"{model_id.lower()}_native_plane_speed_055_seed_{seed}_trace.csv"
                if int(seed) == plane_trace_seed
                else None
            )
            row, _ = evaluate_episode(
                model,
                eval_protocol,
                reward,
                native_plane,
                condition_id=f"{model_id}_NATIVE_PLANE_SPEED_055",
                seed=int(seed),
                max_episode_steps=int(
                    config["flat_plane_comparator"]["max_episode_steps"]
                ),
                cruise_speed=plane_speed,
                trace_path=trace_path,
            )
            row["model_id"] = model_id
            row["speed_m_per_s"] = plane_speed
            row["surface_backend"] = "native_plane"
            row["endpoint_sampled_sustained_slip_step_fraction"] = row[
                "corrected_sustained_slip_step_fraction"
            ]
            plane_rows.append(row)
            if trace_path is not None:
                trace_records.append(
                    {
                        "condition_id": row["condition_id"],
                        "scene_name": "native_plane",
                        "evaluation_seed": int(seed),
                        "path": str(trace_path),
                        "sha256": sha256(trace_path),
                        "rows": int(row["episode_length"]),
                    }
                )
    heightfield_flat_rows: list[dict[str, Any]] = []
    for row in all_rows:
        if row["scene_name"] == "flat" and np.isclose(
            float(row["speed_m_per_s"]), plane_speed
        ):
            copied = dict(row)
            copied["surface_backend"] = "flat_heightfield"
            heightfield_flat_rows.append(copied)
    plane_comparator_rows = heightfield_flat_rows + plane_rows
    write_rows(
        output_root / "logs" / "flat_plane_comparator_episodes.csv",
        plane_comparator_rows,
    )
    plane_summary = summarise_flat_plane_comparator(config, plane_comparator_rows)

    plane_substeps: dict[str, Any] = {}
    for model_id, model in models.items():
        _, substep = evaluate_episode(
            model,
            eval_protocol,
            reward,
            native_plane,
            condition_id=f"{model_id}_NATIVE_PLANE_SUBSTEP",
            seed=int(config["flat_plane_comparator"]["substep_seed"]),
            max_episode_steps=int(
                config["flat_plane_comparator"]["substep_control_steps"]
            ),
            cruise_speed=plane_speed,
            high_frequency_contact=True,
        )
        assert substep is not None
        trace_path = (
            output_root
            / "flat_plane_comparator"
            / f"{model_id.lower()}_native_plane_substeps.csv"
        )
        write_rows(trace_path, substep["rows"])
        heightfield_substep = next(
            record
            for record in high_frequency_records[model_id][f"{plane_speed:.2f}"]
            if record["scene_name"] == "flat"
        )
        plane_substeps[model_id] = {
            "flat_heightfield": heightfield_substep,
            "native_plane": {
                **substep["summary"],
                "trace_path": str(trace_path),
                "trace_sha256": sha256(trace_path),
            },
            "plane_minus_heightfield_full_interval_zero_foot_fraction": float(
                substep["summary"]["full_interval_zero_foot_fraction"]
                - heightfield_substep["full_interval_zero_foot_fraction"]
            ),
        }
    pooled_substep_delta = float(
        np.mean(
            [
                record[
                    "plane_minus_heightfield_full_interval_zero_foot_fraction"
                ]
                for record in plane_substeps.values()
            ]
        )
    )
    plane_summary["high_frequency_contact"] = plane_substeps
    plane_summary["pooled_plane_minus_heightfield_full_interval_zero_foot_fraction"] = (
        pooled_substep_delta
    )
    plane_substep_large = bool(
        abs(pooled_substep_delta)
        >= float(
            config["flat_plane_comparator"]["large_difference_thresholds"][
                "absolute_full_interval_zero_foot_fraction_difference"
            ]
        )
    )
    plane_summary["large_backend_difference"] = bool(
        plane_summary["large_backend_difference"] or plane_substep_large
    )
    if plane_summary["large_backend_difference"]:
        plane_summary["decision"] = (
            "A predeclared endpoint or full-substep difference threshold was met; diagnose heightfield contact transfer before another reward scan."
        )
    write_json(
        output_root / "flat_plane_comparator" / "comparison_summary.json",
        plane_summary,
    )

    code_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_fixed_standard_support_curriculum.py",
        ROOT / "src" / "proxygap" / "fixed_goal_terrain.py",
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
    ]
    manifest = {
        "schema_version": "proxygap-fixed-standard-speed-ablation-manifest-v1",
        "decision": summary["decision"],
        "speed_reduction_recommended": summary["speed_reduction_recommended"],
        "selected_high_frequency_audit_speed_m_per_s": candidate_speed,
        "configuration": {"path": str(config_path), "sha256": sha256(config_path)},
        "frozen_configuration": {
            "path": str(frozen_config),
            "sha256": sha256(frozen_config),
        },
        "frozen_support_protocol": {
            "path": str(frozen_protocol),
            "sha256": sha256(frozen_protocol),
        },
        "models": [
            {
                **record,
                "observed_sha256": sha256(ROOT / record["checkpoint"]),
            }
            for record in config["models"]
        ],
        "code": [
            {"path": str(path), "sha256": sha256(path)} for path in code_paths
        ],
        "git": _git_record(),
        "software": _software_record(),
        "standard_scene_manifest": {
            "path": str(output_root / "standard_scene_manifest.json"),
            "sha256": sha256(output_root / "standard_scene_manifest.json"),
        },
        "episode_log": {
            "path": str(output_root / "logs" / "speed_matrix_episodes.csv"),
            "sha256": sha256(output_root / "logs" / "speed_matrix_episodes.csv"),
        },
        "summary": {
            "path": str(output_root / "speed_ablation_summary.json"),
            "sha256": sha256(output_root / "speed_ablation_summary.json"),
        },
        "high_frequency_contact": {
            "path": str(output_root / "high_frequency_contact" / "paired_summary.json"),
            "sha256": sha256(
                output_root / "high_frequency_contact" / "paired_summary.json"
            ),
        },
        "flat_plane_comparator": {
            "path": str(
                output_root / "flat_plane_comparator" / "comparison_summary.json"
            ),
            "sha256": sha256(
                output_root / "flat_plane_comparator" / "comparison_summary.json"
            ),
            "large_backend_difference": plane_summary["large_backend_difference"],
            "training_use": False,
        },
        "representative_video_inputs": {
            "selection": "all model-speed-scene traces for the predeclared middle seed; no visual cherry-picking",
            "trace_seed": int(config["evaluation"]["representative_trace_seed"]),
            "traces": trace_records,
            "video_rendered": False,
        },
        "training_timesteps": 0,
        "existing_results_overwritten": False,
        "energy_formula_changed": False,
        "reward_changed": False,
        "friction_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "speed_ablation_complete",
                "output_root": str(output_root),
                "recommended": summary["speed_reduction_recommended"],
                "candidate_speed_m_per_s": candidate_speed,
                "large_flat_backend_difference": plane_summary[
                    "large_backend_difference"
                ],
                "summary_sha256": sha256(output_root / "speed_ablation_summary.json"),
                "manifest_sha256": sha256(output_root / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
