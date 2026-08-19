"""Run the final frozen multi-objective V4+PAIR0 full-map evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import stable_baselines3
import torch

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_map_waypoint_route as route_eval  # noqa: E402
import evaluate_post_seal_full_map_v1 as full_map  # noqa: E402
import evaluate_v4_pair0_multiobjective_routes_engineering as engineering  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/v4_pair0_multiobjective_full_map_final_v1_20260820.json"
SELF = "scripts/run_v4_pair0_multiobjective_full_map_final.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_route(path: Path) -> route_eval.Polyline:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return route_eval.Polyline(np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float64))


def validate(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "proxygap-v4-pair0-multiobjective-full-map-final-v1":
        raise ValueError("schema changed")
    source = config["source"]
    for key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("post_seal_evaluation_config", "post_seal_evaluation_config_sha256"),
        ("fixed_map_config", "fixed_map_config_sha256"),
        ("pair0_scene", "pair0_scene_sha256"),
        ("heights", "heights_sha256"),
        ("hfield", "hfield_sha256"),
        ("approved_xml", "approved_xml_sha256"),
    ):
        if sha256(ROOT / source[key]) != source[hash_key]:
            raise ValueError(f"source hash changed: {key}")
    selection = config["candidate_selection"]
    if sha256(ROOT / selection["selection"]) != selection["selection_sha256"]:
        raise ValueError("selection hash changed")
    if selection["formula"] != "w_time*(T/T_min)+w_energy*(W_positive/W_positive_min)":
        raise ValueError("selection formula changed")
    contracts = config["route_contracts"]
    for contract in contracts.values():
        if sha256(ROOT / contract["route"]) != contract["route_sha256"]:
            raise ValueError("route hash changed")
        if float(contract["cruise_speed_m_per_s"]) != 0.5 or float(contract["minimum_slope_speed_m_per_s"]) != 0.28:
            raise ValueError("selected speed contract changed")
    evaluation = config["evaluation"]
    checkpoint_hash = source["checkpoint_sha256"]
    height_hash = source["heights_sha256"]
    expected_seeds = []
    for replicate in range(1, 4):
        material = f"{checkpoint_hash}|{height_hash}|v4_pair0_multiobjective_final_v1|replicate={replicate}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16) % (2**31 - 1)
        record = evaluation["seed_derivations"][replicate - 1]
        if record != {"replicate": replicate, "sha256": digest, "seed": seed}:
            raise ValueError("seed derivation changed")
        expected_seeds.append(seed)
    if evaluation["formal_seeds"] != expected_seeds:
        raise ValueError("formal seeds changed")
    if int(evaluation["horizon_control_steps"]) != 12000 or float(evaluation["lookahead_m"]) != 3.0:
        raise ValueError("evaluation horizon/lookahead changed")
    success = config["success_and_safety"]
    if [success["arrival_radius_m"], success["hold_radius_m"], success["hold_seconds"]] != [1.5, 2.0, 2.0]:
        raise ValueError("arrival contract changed")
    if any(int(success[key]) != 0 for key in ("maximum_fall_count", "maximum_torso_ground_count", "maximum_sustained_nonfoot_contact_count", "maximum_duration_corrected_slip_event_count")):
        raise ValueError("safety gate changed")
    if config["execution"] != {
        "training_permitted": False,
        "checkpoint_write_permitted": False,
        "fixed_map_source_mutation_permitted": False,
        "formal_output_root": "artifacts/dev/v4_pair0_multiobjective_full_map_final_v1_20260820/attempt_0",
        "fail_if_output_root_exists": True,
    }:
        raise ValueError("execution contract changed")


def runtime_paths() -> tuple[str, ...]:
    paths = set(full_map.loaded_project_runtime_paths())
    paths.update({SELF, "scripts/evaluate_v4_pair0_multiobjective_routes_engineering.py", "scripts/evaluate_fixed_map_waypoint_route.py"})
    return tuple(sorted(paths))


def runtime_map(paths: tuple[str, ...]) -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in paths}


def snapshot_runtime(paths: tuple[str, ...], root: Path) -> dict[str, str]:
    snapshot = root / "runtime_snapshot"
    for relative in paths:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    actual = tuple(sorted(path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()))
    if actual != paths:
        raise RuntimeError("runtime snapshot membership changed")
    return {path: sha256(snapshot / path) for path in paths}


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.run:
        raise ValueError("choose exactly one mode")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate(config)
    if args.validate_only:
        print("VALIDATION_OK_NO_EVALUATION")
        return
    output = ROOT / config["execution"]["formal_output_root"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    stage = "attempt_root_created"
    started = time.perf_counter()
    try:
        stage = "freeze_inputs"
        shutil.copy2(DEFAULT_CONFIG, output / "frozen_config.json")
        paths = runtime_paths()
        before = runtime_map(paths)
        snapshot = snapshot_runtime(paths, output)
        if before != snapshot:
            raise RuntimeError("live/snapshot runtime mismatch")
        canonical = json.loads((ROOT / config["source"]["post_seal_evaluation_config"]).read_text(encoding="utf-8"))
        fixed = json.loads((ROOT / config["source"]["fixed_map_config"]).read_text(encoding="utf-8"))
        heights = np.load(ROOT / config["source"]["heights"], allow_pickle=False)
        spacing = 2.0 * float(fixed["approved_map"]["map_half_extent_m"]) / (heights.shape[0] - 1)
        gradient_y, gradient_x = np.gradient(heights, spacing, spacing)
        all_results = []
        stage = "evaluate_two_contracts_three_seeds"
        for contract_id, contract in config["route_contracts"].items():
            route = load_route(ROOT / contract["route"])
            for seed in config["evaluation"]["formal_seeds"]:
                regime = {"id": contract_id, "weights": None, "speed": 0.5, "minimum_speed": 0.28}
                result, controls, substeps = engineering.evaluate_route(
                    canonical_config=canonical, fixed=fixed, route=route, regime=regime,
                    heights=heights, gradient_x=gradient_x, gradient_y=gradient_y, seed=int(seed),
                )
                result["evaluation_seed"] = int(seed)
                result["route_sha256"] = contract["route_sha256"]
                all_results.append(result)
                root = output / contract_id / f"seed_{seed}"
                engineering.write_csv(root / "control_trace.csv", controls)
                engineering.write_csv(root / "substep_trace.csv", substeps)
                write_json(root / "result.json", result)
        stage = "aggregate_and_gate"
        contract_summary = {}
        for contract_id in config["route_contracts"]:
            rows = [row for row in all_results if row["regime"] == contract_id]
            contract_summary[contract_id] = {
                "episode_count": len(rows),
                "success_count": sum(bool(row["safety_qualified_completion"]) for row in rows),
                "all_passed": all(bool(row["safety_qualified_completion"]) for row in rows),
                "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
                "mean_positive_mechanical_work_j": float(np.mean([row["actuator_positive_mechanical_work_total_j"] for row in rows])),
                "mean_path_length_m": float(np.mean([row["path_length_m"] for row in rows])),
                "total_duration_corrected_slip_events": int(sum(row["duration_corrected_slip_event_count"] for row in rows)),
                "fall_count": int(sum(bool(row["fall"]) for row in rows)),
            }
        objectives = {}
        for objective, declaration in config["candidate_selection"]["objectives"].items():
            summary = contract_summary[declaration["contract_id"]]
            objectives[objective] = {**declaration, **summary, "passed": bool(summary["all_passed"])}
        all_passed = all(record["passed"] for record in objectives.values())
        summary = {
            "status": "final_multiobjective_evaluation_complete",
            "all_objectives_passed": all_passed,
            "unique_contract_count": 2,
            "formal_seed_count": 3,
            "episode_count": len(all_results),
            "objectives": objectives,
            "contract_summary": contract_summary,
            "claim_boundary": config["claim_boundary"],
        }
        write_json(output / "summary.json", summary)
        report_lines = [
            "# V4 + PAIR0 multi-objective full-map final evaluation",
            "",
            f"All objectives passed: **{all_passed}**. Two unique route contracts were evaluated over three fresh seeds (six episodes).",
            "",
            "| Objective | Weights (time, energy) | Contract | Success | Mean time (s) | Mean positive work (J, proxy) | Slip events | Falls |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in objectives.items():
            report_lines.append(f"| {name} | {row['weights_time_energy']} | {row['contract_id']} | {row['success_count']}/{row['episode_count']} | {row['mean_elapsed_seconds']:.3f} | {row['mean_positive_mechanical_work_j']:.3f} | {row['total_duration_corrected_slip_events']} | {row['fall_count']} |")
        report_lines.extend(["", "Energy is a mechanical-work proxy, not electrical battery energy. Results concern one frozen, previously inspected map."])
        (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        stage = "verify_provenance"
        after = runtime_map(paths)
        if after != before:
            raise RuntimeError("runtime changed during final evaluation")
        for key, hash_key in (("checkpoint", "checkpoint_sha256"), ("heights", "heights_sha256"), ("hfield", "hfield_sha256"), ("approved_xml", "approved_xml_sha256"), ("pair0_scene", "pair0_scene_sha256")):
            if sha256(ROOT / config["source"][key]) != config["source"][hash_key]:
                raise RuntimeError(f"source changed during evaluation: {key}")
        stage = "write_manifest"
        manifest = {
            "schema_version": "proxygap-v4-pair0-multiobjective-final-artifact-v1",
            "status": "complete" if all_passed else "complete_scientific_gate_failed",
            "configuration_sha256": sha256(DEFAULT_CONFIG),
            "runtime_before": before,
            "runtime_snapshot": snapshot,
            "runtime_after": after,
            "checkpoint_sha256_before_after": config["source"]["checkpoint_sha256"],
            "training_performed": False,
            "checkpoint_written": False,
            "map_source_modified": False,
            "all_objectives_passed": all_passed,
            "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__, "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "numpy": np.__version__},
            "elapsed_wall_seconds": time.perf_counter() - started,
            "artifact_inventory_excludes_manifest": inventory(output),
        }
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "all_objectives_passed": all_passed, "summary": objectives}, ensure_ascii=False))
    except BaseException as error:
        write_json(output / "FAILURE_RECORD.json", {
            "status": "failed_closed_non_evaluable",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "training_performed": False,
            "retry_permitted": False,
        })
        raise


if __name__ == "__main__":
    main()
