"""Read-only standard-slope capability boundary evaluation for final V3 PAIR0.

The runner loads one frozen checkpoint and never trains or saves a policy.  It
evaluates flat and signed planar heightfields with the frozen explicit PAIR0
contact contract, retaining all failures and applying predeclared safety and
progress gates.  Reported bounds are conditional on the tested grid only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_fixed_standard_pair0_adaptation_l2b_extension as l2b  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402
from evaluate_fixed_standard_distal_margin0_paired import (  # noqa: E402
    CANDIDATE_ID,
    prepare_pair,
)
from run_fixed_standard_support_curriculum import (  # noqa: E402
    build_standard_heights,
    prepare_standard_scenes,
    sha256,
    write_json,
    write_rows,
)


DEFAULT_CONFIG = ROOT / "configs" / "fixed_standard_pair0_slope_capability_boundary_v1_20260819.json"
PAIR0_ID = "PAIR0_ADAPT"
EXPECTED_SEEDS = (94131, 94137, 94151, 94153, 94169)
EXPECTED_ANGLES = (4, 8, 12, 16, 20)
ENERGY_KEYS = (
    "cumulative_squared_action",
    "actuator_abs_torque_time_integral_total_n_m_s",
    "actuator_positive_mechanical_work_total_j",
    "actuator_abs_mechanical_work_total_j",
)
RUNTIME_SELF_RELATIVE_PATH = "scripts/evaluate_fixed_standard_pair0_slope_capability_boundary.py"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF_RELATIVE_PATH,
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
FROZEN_NORMALISED_RUNTIME_CONTRACT_SHA256 = (
    "e88582dbbf2b46060fedf1f8bd12e1f80949d33fbf12b837957a79f2b250d014"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser.parse_args()


def _equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen field changed: {label}: {observed!r} != {expected!r}")


def _exact_array(observed: Any, expected: Any, label: str) -> None:
    if not np.array_equal(
        np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    ):
        raise ValueError(f"Frozen array changed: {label}")


def verify_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != str(record[hash_key]):
        raise ValueError(f"SHA-256 changed for {path_key}: {observed}")
    return path


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scene_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [{"scene_name": "flat", "direction": "flat", "angle_degrees": 0}]
    for direction in ("uphill", "downhill"):
        for angle in config["evaluation"]["slope_angles_degrees"]:
            specs.append(
                {
                    "scene_name": f"{direction}_{int(angle)}deg",
                    "direction": direction,
                    "angle_degrees": int(angle),
                }
            )
    return specs


def validate_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    contract = config["runtime_dependency_contract"]
    _equal(
        tuple(contract),
        (
            "copy_preserving_relative_paths",
            "verify_before_and_after",
            "exact_relative_path_sha256",
        ),
        "runtime contract keys/order",
    )
    expected = contract["exact_relative_path_sha256"]
    _equal(tuple(expected), EXPECTED_RUNTIME_PATHS, "runtime exact membership/order")
    normalised = copy.deepcopy(contract)
    normalised["exact_relative_path_sha256"][RUNTIME_SELF_RELATIVE_PATH] = (
        "<RUNNER_SELF_SHA256>"
    )
    _equal(
        _canonical_json_sha256(normalised),
        FROZEN_NORMALISED_RUNTIME_CONTRACT_SHA256,
        "normalised runtime contract digest",
    )
    observed: dict[str, str] = {}
    for relative_path, digest in expected.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"Runtime dependency changed: {relative_path}")
        observed[relative_path] = actual
    return observed


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _equal(config.get("schema_version"), "proxygap-pair0-slope-capability-boundary-v1", "schema")
    _equal(config.get("config_id"), "fixed_standard_pair0_slope_capability_boundary_v1_20260819", "config id")
    _equal(config.get("status"), "frozen_read_only_final_pair0_standard_slope_boundary", "status")
    source = config["source"]
    v3_config_path = verify_file(source, "v3_configuration", "v3_configuration_sha256")
    verify_file(source, "v3_frozen_configuration", "v3_frozen_configuration_sha256")
    verify_file(source, "v3_manifest", "v3_manifest_sha256")
    gate_path = verify_file(source, "v3_final_gate", "v3_final_gate_sha256")
    checkpoint = verify_file(source, "checkpoint", "checkpoint_sha256")
    _equal(int(source["checkpoint_timesteps"]), 2_727_936, "declared checkpoint timesteps")
    _equal(int(source["checkpoint_additional_timesteps"]), 65_536, "declared additional timesteps")
    _equal(int(source["observation_dimension"]), 135, "declared observation dimension")
    _equal(int(source["action_dimension"]), 8, "declared action dimension")
    v3_config = json.loads(v3_config_path.read_text(encoding="utf-8"))
    protocol, reward = l2b.validate_config(v3_config)
    v3_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if v3_gate.get("passed") is not True or int(v3_gate.get("absolute_final_checkpoint", -1)) != 2_727_936:
        raise ValueError("The source V3 final gate/checkpoint contract is not satisfied")
    model = PPO.load(checkpoint, device="cpu")
    _equal(int(model.num_timesteps), 2_727_936, "checkpoint timesteps")
    _equal(tuple(model.observation_space.shape), (135,), "observation shape")
    _equal(tuple(model.action_space.shape), (8,), "action shape")

    evaluation = config["evaluation"]
    _equal(tuple(evaluation["heldout_seeds"]), EXPECTED_SEEDS, "held-out seeds")
    if set(EXPECTED_SEEDS) & set(v3_config["evaluation"]["intermediate_safety_audit"]["seeds"]):
        raise ValueError("Boundary seeds overlap V3 intermediate seeds")
    if set(EXPECTED_SEEDS) & set(v3_config["evaluation"]["final_heldout"]["seeds"]):
        raise ValueError("Boundary seeds overlap V3 final held-out seeds")
    _equal(tuple(evaluation["slope_angles_degrees"]), EXPECTED_ANGLES, "angle grid")
    _equal(
        evaluation["scene_order"],
        [row["scene_name"] for row in scene_specs(config)],
        "scene order",
    )
    _equal(evaluation["seed_status"], "predeclared_new_heldout_not_828xx_or_838xx", "seed status")
    _equal(int(evaluation["max_episode_steps"]), 600, "horizon")
    _equal(float(evaluation["cruise_speed_m_per_s"]), 0.55, "speed")
    _equal(bool(evaluation["deterministic_policy"]), True, "deterministic policy")
    _equal(int(evaluation["physics_substeps_per_control_step"]), 5, "substeps")
    _equal(float(evaluation["physics_timestep_seconds"]), 0.01, "physics timestep")
    _equal(bool(evaluation["all_five_physics_substeps_required"]), True, "all substeps required")
    _equal(evaluation["corrected_slip"], v3_config["evaluation"]["corrected_slip"], "slip definition")

    _equal(config["contact_contract"], v3_config["contact_contract"], "PAIR0 contact contract")
    gates = config["gates"]
    _equal(float(gates["maximum_full_interval_zero_foot_fraction"]), 0.0580555556, "zero-foot gate")
    _equal(float(gates["minimum_uphill_mean_best_progress_m"]), 6.1857992362, "uphill gate")
    _equal(float(gates["minimum_downhill_mean_best_progress_m"]), 8.8113570803, "downhill gate")
    for key in (
        "maximum_fall_count",
        "maximum_torso_ground_episode_count",
        "maximum_sustained_nonfoot_contact_episode_count",
        "maximum_corrected_sustained_slip_substep_count",
        "maximum_corrected_slip_event_count",
    ):
        _equal(int(gates[key]), 0, key)
    _equal(bool(gates["force_qualified_denominator_required"]), True, "denominator gate")
    _equal(float(gates["nonfoot_minimum_sustained_seconds"]), 0.2, "nonfoot duration")
    _equal(gates["flat_progress_gate"], "not_applicable_safety_reference_only", "flat gate")
    _equal(bool(gates["required_all_checks"]), True, "all gate checks")

    _equal(
        config["boundary_inference"],
        {
            "ordered_grid_degrees": [4, 8, 12, 16, 20],
            "lower_bound_requires_all_tested_angles_at_or_below_to_pass": True,
            "first_failure_is_tested_upper_bracket_only": True,
            "nonmonotonic_pass_after_failure_is_reported_not_repaired": True,
            "interpolation": False,
            "physical_maximum_claim_permitted": False,
        },
        "boundary inference",
    )

    invariants = config["invariants"]
    expected_invariants = {
        "training_performed": False,
        "checkpoint_modified": False,
        "reward_changed": False,
        "friction_changed": False,
        "energy_formula_changed": False,
        "energy_status": "measurement_only_not_reward_or_gate",
        "energy_reward_weight": 0.0,
        "fixed_map_evaluated": False,
        "video_rendered": False,
        "candidate_promoted": False,
        "observation_dimension": 135,
        "action_dimension": 8,
        "explicit_pair_count": 4,
    }
    _equal(invariants, expected_invariants, "invariants")
    _equal(tuple(config["energy_measurement"]["raw_components_required"]), ENERGY_KEYS, "energy fields")
    _equal(config["energy_measurement"]["status"], "measurement_only_not_reward_or_gate", "energy status")
    _equal(bool(config["energy_measurement"]["nonfinite_is_failure"]), True, "energy finiteness")
    _equal(bool(config["energy_measurement"]["electrical_battery_energy_claim_permitted"]), False, "battery-energy claim")
    _equal(config["checkpoint_early_stopping"], {"nonfoot_contact_minimum_sustained_seconds": 0.2}, "nonfoot support contract")
    execution = config["execution"]
    _equal(bool(execution["fail_if_output_root_exists"]), True, "overwrite refusal")
    _equal(execution["formal_output_root"], "artifacts/dev/fixed_standard_pair0_slope_capability_boundary_v1_20260819/attempt_0", "output root")
    for key in ("training", "checkpoint_write", "fixed_map", "video", "promotion"):
        _equal(bool(execution[key]), False, f"execution {key}")
    runtime = config["runtime_dependency_contract"]
    _equal(bool(runtime["copy_preserving_relative_paths"]), True, "runtime copy")
    _equal(bool(runtime["verify_before_and_after"]), True, "runtime verification")
    _equal(bool(config["boundary_inference"]["physical_maximum_claim_permitted"]), False, "claim boundary")
    validate_runtime_dependencies(config)
    return protocol, reward, checkpoint


def snapshot_runtime(config: dict[str, Any], output_root: Path) -> dict[str, str]:
    observed = validate_runtime_dependencies(config)
    snapshot_root = output_root / "runtime_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for relative_path, digest in observed.items():
        target = snapshot_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
        if sha256(target) != digest:
            raise RuntimeError(f"Runtime snapshot changed: {relative_path}")
    return observed


def validate_runtime_snapshot(
    config: dict[str, Any], snapshot_root: Path
) -> dict[str, str]:
    actual_paths = tuple(
        sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file()
        )
    )
    if actual_paths != tuple(sorted(EXPECTED_RUNTIME_PATHS)):
        raise RuntimeError("Runtime snapshot exact membership changed")
    expected = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    observed: dict[str, str] = {}
    for relative_path in EXPECTED_RUNTIME_PATHS:
        path = snapshot_root / relative_path
        actual = sha256(path)
        if actual != expected[relative_path]:
            raise RuntimeError(f"Runtime snapshot changed: {relative_path}")
        observed[relative_path] = actual
    return observed


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


def prepare_scenes(
    config: dict[str, Any], protocol: dict[str, Any], output_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    for spec in scene_specs(config):
        local_protocol = copy.deepcopy(protocol)
        standard = local_protocol["standard_scenes"]
        if spec["direction"] == "flat":
            generator_name = "flat"
        elif spec["direction"] == "uphill":
            generator_name = "uphill_8deg"
            standard["uphill_slope_degrees"] = float(spec["angle_degrees"])
        else:
            generator_name = "downhill_8deg"
            standard["downhill_slope_degrees"] = -float(spec["angle_degrees"])
        standard["scene_order"] = [generator_name]
        controls, generation = prepare_standard_scenes(
            local_protocol, output_root / "scene_sources" / spec["scene_name"]
        )
        source = dict(controls[generator_name])
        heights = np.load(source["heights_path"], allow_pickle=False)
        expected = build_standard_heights(standard)[generator_name]
        if not np.array_equal(heights, expected):
            raise RuntimeError(f"Analytic height recipe changed: {spec['scene_name']}")
        source["generator_scene_name"] = generator_name
        source["scene_name"] = spec["scene_name"]
        source["direction"] = spec["direction"]
        source["angle_degrees"] = spec["angle_degrees"]
        pair, audit = prepare_pair(
            source,
            output_root / "condition_assets",
            spec["scene_name"],
            _pair_contract(config),
        )
        candidate = dict(pair[CANDIDATE_ID])
        candidate["condition_id"] = PAIR0_ID
        candidate["scene_name"] = spec["scene_name"]
        candidate["direction"] = spec["direction"]
        candidate["angle_degrees"] = spec["angle_degrees"]
        model = mujoco.MjModel.from_xml_path(candidate["xml_path"])
        l2.compiled_contract_audit(
            model, candidate, PAIR0_ID, config, construction_seed=EXPECTED_SEEDS[0]
        )
        scenes[spec["scene_name"]] = candidate
        audits[spec["scene_name"]] = {
            "spec": spec,
            "generation": generation,
            "explicit_pair_audit": audit,
            "candidate_xml_sha256": candidate["xml_sha256"],
            "analytic_height_match": True,
        }
    write_json(output_root / "prepared_scenes.json", {name: scenes[name] for name in scenes})
    write_json(output_root / "scene_contract_audits.json", audits)
    return scenes, audits


def energy_is_finite(row: dict[str, Any]) -> bool:
    return all(math.isfinite(float(row[key])) for key in ENERGY_KEYS)


def gate_scene(config: dict[str, Any], spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = l2.aggregate_episode_rows(rows)
    sustained_count = sum(int(row["corrected_sustained_slip_physics_substep_count"]) for row in rows)
    energy_finite = all(energy_is_finite(row) for row in rows)
    gates = config["gates"]
    checks = {
        "five_complete_episodes": len(rows) == 5
        and aggregate["total_control_steps"] == 3000
        and aggregate["total_physics_substeps"] == 15000,
        "finite_state_and_reward": aggregate["nonfinite_episode_count"] == 0,
        "finite_energy_measurements": energy_finite,
        "no_falls": aggregate["fall_count"] <= int(gates["maximum_fall_count"]),
        "no_torso_ground": aggregate["torso_ground_episode_count"] <= int(gates["maximum_torso_ground_episode_count"]),
        "no_sustained_nonfoot": aggregate["sustained_nonfoot_contact_episode_count"] <= int(gates["maximum_sustained_nonfoot_contact_episode_count"]),
        "force_qualified_denominator_evaluable": bool(aggregate["force_qualified_slip_evaluable"])
        and int(aggregate["force_qualified_supported_physics_substep_count"]) > 0
        and all(int(row["force_qualified_supported_physics_substep_count"]) > 0 for row in rows),
        "zero_corrected_sustained_slip": sustained_count <= int(gates["maximum_corrected_sustained_slip_substep_count"]),
        "zero_corrected_slip_events": aggregate["corrected_slip_event_count"] <= int(gates["maximum_corrected_slip_event_count"]),
        "zero_foot_within_gate": aggregate["pooled_full_interval_zero_foot_fraction"] <= float(gates["maximum_full_interval_zero_foot_fraction"]),
    }
    if spec["direction"] == "uphill":
        progress_threshold = float(gates["minimum_uphill_mean_best_progress_m"])
        checks["effective_progress"] = aggregate["mean_best_progress_m"] >= progress_threshold
    elif spec["direction"] == "downhill":
        progress_threshold = float(gates["minimum_downhill_mean_best_progress_m"])
        checks["effective_progress"] = aggregate["mean_best_progress_m"] >= progress_threshold
    else:
        progress_threshold = None
        checks["effective_progress"] = True
    aggregate.update(
        {
            "scene_name": spec["scene_name"],
            "direction": spec["direction"],
            "angle_degrees": spec["angle_degrees"],
            "energy_components_finite": energy_finite,
            "corrected_sustained_slip_physics_substep_count": sustained_count,
            "progress_threshold_m": progress_threshold,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "per_seed_failures": {
                str(row.get("evaluation_seed", index)): [
                    name
                    for name, passed in {
                        "complete_horizon": int(row["control_steps"]) == 600
                        and int(row["physics_substeps"]) == 3000,
                        "finite_state_and_reward": bool(row["finite"]),
                        "finite_energy_measurements": energy_is_finite(row),
                        "no_fall": not bool(row["fall"]),
                        "no_torso_ground": not bool(row["torso_ground_any"]),
                        "no_sustained_nonfoot": not bool(row["sustained_nonfoot_contact"]),
                        "force_qualified_denominator_evaluable": int(
                            row["force_qualified_supported_physics_substep_count"]
                        )
                        > 0,
                        "zero_corrected_sustained_slip": int(
                            row["corrected_sustained_slip_physics_substep_count"]
                        )
                        == 0,
                        "zero_corrected_slip_events": int(row["corrected_slip_event_count"])
                        == 0,
                    }.items()
                    if not passed
                ]
                for index, row in enumerate(rows)
            },
            "passed": all(checks.values()),
        }
    )
    return aggregate


def infer_tested_bounds(scene_results: dict[str, dict[str, Any]], direction: str) -> dict[str, Any]:
    ordered = [
        scene_results[f"{direction}_{angle}deg"] for angle in EXPECTED_ANGLES
    ]
    contiguous: list[int] = []
    first_failure: int | None = None
    for result in ordered:
        angle = int(result["angle_degrees"])
        if result["passed"] and first_failure is None:
            contiguous.append(angle)
        elif first_failure is None:
            first_failure = angle
    passing = [int(row["angle_degrees"]) for row in ordered if row["passed"]]
    return {
        "direction": direction,
        "tested_angles_degrees": list(EXPECTED_ANGLES),
        "passing_tested_angles_degrees": passing,
        "failing_tested_angles_degrees": [int(row["angle_degrees"]) for row in ordered if not row["passed"]],
        "highest_passing_tested_angle_degrees": max(passing) if passing else None,
        "conservative_tested_lower_bound_degrees": max(contiguous) if contiguous else None,
        "first_failing_tested_angle_degrees": first_failure,
        "nonmonotonic_pass_after_failure": bool(first_failure is not None and any(angle > first_failure for angle in passing)),
        "claim_boundary": "A passing angle is a lower bound only within this tested grid and protocol; a failing angle is an upper bracket only within this tested grid. Neither is a physical maximum.",
    }


def write_report(output_root: Path, config: dict[str, Any], results: dict[str, dict[str, Any]], bounds: dict[str, Any]) -> Path:
    lines = [
        "# Final PAIR0 standard-slope capability boundary",
        "",
        "This is a read-only, deterministic evaluation of one frozen trained policy. It does not estimate a physical maximum slope or random-map generalisation.",
        "",
        "| Scene | Mean best progress (m) | Zero-foot fraction | Force-qualified denominator | Corrected sustained slip substeps | Slip events | Falls | Torso | Sustained non-foot | Decision | Failed checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for spec in scene_specs(config):
        row = results[spec["scene_name"]]
        lines.append(
            f"| {spec['scene_name']} | {row['mean_best_progress_m']:.6f} | {row['pooled_full_interval_zero_foot_fraction']:.6f} | {row['force_qualified_supported_physics_substep_count']} | {row['corrected_sustained_slip_physics_substep_count']} | {row['corrected_slip_event_count']} | {row['fall_count']} | {row['torso_ground_episode_count']} | {row['sustained_nonfoot_contact_episode_count']} | {'PASS' if row['passed'] else 'FAIL'} | {', '.join(row['failed_checks']) or 'none'} |"
        )
    lines.extend(["", "## Tested brackets", ""])
    for direction in ("uphill", "downhill"):
        record = bounds[direction]
        lines.append(
            f"- {direction}: conservative tested lower bound = {record['conservative_tested_lower_bound_degrees']}; first failing tested angle = {record['first_failing_tested_angle_degrees']}; raw passing angles = {record['passing_tested_angles_degrees']}."
        )
    lines.extend(
        [
            "",
            "All five predeclared held-out seeds are retained per scene. Flat has no added progress threshold and is a safety reference only. Energy quantities are measurements only and do not enter any gate.",
        ]
    )
    path = output_root / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def artifact_inventory(output_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def run(config_path: Path, config: dict[str, Any], protocol: dict[str, Any], reward: dict[str, Any], checkpoint: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    stage = "attempt_root_created"
    runtime_before: dict[str, str] | None = None
    snapshot_before: dict[str, str] | None = None
    try:
        stage = "freeze_configuration"
        shutil.copy2(config_path, output_root / "frozen_config.json")
        if sha256(output_root / "frozen_config.json") != sha256(config_path):
            raise RuntimeError("Frozen configuration copy changed")
        stage = "snapshot_runtime_dependencies"
        runtime_before = snapshot_runtime(config, output_root)
        snapshot_before = validate_runtime_snapshot(
            config, output_root / "runtime_snapshot"
        )
        if snapshot_before != runtime_before:
            raise RuntimeError("Live and snapshotted runtime closures differ")
        stage = "prepare_and_audit_scenes"
        scenes, _ = prepare_scenes(config, protocol, output_root)
        stage = "load_read_only_checkpoint"
        model = PPO.load(checkpoint, device="cpu")
        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        stage = "evaluate_55_episodes"
        for spec in scene_specs(config):
            for seed in EXPECTED_SEEDS:
                row, _, local_events = l2.evaluate_episode(
                    model,
                    config,
                    protocol,
                    reward,
                    scenes[spec["scene_name"]],
                    condition_id=PAIR0_ID,
                    seed=seed,
                    checkpoint_additional_timesteps=65_536,
                    max_episode_steps=600,
                    retain_substeps=False,
                )
                if int(row["checkpoint_timesteps"]) != 2_727_936:
                    raise RuntimeError(
                        "Loaded checkpoint timestep metadata changed during evaluation"
                    )
                rows.append(row)
                events.extend(local_events)
        if len(rows) != 55:
            raise RuntimeError("Evaluation matrix is incomplete")
        stage = "write_raw_metrics"
        write_rows(output_root / "episode_metrics.csv", rows)
        l2.write_event_rows(output_root / "corrected_slip_events.csv", events)
        stage = "apply_frozen_gates"
        results = {
            spec["scene_name"]: gate_scene(
                config,
                spec,
                [row for row in rows if row["scene_name"] == spec["scene_name"]],
            )
            for spec in scene_specs(config)
        }
        bounds = {
            direction: infer_tested_bounds(results, direction)
            for direction in ("uphill", "downhill")
        }
        summary = {
            "schema_version": "proxygap-pair0-slope-capability-boundary-result-v1",
            "checkpoint_timesteps": 2_727_936,
            "evaluation_seeds": list(EXPECTED_SEEDS),
            "scene_results": results,
            "tested_bounds": bounds,
            "all_scene_decisions_reported": True,
            "physical_maximum_claimed": False,
        }
        write_json(output_root / "summary.json", summary)
        report = write_report(output_root, config, results, bounds)
        stage = "post_run_provenance_verification"
        runtime_after = validate_runtime_dependencies(config)
        snapshot_after = validate_runtime_snapshot(
            config, output_root / "runtime_snapshot"
        )
        if runtime_after != runtime_before or snapshot_after != snapshot_before:
            raise RuntimeError("Runtime dependencies changed during evaluation")
        if sha256(checkpoint) != config["source"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
        stage = "write_success_manifest"
        manifest = {
            "schema_version": "proxygap-pair0-slope-capability-boundary-artifact-v1",
            "status": "read_only_standard_slope_boundary_formally_evaluated",
            "training_performed": False,
            "checkpoint_modified": False,
            "reward_changed": False,
            "friction_changed": False,
            "energy_formula_changed": False,
            "energy_status": "measurement_only_not_reward_or_gate",
            "fixed_map_evaluated": False,
            "video_rendered": False,
            "candidate_promoted": False,
            "runtime_dependency_sha256_before": runtime_before,
            "runtime_snapshot_sha256_before": snapshot_before,
            "runtime_dependency_sha256_after": runtime_after,
            "runtime_snapshot_sha256_after": snapshot_after,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "report": report.name,
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mujoco": mujoco.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "artifact_inventory_excludes_manifest_itself": artifact_inventory(
                output_root
            ),
        }
        write_json(output_root / "manifest.json", manifest)
        return summary
    except BaseException as error:
        failure_record = {
            "schema_version": "proxygap-pair0-slope-capability-boundary-failure-v1",
            "status": "formal_attempt_failed_closed_non_evaluable",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "scientifically_evaluable": False,
            "all_slope_decisions_withheld": True,
            "retry_permitted": False,
            "canonical_attempt_root_permanently_reserved": True,
            "training_performed": False,
            "checkpoint_write_performed": False,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_expected_sha256": config["source"]["checkpoint_sha256"],
            "checkpoint_observed_sha256": sha256(checkpoint)
            if checkpoint.is_file()
            else None,
            "configuration": str(config_path.resolve()),
            "configuration_sha256": sha256(config_path),
            "runtime_dependency_sha256_before": runtime_before,
            "runtime_snapshot_sha256_before": snapshot_before,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output_root / "FAILURE_RECORD.json", failure_record)
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("Only the canonical frozen configuration may be executed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward, checkpoint = validate_config(config)
    if args.validate_only:
        print("VALIDATION_OK")
        return
    expected_root = ROOT / config["execution"]["formal_output_root"]
    output_root = args.output_root.resolve() if args.output_root else expected_root.resolve()
    if output_root != expected_root.resolve():
        raise ValueError("Formal evaluation must use the canonical output root")
    summary = run(config_path, config, protocol, reward, checkpoint, output_root)
    print(json.dumps({"status": "FORMAL_EVALUATION_COMPLETE", "tested_bounds": summary["tested_bounds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
