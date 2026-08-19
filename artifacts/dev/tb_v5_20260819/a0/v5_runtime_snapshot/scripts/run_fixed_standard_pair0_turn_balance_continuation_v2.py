"""Once-only executable C0/C1 PAIR0 turn-balance continuation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
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
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_fixed_standard_pair0_flat_turn_diagnostic as turn  # noqa: E402
import evaluate_fixed_standard_pair0_slope_capability_boundary as slope  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation as design  # noqa: E402
from proxygap.paired_turn_balance import (  # noqa: E402
    BALANCED_CONDITION_ID,
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    expected_worker_exposure_steps,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v2_20260819.json"
)
RUNTIME_SELF = "scripts/run_fixed_standard_pair0_turn_balance_continuation_v2.py"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF,
    "configs/fixed_standard_pair0_turn_balance_continuation_v1_20260819.json",
    "scripts/run_fixed_standard_pair0_turn_balance_continuation.py",
    "src/proxygap/paired_turn_balance.py",
    "scripts/evaluate_fixed_standard_pair0_flat_turn_diagnostic.py",
    "scripts/evaluate_fixed_standard_pair0_slope_capability_boundary.py",
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
EXPECTED_TURN_SPECS = (
    ("straight_055", "straight", 0.55, 0.0, 0.0, False),
    ("curve_left_010", "constant_curvature", 0.55, 0.1, 0.055, False),
    ("curve_right_010", "constant_curvature", 0.55, -0.1, -0.055, False),
    ("curve_left_020", "constant_curvature", 0.55, 0.2, 0.11, False),
    ("curve_right_020", "constant_curvature", 0.55, -0.2, -0.11, False),
    ("curve_left_035", "constant_curvature", 0.55, 0.35, 0.1925, False),
    ("curve_right_035", "constant_curvature", 0.55, -0.35, -0.1925, False),
    (
        "low_speed_yaw_left",
        "positive_speed_yaw_rate_probe_not_in_place_turn",
        0.1,
        1.0,
        0.1,
        True,
    ),
    (
        "low_speed_yaw_right",
        "positive_speed_yaw_rate_probe_not_in_place_turn",
        0.1,
        -1.0,
        -0.1,
        True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--condition-worker", action="store_true")
    parser.add_argument("--condition-id", choices=CONDITION_IDS)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--training-scene-json", type=Path)
    parser.add_argument("--worker-smoke", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} changed: {observed!r} != {expected!r}")


def configure_torch_threads(thread_count: int) -> int:
    torch.set_num_threads(int(thread_count))
    observed = int(torch.get_num_threads())
    equal(observed, int(thread_count), "torch thread count")
    return observed


def validate_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    contract = config["runtime_dependency_contract"]
    equal(
        tuple(contract),
        (
            "copy_preserving_relative_paths",
            "verify_live_and_snapshot_before_and_after",
            "exact_relative_path_sha256",
        ),
        "runtime contract fields/order",
    )
    expected = contract["exact_relative_path_sha256"]
    equal(tuple(expected), EXPECTED_RUNTIME_PATHS, "runtime exact membership/order")
    observed: dict[str, str] = {}
    for relative_path in EXPECTED_RUNTIME_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        equal(digest, str(expected[relative_path]), f"runtime {relative_path}")
        observed[relative_path] = digest
    return observed


def snapshot_runtime_dependencies(
    config: dict[str, Any], attempt_root: Path
) -> tuple[Path, dict[str, str]]:
    live = validate_runtime_dependencies(config)
    snapshot = attempt_root / "runtime_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    for relative_path in EXPECTED_RUNTIME_PATHS:
        target = snapshot / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
        equal(sha256(target), live[relative_path], f"snapshot {relative_path}")
    return snapshot, live


def validate_runtime_snapshot(
    config: dict[str, Any], snapshot: Path
) -> dict[str, str]:
    actual = tuple(
        sorted(
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        )
    )
    equal(actual, tuple(sorted(EXPECTED_RUNTIME_PATHS)), "runtime snapshot membership")
    expected = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    observed: dict[str, str] = {}
    for relative_path in EXPECTED_RUNTIME_PATHS:
        digest = sha256(snapshot / relative_path)
        equal(digest, expected[relative_path], f"runtime snapshot {relative_path}")
        observed[relative_path] = digest
    return observed


def expected_turn_conditions() -> list[dict[str, Any]]:
    return [
        {
            "condition_name": name,
            "kind": kind,
            "speed_m_per_s": speed,
            "target_curvature_per_m": curvature,
            "target_yaw_rate_rad_per_s": yaw_rate,
            "out_of_training_command_envelope": out_of_envelope,
        }
        for name, kind, speed, curvature, yaw_rate, out_of_envelope in EXPECTED_TURN_SPECS
    ]


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    equal(
        config.get("status"),
        "frozen_executable_last_optimisation_round_pending_independent_go",
        "V2 status",
    )
    source = config["design_source"]
    design_path = ROOT / source["configuration"]
    equal(sha256(design_path), source["configuration_sha256"], "V1 design SHA")
    equal(
        sha256(ROOT / source["validation_runner"]),
        source["validation_runner_sha256"],
        "V1 runner SHA",
    )
    equal(
        sha256(ROOT / source["command_wrapper"]),
        source["command_wrapper_sha256"],
        "command wrapper SHA",
    )
    v1 = json.loads(design_path.read_text(encoding="utf-8"))
    protocol, reward, checkpoint, _ = design.validate_config(v1)
    equal(bool(v1["execution"]["formal_training_authorised"]), False, "V1 non-executable status")
    equal(tuple(config["execution"]["condition_process_order"]), CONDITION_IDS, "condition order")
    equal(int(config["execution"]["maximum_attempt_index"]), 0, "once-only attempt")
    equal(bool(config["execution"]["formal_requires_successful_canonical_smoke"]), True, "smoke prerequisite")
    equal(
        bool(
            config["execution"][
                "formal_smoke_prerequisite_full_manifest_branch_runtime_and_inventory_validation"
            ]
        ),
        True,
        "strict smoke prerequisite validation",
    )
    equal(int(config["smoke"]["additional_timesteps_per_condition"]), 8192, "smoke budget")
    equal(int(config["smoke"]["rollouts_per_condition"]), 4, "smoke rollouts")
    equal(int(config["smoke"]["steps_per_worker"]), 1024, "smoke worker steps")
    equal(int(config["smoke"]["complete_episodes_per_worker"]), 2, "smoke episodes")
    equal(
        bool(config["smoke"]["C1_each_turning_worker_covers_both_signs"]),
        True,
        "smoke sign coverage",
    )
    equal(bool(config["smoke"]["scientific_gate_applied"]), False, "smoke science boundary")
    equal(bool(config["smoke"]["checkpoint_written"]), False, "smoke checkpoint boundary")
    equal(int(config["formal"]["additional_timesteps_per_condition"]), 65_536, "formal budget")
    equal(int(config["formal"]["rollouts_per_condition"]), 32, "formal rollouts")
    equal(int(config["formal"]["steps_per_worker"]), 8192, "formal worker steps")
    equal(int(config["formal"]["complete_episodes_per_worker"]), 16, "formal episodes")
    equal(int(config["formal"]["absolute_final_checkpoint_timesteps"]), 2_793_472, "formal final timestep")
    equal(config["formal"]["final_checkpoint_filename"], "checkpoint_2793472.zip", "checkpoint name")
    equal(bool(config["formal"]["save_intermediate_checkpoints"]), False, "intermediate checkpoint save")
    equal(bool(config["formal"]["evaluate_intermediate_checkpoints"]), False, "intermediate evaluation")
    equal(bool(config["formal"]["select_intermediate_checkpoint"]), False, "intermediate selection")
    equal(config["evaluation_command_contract"]["turn_conditions"], expected_turn_conditions(), "turn command specifications")
    equal(
        tuple(config["evaluation_command_contract"]["standard_slope_scenes"]),
        tuple(v1["final_evaluation"]["slope_scene_order"]),
        "slope scene order",
    )
    equal(
        int(config["evaluation_command_contract"]["representative_substep_trace_seed"]),
        96149,
        "trace seed",
    )
    equal(bool(config["hard_stop_and_archive"]["last_locomotion_optimisation_round"]), True, "last round")
    equal(bool(config["hard_stop_and_archive"]["hard_stop_after_formal_pass_or_fail"]), True, "hard stop")
    equal(bool(config["hard_stop_and_archive"]["post_result_read_only_video_contract_required"]), True, "video archive")
    equal(bool(config["hard_stop_and_archive"]["video_participates_in_scientific_gate"]), False, "video gate boundary")
    equal(
        int(config["hard_stop_and_archive"]["post_result_video_seed_predeclared_before_training"]),
        96131,
        "predeclared video seed",
    )
    equal(
        config["hard_stop_and_archive"]["post_result_video_turn_conditions"],
        ["curve_left_020", "curve_right_020"],
        "predeclared video turn conditions",
    )
    equal(
        bool(
            config["hard_stop_and_archive"][
                "post_result_video_seed_or_condition_change_after_numeric_results_permitted"
            ]
        ),
        False,
        "post-result video selection boundary",
    )
    equal(int(v1["training"]["training_seed_count"]), 1, "single training seed")
    equal(int(v1["ppo"]["torch_num_threads"]), 2, "training torch threads")
    failure = config["attempt_failure_contract"]
    equal(bool(failure["catch_base_exception_after_root_creation"]), True, "failure capture")
    equal(bool(failure["retry_permitted"]), False, "failure retry")
    equal(bool(failure["partial_root_permanently_reserved"]), True, "failure root reservation")
    formal_contract = config["formal_result_contract"]
    equal(int(formal_contract["complete_turn_rows_per_condition"]), 45, "turn row count")
    equal(int(formal_contract["complete_slope_rows_per_condition"]), 20, "slope row count")
    equal(int(formal_contract["complete_training_branches"]), 2, "training branch count")
    equal(bool(formal_contract["all_energy_components_finite"]), True, "energy finiteness")
    equal(bool(formal_contract["hard_stop_marker_required"]), True, "hard-stop marker")
    equal(bool(config["execution"]["fixed_map"]), False, "fixed-map boundary")
    equal(bool(config["execution"]["training_video"]), False, "training-video boundary")
    equal(bool(config["execution"]["promotion"]), False, "promotion boundary")
    validate_runtime_dependencies(config)
    return v1, protocol, reward, checkpoint


def prepare_standard_pair0_scenes(
    v1: dict[str, Any], protocol: dict[str, Any], output_root: Path
) -> dict[str, dict[str, Any]]:
    controls, generation = slope.prepare_standard_scenes(
        protocol, output_root / "generated"
    )
    scenes: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    for scene_name in l2.EXPECTED_SCENES:
        pair, audit = slope.prepare_pair(
            controls[scene_name],
            output_root / "condition_assets",
            f"turn_balance_{scene_name}",
            l2._pair_contract(v1),
        )
        scene = dict(pair[slope.CANDIDATE_ID])
        scene["condition_id"] = l2.PAIR0_ID
        scene["scene_name"] = scene_name
        scenes[scene_name] = scene
        audits[scene_name] = audit
    equal(tuple(scenes), tuple(l2.EXPECTED_SCENES), "standard scene order")
    if not all(bool(audit.get("passed")) for audit in audits.values()):
        raise RuntimeError("one or more standard PAIR0 scene/contact audits failed")
    write_json(output_root / "scene_generation.json", generation)
    write_json(output_root / "explicit_pair_audits.json", audits)
    write_json(output_root / "prepared_scenes.json", scenes)
    return scenes


def make_training_vec_env(
    v1: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    monitor_path: Path,
    start_method: str,
) -> tuple[VecMonitor, list[dict[str, Any]]]:
    factories: list[Callable[[], gym.Env]] = []
    master = int(v1["training"]["master_seed"])
    for rank in range(8):
        construction_seed = master + 1000 * rank

        def factory(
            local_rank: int = rank,
            local_seed: int = construction_seed,
        ) -> gym.Env:
            env = design.make_turn_balance_env(
                v1,
                protocol,
                reward,
                scene,
                condition_id=condition_id,
                worker_rank=local_rank,
                seed=local_seed,
            )
            audit = l2.compiled_contract_audit(
                env.unwrapped.model,
                scene,
                l2.PAIR0_ID,
                v1,
                construction_seed=local_seed,
            )
            audit["worker_rank"] = local_rank
            setattr(env, "_turn_balance_worker_audit", audit)
            return env

        factories.append(factory)
    base = SubprocVecEnv(factories, start_method=start_method)
    audits = list(base.get_attr("_turn_balance_worker_audit"))
    if len(audits) != 8 or not all(bool(row["passed"]) for row in audits):
        base.close()
        raise RuntimeError("not all eight training workers passed compiled audit")
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return VecMonitor(base, filename=str(monitor_path)), audits


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(state: dict[str, Any]) -> str:
    stream = io.BytesIO()
    torch.save(state, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def expected_worker_exposure(
    condition_id: str, rank: int, *, smoke: bool
) -> dict[str, int]:
    return expected_worker_exposure_steps(
        condition_id,
        rank,
        episodes_per_worker=2 if smoke else 16,
        episode_steps=512,
    )


def validate_exposure_states(
    states: list[dict[str, Any]], condition_id: str, *, smoke: bool
) -> None:
    expected_episodes = 2 if smoke else 16
    if len(states) != 8:
        raise RuntimeError("training exposure state count changed")
    for rank, state in enumerate(states):
        checks = {
            "condition": state["condition_id"] == condition_id,
            "rank": int(state["worker_rank"]) == rank,
            "completed_episodes": int(state["completed_episode_count"])
            == expected_episodes,
            "auto_reset_episode_index": int(state["active_episode_index"])
            == expected_episodes,
            "zero_transition_auto_reset": int(state["active_episode_steps"]) == 0,
            "first_reset_seed": int(state["first_reset_seed"]) == 63806 + rank,
            "actual_exposure": state["executed_exposure_steps"]
            == expected_worker_exposure(condition_id, rank, smoke=smoke),
        }
        if not all(checks.values()):
            raise RuntimeError(f"worker {rank} exposure contract failed: {checks}")


def train_condition_worker(
    config_path: Path,
    config: dict[str, Any],
    v1: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    checkpoint: Path,
    attempt_root: Path,
    scene: dict[str, Any],
    condition_id: str,
    *,
    smoke: bool,
) -> None:
    condition_root = attempt_root / condition_id.lower()
    condition_root.mkdir(parents=True, exist_ok=False)
    stage = "condition_root_created"
    live_before: dict[str, str] | None = None
    snapshot_before: dict[str, str] | None = None
    started = time.perf_counter()
    try:
        snapshot = attempt_root / "runtime_snapshot"
        stage = "worker_runtime_before"
        live_before = validate_runtime_dependencies(config)
        snapshot_before = validate_runtime_snapshot(config, snapshot)
        equal(live_before, snapshot_before, "worker live/snapshot before")
        torch.set_num_threads(int(v1["ppo"]["torch_num_threads"]))
        equal(torch.get_num_threads(), 2, "worker torch threads")
        stage = "construct_eight_training_workers"
        vec_env, audits = make_training_vec_env(
            v1,
            protocol,
            reward,
            scene,
            condition_id=condition_id,
            monitor_path=condition_root / "monitor.csv",
            start_method=str(config["execution"]["subprocess_start_method"]),
        )
        model: PPO | None = None
        try:
            source_model = PPO.load(checkpoint, device="cpu")
            source_policy_hash = tensor_state_sha256(source_model.policy.state_dict())
            source_optimizer_hash = optimizer_state_sha256(
                source_model.policy.optimizer.state_dict()
            )
            stage = "fresh_load_source_with_8x256_buffer"
            model = design.load_continuation_model(checkpoint, vec_env, v1)
            policy_exact = design._nested_equal(
                source_model.policy.state_dict(), model.policy.state_dict()
            )
            optimizer_exact = design._nested_equal(
                source_model.policy.optimizer.state_dict(),
                model.policy.optimizer.state_dict(),
            )
            probe = np.zeros((1, 135), dtype=np.float32)
            source_action, _ = source_model.predict(probe, deterministic=True)
            loaded_action, _ = model.predict(probe, deterministic=True)
            action_exact = bool(np.array_equal(source_action, loaded_action))
            if not (policy_exact and optimizer_exact and action_exact):
                raise RuntimeError("source policy/optimizer/action continuation mismatch")
            master = int(v1["training"]["master_seed"])
            equal(int(model.seed), master, "loaded training RNG seed")
            effective = [int(value) for value in vec_env.seed(master)]
            equal(
                effective,
                list(v1["training"]["worker_effective_first_reset_seeds"]),
                "effective first reset seeds",
            )
            budget = int(
                config["smoke" if smoke else "formal"][
                    "additional_timesteps_per_condition"
                ]
            )
            stage = "smoke_training" if smoke else "formal_training"
            model.learn(
                total_timesteps=budget,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            expected_final = int(v1["source"]["checkpoint_timesteps"]) + budget
            equal(int(model.num_timesteps), expected_final, "post-training timestep")
            stage = "exposure_and_auto_reset_audit"
            states = list(vec_env.env_method("turn_balance_state"))
            validate_exposure_states(states, condition_id, smoke=smoke)
            observed_first_reset_seeds = [
                int(state["first_reset_seed"]) for state in states
            ]
            equal(
                observed_first_reset_seeds,
                effective,
                "actual first reset seeds consumed by workers",
            )
            final_checkpoint: Path | None = None
            final_checkpoint_hash: str | None = None
            if not smoke:
                stage = "save_only_final_checkpoint"
                model_root = condition_root / "models"
                model_root.mkdir(parents=True, exist_ok=False)
                target = model_root / "checkpoint_2793472"
                model.save(target)
                final_checkpoint = target.with_suffix(".zip")
                if not final_checkpoint.is_file():
                    raise FileNotFoundError(final_checkpoint)
                final_checkpoint_hash = sha256(final_checkpoint)
                reloaded = PPO.load(final_checkpoint, device="cpu")
                equal(int(reloaded.num_timesteps), 2_793_472, "reloaded final timestep")
                equal(tuple(reloaded.observation_space.shape or ()), (135,), "reloaded observation")
                equal(tuple(reloaded.action_space.shape or ()), (8,), "reloaded action")
                if len(reloaded.policy.optimizer.state) == 0:
                    raise RuntimeError("saved final optimizer state is empty")
            stage = "worker_runtime_after"
            live_after = validate_runtime_dependencies(config)
            snapshot_after = validate_runtime_snapshot(config, snapshot)
            equal(live_after, live_before, "worker live runtime after")
            equal(snapshot_after, snapshot_before, "worker snapshot runtime after")
            equal(sha256(checkpoint), v1["source"]["checkpoint_sha256"], "source checkpoint after worker")
            record = {
                "schema_version": "proxygap-turn-balance-condition-training-v2",
                "condition_id": condition_id,
                "mode": "engineering_smoke" if smoke else "formal",
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": sha256(checkpoint),
                "source_checkpoint_timesteps": 2_727_936,
                "source_policy_tensor_sha256": source_policy_hash,
                "source_optimizer_state_sha256": source_optimizer_hash,
                "source_optimizer_state_entries": len(
                    source_model.policy.optimizer.state
                ),
                "fresh_loaded_policy_exact": policy_exact,
                "fresh_loaded_optimizer_exact": optimizer_exact,
                "fresh_loaded_deterministic_action_exact": action_exact,
                "training_master_seed": master,
                "configured_pending_first_reset_seeds": effective,
                "observed_first_reset_seeds": observed_first_reset_seeds,
                "torch_num_threads": int(torch.get_num_threads()),
                "additional_timesteps": budget,
                "absolute_final_timesteps": expected_final,
                "worker_contract_audits": audits,
                "worker_exposure_states": states,
                "final_checkpoint": (
                    str(final_checkpoint.resolve())
                    if final_checkpoint is not None
                    else None
                ),
                "final_checkpoint_sha256": final_checkpoint_hash,
                "intermediate_checkpoint_saved": False,
                "intermediate_checkpoint_evaluated": False,
                "energy_status": "measurement_only_not_reward_or_gate",
                "runtime_live_before": live_before,
                "runtime_snapshot_before": snapshot_before,
                "runtime_live_after": live_after,
                "runtime_snapshot_after": snapshot_after,
                "elapsed_seconds": time.perf_counter() - started,
            }
            write_json(condition_root / "training_record.json", record)
        finally:
            vec_env.close()
        if model is None:
            raise RuntimeError("condition model was never loaded")
    except BaseException as error:
        write_json(
            condition_root / "WORKER_FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-turn-balance-worker-failure-v2",
                "condition_id": condition_id,
                "mode": "engineering_smoke" if smoke else "formal",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "scientifically_evaluable": False,
                "retry_permitted": False,
                "runtime_live_before": live_before,
                "runtime_snapshot_before": snapshot_before,
                "configuration": str(config_path.resolve()),
                "configuration_sha256": sha256(config_path),
            },
        )
        raise


def run_condition_subprocess(
    frozen_config: Path,
    attempt_root: Path,
    training_scene_json: Path,
    condition_id: str,
    *,
    smoke: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(frozen_config),
        "--condition-worker",
        "--condition-id",
        condition_id,
        "--attempt-root",
        str(attempt_root),
        "--training-scene-json",
        str(training_scene_json),
    ]
    if smoke:
        command.append("--worker-smoke")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=environment,
        check=False,
    )
    log_root = attempt_root / "condition_process_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / f"{condition_id}.stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_root / f"{condition_id}.stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{condition_id} clean condition worker failed with exit {completed.returncode}"
        )
    record_path = attempt_root / condition_id.lower() / "training_record.json"
    if not record_path.is_file():
        raise RuntimeError(f"{condition_id} did not write training_record.json")
    return json.loads(record_path.read_text(encoding="utf-8"))


def turn_adapter(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    final = v1["final_evaluation"]
    return {
        "evaluation": {
            "diagnostic_command_adapter_maximum_abs_curvature_per_m": float(
                v2["evaluation_command_contract"][
                    "diagnostic_adapter_maximum_abs_curvature_per_m"
                ]
            ),
            "max_episode_steps": int(final["max_episode_steps"]),
            "control_timestep_seconds": float(final["control_timestep_seconds"]),
            "physics_timestep_seconds": float(final["physics_timestep_seconds"]),
            "corrected_slip": dict(final["corrected_slip"]),
        },
        "contact_contract": dict(v1["contact_contract"]),
        "checkpoint_early_stopping": {
            "nonfoot_contact_minimum_sustained_seconds": 0.2
        },
        "safety_gates": dict(
            v1["prospective_final_gate"]["turn_safety_each_of_nine_conditions"]
        ),
    }


def slope_adapter(v1: dict[str, Any]) -> dict[str, Any]:
    final = v1["final_evaluation"]
    return {
        "evaluation": {
            "cruise_speed_m_per_s": float(v1["training"]["cruise_speed_m_per_s"]),
            "physics_timestep_seconds": float(final["physics_timestep_seconds"]),
            "corrected_slip": dict(final["corrected_slip"]),
        },
        "contact_contract": dict(v1["contact_contract"]),
        "checkpoint_early_stopping": {
            "nonfoot_contact_minimum_sustained_seconds": 0.2
        },
    }


def evaluate_turn_matrix(
    model: PPO,
    v1: dict[str, Any],
    v2: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    branch_id: str,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter = turn_adapter(v1, v2)
    seeds = [int(value) for value in v1["final_evaluation"]["heldout_seeds"]]
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for condition in expected_turn_conditions():
        for seed in seeds:
            row, local_events = turn.evaluate_episode(
                model,
                adapter,
                protocol,
                reward,
                scene,
                condition,
                seed,
            )
            row["branch_id"] = branch_id
            rows.append(row)
            for event in local_events:
                event["branch_id"] = branch_id
            events.extend(local_events)
    if len(rows) != 45:
        raise RuntimeError("turn evaluation matrix is incomplete")
    if not all(energy_row_finite(row) for row in rows):
        raise RuntimeError("turn evaluation produced non-finite energy measurements")
    results = {
        condition["condition_name"]: turn.aggregate_condition(
            adapter,
            condition,
            [row for row in rows if row["condition_name"] == condition["condition_name"]],
        )
        for condition in expected_turn_conditions()
    }
    write_rows(output_root / "turn_episode_metrics.csv", rows)
    l2.write_event_rows(output_root / "turn_corrected_slip_events.csv", events)
    write_json(output_root / "turn_aggregates.json", results)
    return results, rows


def apply_turn_gate(
    v1: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    gate = v1["prospective_final_gate"]["turn_effectiveness"]
    required = {name for name, *_ in EXPECTED_TURN_SPECS}
    if set(results) != required:
        return {"evaluable": False, "passed": False, "reason": "incomplete turn conditions"}
    safety = {name: bool(results[name]["safety_passed"]) for name in sorted(required)}
    force_qualified = {
        name: bool(
            results[name]["safety_checks"][
                "force_qualified_denominator_evaluable"
            ]
        )
        for name in sorted(required)
    }
    formal_turns = (
        "curve_left_010",
        "curve_right_010",
        "curve_left_020",
        "curve_right_020",
        "curve_left_035",
        "curve_right_035",
    )
    numeric_values = [
        results["straight_055"]["mean_actual_cumulative_yaw_change_rad"],
        *(results[name]["mean_yaw_change_target_ratio"] for name in formal_turns),
    ]
    if not all(force_qualified.values()):
        return {
            "evaluable": False,
            "passed": False,
            "reason": "force-qualified denominator missing for at least one turn seed/condition",
            "safety_by_condition": safety,
            "force_qualified_denominator_by_condition": force_qualified,
            "checks": {"force_qualified_denominator_every_condition": False},
            "failed_checks": ["force_qualified_denominator_every_condition"],
            "low_speed_tracking_gate_applied": False,
        }
    if not all(value is not None and math.isfinite(float(value)) for value in numeric_values):
        return {
            "evaluable": False,
            "passed": False,
            "reason": "required turn tracking metric is missing or non-finite",
            "safety_by_condition": safety,
            "force_qualified_denominator_by_condition": force_qualified,
            "checks": {"required_turn_tracking_metrics_finite": False},
            "failed_checks": ["required_turn_tracking_metrics_finite"],
            "low_speed_tracking_gate_applied": False,
        }
    checks: dict[str, bool] = {
        "all_nine_condition_safety_gates": all(safety.values()),
        "force_qualified_denominator_every_condition": all(
            force_qualified.values()
        ),
        "straight_mean_yaw": abs(
            float(results["straight_055"]["mean_actual_cumulative_yaw_change_rad"])
        )
        <= float(gate["maximum_absolute_straight_mean_yaw_change_rad"]),
    }
    for name in formal_turns:
        checks[f"{name}_5_of_5_same_sign"] = int(
            results[name]["same_sign_episode_count"]
        ) == int(gate["required_same_sign_episode_count_each_nonzero_turn_condition"])
    for magnitude in ("010", "020"):
        left = float(results[f"curve_left_{magnitude}"]["mean_yaw_change_target_ratio"])
        right = float(results[f"curve_right_{magnitude}"]["mean_yaw_change_target_ratio"])
        checks[f"ratio_{magnitude}_left"] = float(
            gate["minimum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]
        ) <= left <= float(
            gate["maximum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]
        )
        checks[f"ratio_{magnitude}_right"] = float(
            gate["minimum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]
        ) <= right <= float(
            gate["maximum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]
        )
        checks[f"left_right_ratio_difference_{magnitude}"] = abs(left - right) <= float(
            gate["maximum_left_right_mean_ratio_difference_abs_curvature_010_and_020"]
        )
    left_035 = float(results["curve_left_035"]["mean_yaw_change_target_ratio"])
    right_035 = float(results["curve_right_035"]["mean_yaw_change_target_ratio"])
    minimum_035 = float(
        gate["minimum_absolute_mean_yaw_change_target_ratio_abs_curvature_035"]
    )
    checks["ratio_035_left"] = abs(left_035) >= minimum_035
    checks["ratio_035_right"] = abs(right_035) >= minimum_035
    checks["left_right_ratio_difference_035"] = abs(
        abs(left_035) - abs(right_035)
    ) <= float(
        gate["maximum_left_right_absolute_mean_ratio_difference_abs_curvature_035"]
    )
    checks["formal_turn_ratios_finite"] = True
    return {
        "evaluable": True,
        "safety_by_condition": safety,
        "force_qualified_denominator_by_condition": force_qualified,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "low_speed_tracking_gate_applied": False,
    }


def evaluate_slope_matrix(
    model: PPO,
    v1: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scenes: dict[str, dict[str, Any]],
    branch_id: str,
    output_root: Path,
    trace_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter = slope_adapter(v1)
    seeds = [int(value) for value in v1["final_evaluation"]["heldout_seeds"]]
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for scene_name in l2.EXPECTED_SCENES:
        for seed in seeds:
            row, local_trace, local_events = l2.evaluate_episode(
                model,
                adapter,
                protocol,
                reward,
                scenes[scene_name],
                condition_id=l2.PAIR0_ID,
                seed=seed,
                checkpoint_additional_timesteps=65_536,
                max_episode_steps=600,
                retain_substeps=seed == trace_seed,
            )
            row["branch_id"] = branch_id
            rows.append(row)
            for trace_row in local_trace:
                trace_row["branch_id"] = branch_id
            traces.extend(local_trace)
            for event in local_events:
                event["branch_id"] = branch_id
            events.extend(local_events)
    if len(rows) != 20:
        raise RuntimeError("standard-slope evaluation matrix is incomplete")
    if not all(energy_row_finite(row) for row in rows):
        raise RuntimeError("slope evaluation produced non-finite energy measurements")
    aggregate = l2.aggregate_episode_rows(rows)
    aggregate["per_scene"] = {
        scene_name: l2.aggregate_episode_rows(
            [row for row in rows if row["scene_name"] == scene_name]
        )
        for scene_name in l2.EXPECTED_SCENES
    }
    write_rows(output_root / "slope_episode_metrics.csv", rows)
    write_rows(output_root / "slope_representative_substep_traces.csv", traces)
    l2.write_event_rows(output_root / "slope_corrected_slip_events.csv", events)
    write_json(output_root / "slope_aggregate.json", aggregate)
    return aggregate, rows


def energy_row_finite(row: dict[str, Any]) -> bool:
    return all(math.isfinite(float(row[key])) for key in turn.ENERGY_KEYS)


def apply_slope_gate(
    v1: dict[str, Any], aggregate: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gate = v1["prospective_final_gate"]["standard_slope_continuity"]
    uphill = aggregate["per_scene"]["uphill_8deg"]
    downhill = aggregate["per_scene"]["downhill_8deg"]
    denominator_evaluable = bool(aggregate["force_qualified_slip_evaluable"]) and all(
        int(row["force_qualified_supported_physics_substep_count"]) > 0
        for row in rows
    )
    if not denominator_evaluable:
        return {
            "evaluable": False,
            "passed": False,
            "reason": "force-qualified denominator missing for at least one standard-slope seed",
            "checks": {"force_qualified_denominator_every_seed": False},
            "failed_checks": ["force_qualified_denominator_every_seed"],
        }
    required_numeric = (
        aggregate["mean_best_progress_m"],
        uphill["mean_best_progress_m"],
        downhill["mean_best_progress_m"],
        aggregate["pooled_full_interval_zero_foot_fraction"],
        aggregate["corrected_sustained_slip_per_force_qualified_supported_fraction"],
        aggregate[
            "corrected_slip_events_per_100_force_qualified_supported_substeps"
        ],
    )
    if not all(value is not None and math.isfinite(float(value)) for value in required_numeric):
        return {
            "evaluable": False,
            "passed": False,
            "reason": "required standard-slope metric is missing or non-finite",
            "checks": {"required_standard_slope_metrics_finite": False},
            "failed_checks": ["required_standard_slope_metrics_finite"],
        }
    checks = {
        "complete_20_episode_matrix": len(rows) == 20
        and int(aggregate["total_control_steps"]) == 12_000
        and int(aggregate["total_physics_substeps"]) == 60_000,
        "finite_state_and_reward": int(aggregate["nonfinite_episode_count"]) == 0,
        "finite_energy_measurements": all(energy_row_finite(row) for row in rows),
        "overall_progress_floor": float(aggregate["mean_best_progress_m"])
        >= float(gate["minimum_pair0_mean_best_progress_m"]),
        "uphill_progress_floor": float(uphill["mean_best_progress_m"])
        >= float(gate["minimum_pair0_uphill_mean_best_progress_m"]),
        "downhill_progress_floor": float(downhill["mean_best_progress_m"])
        >= float(gate["minimum_pair0_downhill_mean_best_progress_m"]),
        "no_falls": int(aggregate["fall_count"])
        <= int(gate["maximum_pair0_fall_count"]),
        "no_torso_ground": int(aggregate["torso_ground_episode_count"])
        <= int(gate["maximum_pair0_torso_ground_episode_count"]),
        "no_sustained_nonfoot": int(
            aggregate["sustained_nonfoot_contact_episode_count"]
        )
        <= int(gate["maximum_pair0_sustained_nonfoot_contact_episode_count"]),
        "zero_foot_within_gate": float(
            aggregate["pooled_full_interval_zero_foot_fraction"]
        )
        <= float(gate["maximum_pair0_pooled_full_interval_zero_foot_fraction"]),
        "force_qualified_denominator_every_seed": denominator_evaluable,
        "corrected_sustained_slip": float(
            aggregate[
                "corrected_sustained_slip_per_force_qualified_supported_fraction"
            ]
        )
        <= float(
            gate[
                "maximum_pair0_corrected_sustained_slip_per_force_qualified_supported_fraction"
            ]
        ),
        "corrected_slip_event_rate": float(
            aggregate[
                "corrected_slip_events_per_100_force_qualified_supported_substeps"
            ]
        )
        <= float(
            gate[
                "maximum_pair0_corrected_slip_events_per_100_force_qualified_supported_substeps"
            ]
        ),
    }
    return {
        "evaluable": True,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def combined_final_gate(branches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(branches) != set(CONDITION_IDS):
        return {
            "evaluable": False,
            "passed": False,
            "decision": "incomplete_branch_results",
            "fixed_map_authorised": False,
            "candidate_promoted": False,
            "hard_stop": True,
            "further_optimisation_authorised": False,
            "post_result_video_archive_required": True,
        }
    branch_evaluable = {
        branch: bool(
            record["turn_gate"].get("evaluable", False)
            and record["slope_gate"].get("evaluable", False)
        )
        for branch, record in branches.items()
    }
    if not all(branch_evaluable.values()):
        return {
            "evaluable": False,
            "branch_evaluable": branch_evaluable,
            "branch_pass": {branch: False for branch in CONDITION_IDS},
            "C1_passed_both_turn_and_slope": False,
            "passed": False,
            "decision": "scientific_decisions_withheld_non_evaluable_turning_HOLD",
            "source_PAIR0_retained_as_known_best_slope_candidate": True,
            "fixed_map_authorised": False,
            "candidate_promoted": False,
            "hard_stop": True,
            "further_optimisation_authorised": False,
            "post_result_video_archive_required": True,
            "video_participated_in_gate": False,
        }
    branch_pass = {
        branch: bool(
            record["turn_gate"]["passed"] and record["slope_gate"]["passed"]
        )
        for branch, record in branches.items()
    }
    c0 = branch_pass[CONTROL_CONDITION_ID]
    c1 = branch_pass[BALANCED_CONDITION_ID]
    if c1 and not c0:
        decision = "balanced_intervention_supported_archive_without_more_optimisation"
    elif c1 and c0:
        decision = "both_pass_effect_not_isolated_archive_without_more_optimisation"
    elif not c1 and c0:
        decision = "balanced_intervention_rejected_turning_HOLD_retain_source_PAIR0"
    else:
        decision = "both_fail_turning_HOLD_retain_source_PAIR0"
    return {
        "evaluable": True,
        "branch_evaluable": branch_evaluable,
        "branch_pass": branch_pass,
        "C1_passed_both_turn_and_slope": c1,
        "passed": c1,
        "decision": decision,
        "source_PAIR0_retained_as_known_best_slope_candidate": not c1,
        "fixed_map_authorised": False,
        "candidate_promoted": False,
        "hard_stop": True,
        "further_optimisation_authorised": False,
        "post_result_video_archive_required": True,
        "video_participated_in_gate": False,
    }


def artifact_inventory(output_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.relative_to(output_root).as_posix() != "manifest.json"
    ]


def git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            return f"unavailable:{type(error).__name__}"

    status = run("status", "--short")
    return {"head": run("rev-parse", "HEAD"), "status_short": status, "dirty": bool(status)}


def write_formal_report(
    output_root: Path, branch_results: dict[str, dict[str, Any]], gate: dict[str, Any]
) -> Path:
    lines = [
        "# PAIR0 turn-balance final result",
        "",
        "This is the final locomotion-optimisation round. Both branches were trained from the same frozen source and only final checkpoints were evaluated.",
        "",
        "| Branch | Turn gate | Standard-slope gate | Combined |",
        "|---|---|---|---|",
    ]
    for branch in CONDITION_IDS:
        record = branch_results[branch]
        branch_evaluable = bool(
            record["turn_gate"].get("evaluable", False)
            and record["slope_gate"].get("evaluable", False)
        )
        turn_pass = bool(record["turn_gate"]["passed"])
        slope_pass = bool(record["slope_gate"]["passed"])
        lines.append(
            f"| {branch} | "
            f"{'PASS' if turn_pass else ('FAIL' if record['turn_gate'].get('evaluable', False) else 'NON-EVALUABLE')} | "
            f"{'PASS' if slope_pass else ('FAIL' if record['slope_gate'].get('evaluable', False) else 'NON-EVALUABLE')} | "
            f"{'PASS' if turn_pass and slope_pass else ('FAIL' if branch_evaluable else 'NON-EVALUABLE')} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{gate['decision']}`.",
            "",
            "Energy remained measurement-only and did not enter reward, checkpoint selection or gates. No fixed-map evaluation, video rendering or promotion occurred in this run.",
            "",
            "Optimisation is now hard-stopped regardless of PASS, FAIL or non-evaluable completion. A separate read-only video archive is frozen to seed 96131 and the left/right 0.20 per metre conditions; it cannot be reselected after seeing results.",
        ]
    )
    path = output_root / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_parent_attempt(
    config_path: Path,
    config: dict[str, Any],
    v1: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    checkpoint: Path,
    output_root: Path,
    *,
    smoke: bool,
    smoke_prerequisite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_torch_threads(int(v1["ppo"]["torch_num_threads"]))
    if smoke and smoke_prerequisite is not None:
        raise ValueError("smoke attempt cannot consume a smoke prerequisite")
    if not smoke and smoke_prerequisite is None:
        raise ValueError("formal attempt requires validated full smoke evidence")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite canonical root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    stage = "attempt_root_created"
    live_before: dict[str, str] | None = None
    snapshot_before: dict[str, str] | None = None
    try:
        stage = "freeze_configuration"
        frozen_config = output_root / "frozen_config.json"
        shutil.copy2(config_path, frozen_config)
        equal(sha256(frozen_config), sha256(config_path), "frozen V2 config")
        stage = "snapshot_runtime"
        snapshot, live_before = snapshot_runtime_dependencies(config, output_root)
        snapshot_before = validate_runtime_snapshot(config, snapshot)
        equal(live_before, snapshot_before, "parent live/snapshot before")
        stage = "prepare_flat_training_scene"
        training_scene = design.prepare_flat_pair0_scene(
            v1, protocol, output_root / "training_scene_assets"
        )
        training_scene_json = output_root / "training_scene.json"
        write_json(training_scene_json, training_scene)
        standard_scenes: dict[str, dict[str, Any]] | None = None
        if not smoke:
            stage = "prepare_standard_slope_scenes"
            standard_scenes = prepare_standard_pair0_scenes(
                v1, protocol, output_root / "standard_slope_assets"
            )
        stage = "run_clean_condition_processes"
        records: dict[str, dict[str, Any]] = {}
        for condition_id in CONDITION_IDS:
            records[condition_id] = run_condition_subprocess(
                frozen_config,
                output_root,
                training_scene_json,
                condition_id,
                smoke=smoke,
            )
        c0_record = records[CONTROL_CONDITION_ID]
        c1_record = records[BALANCED_CONDITION_ID]
        for key in (
            "source_checkpoint_sha256",
            "source_policy_tensor_sha256",
            "source_optimizer_state_sha256",
            "training_master_seed",
            "configured_pending_first_reset_seeds",
            "observed_first_reset_seeds",
            "torch_num_threads",
            "additional_timesteps",
        ):
            equal(c1_record[key], c0_record[key], f"paired branch {key}")
        if smoke:
            summary = {
                "schema_version": "proxygap-turn-balance-smoke-result-v2",
                "status": "engineering_smoke_complete_no_science",
                "conditions": records,
                "scientifically_evaluable": False,
                "heldout_evaluation_run": False,
                "checkpoint_written": False,
            }
            write_json(output_root / "summary.json", summary)
            branch_results: dict[str, Any] | None = None
            final_gate: dict[str, Any] | None = None
            report: Path | None = None
        else:
            if standard_scenes is None:
                raise RuntimeError("standard scenes missing for formal evaluation")
            stage = "evaluate_only_both_final_checkpoints"
            branch_results = {}
            trace_seed = int(
                config["evaluation_command_contract"][
                    "representative_substep_trace_seed"
                ]
            )
            for condition_id in CONDITION_IDS:
                checkpoint_path = Path(records[condition_id]["final_checkpoint"])
                equal(
                    sha256(checkpoint_path),
                    records[condition_id]["final_checkpoint_sha256"],
                    f"{condition_id} final checkpoint",
                )
                model = PPO.load(checkpoint_path, device="cpu")
                equal(int(model.num_timesteps), 2_793_472, f"{condition_id} final timestep")
                evaluation_root = output_root / condition_id.lower() / "final_evaluation"
                evaluation_root.mkdir(parents=True, exist_ok=False)
                turn_results, turn_rows = evaluate_turn_matrix(
                    model,
                    v1,
                    config,
                    protocol,
                    reward,
                    training_scene,
                    condition_id,
                    evaluation_root,
                )
                turn_gate = apply_turn_gate(v1, turn_results)
                slope_result, slope_rows = evaluate_slope_matrix(
                    model,
                    v1,
                    protocol,
                    reward,
                    standard_scenes,
                    condition_id,
                    evaluation_root,
                    trace_seed,
                )
                slope_gate = apply_slope_gate(v1, slope_result, slope_rows)
                branch_results[condition_id] = {
                    "final_checkpoint": str(checkpoint_path.resolve()),
                    "final_checkpoint_sha256": sha256(checkpoint_path),
                    "turn_aggregates": turn_results,
                    "turn_gate": turn_gate,
                    "slope_aggregate": slope_result,
                    "slope_gate": slope_gate,
                    "turn_row_count": len(turn_rows),
                    "slope_row_count": len(slope_rows),
                }
                write_json(evaluation_root / "branch_gate.json", branch_results[condition_id])
            stage = "apply_predeclared_combined_gate"
            final_gate = combined_final_gate(branch_results)
            write_json(output_root / "final_gate.json", final_gate)
            report = write_formal_report(output_root, branch_results, final_gate)
            write_json(
                output_root / "FINAL_OPTIMISATION_HARD_STOP.json",
                {
                    "schema_version": "proxygap-final-optimisation-hard-stop-v1",
                    "hard_stop": True,
                    "further_optimisation_authorised": False,
                    "applies_after_pass_or_fail": True,
                    "decision": final_gate["decision"],
                    "source_checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": sha256(checkpoint),
                    "final_condition_checkpoints": {
                        condition_id: {
                            "path": branch_results[condition_id]["final_checkpoint"],
                            "sha256": branch_results[condition_id]["final_checkpoint_sha256"],
                        }
                        for condition_id in CONDITION_IDS
                    },
                    "post_result_read_only_video_archive_required": True,
                    "post_result_video_seed": 96131,
                    "post_result_video_turn_conditions": [
                        "curve_left_020",
                        "curve_right_020",
                    ],
                    "post_result_video_selection_locked_before_training": True,
                    "video_not_yet_rendered": True,
                    "fixed_map_authorised": False,
                    "candidate_promoted": False,
                },
            )
            summary = {
                "schema_version": "proxygap-turn-balance-formal-result-v2",
                "status": "final_optimisation_round_complete_hard_stopped",
                "condition_training": records,
                "condition_final_results": branch_results,
                "final_gate": final_gate,
            }
            write_json(output_root / "summary.json", summary)
        stage = "post_run_provenance"
        live_after = validate_runtime_dependencies(config)
        snapshot_after = validate_runtime_snapshot(config, snapshot)
        equal(live_after, live_before, "parent live runtime after")
        equal(snapshot_after, snapshot_before, "parent snapshot runtime after")
        equal(sha256(checkpoint), v1["source"]["checkpoint_sha256"], "source checkpoint final")
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
        write_json(output_root / "environment.json", environment)
        stage = "write_success_manifest"
        manifest = {
            "schema_version": "proxygap-turn-balance-attempt-manifest-v2",
            "status": (
                "engineering_smoke_complete_no_science"
                if smoke
                else "formal_final_optimisation_round_complete_hard_stopped"
            ),
            "mode": "smoke" if smoke else "formal",
            "configuration": str(config_path.resolve()),
            "configuration_sha256": sha256(config_path),
            "frozen_configuration": str(frozen_config.resolve()),
            "frozen_configuration_sha256": sha256(frozen_config),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(checkpoint),
            "validated_smoke_prerequisite": smoke_prerequisite,
            "condition_training": records,
            "condition_final_results": branch_results,
            "final_gate": final_gate,
            "report": report.name if report is not None else None,
            "runtime_live_before": live_before,
            "runtime_snapshot_before": snapshot_before,
            "runtime_live_after": live_after,
            "runtime_snapshot_after": snapshot_after,
            "training_performed": True,
            "scientifically_evaluable": bool(
                not smoke and final_gate is not None and final_gate.get("evaluable", False)
            ),
            "all_scientific_decisions_withheld": bool(
                not smoke
                and final_gate is not None
                and not final_gate.get("evaluable", False)
            ),
            "intermediate_checkpoint_saved_or_selected": False,
            "reward_changed": False,
            "friction_changed": False,
            "energy_formula_changed": False,
            "energy_status": "measurement_only_not_reward_or_gate",
            "fixed_map_evaluated": False,
            "video_rendered": False,
            "candidate_promoted": False,
            "hard_stop": not smoke,
            "post_result_video_archive_pending": not smoke,
            "environment": environment,
            "git": git_record(),
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_inventory_excludes_manifest_itself": artifact_inventory(
                output_root
            ),
        }
        write_json(output_root / "manifest.json", manifest)
        return summary
    except BaseException as error:
        write_json(
            output_root / "FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-turn-balance-attempt-failure-v2",
                "mode": "smoke" if smoke else "formal",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "scientifically_evaluable": False,
                "all_decisions_withheld": True,
                "retry_permitted": False,
                "partial_root_permanently_reserved": True,
                "configuration": str(config_path.resolve()),
                "configuration_sha256": sha256(config_path),
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_expected_sha256": v1["source"][
                    "checkpoint_sha256"
                ],
                "source_checkpoint_observed_sha256": (
                    sha256(checkpoint) if checkpoint.is_file() else None
                ),
                "validated_smoke_prerequisite": smoke_prerequisite,
                "runtime_live_before": live_before,
                "runtime_snapshot_before": snapshot_before,
                "fixed_map_evaluated": False,
                "video_rendered": False,
                "candidate_promoted": False,
                "further_optimisation_authorised": False if not smoke else None,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def canonical_root(config: dict[str, Any], *, smoke: bool) -> Path:
    key = "canonical_smoke_root" if smoke else "canonical_formal_root"
    return (ROOT / config["execution"][key]).resolve()


def validate_smoke_manifest_and_inventory(
    config: dict[str, Any], config_sha: str, smoke_root: Path
) -> dict[str, Any]:
    """Fail closed unless the complete canonical engineering smoke is genuine."""

    smoke_root = smoke_root.resolve()
    manifest_path = smoke_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("formal run requires the successful canonical smoke manifest")
    if (smoke_root / "FAILURE_RECORD.json").exists():
        raise RuntimeError("canonical smoke root contains a failure record")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    equal(
        manifest.get("schema_version"),
        "proxygap-turn-balance-attempt-manifest-v2",
        "smoke manifest schema",
    )
    equal(manifest.get("status"), "engineering_smoke_complete_no_science", "smoke status")
    equal(manifest.get("mode"), "smoke", "smoke mode")
    equal(manifest.get("configuration_sha256"), config_sha, "smoke V2 config SHA")
    equal(
        Path(str(manifest["configuration"])).resolve(),
        DEFAULT_CONFIG.resolve(),
        "smoke canonical configuration path",
    )
    frozen_config = smoke_root / "frozen_config.json"
    equal(
        Path(str(manifest["frozen_configuration"])).resolve(),
        frozen_config.resolve(),
        "smoke frozen configuration path",
    )
    equal(sha256(frozen_config), config_sha, "smoke frozen configuration SHA")
    equal(manifest["frozen_configuration_sha256"], config_sha, "smoke frozen manifest SHA")

    v1 = json.loads(
        (ROOT / config["design_source"]["configuration"]).read_text(encoding="utf-8")
    )
    source_checkpoint = (ROOT / v1["source"]["checkpoint"]).resolve()
    equal(
        Path(str(manifest["source_checkpoint"])).resolve(),
        source_checkpoint,
        "smoke source checkpoint path",
    )
    equal(
        manifest["source_checkpoint_sha256"],
        v1["source"]["checkpoint_sha256"],
        "smoke source checkpoint SHA",
    )
    equal(sha256(source_checkpoint), v1["source"]["checkpoint_sha256"], "live source SHA")

    expected_runtime = config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    for field in (
        "runtime_live_before",
        "runtime_snapshot_before",
        "runtime_live_after",
        "runtime_snapshot_after",
    ):
        equal(manifest[field], expected_runtime, f"smoke manifest {field}")
    equal(
        validate_runtime_snapshot(config, smoke_root / "runtime_snapshot"),
        expected_runtime,
        "smoke runtime snapshot files",
    )

    records = manifest.get("condition_training")
    equal(tuple(records or ()), CONDITION_IDS, "smoke branch membership/order")
    for condition_id in CONDITION_IDS:
        record = records[condition_id]
        on_disk_path = smoke_root / condition_id.lower() / "training_record.json"
        if not on_disk_path.is_file():
            raise FileNotFoundError(on_disk_path)
        equal(
            json.loads(on_disk_path.read_text(encoding="utf-8")),
            record,
            f"{condition_id} smoke record file",
        )
        equal(record["schema_version"], "proxygap-turn-balance-condition-training-v2", "worker schema")
        equal(record["condition_id"], condition_id, "worker condition")
        equal(record["mode"], "engineering_smoke", "worker mode")
        equal(record["source_checkpoint_sha256"], v1["source"]["checkpoint_sha256"], "worker source SHA")
        equal(int(record["source_checkpoint_timesteps"]), 2_727_936, "worker source timestep")
        for key in (
            "fresh_loaded_policy_exact",
            "fresh_loaded_optimizer_exact",
            "fresh_loaded_deterministic_action_exact",
        ):
            equal(bool(record[key]), True, f"{condition_id} {key}")
        if int(record["source_optimizer_state_entries"]) <= 0:
            raise RuntimeError(f"{condition_id} source optimizer state is empty")
        equal(int(record["training_master_seed"]), 63806, "worker master seed")
        expected_seeds = list(range(63806, 63814))
        equal(record["configured_pending_first_reset_seeds"], expected_seeds, "pending worker seeds")
        equal(record["observed_first_reset_seeds"], expected_seeds, "observed worker seeds")
        equal(int(record["torch_num_threads"]), 2, "worker torch threads")
        equal(int(record["additional_timesteps"]), 8192, "worker smoke budget")
        equal(int(record["absolute_final_timesteps"]), 2_736_128, "worker smoke final timestep")
        equal(record["final_checkpoint"], None, "smoke final checkpoint path")
        equal(record["final_checkpoint_sha256"], None, "smoke final checkpoint SHA")
        equal(bool(record["intermediate_checkpoint_saved"]), False, "smoke intermediate save")
        equal(bool(record["intermediate_checkpoint_evaluated"]), False, "smoke intermediate eval")
        equal(record["energy_status"], "measurement_only_not_reward_or_gate", "smoke energy")
        audits = record["worker_contract_audits"]
        equal(len(audits), 8, "smoke worker audit count")
        equal(
            sorted(int(audit["worker_rank"]) for audit in audits),
            list(range(8)),
            "smoke worker audit ranks",
        )
        if not all(bool(audit["passed"]) for audit in audits):
            raise RuntimeError(f"{condition_id} contains a failed worker contract audit")
        states = record["worker_exposure_states"]
        validate_exposure_states(states, condition_id, smoke=True)
        for field in (
            "runtime_live_before",
            "runtime_snapshot_before",
            "runtime_live_after",
            "runtime_snapshot_after",
        ):
            equal(record[field], expected_runtime, f"{condition_id} {field}")

    c0 = records[CONTROL_CONDITION_ID]
    c1 = records[BALANCED_CONDITION_ID]
    for key in (
        "source_checkpoint_sha256",
        "source_policy_tensor_sha256",
        "source_optimizer_state_sha256",
        "training_master_seed",
        "configured_pending_first_reset_seeds",
        "observed_first_reset_seeds",
        "torch_num_threads",
        "additional_timesteps",
    ):
        equal(c1[key], c0[key], f"smoke paired branch {key}")

    summary_path = smoke_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    equal(summary["schema_version"], "proxygap-turn-balance-smoke-result-v2", "smoke summary schema")
    equal(summary["status"], "engineering_smoke_complete_no_science", "smoke summary status")
    equal(summary["conditions"], records, "smoke summary records")
    equal(bool(summary["scientifically_evaluable"]), False, "smoke summary science")
    equal(bool(summary["heldout_evaluation_run"]), False, "smoke heldout boundary")
    equal(bool(summary["checkpoint_written"]), False, "smoke checkpoint boundary")
    equal(manifest["condition_final_results"], None, "smoke final results boundary")
    equal(manifest["final_gate"], None, "smoke final gate boundary")
    equal(manifest["report"], None, "smoke report boundary")
    equal(manifest["validated_smoke_prerequisite"], None, "smoke recursive prerequisite")
    equal(bool(manifest["training_performed"]), True, "smoke training performed")
    equal(bool(manifest["scientifically_evaluable"]), False, "smoke manifest science")
    equal(bool(manifest["all_scientific_decisions_withheld"]), False, "smoke decision field")
    for key in (
        "intermediate_checkpoint_saved_or_selected",
        "reward_changed",
        "friction_changed",
        "energy_formula_changed",
        "fixed_map_evaluated",
        "video_rendered",
        "candidate_promoted",
        "hard_stop",
        "post_result_video_archive_pending",
    ):
        equal(bool(manifest[key]), False, f"smoke manifest {key}")
    equal(manifest["energy_status"], "measurement_only_not_reward_or_gate", "smoke manifest energy")
    environment = manifest["environment"]
    for key in ("python", "platform", "gymnasium", "mujoco", "stable_baselines3", "torch", "numpy"):
        if not str(environment.get(key, "")):
            raise RuntimeError(f"smoke environment field is empty: {key}")
    equal(int(environment["torch_num_threads"]), 2, "smoke parent torch threads")
    equal(
        json.loads((smoke_root / "environment.json").read_text(encoding="utf-8")),
        environment,
        "smoke environment file",
    )
    equal(tuple(manifest["git"]), ("head", "status_short", "dirty"), "smoke git fields/order")

    declared_inventory = manifest["artifact_inventory_excludes_manifest_itself"]
    actual_inventory = artifact_inventory(smoke_root)
    equal(declared_inventory, actual_inventory, "smoke artifact inventory size/hash/membership")
    relative_paths = [str(row["relative_path"]) for row in actual_inventory]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("smoke artifact inventory contains duplicate paths")
    required_paths = {
        "frozen_config.json",
        "summary.json",
        "environment.json",
        "training_scene.json",
        "condition_process_logs/C0_STRAIGHT_CONTINUE.stdout.txt",
        "condition_process_logs/C0_STRAIGHT_CONTINUE.stderr.txt",
        "condition_process_logs/C1_BALANCED_TURN.stdout.txt",
        "condition_process_logs/C1_BALANCED_TURN.stderr.txt",
        "c0_straight_continue/training_record.json",
        "c0_straight_continue/monitor.csv",
        "c1_balanced_turn/training_record.json",
        "c1_balanced_turn/monitor.csv",
        *(f"runtime_snapshot/{path}" for path in EXPECTED_RUNTIME_PATHS),
    }
    missing = sorted(required_paths - set(relative_paths))
    if missing:
        raise RuntimeError(f"smoke artifact inventory misses required paths: {missing}")
    forbidden = [
        path
        for path in relative_paths
        if path.lower().endswith(".zip")
        or "/models/" in f"/{path.lower()}/"
        or "final_evaluation" in path.lower()
        or path.lower().endswith((".mp4", ".avi", ".mov"))
        or path.lower().endswith("manifest.json")
        or "fixed_map" in path.lower()
        or "promotion" in path.lower()
    ]
    if forbidden:
        raise RuntimeError(f"smoke contains forbidden scientific/checkpoint/media artifacts: {forbidden}")
    return {
        "smoke_root": str(smoke_root),
        "smoke_manifest": str(manifest_path.resolve()),
        "smoke_manifest_sha256": sha256(manifest_path),
        "smoke_artifact_count_excluding_manifest": len(actual_inventory),
        "smoke_configuration_sha256": config_sha,
        "smoke_branch_ids": list(CONDITION_IDS),
        "smoke_budget_per_branch": 8192,
        "smoke_absolute_final_timesteps": 2_736_128,
        "scientifically_evaluable": False,
        "checkpoint_written": False,
        "full_manifest_and_inventory_validated": True,
    }


def require_smoke_success(config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    return validate_smoke_manifest_and_inventory(
        config, config_sha, canonical_root(config, smoke=True)
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    canonical_config = DEFAULT_CONFIG.resolve()
    if args.condition_worker:
        if args.attempt_root is None or args.training_scene_json is None or args.condition_id is None:
            raise ValueError("condition worker arguments are incomplete")
        if sha256(config_path) != sha256(canonical_config):
            raise ValueError("condition worker frozen config differs from canonical V2")
    elif config_path != canonical_config:
        raise ValueError("public execution requires the canonical V2 configuration")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configure_torch_threads(2)
    v1, protocol, reward, checkpoint = validate_config(config)
    if args.validate_only:
        print("V2_VALIDATION_OK_PENDING_INDEPENDENT_GO")
        return
    if args.condition_worker:
        scene = json.loads(args.training_scene_json.read_text(encoding="utf-8"))
        train_condition_worker(
            config_path,
            config,
            v1,
            protocol,
            reward,
            checkpoint,
            args.attempt_root.resolve(),
            scene,
            str(args.condition_id),
            smoke=bool(args.worker_smoke),
        )
        print(json.dumps({"status": "CONDITION_WORKER_COMPLETE", "condition_id": args.condition_id}))
        return
    smoke = bool(args.smoke)
    smoke_prerequisite: dict[str, Any] | None = None
    if args.formal:
        smoke_prerequisite = require_smoke_success(config, sha256(config_path))
    output_root = canonical_root(config, smoke=smoke)
    result = run_parent_attempt(
        config_path,
        config,
        v1,
        protocol,
        reward,
        checkpoint,
        output_root,
        smoke=smoke,
        smoke_prerequisite=smoke_prerequisite,
    )
    print(json.dumps({"status": result["status"], "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
