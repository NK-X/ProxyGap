"""Validate and preflight the frozen PAIR0 turn-balance continuation design.

Version 1 is intentionally non-executable for training.  It validates all
scientific decisions and can build the real flat PAIR0 environment plus a
read-only eight-environment PPO loader inside a temporary directory.  A later
training-capable configuration requires a new freeze and an independent GO.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_fixed_standard_pair0_slope_capability_boundary as slope  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402
from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.paired_turn_balance import (  # noqa: E402
    BALANCED_CONDITION_ID,
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    EPISODE_STEPS,
    EPISODES_PER_WORKER,
    PARALLEL_ENVIRONMENTS,
    PairedTurnBalanceTerrainWrapper,
    expected_condition_exposure_steps,
    expected_worker_exposure_steps,
    scheduled_curvature_per_m,
)
from run_curved_gait_training import common_env_kwargs  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v1_20260819.json"
)
RUNTIME_SELF = "scripts/run_fixed_standard_pair0_turn_balance_continuation.py"
EXPECTED_RUNTIME_PATHS = (
    RUNTIME_SELF,
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
EXPECTED_HELDOUT_SEEDS = (96131, 96137, 96149, 96153, 96177)
EXPECTED_TURN_CONDITIONS = (
    "straight_055",
    "curve_left_010",
    "curve_right_010",
    "curve_left_020",
    "curve_right_020",
    "curve_left_035",
    "curve_right_035",
    "low_speed_yaw_left",
    "low_speed_yaw_right",
)
EXPECTED_SLOPE_SCENES = ("flat", "uphill_8deg", "downhill_8deg", "bowl_exit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--engineering-preflight", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} changed: {observed!r} != {expected!r}")


def verified_json(record: dict[str, Any], path_key: str, hash_key: str) -> dict[str, Any]:
    path = ROOT / str(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    equal(sha256(path), str(record[hash_key]), f"{path_key} SHA-256")
    return json.loads(path.read_text(encoding="utf-8"))


def schedule_rows() -> list[dict[str, Any]]:
    return [
        {
            "condition_id": condition_id,
            "worker_rank": rank,
            "episodes": [
                scheduled_curvature_per_m(condition_id, rank, episode)
                for episode in range(EPISODES_PER_WORKER)
            ],
            "exposure_steps": expected_worker_exposure_steps(condition_id, rank),
        }
        for condition_id in CONDITION_IDS
        for rank in range(PARALLEL_ENVIRONMENTS)
    ]


def validate_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    contract = config["runtime_dependency_contract"]
    equal(
        tuple(contract),
        (
            "copy_preserving_relative_paths",
            "verify_before_and_after_future_training",
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


def _validate_flat_turn_evidence(summary: dict[str, Any]) -> None:
    results = summary["condition_results"]
    equal(results["curve_left_010"]["same_sign_episode_count"], 0, "left 0.10 evidence")
    if not float(results["curve_left_010"]["mean_actual_cumulative_yaw_change_rad"]) < 0.0:
        raise ValueError("Frozen left 0.10 evidence no longer has negative yaw")
    if not float(results["curve_right_010"]["mean_actual_cumulative_yaw_change_rad"]) < 0.0:
        raise ValueError("Frozen right 0.10 evidence no longer has negative yaw")
    equal(bool(results["curve_left_010"]["safety_passed"]), True, "left safety evidence")
    equal(bool(results["curve_right_010"]["safety_passed"]), True, "right safety evidence")


def _validate_contact_contract(config: dict[str, Any], v3: dict[str, Any]) -> None:
    expected = dict(v3["contact_contract"])
    expected["explicit_pair_count"] = 4
    expected["friction_randomisation"] = False
    equal(config["contact_contract"], expected, "PAIR0 contact/friction contract")


def _validate_schedule(config: dict[str, Any]) -> None:
    schedule = config["command_schedule"]
    equal(
        schedule["worker_magnitude_assignment"],
        [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.35, 0.35],
        "worker magnitude assignment",
    )
    equal(
        schedule["expected_worker_episode_and_exposure_table_canonical_sha256"],
        canonical_sha256(schedule_rows()),
        "schedule table digest",
    )
    expected = {
        condition_id: expected_condition_exposure_steps(condition_id)
        for condition_id in CONDITION_IDS
    }
    equal(schedule["expected_control_exposure_steps"], expected, "condition exposure")
    balanced = expected[BALANCED_CONDITION_ID]
    equal(sum(balanced.values()), 65_536, "balanced total exposure")
    equal(
        sum(value for key, value in balanced.items() if key.startswith("left")),
        sum(value for key, value in balanced.items() if key.startswith("right")),
        "left/right exposure",
    )
    for first, second in ((2, 3), (4, 5), (6, 7)):
        for episode in range(EPISODES_PER_WORKER):
            equal(
                scheduled_curvature_per_m(BALANCED_CONDITION_ID, first, episode),
                -scheduled_curvature_per_m(BALANCED_CONDITION_ID, second, episode),
                f"paired phase ranks {first}/{second} episode {episode}",
            )


def validate_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, PPO]:
    equal(
        config.get("status"),
        "predeclared_validate_and_preflight_only_not_authorised_to_train",
        "configuration status",
    )
    source = config["source"]
    v3 = verified_json(source, "v3_configuration", "v3_configuration_sha256")
    manifest = verified_json(source, "v3_manifest", "v3_manifest_sha256")
    gate = verified_json(source, "v3_final_gate", "v3_final_gate_sha256")
    reward = verified_json(source, "reward_configuration", "reward_configuration_sha256")
    protocol = verified_json(source, "standard_protocol", "standard_protocol_sha256")
    verified_json(source, "flat_turn_diagnostic_manifest", "flat_turn_diagnostic_manifest_sha256")
    flat_summary = verified_json(
        source,
        "flat_turn_diagnostic_summary",
        "flat_turn_diagnostic_summary_sha256",
    )
    _validate_flat_turn_evidence(flat_summary)
    equal(bool(gate.get("passed")), True, "V3 source gate")
    equal(int(gate.get("absolute_final_checkpoint", -1)), 2_727_936, "V3 final time")
    equal(bool(manifest.get("candidate_promoted", False)), False, "source promotion boundary")

    checkpoint = ROOT / str(source["checkpoint"])
    equal(sha256(checkpoint), source["checkpoint_sha256"], "source checkpoint SHA-256")
    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    equal(torch.get_num_threads(), 2, "preflight torch threads")
    source_model = PPO.load(checkpoint, device="cpu")
    equal(int(source_model.num_timesteps), 2_727_936, "source num_timesteps")
    equal(int(source_model.n_envs), 4, "source saved n_envs")
    equal(int(source_model.n_steps), 512, "source saved n_steps")
    equal(tuple(source_model.observation_space.shape or ()), (135,), "source observation shape")
    equal(tuple(source_model.action_space.shape or ()), (8,), "source action shape")
    if len(source_model.policy.optimizer.state) == 0:
        raise ValueError("source optimiser state is empty")

    equal([row["condition_id"] for row in config["conditions"]], list(CONDITION_IDS), "condition order")
    training = config["training"]
    equal(int(training["parallel_environments"]), 8, "parallel environments")
    equal(int(training["episode_steps"]), 512, "episode horizon")
    equal(int(training["episodes_per_worker"]), 16, "episodes per worker")
    equal(int(training["steps_per_worker"]), 8192, "steps per worker")
    equal(int(training["additional_timesteps_per_condition"]), 65_536, "budget")
    equal(int(training["absolute_final_checkpoint_timesteps"]), 2_793_472, "final time")
    equal(
        training["worker_effective_first_reset_seeds"],
        [int(training["master_seed"]) + index for index in range(8)],
        "effective first seeds",
    )
    _validate_schedule(config)

    ppo = config["ppo"]
    for key, expected in {
        "n_steps": 256,
        "parallel_environments": 8,
        "rollout_transitions": 2048,
        "batch_size": 1024,
        "minibatches_per_epoch": 2,
        "n_epochs": 10,
        "gradient_minibatches_per_condition": 640,
    }.items():
        equal(int(ppo[key]), expected, f"PPO {key}")
    equal(int(ppo["n_steps"]) * int(ppo["parallel_environments"]), 2048, "rollout arithmetic")
    equal(65_536 % 2048, 0, "complete rollout budget")
    for key in ("batch_size", "n_epochs"):
        equal(int(getattr(source_model, key)), int(ppo[key]), f"source PPO {key}")
    for key in ("gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm"):
        equal(float(getattr(source_model, key)), float(ppo[key]), f"source PPO {key}")
    equal(bool(source_model.normalize_advantage), bool(ppo["normalize_advantage"]), "normalise advantage")
    equal(float(source_model.clip_range(1.0)), float(ppo["clip_range"]), "clip range")
    equal(
        ppo["load_call_contract"],
        "PPO.load(source_env8_device_cpu_force_reset_true_n_steps256_seed63806)",
        "continuation load contract",
    )
    equal(
        ppo["only_permitted_checkpoint_ppo_overrides"],
        ["n_steps_from_512_to_256_before_setup_model", "seed_from_62802_to_63806"],
        "permitted checkpoint PPO overrides",
    )
    equal(int(training["training_seed_count"]), 1, "training seed count")
    equal(
        bool(training["multi_seed_training_robustness_claim_permitted"]),
        False,
        "single-seed claim boundary",
    )

    _validate_contact_contract(config, v3)
    final = config["final_evaluation"]
    equal(tuple(final["heldout_seeds"]), EXPECTED_HELDOUT_SEEDS, "new heldout seeds")
    equal(len(set(final["heldout_seeds"])), 5, "heldout seed uniqueness")
    if set(final["heldout_seeds"]) & set(range(82800, 82900)):
        raise ValueError("828xx continuity seeds cannot be relabelled held out")
    if set(final["heldout_seeds"]) & set(range(83800, 83900)):
        raise ValueError("838xx V3 heldout seeds cannot be reused")
    if set(final["heldout_seeds"]) & set(range(94100, 94200)):
        raise ValueError("941xx slope diagnostic seeds cannot be reused")
    if set(final["heldout_seeds"]) & set(range(95100, 95200)):
        raise ValueError("951xx turn diagnostic seeds cannot be reused")
    equal(tuple(final["turn_condition_order"]), EXPECTED_TURN_CONDITIONS, "turn conditions")
    equal(tuple(final["slope_scene_order"]), EXPECTED_SLOPE_SCENES, "slope scenes")
    equal(int(final["max_episode_steps"]), 600, "evaluation horizon")

    gates = config["prospective_final_gate"]
    equal(float(gates["turn_effectiveness"]["maximum_absolute_straight_mean_yaw_change_rad"]), 0.5, "straight yaw gate")
    equal(int(gates["turn_effectiveness"]["required_same_sign_episode_count_each_nonzero_turn_condition"]), 5, "turn sign gate")
    equal(float(gates["turn_effectiveness"]["minimum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]), 0.7, "turn ratio low")
    equal(float(gates["turn_effectiveness"]["maximum_mean_yaw_change_target_ratio_abs_curvature_010_and_020"]), 1.3, "turn ratio high")
    equal(float(gates["turn_effectiveness"]["maximum_left_right_mean_ratio_difference_abs_curvature_010_and_020"]), 0.3, "turn symmetry 0.1/0.2")
    equal(float(gates["turn_effectiveness"]["minimum_absolute_mean_yaw_change_target_ratio_abs_curvature_035"]), 0.5, "turn ratio 0.35")
    equal(float(gates["turn_effectiveness"]["maximum_left_right_absolute_mean_ratio_difference_abs_curvature_035"]), 0.5, "turn symmetry 0.35")
    equal(config["energy_boundary"]["status"], "measurement_only_not_reward_or_gate", "energy status")
    equal(float(config["energy_boundary"]["reward_weight"]), 0.0, "energy reward weight")
    equal(config["invariants"]["only_condition_difference"], "per_episode_external_curvature_schedule", "only condition difference")
    equal(bool(config["execution"]["formal_training_authorised"]), False, "training authorisation")
    equal(
        config["execution"]["permitted_modes"],
        ["validate_only", "engineering_preflight_without_training_or_artifact_root"],
        "permitted execution modes",
    )
    terminal = config["terminal_project_decision"]
    for key in (
        "this_is_the_last_locomotion_optimisation_round",
        "hard_stop_after_final_C0_C1_training_and_predeclared_turn_slope_safety_retest",
        "no_further_structure_reward_contact_DOF_or_training_intervention_after_pass_or_fail",
        "retain_both_final_C0_and_C1_checkpoints_and_complete_evidence_regardless_of_result",
    ):
        equal(bool(terminal[key]), True, f"terminal decision {key}")
    equal(bool(terminal["fixed_map_entry_after_this_protocol"]), False, "terminal fixed-map boundary")
    video = config["post_result_video_archive_contract"]
    equal(bool(video["training_stage_video_rendering"]), False, "training video boundary")
    for key in (
        "render_only_after_all_numeric_results_and_gates_are_frozen",
        "video_not_used_by_any_gate_or_seed_selection",
        "C0_C1_same_seed_left_right_comparison_required",
        "seed_predeclared_before_training",
        "trace_manifest_sha256_and_full_decode_validation_required",
    ):
        equal(bool(video[key]), True, f"post-result video {key}")
    equal(int(video["predeclared_seed"]), 96131, "post-result video seed")
    equal(
        video["predeclared_turn_conditions"],
        ["curve_left_020", "curve_right_020"],
        "post-result video turn conditions",
    )
    equal(bool(video["video_changes_scientific_result"]), False, "video scientific boundary")
    validate_runtime_dependencies(config)
    return protocol, reward, checkpoint, source_model


def prepare_flat_pair0_scene(
    config: dict[str, Any], protocol: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    training = config["training"]
    local_protocol = copy.deepcopy(protocol)
    standard = local_protocol["standard_scenes"]
    standard["scene_order"] = ["flat"]
    standard["grid_rows"] = int(training["flat_scene_grid_rows"])
    standard["grid_cols"] = int(training["flat_scene_grid_cols"])
    standard["map_half_extent_m"] = float(training["flat_scene_map_half_extent_m"])
    standard["start_xy_m"] = list(training["flat_scene_start_xy_m"])
    standard["goal_xy_m"] = list(training["flat_scene_goal_marker_xy_m"])
    controls, _ = slope.prepare_standard_scenes(
        local_protocol, output_root / "scene_source"
    )
    source = dict(controls["flat"])
    observed = np.load(source["heights_path"], allow_pickle=False)
    expected = slope.build_standard_heights(local_protocol["standard_scenes"])[
        "flat"
    ]
    if (
        observed.shape != (513, 513)
        or not np.array_equal(observed, expected)
        or float(np.ptp(observed)) > 3.0e-6 + 1e-15
    ):
        raise RuntimeError("training flat heightfield contract changed")
    source.update(
        {
            "scene_name": "turn_balance_flat",
            "direction": "flat",
            "angle_degrees": 0,
        }
    )
    pair, _ = slope.prepare_pair(
        source,
        output_root / "condition_assets",
        "turn_balance_flat",
        l2._pair_contract(config),
    )
    scene = dict(pair[slope.CANDIDATE_ID])
    scene.update(
        {
            "condition_id": l2.PAIR0_ID,
            "scene_name": "turn_balance_flat",
            "direction": "flat",
            "angle_degrees": 0,
        }
    )
    return scene


def make_turn_balance_env(
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    scene: dict[str, Any],
    *,
    condition_id: str,
    worker_rank: int,
    seed: int,
) -> PairedTurnBalanceTerrainWrapper:
    speed = float(config["training"]["cruise_speed_m_per_s"])
    task = protocol["task_adapter"]
    curve_env = make_curved_gait_env(
        condition_id=l2.PAIR0_ID,
        seed=int(seed),
        render_mode=None,
        xml_file=Path(scene["xml_path"]),
        max_episode_steps=int(config["training"]["episode_steps"]),
        terminate_when_unhealthy=False,
        profile="external",
        speed_min=speed,
        speed_max=speed,
        max_abs_curvature=0.35,
        max_abs_lateral_speed=0.0,
        fixed_lateral_speed=0.0,
        heading_termination_enabled=False,
        terrain_frame_shaping_enabled=False,
        **common_env_kwargs(reward),
    )
    env = PairedTurnBalanceTerrainWrapper(
        curve_env,
        condition_id=condition_id,
        worker_rank=worker_rank,
        expected_episode_steps=EPISODE_STEPS,
        fixed_speed_m_per_s=speed,
        fail_closed_training_contract=True,
        heights_path=Path(scene["heights_path"]),
        expected_height_sha256=scene["heights_sha256"],
        map_half_extent_m=float(scene["map_half_extent_m"]),
        start_xy_m=scene["start_xy_m"],
        goal_xy_m=scene["goal_xy_m"],
        spawn_fraction=0.0,
        cruise_speed_m_per_s=speed,
        maximum_abs_curvature_per_m=0.35,
        yaw_gain_per_second=float(task["yaw_gain_per_second"]),
        slow_radius_m=float(task["slow_radius_m"]),
        arrival_radius_m=float(task["arrival_radius_m"]),
        hold_radius_m=float(task["hold_radius_m"]),
        hold_seconds=float(task["hold_seconds"]),
        hold_speed_m_per_s=float(task["hold_speed_m_per_s"]),
        terminate_on_success=False,
        terrain_relative_healthy_clearance_m=tuple(task["terrain_relative_healthy_clearance_m"]),
        maximum_healthy_tilt_degrees=float(task["maximum_healthy_tilt_degrees"]),
        unhealthy_grace_steps=int(task["unhealthy_grace_steps"]),
        slip_speed_threshold_m_per_s=float(task["slip_speed_threshold_m_per_s"]),
        augment_local_terrain_observation=True,
        terrain_frame_shaping_enabled=False,
        terrain_preview_longitudinal_m=tuple(task["terrain_preview_longitudinal_m"]),
        terrain_preview_lateral_m=tuple(task["terrain_preview_lateral_m"]),
        local_terrain_height_bound_m=float(task["local_terrain_height_bound_m"]),
    )
    audit = l2.compiled_contract_audit(
        env.unwrapped.model,
        scene,
        l2.PAIR0_ID,
        config,
        construction_seed=seed,
    )
    if not bool(audit["passed"]):
        env.close()
        raise RuntimeError("compiled PAIR0 contract preflight failed")
    return env


class StaticContractEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata: dict[str, Any] = {}

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__()
        self.observation_space = copy.deepcopy(observation_space)
        self.action_space = copy.deepcopy(action_space)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action: np.ndarray):
        del action
        return (
            np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype),
            0.0,
            False,
            False,
            {},
        )


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _box_contract_equal(left: gym.spaces.Space, right: gym.spaces.Space) -> bool:
    return bool(
        isinstance(left, gym.spaces.Box)
        and isinstance(right, gym.spaces.Box)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left.low, right.low)
        and np.array_equal(left.high, right.high)
    )


def load_continuation_model(
    checkpoint: Path,
    vec_env: DummyVecEnv,
    config: dict[str, Any],
) -> PPO:
    """Fresh-load one branch with frozen pre-setup ``n_steps`` and RNG seed."""

    torch.set_num_threads(int(config["ppo"]["torch_num_threads"]))
    equal(torch.get_num_threads(), 2, "continuation loader torch threads")
    master_seed = int(config["training"]["master_seed"])
    model = PPO.load(
        checkpoint,
        env=vec_env,
        device=str(config["ppo"]["device"]),
        force_reset=True,
        n_steps=int(config["ppo"]["n_steps"]),
        seed=master_seed,
    )
    checks = {
        "source_timestep": int(model.num_timesteps) == 2_727_936,
        "n_envs": int(model.n_envs) == 8,
        "n_steps": int(model.n_steps) == 256,
        "buffer_size": int(model.rollout_buffer.buffer_size) == 256,
        "buffer_n_envs": int(model.rollout_buffer.n_envs) == 8,
        "batch_size": int(model.batch_size) == 1024,
        "n_epochs": int(model.n_epochs) == 10,
        "seed": int(model.seed) == master_seed,
        "force_reset": model._last_obs is None,
        "optimizer_nonempty": len(model.policy.optimizer.state) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"loaded continuation contract failed: {checks}")
    return model


def loader_preflight(
    config: dict[str, Any], checkpoint: Path, source_model: PPO
) -> dict[str, Any]:
    factories: list[Callable[[], gym.Env]] = [
        lambda: StaticContractEnv(
            source_model.observation_space, source_model.action_space
        )
        for _ in range(8)
    ]
    vec_env = DummyVecEnv(factories)
    try:
        loaded = load_continuation_model(checkpoint, vec_env, config)
        effective_first_reset_seeds = [
            int(value)
            for value in vec_env.seed(int(config["training"]["master_seed"]))
        ]
        policy_exact = _nested_equal(
            source_model.policy.state_dict(), loaded.policy.state_dict()
        )
        optimiser_exact = _nested_equal(
            source_model.policy.optimizer.state_dict(),
            loaded.policy.optimizer.state_dict(),
        )
        probe = np.zeros((1, 135), dtype=np.float32)
        source_action, _ = source_model.predict(probe, deterministic=True)
        loaded_action, _ = loaded.predict(probe, deterministic=True)
        action_exact = bool(np.array_equal(source_action, loaded_action))
        checks = {
            "policy_state_exact": policy_exact,
            "optimizer_state_exact": optimiser_exact,
            "deterministic_action_exact": action_exact,
            "num_timesteps_preserved": int(loaded.num_timesteps) == 2_727_936,
            "n_envs_is_8": int(loaded.n_envs) == 8,
            "n_steps_is_256": int(loaded.n_steps) == 256,
            "rollout_buffer_is_256_by_8": (
                int(loaded.rollout_buffer.buffer_size) == 256
                and int(loaded.rollout_buffer.n_envs) == 8
            ),
            "batch_size_preserved": int(loaded.batch_size) == 1024,
            "n_epochs_preserved": int(loaded.n_epochs) == 10,
            "seed_override_is_master_seed": int(loaded.seed)
            == int(config["training"]["master_seed"]),
            "pending_first_reset_seeds_match": effective_first_reset_seeds
            == list(config["training"]["worker_effective_first_reset_seeds"]),
            "force_reset_removed_last_observation": loaded._last_obs is None,
        }
        if not all(checks.values()):
            raise RuntimeError(f"continuation loader preflight failed: {checks}")
        loaded._setup_learn(
            int(config["training"]["additional_timesteps_per_condition"]),
            reset_num_timesteps=False,
            tb_log_name="turn_balance_preflight",
            progress_bar=False,
        )
        setup_checks = {
            "resolved_total_timesteps": int(loaded._total_timesteps) == 2_793_472,
            "counter_unchanged_before_rollout": int(loaded.num_timesteps) == 2_727_936,
            "last_observation_shape": tuple(loaded._last_obs.shape) == (8, 135),
            "episode_start_shape": tuple(loaded._last_episode_starts.shape) == (8,),
        }
        if not all(setup_checks.values()):
            raise RuntimeError(f"continuation setup preflight failed: {setup_checks}")
        return {
            "checks": checks,
            "setup_learn_without_rollout_or_gradient_checks": setup_checks,
            "optimizer_state_entries": len(loaded.policy.optimizer.state),
            "torch_num_threads": int(torch.get_num_threads()),
            "effective_first_reset_seeds": effective_first_reset_seeds,
            "learn_called": False,
            "gradient_update_performed": False,
            "checkpoint_written": False,
        }
    finally:
        vec_env.close()


def engineering_preflight(
    config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    checkpoint: Path,
    source_model: PPO,
) -> dict[str, Any]:
    forbidden = [
        ROOT / config["execution"]["future_smoke_output_root"],
        ROOT / config["execution"]["future_formal_output_root"],
    ]
    before = {str(path): path.exists() for path in forbidden}
    if any(before.values()):
        raise FileExistsError("future smoke/formal root already exists")
    with tempfile.TemporaryDirectory(prefix="proxygap_turn_balance_preflight_") as raw:
        temporary_root = Path(raw)
        scene = prepare_flat_pair0_scene(config, protocol, temporary_root)
        model = mujoco.MjModel.from_xml_path(scene["xml_path"])
        equal(int(model.npair), 4, "preflight explicit pair count")
        env = make_turn_balance_env(
            config,
            protocol,
            reward,
            scene,
            condition_id=BALANCED_CONDITION_ID,
            worker_rank=2,
            seed=int(config["training"]["master_seed"]),
        )
        try:
            observation, info = env.reset(seed=int(config["training"]["master_seed"]))
            if not _box_contract_equal(env.observation_space, source_model.observation_space):
                raise RuntimeError("real environment observation Box differs from checkpoint")
            if not _box_contract_equal(env.action_space, source_model.action_space):
                raise RuntimeError("real environment action Box differs from checkpoint")
            equal(observation.shape, (135,), "real reset observation")
            equal(float(info["proxygap_turn_balance_curvature_per_m"]), 0.1, "first command curvature")
            if not math.isclose(
                float(observation[115]), 0.055, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError("first observation yaw-rate command changed")
            if not np.all(np.isfinite(observation)):
                raise RuntimeError("real reset observation is non-finite")
        finally:
            env.close()
        loader = loader_preflight(config, checkpoint, source_model)
    after = {str(path): path.exists() for path in forbidden}
    equal(after, before, "preflight formal/smoke root non-mutation")
    equal(sha256(checkpoint), config["source"]["checkpoint_sha256"], "checkpoint after preflight")
    return {
        "status": "ENGINEERING_PREFLIGHT_OK_NO_TRAINING",
        "real_pair0_environment": {
            "npair": int(model.npair),
            "observation_dimension": 135,
            "action_dimension": 8,
            "first_worker2_curvature_per_m": 0.1,
            "first_worker2_yaw_rate_rad_per_s": 0.055,
        },
        "loader": loader,
        "temporary_directory_deleted": True,
        "smoke_or_formal_root_created": False,
        "learn_called": False,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical design configuration may be checked")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol, reward, checkpoint, source_model = validate_config(config)
    if args.validate_only:
        print("VALIDATION_OK_NO_TRAINING_AUTHORISED")
        return
    result = engineering_preflight(
        config, protocol, reward, checkpoint, source_model
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
