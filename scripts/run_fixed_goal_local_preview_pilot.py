"""Run the observation-only 122-to-135 local-terrain-preview pilot.

This runner creates a fresh PPO optimiser, performs a fail-closed zero-column
policy migration, verifies action-distribution and value parity, and only then
starts the bounded fixed-map pilot.  Existing checkpoints and frozen terrain
assets are read-only inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.experiment import write_rows  # noqa: E402
from proxygap.planar_transition import make_ppo_from_config  # noqa: E402
from proxygap.ppo_observation_transfer import (  # noqa: E402
    transfer_ppo_with_appended_observations,
    verify_ppo_appended_observation_equivalence,
)
from run_fixed_goal_terrain_training import (  # noqa: E402
    evaluate_checkpoint,
    keep_windows_awake,
    prepare_task_scenes,
    sha256,
    vector_env,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_local_preview_pilot_v1_20260819.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--migration-only",
        action="store_true",
        help="Stop after migration, parity verification and initial evaluation.",
    )
    return parser.parse_args()


def _load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed.lower() != str(expected_sha256).lower():
        raise ValueError(f"SHA-256 mismatch for {path}: {observed}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the observation-only intervention and all frozen inputs."""

    if config.get("status") != "frozen_local_terrain_preview_pilot":
        raise ValueError("Local terrain preview pilot configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("The pilot must prohibit a formal generalisation claim")

    comparison = config["comparison_control"]
    control_path = ROOT / comparison["configuration"]
    control = _load_verified_json(
        control_path,
        comparison["configuration_sha256"],
    )
    if comparison.get("permitted_intervention") != "append_local_terrain_observation_only":
        raise ValueError("Unexpected pilot intervention declaration")
    if config["approved_map"] != control["approved_map"]:
        raise ValueError("Approved map, friction or XML differs from the control")

    task = config["task_adapter"]
    control_task = control["task_adapter"]
    for key, value in control_task.items():
        if task.get(key) != value:
            raise ValueError(f"Task adapter changed control field: {key}")
    permitted_task_additions = {
        "augment_local_terrain_observation",
        "terrain_preview_longitudinal_m",
        "terrain_preview_lateral_m",
    }
    if set(task) != set(control_task) | permitted_task_additions:
        raise ValueError("Task adapter contains an unapproved intervention")
    if task["augment_local_terrain_observation"] is not True:
        raise ValueError("Local terrain preview must be enabled")
    if float(task["additional_task_reward"]) != 0.0:
        raise ValueError("The observation pilot must not add task reward")

    control_training = control["training"]
    training = config["training"]
    for key in ("parallel_environments", "spawn_fractions", "max_episode_steps"):
        if training[key] != control_training[key]:
            raise ValueError(f"Training control field changed: {key}")
    control_stage_values = [
        (
            int(stage["additional_target_timesteps"]),
            float(stage["cruise_speed_m_per_s"]),
        )
        for stage in control_training["stages"]
    ]
    preview_stage_values = [
        (
            int(stage["additional_target_timesteps"]),
            float(stage["cruise_speed_m_per_s"]),
        )
        for stage in training["stages"]
    ]
    if preview_stage_values != control_stage_values:
        raise ValueError("Training budget or cruise-speed curriculum changed")
    if config["evaluation"] != control["evaluation"]:
        raise ValueError("Evaluation protocol differs from the control")
    for key, value in control["ppo"].items():
        if config["ppo"].get(key) != value:
            raise ValueError(f"PPO control field changed: {key}")

    base = config["base_policy"]
    v22_path = ROOT / base["configuration"]
    v22_config = _load_verified_json(v22_path, base["configuration_sha256"])
    _load_verified_json(
        ROOT / base["source_run_configuration"],
        base["source_run_configuration_sha256"],
    )
    source_path = ROOT / base["model_path"]
    if not source_path.is_file() or sha256(source_path) != base["model_sha256"]:
        raise ValueError("Source checkpoint is missing or has changed")
    source_model = PPO.load(source_path, device="cpu")
    if int(source_model.observation_space.shape[0]) != int(
        base["observation_dimension"]
    ):
        raise ValueError("Source observation dimension mismatch")
    if int(source_model.action_space.shape[0]) != int(base["action_dimension"]):
        raise ValueError("Source action dimension mismatch")
    if int(source_model.num_timesteps) != int(base["source_timesteps"]):
        raise ValueError("Source checkpoint timestep count mismatch")

    transfer = config["observation_transfer"]
    longitudinal = list(task["terrain_preview_longitudinal_m"])
    lateral = list(task["terrain_preview_lateral_m"])
    expected_appended = len(longitudinal) * len(lateral) + 4
    if expected_appended != 13:
        raise ValueError("The local preview must contain nine heights plus four descriptors")
    if int(transfer["source_observation_dimension"]) != int(
        base["observation_dimension"]
    ):
        raise ValueError("Transfer source dimension differs from the checkpoint")
    if int(transfer["appended_columns"]) != expected_appended:
        raise ValueError("Transfer appended-column count is inconsistent")
    if len(transfer["appended_feature_names"]) != expected_appended:
        raise ValueError("Transfer feature-name count is inconsistent")
    if int(transfer["target_observation_dimension"]) != int(
        transfer["source_observation_dimension"]
    ) + expected_appended:
        raise ValueError("Transfer target dimension is inconsistent")
    if float(transfer["new_columns_initial_value"]) != 0.0:
        raise ValueError("New observation weights must be initialised to zero")
    if transfer["optimizer_state"] != "fresh":
        raise ValueError("Observation migration requires a fresh optimiser")
    return control, v22_config


def _effective_ppo_config(
    config: dict[str, Any],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo = dict(config["ppo"])
    smoke_overrides: dict[str, Any] = {}
    if smoke:
        smoke_overrides = {"batch_size": int(ppo["n_steps"]), "n_epochs": 1}
        ppo.update(smoke_overrides)
    return ppo, smoke_overrides


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, v22_config = validate_config(config)
    keep_windows_awake()

    if args.output_root:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = ROOT / "artifacts" / "smoke" / config["config_id"]
    else:
        output_root = (
            ROOT
            / config["execution"]["output_root"]
            / f"seed_{config['training']['training_seed']}"
        )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    (output_root / "logs").mkdir(parents=True)
    (output_root / "models").mkdir(parents=True)
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())

    training = config["training"]
    evaluation = config["evaluation"]
    transfer_config = config["observation_transfer"]
    ppo_config, smoke_overrides = _effective_ppo_config(
        config,
        smoke=bool(args.smoke),
    )
    training_seed = int(training["training_seed"])
    spawn_fractions = [float(value) for value in training["spawn_fractions"]]
    if args.smoke:
        spawn_fractions = spawn_fractions[:1]
    scene_paths, spawn_metadata = prepare_task_scenes(
        config,
        output_root,
        spawn_fractions,
    )
    (output_root / "spawn_manifest.json").write_text(
        json.dumps(spawn_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    initial_speed = float(training["stages"][0]["cruise_speed_m_per_s"])
    training_max_steps = 160 if args.smoke else int(training["max_episode_steps"])
    env = vector_env(
        config,
        v22_config,
        scene_paths=scene_paths,
        spawn_fractions=spawn_fractions,
        seed=training_seed,
        max_episode_steps=training_max_steps,
        cruise_speed=initial_speed,
        monitor_path=output_root / "logs" / "training_vecmonitor.csv",
    )
    source_path = ROOT / config["base_policy"]["model_path"]
    torch.set_num_threads(int(ppo_config["torch_num_threads"]))
    source_model = PPO.load(source_path, device="cpu")
    model = make_ppo_from_config(env, ppo_config, seed=training_seed)

    migration = transfer_ppo_with_appended_observations(
        source_model,
        model,
        appended_feature_names=transfer_config["appended_feature_names"],
        restore_num_timesteps=bool(
            transfer_config["restore_source_num_timesteps"]
        ),
    )
    initial_observations = env.reset()
    source_observations = initial_observations[
        :, : int(transfer_config["source_observation_dimension"])
    ]
    configured_tolerance = float(
        transfer_config["required_equivalence_tolerance"]
    )
    effective_tolerance = max(
        configured_tolerance,
        float(np.finfo(np.float32).eps),
    )
    parity = verify_ppo_appended_observation_equivalence(
        source_model,
        model,
        source_observations=source_observations,
        target_observations=initial_observations,
        tolerance=effective_tolerance,
    )
    if not parity["equivalent_within_tolerance"]:
        raise RuntimeError(
            "122-to-135 policy migration failed initial equivalence: "
            f"{parity}"
        )

    initial_model_path = (
        output_root
        / "models"
        / f"initial_zero_column_transfer_{int(model.num_timesteps)}.zip"
    )
    model.save(initial_model_path)
    reloaded_initial_model = PPO.load(initial_model_path, device="cpu")
    reloaded_parity = verify_ppo_appended_observation_equivalence(
        source_model,
        reloaded_initial_model,
        source_observations=source_observations,
        target_observations=initial_observations,
        tolerance=effective_tolerance,
    )
    if not reloaded_parity["equivalent_within_tolerance"]:
        raise RuntimeError(
            "Saved 135-value checkpoint failed reload equivalence: "
            f"{reloaded_parity}"
        )
    runner_path = Path(__file__).resolve()
    transfer_module_path = (
        ROOT / "src" / "proxygap" / "ppo_observation_transfer.py"
    )
    migration.update(
        {
            "source_model": str(source_path),
            "source_model_sha256": sha256(source_path),
            "source_configuration": str(
                ROOT / config["base_policy"]["source_run_configuration"]
            ),
            "source_configuration_sha256": config["base_policy"][
                "source_run_configuration_sha256"
            ],
            "pilot_configuration": str(config_path),
            "pilot_configuration_sha256": sha256(config_path),
            "approved_height_sha256": config["approved_map"]["heights_sha256"],
            "approved_xml_sha256": config["approved_map"]["xml_sha256"],
            "fixed_friction": config["approved_map"]["fixed_friction"],
            "condim": config["approved_map"]["condim"],
            "migration_runner": str(runner_path),
            "migration_runner_sha256": sha256(runner_path),
            "transfer_module": str(transfer_module_path),
            "transfer_module_sha256": sha256(transfer_module_path),
            "initial_equivalence": parity,
            "initial_migrated_model": str(initial_model_path),
            "initial_migrated_model_sha256": sha256(initial_model_path),
            "reloaded_initial_equivalence": reloaded_parity,
            "reloaded_optimizer_state_entries": len(
                reloaded_initial_model.policy.optimizer.state
            ),
        }
    )
    migration_path = output_root / "migration_manifest.json"
    migration_path.write_text(
        json.dumps(migration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    execution: dict[str, Any] = {
        "schema_version": "proxygap-fixed-goal-local-preview-pilot-v1",
        "status": "started",
        "smoke": bool(args.smoke),
        "migration_only": bool(args.migration_only),
        "training_seed": training_seed,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_model_sha256": sha256(source_path),
        "approved_height_sha256": config["approved_map"]["heights_sha256"],
        "observation_dimension": int(model.observation_space.shape[0]),
        "action_dimension": int(model.action_space.shape[0]),
        "fresh_optimizer_state_entries": len(model.policy.optimizer.state),
        "effective_ppo": {
            key: ppo_config[key]
            for key in (
                "n_steps",
                "batch_size",
                "n_epochs",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
                "normalize_advantage",
                "device",
                "torch_num_threads",
            )
        },
        "smoke_only_ppo_overrides": smoke_overrides,
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")

    evaluation_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    try:
        eval_seeds = (
            [int(evaluation["validation_seeds"][0])]
            if args.smoke
            else [int(value) for value in evaluation["validation_seeds"]]
        )
        eval_max_steps = 160 if args.smoke else int(
            evaluation["full_route_max_episode_steps"]
        )
        evaluation_rows.extend(
            evaluate_checkpoint(
                model,
                config,
                v22_config,
                start_scene=scene_paths[0],
                checkpoint_label="initial_zero_column_transfer",
                checkpoint_timesteps=int(model.num_timesteps),
                seeds=eval_seeds,
                max_episode_steps=eval_max_steps,
                cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
            )
        )
        write_rows(output_root / "logs" / "evaluation_episodes.csv", evaluation_rows)

        if args.migration_only:
            execution.update(
                {
                    "status": "migration_verified",
                    "completed_model_timesteps": int(model.num_timesteps),
                    "added_training_timesteps": 0,
                    "evaluation_episode_rows": len(evaluation_rows),
                }
            )
        else:
            source_timesteps = int(model.num_timesteps)
            if args.smoke:
                stages = [
                    {
                        "name": "smoke_local_preview_512",
                        "additional_target_timesteps": int(ppo_config["n_steps"]),
                        "cruise_speed_m_per_s": initial_speed,
                    }
                ]
            else:
                stages = list(training["stages"])
            rollout_size = len(spawn_fractions) * int(ppo_config["n_steps"])
            for stage_index, stage in enumerate(stages):
                additional_target = int(stage["additional_target_timesteps"])
                if additional_target % rollout_size:
                    raise ValueError(
                        f"Additional target {additional_target} is not divisible by "
                        f"rollout size {rollout_size}"
                    )
                absolute_target = source_timesteps + additional_target
                requested = absolute_target - int(model.num_timesteps)
                cruise_speed = float(stage["cruise_speed_m_per_s"])
                env.env_method("set_task_speed", cruise_speed)
                started = time.perf_counter()
                model.learn(total_timesteps=requested, reset_num_timesteps=False)
                elapsed = time.perf_counter() - started
                checkpoint_path = (
                    output_root / "models" / f"checkpoint_{absolute_target}.zip"
                )
                model.save(checkpoint_path)
                runtime_rows.append(
                    {
                        "stage_index": stage_index,
                        "stage_name": stage["name"],
                        "source_timesteps": source_timesteps,
                        "additional_target_timesteps": additional_target,
                        "absolute_target_timesteps": absolute_target,
                        "actual_model_timesteps": int(model.num_timesteps),
                        "requested_timesteps": requested,
                        "cruise_speed_m_per_s": cruise_speed,
                        "train_elapsed_seconds": elapsed,
                        "train_steps_per_second": requested / max(elapsed, 1e-12),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256(checkpoint_path),
                    }
                )
                write_rows(
                    output_root / "logs" / "training_runtime.csv",
                    runtime_rows,
                )
                evaluation_rows.extend(
                    evaluate_checkpoint(
                        model,
                        config,
                        v22_config,
                        start_scene=scene_paths[0],
                        checkpoint_label=str(stage["name"]),
                        checkpoint_timesteps=int(model.num_timesteps),
                        seeds=eval_seeds,
                        max_episode_steps=eval_max_steps,
                        cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
                    )
                )
                write_rows(
                    output_root / "logs" / "evaluation_episodes.csv",
                    evaluation_rows,
                )
                print(
                    json.dumps(
                        {
                            "stage": stage["name"],
                            "timesteps": int(model.num_timesteps),
                            "steps_per_second": runtime_rows[-1][
                                "train_steps_per_second"
                            ],
                        }
                    ),
                    flush=True,
                )
            execution.update(
                {
                    "status": "complete",
                    "completed_model_timesteps": int(model.num_timesteps),
                    "added_training_timesteps": int(model.num_timesteps)
                    - source_timesteps,
                    "final_model": runtime_rows[-1]["checkpoint_path"],
                    "final_model_sha256": runtime_rows[-1]["checkpoint_sha256"],
                    "evaluation_episode_rows": len(evaluation_rows),
                }
            )
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "completed_model_timesteps": int(model.num_timesteps),
            }
        )
        raise
    finally:
        record_path.write_text(
            json.dumps(execution, indent=2) + "\n",
            encoding="utf-8",
        )
        env.close()


if __name__ == "__main__":
    main()
