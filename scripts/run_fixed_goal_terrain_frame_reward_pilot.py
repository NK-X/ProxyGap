"""Run the predeclared terrain/target-frame reward-coordinate ablation.

The source policy and archived W4 continuation are evaluation controls. Only
the terrain-frame intervention is trained, from the same 135D source with the
same seed, four spawn fractions, speed, PPO hyperparameters and 131072-step
budget used by the archived W4 continuation. Relative-energy V2 remains a
measurement-only design.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from run_fixed_goal_support_priority_pilot import (  # noqa: E402
    AGGREGATE_FIELDS,
    _configure_continuation_model,
    _load_verified_json,
    aggregate_condition_rows,
    evaluate_model,
    recursive_json_differences,
)
from run_fixed_goal_terrain_training import (  # noqa: E402
    keep_windows_awake,
    prepare_task_scenes,
    sha256,
    vector_env,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _without_terrain_frame(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["task_adapter"]["terrain_frame_shaping_enabled"] = False
    return result


def _mean_support_count(row: dict[str, Any]) -> float:
    fractions = np.asarray(
        row["support_count_step_fractions_0_to_4"],
        dtype=np.float64,
    )
    if fractions.shape != (5,) or not np.all(np.isfinite(fractions)):
        raise ValueError("support-count fractions must contain five finite values")
    return float(np.dot(np.arange(5, dtype=np.float64), fractions))


def _add_support_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["mean_support_count"] = _mean_support_count(row)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the archived W4 and new intervention are matched."""

    if config.get("status") != "frozen_terrain_frame_grouped_ablation":
        raise ValueError("terrain-frame pilot is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("fixed-map pilot must prohibit generalisation claims")
    comparison = config["comparison_control"]
    if comparison["permitted_intervention"] != "terrain_frame_shaping_context_only":
        raise ValueError("unexpected intervention declaration")
    if comparison["permitted_configuration_path"] != (
        "task_adapter.terrain_frame_shaping_enabled"
    ):
        raise ValueError("terrain-frame permission is not fail-closed")
    if comparison["source_and_w4_flag"] is not False:
        raise ValueError("world-frame controls must keep the flag false")
    if comparison["intervention_flag"] is not True:
        raise ValueError("terrain-frame intervention must keep the flag true")

    preview = _load_verified_json(
        ROOT / comparison["local_preview_configuration"],
        comparison["local_preview_configuration_sha256"],
    )
    source_run = _load_verified_json(
        ROOT / comparison["source_run_configuration"],
        comparison["source_run_configuration_sha256"],
    )
    archived_w4 = _load_verified_json(
        ROOT / comparison["archived_w4_configuration"],
        comparison["archived_w4_configuration_sha256"],
    )
    if source_run != preview:
        raise ValueError("source frozen configuration differs from preview control")
    for key in ("approved_map", "ppo"):
        if config[key] != archived_w4[key] or config[key] != preview[key]:
            raise ValueError(f"terrain-frame pilot changed frozen field: {key}")

    expected_task = copy.deepcopy(preview["task_adapter"])
    expected_task["terrain_frame_shaping_enabled"] = True
    expected_task["terrain_frame_components"] = config["task_adapter"][
        "terrain_frame_components"
    ]
    if config["task_adapter"] != expected_task:
        differences = recursive_json_differences(
            expected_task,
            config["task_adapter"],
        )
        raise ValueError(f"terrain-frame task changed unapproved fields: {differences}")

    training = config["training"]
    archived_training = archived_w4["training"]
    for key in (
        "training_seed",
        "parallel_environments",
        "spawn_fractions",
        "max_episode_steps",
        "additional_target_timesteps",
        "cruise_speed_m_per_s",
    ):
        if training[key] != archived_training[key]:
            raise ValueError(f"training control changed: {key}")
    rollout = int(config["ppo"]["n_steps"]) * int(
        training["parallel_environments"]
    )
    if int(training["additional_target_timesteps"]) % rollout:
        raise ValueError("training budget is not divisible by vector rollout")

    evaluation = config["evaluation"]
    archived_evaluation = archived_w4["evaluation"]
    for key in (
        "max_episode_steps",
        "cruise_speed_m_per_s",
        "validation_seeds",
        "deterministic_policy",
        "representative_trace_seed",
        "slip_transient_minimum_seconds",
    ):
        if evaluation[key] != archived_evaluation[key]:
            raise ValueError(f"evaluation control changed: {key}")
    if evaluation["conditions"] != [
        "SOURCE_STAGE1_WORLD_FRAME",
        "W4_ARCHIVED_WORLD_FRAME_CONTROL",
        "TERRAIN_FRAME_REWARD_INTERVENTION",
    ]:
        raise ValueError("evaluation conditions differ from the predeclaration")

    base = config["base_policy"]
    reward_config = _load_verified_json(
        ROOT / base["reward_configuration"],
        base["reward_configuration_sha256"],
    )
    if float(reward_config["preserved_pre_pitch_reward"]["ctrl_cost_weight"]) != 0.5:
        raise ValueError("ctrl_cost_weight changed")
    if float(
        reward_config["preserved_pre_pitch_reward"]["airborne_shaping_weight"]
    ) != 4.0:
        raise ValueError("active W4 reward weight changed")
    if config["energy_boundary"]["relative_mission_energy_v2_status"] != (
        "measurement_only_not_implemented_as_reward"
    ):
        raise ValueError("energy V2 must remain measurement-only")

    source_path = ROOT / base["model_path"]
    w4_path = ROOT / comparison["archived_w4_model_path"]
    for path, expected_hash, expected_steps in (
        (source_path, base["model_sha256"], base["source_timesteps"]),
        (
            w4_path,
            comparison["archived_w4_model_sha256"],
            comparison["archived_w4_timesteps"],
        ),
    ):
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"checkpoint missing or changed: {path}")
        model = PPO.load(path, device="cpu")
        if int(model.num_timesteps) != int(expected_steps):
            raise ValueError(f"checkpoint timestep mismatch: {path}")
        if tuple(model.observation_space.shape) != (135,):
            raise ValueError(f"checkpoint observation dimension mismatch: {path}")
        if tuple(model.action_space.shape) != (8,):
            raise ValueError(f"checkpoint action dimension mismatch: {path}")
    return reward_config


def build_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    aggregates = {
        condition: aggregate_condition_rows(condition_rows)
        for condition, condition_rows in by_condition.items()
    }
    for condition, condition_rows in by_condition.items():
        support = np.asarray(
            [float(row["mean_support_count"]) for row in condition_rows],
            dtype=np.float64,
        )
        aggregates[condition]["mean_support_count_mean"] = float(np.mean(support))
        aggregates[condition]["mean_support_count_std_population"] = float(
            np.std(support)
        )

    control_id = "W4_ARCHIVED_WORLD_FRAME_CONTROL"
    intervention_id = "TERRAIN_FRAME_REWARD_INTERVENTION"
    control = aggregates[control_id]
    intervention = aggregates[intervention_id]
    gate = config["evaluation"]["retention_gate"]
    airborne_reduction = float(control["task_airborne_step_fraction_mean"]) - float(
        intervention["task_airborne_step_fraction_mean"]
    )
    progress_ratio = float(intervention["fixed_goal_best_progress_m_mean"]) / max(
        float(control["fixed_goal_best_progress_m_mean"]),
        1e-12,
    )
    additional_falls = int(intervention["fall_count"]) - int(control["fall_count"])
    passed = bool(
        airborne_reduction
        >= float(gate["minimum_absolute_airborne_fraction_reduction_vs_w4"])
        and progress_ratio >= float(gate["minimum_best_progress_ratio_vs_w4"])
        and additional_falls <= int(gate["maximum_additional_falls_vs_w4"])
    )

    paired: list[dict[str, Any]] = []
    control_by_seed = {
        int(row["evaluation_seed"]): row for row in by_condition[control_id]
    }
    intervention_by_seed = {
        int(row["evaluation_seed"]): row for row in by_condition[intervention_id]
    }
    paired_fields = (*AGGREGATE_FIELDS, "mean_support_count")
    for seed in sorted(set(control_by_seed) & set(intervention_by_seed)):
        left = control_by_seed[seed]
        right = intervention_by_seed[seed]
        paired.append(
            {
                "evaluation_seed": seed,
                **{
                    f"delta_{field}_terrain_frame_minus_w4": float(right[field])
                    - float(left[field])
                    for field in paired_fields
                },
                "delta_fall_terrain_frame_minus_w4": int(bool(right["fall"]))
                - int(bool(left["fall"])),
            }
        )
    return {
        "schema_version": "proxygap-terrain-frame-reward-comparison-v1",
        "condition_aggregates": aggregates,
        "paired_deltas": paired,
        "retention_gate": {
            "predeclared_rule": gate,
            "observed_absolute_airborne_fraction_reduction_vs_w4": airborne_reduction,
            "observed_best_progress_ratio_vs_w4": progress_ratio,
            "observed_additional_falls_vs_w4": additional_falls,
            "passed": passed,
            "retained_condition": intervention_id if passed else "SOURCE_STAGE1_WORLD_FRAME",
            "interpretation": (
                "Terrain-frame shaping passes the development retention gate"
                if passed
                else "Terrain-frame shaping is not retained; preserve the source policy"
            ),
        },
        "coordinate_frame_choice": (
            "Full terrain normal is used for torso-up alignment; foot height and "
            "normal speed are local to each distal foot XY, while root velocity "
            "and angular speed use the root local normal."
        ),
        "free_joint_velocity_verification": (
            "MuJoCo free-joint qvel[3:6] is rotated by data.xmat[torso] before "
            "world-normal projection; a regression test matches mj_objectVelocity."
        ),
        "energy_boundary": (
            "Action, torque and mechanical-work values are diagnostics only. "
            "ctrl_cost_weight remains 0.5 and relative-energy V2 is not a reward."
        ),
        "claim_boundary": config["evaluation"]["claim_boundary"],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reward_config = validate_config(config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config": str(config_path)}))
        return
    keep_windows_awake()

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT
        / ("artifacts/smoke" if args.smoke else config["execution"]["output_root"])
        / (config["config_id"] if args.smoke else f"seed_{config['training']['training_seed']}")
    )
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    for directory in ("logs", "models", "traces"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())

    training = config["training"]
    evaluation = config["evaluation"]
    ppo = config["ppo"]
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
    compiled = mujoco.MjModel.from_xml_path(str(scene_paths[0]))
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("compiled scene lacks floor geom")
    np.testing.assert_allclose(
        compiled.geom_friction[floor_id],
        np.asarray(config["approved_map"]["fixed_friction"]),
        atol=0.0,
        rtol=0.0,
    )
    if int(compiled.geom_condim[floor_id]) != int(config["approved_map"]["condim"]):
        raise ValueError("compiled floor condim changed")

    eval_seeds = (
        [int(evaluation["validation_seeds"][0])]
        if args.smoke
        else [int(value) for value in evaluation["validation_seeds"]]
    )
    eval_steps = 160 if args.smoke else int(evaluation["max_episode_steps"])
    train_steps = int(ppo["n_steps"]) if args.smoke else int(
        training["additional_target_timesteps"]
    )
    train_horizon = 160 if args.smoke else int(training["max_episode_steps"])
    if args.smoke:
        evaluation["representative_trace_seed"] = eval_seeds[0]
    source_path = ROOT / config["base_policy"]["model_path"]
    w4_path = ROOT / config["comparison_control"]["archived_w4_model_path"]
    condition_config = _without_terrain_frame(config)

    execution: dict[str, Any] = {
        "schema_version": "proxygap-terrain-frame-reward-pilot-execution-v1",
        "status": "started",
        "smoke": bool(args.smoke),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_model": str(source_path),
        "source_model_sha256": sha256(source_path),
        "archived_w4_model": str(w4_path),
        "archived_w4_model_sha256": sha256(w4_path),
        "approved_height_sha256": config["approved_map"]["heights_sha256"],
        "approved_xml_sha256": config["approved_map"]["xml_sha256"],
        "fixed_friction": config["approved_map"]["fixed_friction"],
        "condim": config["approved_map"]["condim"],
        "observation_dimension": 135,
        "action_dimension": 8,
        "training_seed": int(training["training_seed"]),
        "evaluation_seeds": eval_seeds,
        "training_budget": train_steps,
        "terrain_frame_shaping_only_intervention": True,
        "energy_v2_used_as_reward": False,
        "ctrl_cost_weight": 0.5,
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(int(ppo["torch_num_threads"]))
        for model_path, condition_id in (
            (source_path, "SOURCE_STAGE1_WORLD_FRAME"),
            (w4_path, "W4_ARCHIVED_WORLD_FRAME_CONTROL"),
        ):
            model = PPO.load(model_path, device="cpu")
            rows, _ = evaluate_model(
                model,
                condition_config,
                reward_config,
                start_scene=scene_paths[0],
                condition_id=condition_id,
                output_root=output_root,
                seeds=eval_seeds,
                max_episode_steps=eval_steps,
                cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
                write_representative_trace=False,
            )
            _add_support_metrics(rows)
            all_rows.extend(rows)
            write_rows(output_root / "logs" / "evaluation_episodes.csv", all_rows)

        env = vector_env(
            config,
            reward_config,
            scene_paths=scene_paths,
            spawn_fractions=spawn_fractions,
            seed=int(training["training_seed"]),
            max_episode_steps=train_horizon,
            cruise_speed=float(training["cruise_speed_m_per_s"]),
            monitor_path=output_root / "logs" / "terrain_frame_vecmonitor.csv",
        )
        try:
            model = _configure_continuation_model(
                source_path,
                env,
                ppo,
                training_seed=int(training["training_seed"]),
                smoke=bool(args.smoke),
            )
            source_timesteps = int(model.num_timesteps)
            optimizer_entries = len(model.policy.optimizer.state)
            started = time.perf_counter()
            model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
            elapsed = time.perf_counter() - started
            model_dir = output_root / "models" / "terrain_frame_reward_intervention"
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / f"checkpoint_{int(model.num_timesteps)}.zip"
            model.save(checkpoint)
        finally:
            env.close()
        runtime_row = {
            "condition_id": training["condition_id"],
            "source_timesteps": source_timesteps,
            "additional_training_timesteps": train_steps,
            "final_timesteps": int(model.num_timesteps),
            "train_elapsed_seconds": elapsed,
            "train_steps_per_second": train_steps / max(elapsed, 1e-12),
            "restored_optimizer_state_entries": optimizer_entries,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        }
        write_rows(output_root / "logs" / "training_runtime.csv", [runtime_row])

        rows, trace_record = evaluate_model(
            model,
            config,
            reward_config,
            start_scene=scene_paths[0],
            condition_id=training["condition_id"],
            output_root=output_root,
            seeds=eval_seeds,
            max_episode_steps=eval_steps,
            cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
            write_representative_trace=True,
        )
        _add_support_metrics(rows)
        all_rows.extend(rows)
        if trace_record is not None:
            trace_records.append(trace_record)
        write_rows(output_root / "logs" / "evaluation_episodes.csv", all_rows)
        summary = build_summary(all_rows, config)
        summary_path = output_root / "comparison_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "proxygap-terrain-frame-reward-manifest-v1",
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_model_path": str(source_path),
            "source_model_sha256": sha256(source_path),
            "archived_w4_model_path": str(w4_path),
            "archived_w4_model_sha256": sha256(w4_path),
            "intervention_model_path": str(checkpoint),
            "intervention_model_sha256": sha256(checkpoint),
            "intervention_model_timesteps": int(model.num_timesteps),
            "approved_height_sha256": config["approved_map"]["heights_sha256"],
            "approved_xml_sha256": config["approved_map"]["xml_sha256"],
            "fixed_friction": config["approved_map"]["fixed_friction"],
            "condim": config["approved_map"]["condim"],
            "traces": trace_records,
            "evaluation_csv": str(output_root / "logs" / "evaluation_episodes.csv"),
            "evaluation_csv_sha256": sha256(output_root / "logs" / "evaluation_episodes.csv"),
            "comparison_summary": str(summary_path),
            "comparison_summary_sha256": sha256(summary_path),
            "energy_boundary": config["energy_boundary"],
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        execution.update(
            {
                "status": "complete",
                "final_model": str(checkpoint),
                "final_model_sha256": sha256(checkpoint),
                "final_model_timesteps": int(model.num_timesteps),
                "evaluation_episode_rows": len(all_rows),
                "comparison_summary": str(summary_path),
                "comparison_summary_sha256": sha256(summary_path),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "timesteps": int(model.num_timesteps),
                    "steps_per_second": runtime_row["train_steps_per_second"],
                    "retention_passed": summary["retention_gate"]["passed"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        raise
    finally:
        record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
