"""Fine-tune and evaluate V22 on the approved fixed quadrant terrain V2."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.experiment import write_rows  # noqa: E402
from proxygap.fixed_goal_terrain import FixedGoalTerrainWrapper  # noqa: E402
from run_curved_gait_training import common_env_kwargs  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "fixed_quad_terrain_v2_training_20260818.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keep_windows_awake() -> None:
    if os.name != "nt":
        return
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        0x80000000 | 0x00000001
    )
    if result == 0:
        raise OSError("Windows rejected the sleep-prevention request")


def terrain_value(array: np.ndarray, x: float, y: float, half_extent: float) -> float:
    rows, cols = array.shape
    col_f = np.clip((x + half_extent) / (2.0 * half_extent) * (cols - 1), 0, cols - 1)
    row_f = np.clip((y + half_extent) / (2.0 * half_extent) * (rows - 1), 0, rows - 1)
    col0 = min(int(math.floor(col_f)), cols - 2)
    row0 = min(int(math.floor(row_f)), rows - 2)
    tx = float(col_f - col0)
    ty = float(row_f - row0)
    return float(
        (1.0 - ty) * ((1.0 - tx) * array[row0, col0] + tx * array[row0, col0 + 1])
        + ty * ((1.0 - tx) * array[row0 + 1, col0] + tx * array[row0 + 1, col0 + 1])
    )


def surface_pose(
    heights: np.ndarray,
    *,
    half_extent: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    point = start_xy + float(fraction) * (goal_xy - start_xy)
    spacing = 2.0 * half_extent / (heights.shape[0] - 1)
    dz_dy, dz_dx = np.gradient(heights, spacing, spacing)
    gx = terrain_value(dz_dx, float(point[0]), float(point[1]), half_extent)
    gy = terrain_value(dz_dy, float(point[0]), float(point[1]), half_extent)
    z = terrain_value(heights, float(point[0]), float(point[1]), half_extent)

    planar_forward = goal_xy - start_xy
    planar_forward /= np.linalg.norm(planar_forward)
    forward = np.asarray(
        [planar_forward[0], planar_forward[1], gx * planar_forward[0] + gy * planar_forward[1]],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    normal = np.asarray([-gx, -gy, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, normal)
    forward /= np.linalg.norm(forward)
    rotation = np.column_stack((forward, left, normal))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    position = np.asarray([point[0], point[1], z + 0.75], dtype=np.float64)
    metadata = {
        "spawn_fraction": float(fraction),
        "x_m": float(point[0]),
        "y_m": float(point[1]),
        "terrain_height_m": z,
        "gradient_x": gx,
        "gradient_y": gy,
        "gradient_degrees": float(math.degrees(math.atan(math.hypot(gx, gy)))),
    }
    return position, quaternion, metadata


def prepare_task_scenes(
    config: dict[str, Any],
    output_root: Path,
    spawn_fractions: list[float],
) -> tuple[list[Path], list[dict[str, float]]]:
    approved = config["approved_map"]
    source_xml = ROOT / approved["xml_path"]
    source_scene = source_xml.parent
    heights_source = ROOT / approved["heights_path"]
    if sha256(source_xml) != approved["xml_sha256"]:
        raise ValueError("Approved map XML SHA-256 mismatch")
    if sha256(heights_source) != approved["heights_sha256"]:
        raise ValueError("Approved map height SHA-256 mismatch")
    if sha256(source_scene / "terrain.hfield") != approved["hfield_sha256"]:
        raise ValueError("Approved map hfield SHA-256 mismatch")

    scene_dir = output_root / "task_scenes"
    scene_dir.mkdir(parents=True)
    for name in ("terrain.hfield", "terrain_contours.png", "heights_m.npy"):
        shutil.copy2(source_scene / name, scene_dir / name)
    heights = np.load(heights_source, allow_pickle=False)
    start_xy = np.asarray(approved["start_xy_m"], dtype=np.float64)
    goal_xy = np.asarray(approved["goal_xy_m"], dtype=np.float64)
    half_extent = float(approved["map_half_extent_m"])

    scene_paths: list[Path] = []
    spawn_metadata: list[dict[str, float]] = []
    for index, fraction in enumerate(spawn_fractions):
        position, quaternion, metadata = surface_pose(
            heights,
            half_extent=half_extent,
            start_xy=start_xy,
            goal_xy=goal_xy,
            fraction=float(fraction),
        )
        tree = ET.parse(source_xml)
        torso = tree.getroot().find("./worldbody/body[@name='torso']")
        if torso is None:
            raise ValueError("Approved XML lacks torso body")
        torso.set("pos", " ".join(f"{value:.12g}" for value in position))
        torso.set("quat", " ".join(f"{value:.12g}" for value in quaternion))
        ET.indent(tree, space="  ")
        path = scene_dir / f"spawn_{index}_{fraction:.3f}.xml"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        compiled = mujoco.MjModel.from_xml_path(str(path))
        floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if not np.allclose(
            compiled.geom_friction[floor_id],
            np.asarray(approved["fixed_friction"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Task scene changed frozen floor friction")
        if int(compiled.geom_condim[floor_id]) != int(approved["condim"]):
            raise RuntimeError("Task scene changed frozen floor condim")
        metadata["scene_sha256"] = sha256(path)
        scene_paths.append(path)
        spawn_metadata.append(metadata)
    return scene_paths, spawn_metadata


def make_task_env(
    config: dict[str, Any],
    v22_config: dict[str, Any],
    *,
    xml_path: Path,
    seed: int,
    spawn_fraction: float,
    max_episode_steps: int,
    cruise_speed: float,
    terminate_on_success: bool,
    render_mode: str | None = None,
) -> FixedGoalTerrainWrapper:
    task = config["task_adapter"]
    approved = config["approved_map"]
    curve_env = make_curved_gait_env(
        condition_id="FIXED_QUAD_TERRAIN_V2",
        seed=seed,
        render_mode=render_mode,
        xml_file=xml_path,
        max_episode_steps=max_episode_steps,
        terminate_when_unhealthy=False,
        profile="external",
        speed_min=cruise_speed,
        speed_max=cruise_speed,
        max_abs_curvature=float(task["maximum_abs_curvature_per_m"]),
        max_abs_lateral_speed=0.0,
        fixed_lateral_speed=0.0,
        heading_termination_enabled=False,
        terrain_frame_shaping_enabled=bool(
            task.get("terrain_frame_shaping_enabled", False)
        ),
        **common_env_kwargs(v22_config),
    )
    return FixedGoalTerrainWrapper(
        curve_env,
        heights_path=ROOT / approved["heights_path"],
        expected_height_sha256=approved["heights_sha256"],
        map_half_extent_m=float(approved["map_half_extent_m"]),
        start_xy_m=approved["start_xy_m"],
        goal_xy_m=approved["goal_xy_m"],
        spawn_fraction=spawn_fraction,
        cruise_speed_m_per_s=cruise_speed,
        maximum_abs_curvature_per_m=float(task["maximum_abs_curvature_per_m"]),
        yaw_gain_per_second=float(task["yaw_gain_per_second"]),
        yaw_deadband_degrees=float(task.get("yaw_deadband_degrees", 0.0)),
        curvature_speed_reduction_gain=float(
            task.get("curvature_speed_reduction_gain", 0.0)
        ),
        minimum_turn_speed_fraction=float(
            task.get("minimum_turn_speed_fraction", 1.0)
        ),
        slow_radius_m=float(task["slow_radius_m"]),
        arrival_radius_m=float(task["arrival_radius_m"]),
        hold_radius_m=float(task["hold_radius_m"]),
        hold_seconds=float(task["hold_seconds"]),
        hold_speed_m_per_s=float(task["hold_speed_m_per_s"]),
        terminate_on_success=terminate_on_success,
        terrain_relative_healthy_clearance_m=tuple(
            float(value) for value in task["terrain_relative_healthy_clearance_m"]
        ),
        maximum_healthy_tilt_degrees=float(task["maximum_healthy_tilt_degrees"]),
        unhealthy_grace_steps=int(task["unhealthy_grace_steps"]),
        slip_speed_threshold_m_per_s=float(task["slip_speed_threshold_m_per_s"]),
        augment_local_terrain_observation=bool(
            task.get("augment_local_terrain_observation", False)
        ),
        terrain_frame_shaping_enabled=bool(
            task.get("terrain_frame_shaping_enabled", False)
        ),
        terrain_preview_longitudinal_m=tuple(
            float(value)
            for value in task.get(
                "terrain_preview_longitudinal_m",
                [0.5, 1.0, 1.5],
            )
        ),
        terrain_preview_lateral_m=tuple(
            float(value)
            for value in task.get(
                "terrain_preview_lateral_m",
                [-0.4, 0.0, 0.4],
            )
        ),
    )


def vector_env(
    config: dict[str, Any],
    v22_config: dict[str, Any],
    *,
    scene_paths: list[Path],
    spawn_fractions: list[float],
    seed: int,
    max_episode_steps: int,
    cruise_speed: float,
    monitor_path: Path,
) -> VecMonitor:
    factories: list[Callable[[], gym.Env]] = []
    for rank, (scene_path, fraction) in enumerate(zip(scene_paths, spawn_fractions, strict=True)):
        local_seed = seed + 1000 * rank

        def factory(
            path: Path = scene_path,
            spawn: float = fraction,
            env_seed: int = local_seed,
        ) -> gym.Env:
            return make_task_env(
                config,
                v22_config,
                xml_path=path,
                seed=env_seed,
                spawn_fraction=spawn,
                max_episode_steps=max_episode_steps,
                cruise_speed=cruise_speed,
                terminate_on_success=False,
            )

        factories.append(factory)
    if len(factories) == 1:
        base = DummyVecEnv(factories)
    else:
        base = SubprocVecEnv(
            factories,
            start_method=str(config["execution"]["subprocess_start_method"]),
        )
    return VecMonitor(base, filename=str(monitor_path))


def evaluate_checkpoint(
    model: PPO,
    config: dict[str, Any],
    v22_config: dict[str, Any],
    *,
    start_scene: Path,
    checkpoint_label: str,
    checkpoint_timesteps: int,
    seeds: list[int],
    max_episode_steps: int,
    cruise_speed: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deterministic = bool(config["evaluation"]["deterministic_policy"])
    for seed in seeds:
        env = make_task_env(
            config,
            v22_config,
            xml_path=start_scene,
            seed=seed,
            spawn_fraction=0.0,
            max_episode_steps=max_episode_steps,
            cruise_speed=cruise_speed,
            terminate_on_success=True,
        )
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, _, terminated, truncated, _ = env.step(action)
        summary = env.episode_summary()
        rows.append(
            {
                "checkpoint_label": checkpoint_label,
                "checkpoint_timesteps": checkpoint_timesteps,
                "evaluation_seed": seed,
                **summary,
            }
        )
        env.close()
    return rows


def validate_config(config: dict[str, Any], v22_config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_user_approved_map_pilot":
        raise ValueError("Fixed-map training configuration is not frozen")
    if config.get("formal_generalisation_claim") != "prohibited":
        raise ValueError("Pilot must prohibit a formal generalisation claim")
    if config["task_adapter"]["additional_task_reward"] != 0.0:
        raise ValueError("This pilot must preserve the V22 reward without task bonus")
    base = ROOT / config["base_policy"]["model_path"]
    if not base.is_file() or sha256(base) != config["base_policy"]["model_sha256"]:
        raise ValueError("V22 base checkpoint is missing or has changed")
    model = PPO.load(base, device="cpu")
    if int(model.observation_space.shape[0]) != int(config["base_policy"]["observation_dimension"]):
        raise ValueError("V22 observation dimension mismatch")
    if int(model.action_space.shape[0]) != int(config["base_policy"]["action_dimension"]):
        raise ValueError("V22 action dimension mismatch")
    if not bool(v22_config["commands"]["augment_foot_contact_mask"]):
        raise ValueError("Referenced V22 configuration lacks contact observations")
    fractions = [float(value) for value in config["training"]["spawn_fractions"]]
    if len(fractions) != int(config["training"]["parallel_environments"]):
        raise ValueError("One declared spawn fraction is required per environment")
    if fractions[0] != 0.0 or fractions != sorted(fractions):
        raise ValueError("Spawn fractions must be sorted and begin at the fixed start")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    v22_path = ROOT / config["base_policy"]["configuration"]
    v22_config = json.loads(v22_path.read_text(encoding="utf-8"))
    validate_config(config, v22_config)
    keep_windows_awake()

    if args.output_root:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = ROOT / "artifacts" / "smoke" / config["config_id"]
    else:
        output_root = ROOT / config["execution"]["output_root"] / f"seed_{config['training']['training_seed']}"
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    (output_root / "logs").mkdir(parents=True)
    (output_root / "models").mkdir(parents=True)
    (output_root / "frozen_run_config.json").write_bytes(config_path.read_bytes())

    training = config["training"]
    evaluation = config["evaluation"]
    ppo_config = config["ppo"]
    training_seed = int(training["training_seed"])
    spawn_fractions = [float(value) for value in training["spawn_fractions"]]
    if args.smoke:
        spawn_fractions = spawn_fractions[:1]
    scene_paths, spawn_metadata = prepare_task_scenes(config, output_root, spawn_fractions)
    (output_root / "spawn_manifest.json").write_text(
        json.dumps(spawn_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    initial_speed = float(training["stages"][0]["cruise_speed_m_per_s"])
    training_max_steps = 160 if args.smoke else int(training["max_episode_steps"])
    monitor_path = output_root / "logs" / "training_vecmonitor.csv"
    env = vector_env(
        config,
        v22_config,
        scene_paths=scene_paths,
        spawn_fractions=spawn_fractions,
        seed=training_seed,
        max_episode_steps=training_max_steps,
        cruise_speed=initial_speed,
        monitor_path=monitor_path,
    )
    source_path = ROOT / config["base_policy"]["model_path"]
    torch.set_num_threads(int(ppo_config["torch_num_threads"]))
    model = PPO.load(source_path, env=env, device=str(ppo_config["device"]))
    model.learning_rate = float(ppo_config["learning_rate"])
    model._setup_lr_schedule()
    model.set_random_seed(training_seed)
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "normalize_advantage",
    ):
        if float(getattr(model, name)) != float(ppo_config[name]):
            raise ValueError(f"Continuation PPO parameter mismatch: {name}")
    clip_value = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    if float(clip_value) != float(ppo_config["clip_range"]):
        raise ValueError("Continuation PPO clip_range mismatch")

    execution = {
        "status": "started",
        "smoke": bool(args.smoke),
        "training_seed": training_seed,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "v22_config_sha256": sha256(v22_path),
        "source_model": str(source_path),
        "source_model_sha256": sha256(source_path),
        "source_model_timesteps": int(model.num_timesteps),
        "approved_height_sha256": config["approved_map"]["heights_sha256"],
        "parallel_environments": len(spawn_fractions),
        "spawn_fractions": spawn_fractions,
        "device": str(model.device),
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
        if args.smoke:
            eval_seeds = [int(evaluation["validation_seeds"][0])]
            eval_max_steps = 160
            stages = [
                {
                    "name": "smoke_fixed_map_4k",
                    "additional_target_timesteps": 4096,
                    "cruise_speed_m_per_s": initial_speed,
                }
            ]
        else:
            eval_seeds = [int(value) for value in evaluation["paired_test_seeds"]]
            eval_max_steps = int(evaluation["full_route_max_episode_steps"])
            stages = list(training["stages"])

        evaluation_rows.extend(
            evaluate_checkpoint(
                model,
                config,
                v22_config,
                start_scene=scene_paths[0],
                checkpoint_label="v22_baseline",
                checkpoint_timesteps=int(model.num_timesteps),
                seeds=eval_seeds,
                max_episode_steps=eval_max_steps,
                cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
            )
        )
        write_rows(output_root / "logs" / "evaluation_episodes.csv", evaluation_rows)

        source_timesteps = int(model.num_timesteps)
        rollout_size = len(spawn_fractions) * int(ppo_config["n_steps"])
        for stage_index, stage in enumerate(stages):
            additional_target = int(stage["additional_target_timesteps"])
            if additional_target % rollout_size:
                raise ValueError(
                    f"Additional target {additional_target} is not divisible by rollout size {rollout_size}"
                )
            absolute_target = source_timesteps + additional_target
            requested = absolute_target - int(model.num_timesteps)
            cruise_speed = float(stage["cruise_speed_m_per_s"])
            env.env_method("set_task_speed", cruise_speed)
            started = time.perf_counter()
            model.learn(total_timesteps=requested, reset_num_timesteps=False)
            elapsed = time.perf_counter() - started
            checkpoint_path = output_root / "models" / f"checkpoint_{absolute_target}.zip"
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
            write_rows(output_root / "logs" / "training_runtime.csv", runtime_rows)

            stage_seeds = (
                [int(value) for value in evaluation["validation_seeds"]]
                if not args.smoke
                else eval_seeds
            )
            evaluation_rows.extend(
                evaluate_checkpoint(
                    model,
                    config,
                    v22_config,
                    start_scene=scene_paths[0],
                    checkpoint_label=stage["name"],
                    checkpoint_timesteps=int(model.num_timesteps),
                    seeds=stage_seeds,
                    max_episode_steps=eval_max_steps,
                    cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
                )
            )
            write_rows(output_root / "logs" / "evaluation_episodes.csv", evaluation_rows)
            print(
                json.dumps(
                    {
                        "stage": stage["name"],
                        "timesteps": int(model.num_timesteps),
                        "steps_per_second": runtime_rows[-1]["train_steps_per_second"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if not args.smoke:
            evaluation_rows.extend(
                evaluate_checkpoint(
                    model,
                    config,
                    v22_config,
                    start_scene=scene_paths[0],
                    checkpoint_label="final_paired_test",
                    checkpoint_timesteps=int(model.num_timesteps),
                    seeds=[int(value) for value in evaluation["paired_test_seeds"]],
                    max_episode_steps=int(evaluation["full_route_max_episode_steps"]),
                    cruise_speed=float(evaluation["cruise_speed_m_per_s"]),
                )
            )
            write_rows(output_root / "logs" / "evaluation_episodes.csv", evaluation_rows)

        execution.update(
            {
                "status": "complete",
                "completed_model_timesteps": int(model.num_timesteps),
                "added_training_timesteps": int(model.num_timesteps) - int(config["base_policy"]["source_timesteps"]),
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
        record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        env.close()

    if not args.smoke:
        paired_seeds = [int(value) for value in evaluation["paired_test_seeds"]]
        video_seed = paired_seeds[len(paired_seeds) // 2]
        video_script = ROOT / "scripts" / "render_fixed_goal_training_video.py"
        video_log_path = output_root / "logs" / "training_video_render.log"
        video_command = [
            sys.executable,
            str(video_script),
            "--run-root",
            str(output_root),
            "--evaluation-seed",
            str(video_seed),
            "--physical-seconds",
            "45.0",
        ]
        try:
            completed_video = subprocess.run(
                video_command,
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            video_log_path.write_text(completed_video.stdout, encoding="utf-8")
            video_dir = output_root / "videos" / f"representative_seed_{video_seed}"
            manifest_path = (
                video_dir
                / f"fixed_map_final_policy_seed_{video_seed}_video_manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            video_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not bool(video_manifest["qa"]["duration_at_least_10_seconds"]):
                raise RuntimeError("Training video did not pass its duration gate")
            execution.update(
                {
                    "required_video_artifact_status": "complete",
                    "required_video_evaluation_seed": video_seed,
                    "required_video_manifest": str(manifest_path),
                    "required_video_path": video_manifest["video"]["path"],
                    "required_video_sha256": video_manifest["video"]["sha256"],
                }
            )
        except Exception as error:
            captured_output = getattr(error, "stdout", None)
            if captured_output:
                video_log_path.write_text(str(captured_output), encoding="utf-8")
            execution.update(
                {
                    "required_video_artifact_status": "failed",
                    "required_video_evaluation_seed": video_seed,
                    "required_video_error_type": type(error).__name__,
                    "required_video_error_message": str(error),
                }
            )
            record_path.write_text(
                json.dumps(execution, indent=2) + "\n", encoding="utf-8"
            )
            raise
        record_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(execution, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
