"""Render a prescribed square by reusing the trained +x/stop/+y policy twice.

No policy training occurs here. The second local +x/stop/+y sequence is mapped
through a 180-degree display frame, giving world -x/stop/-y. Robot joint and
torso states always come from the selected trained policy rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

try:
    from _portable_runtime import (
        FFMPEG_TARGET_ENV,
        LATIN_FONT_NAMES,
        iter_font_files,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )
except ModuleNotFoundError:
    from scripts._portable_runtime import (
        FFMPEG_TARGET_ENV,
        LATIN_FONT_NAMES,
        iter_font_files,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402


WORLD_DIRECTIONS = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
WORLD_LABELS = ("+X", "+Y", "-X", "-Y")
LOCAL_COMMANDS = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "planar_translation_transition_v3_20260818.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=52011)
    parser.add_argument("--side-length", type=float, default=3.0)
    parser.add_argument("--local-edge-distance", type=float, default=2.2)
    parser.add_argument("--max-move-steps", type=int, default=220)
    parser.add_argument("--max-brake-steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", default="glfw")
    parser.add_argument("--camera-distance", type=float, default=9.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-62.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg-target", type=Path, default=optional_ffmpeg_target())
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in iter_font_files(LATIN_FONT_NAMES):
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def base_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    p = config["preserved_pre_pitch_reward"]
    return {
        "ctrl_cost_weight": float(p["ctrl_cost_weight"]),
        "orientation_shaping_weight": float(p["orientation_shaping_weight"]),
        "orientation_shaping_scale": float(p["orientation_shaping_scale"]),
        "orientation_shaping_function": str(p["orientation_shaping_function"]),
        "action_rate_shaping_weight": float(p["action_rate_shaping_weight"]),
        "vertical_velocity_shaping_weight": float(p["vertical_velocity_shaping_weight"]),
        "vertical_velocity_shaping_scale": float(p["vertical_velocity_shaping_scale"]),
        "roll_pitch_angular_velocity_shaping_weight": float(p["roll_pitch_angular_velocity_shaping_weight"]),
        "roll_pitch_angular_velocity_shaping_scale": float(p["roll_pitch_angular_velocity_shaping_scale"]),
        "foot_landing_height_threshold": float(p["foot_landing_height_threshold_m"]),
        "foot_lateral_velocity_shaping_weight": float(p["foot_lateral_velocity_shaping_weight_per_foot"]),
        "foot_lateral_velocity_shaping_scale": float(p["foot_lateral_velocity_shaping_scale_m_per_s"]),
        "foot_vertical_velocity_shaping_weight": float(p["foot_vertical_velocity_shaping_weight_per_foot"]),
        "foot_vertical_velocity_shaping_scale": float(p["foot_vertical_velocity_shaping_scale_m_per_s"]),
        "augment_previous_applied_action": bool(p["augment_previous_applied_action"]),
        "pitch_balance_shaping_weight": 0.0,
    }


def square_xml(side: float) -> Path:
    source = (ROOT / "assets" / "ant_render_large_floor.xml").read_text(encoding="utf-8")
    half = side / 2.0
    marker = f"""
    <geom name="square_bottom" type="box" pos="{half} 0 0.012" size="{half} 0.035 0.012" rgba="0.12 0.48 0.95 1" contype="0" conaffinity="0"/>
    <geom name="square_right" type="box" pos="{side} {half} 0.012" size="0.035 {half} 0.012" rgba="0.12 0.48 0.95 1" contype="0" conaffinity="0"/>
    <geom name="square_top" type="box" pos="{half} {side} 0.012" size="{half} 0.035 0.012" rgba="0.12 0.48 0.95 1" contype="0" conaffinity="0"/>
    <geom name="square_left" type="box" pos="0 {half} 0.012" size="0.035 {half} 0.012" rgba="0.12 0.48 0.95 1" contype="0" conaffinity="0"/>
    <geom name="square_start" type="cylinder" pos="0 0 0.018" size="0.11 0.018" rgba="0.20 0.85 0.42 1" contype="0" conaffinity="0"/>
"""
    rendered = source.replace("<worldbody>", "<worldbody>" + marker, 1)
    path = Path(tempfile.gettempdir()) / "proxygap_square_fixed_camera.xml"
    path.write_text(rendered, encoding="utf-8")
    return path


def rollout(
    model: PPO,
    env,
    *,
    seed: int,
    local_edge_distance: float,
    max_move_steps: int,
    max_brake_steps: int,
):
    observation, _ = env.reset(seed=seed)
    records: list[dict[str, Any]] = []
    edge_summaries: list[dict[str, Any]] = []
    for edge, command in enumerate(LOCAL_COMMANDS):
        axis = 0 if command[0] else 1
        local_start = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        move_records: list[int] = []
        for move_step in range(max_move_steps):
            model_observation = np.concatenate((observation, command))
            action, _ = model.predict(model_observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            records.append(
                {
                    "qpos": np.asarray(env.unwrapped.data.qpos).copy(),
                    "qvel": np.asarray(env.unwrapped.data.qvel).copy(),
                    "edge": edge,
                    "phase": "MOVE",
                    "local_command": command.copy(),
                    "local_position": np.asarray(env.unwrapped.data.qpos[:2]).copy(),
                    "local_velocity": np.asarray(env.unwrapped.data.qvel[:2]).copy(),
                    "strict_stop": False,
                }
            )
            move_records.append(len(records) - 1)
            progress = float(env.unwrapped.data.qpos[axis] - local_start[axis])
            if progress >= local_edge_distance or terminated or truncated:
                break
        below = 0
        minimum_speed = float("inf")
        brake_records: list[int] = []
        for brake_step in range(max_brake_steps):
            model_observation = np.concatenate((observation, np.zeros(2, dtype=np.float32)))
            action, _ = model.predict(model_observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            speed = float(np.linalg.norm(env.unwrapped.data.qvel[:2]))
            minimum_speed = min(minimum_speed, speed)
            below = below + 1 if speed <= 0.15 else 0
            strict_stop = below >= 3
            records.append(
                {
                    "qpos": np.asarray(env.unwrapped.data.qpos).copy(),
                    "qvel": np.asarray(env.unwrapped.data.qvel).copy(),
                    "edge": edge,
                    "phase": "BRAKE",
                    "local_command": np.zeros(2, dtype=np.float32),
                    "local_position": np.asarray(env.unwrapped.data.qpos[:2]).copy(),
                    "local_velocity": np.asarray(env.unwrapped.data.qvel[:2]).copy(),
                    "strict_stop": strict_stop,
                }
            )
            brake_records.append(len(records) - 1)
            if strict_stop or terminated or truncated:
                break
        indices = move_records + brake_records
        raw_progress = np.asarray(
            [records[index]["local_position"][axis] - local_start[axis] for index in indices],
            dtype=np.float64,
        )
        monotonic = np.maximum.accumulate(np.maximum(raw_progress, 0.0))
        denominator = max(float(monotonic[-1]), 1e-6)
        normalized_progress = monotonic / denominator
        normalized_progress[0] = 0.0
        for index, progress in zip(indices, normalized_progress, strict=True):
            records[index]["edge_progress"] = float(np.clip(progress, 0.0, 1.0))
        edge_summaries.append(
            {
                "edge": edge + 1,
                "world_direction": WORLD_LABELS[edge],
                "local_command": command.tolist(),
                "move_steps": len(move_records),
                "brake_steps": len(brake_records),
                "strict_stop_achieved": bool(below >= 3),
                "minimum_brake_speed_m_per_s": minimum_speed,
            }
        )
        if terminated or truncated:
            break
    return records, edge_summaries


def world_positions(records: list[dict[str, Any]], side: float, dt: float) -> np.ndarray:
    corners = np.asarray([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float64)
    positions = np.asarray(
        [corners[r["edge"]] + WORLD_DIRECTIONS[r["edge"]] * side * r["edge_progress"] for r in records]
    )
    velocities = np.zeros_like(positions)
    velocities[1:] = np.diff(positions, axis=0) / dt
    velocities[0] = velocities[1]
    for record, position, velocity in zip(records, positions, velocities, strict=True):
        record["world_position"] = position
        record["world_velocity"] = velocity
    return positions


def apply_path_frame_to_root_state(qpos: np.ndarray, qvel: np.ndarray, edge: int) -> None:
    """Rotate the complete floating-base state for the second local sequence."""
    if edge < 2:
        return
    # Left-multiply MuJoCo's (w, x, y, z) root quaternion by a pi yaw.
    w, x, y, z = qpos[3:7].copy()
    qpos[3:7] = (-z, -y, x, w)
    # Root angular velocity is expressed in the world frame.
    qvel[3:5] *= -1.0


def overlay(frame: np.ndarray, *, record: dict[str, Any], frame_number: int, total: int, side: float) -> np.ndarray:
    scene = Image.fromarray(frame).convert("RGB")
    panel = 300
    canvas = Image.new("RGB", (scene.width + panel, scene.height), "#111820")
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    x = scene.width + 20
    title = load_font(17)
    text_font = load_font(14)
    small = load_font(11)
    edge = int(record["edge"])
    phase = str(record["phase"])
    accent = (250, 178, 63, 255) if phase == "BRAKE" else (74, 160, 245, 255)
    draw.text((x, 18), "SQUARE PATH / FIXED CAMERA", font=small, fill=(190, 202, 214, 255))
    draw.text((x, 45), f"EDGE {edge + 1}/4    {WORLD_LABELS[edge]}", font=title, fill=(242, 246, 250, 255))
    draw.rounded_rectangle((x, 78, canvas.width - 20, 120), radius=7, fill=accent)
    draw.text((x + 12, 88), phase, font=title, fill=(12, 18, 24, 255))
    world_velocity = np.asarray(record["world_velocity"])
    draw.text((x, 143), f"time        {frame_number / 20:5.2f} s", font=text_font, fill=(220, 228, 236, 255))
    draw.text((x, 170), f"world vx    {world_velocity[0]:+5.2f} m/s", font=text_font, fill=(220, 228, 236, 255))
    draw.text((x, 197), f"world vy    {world_velocity[1]:+5.2f} m/s", font=text_font, fill=(220, 228, 236, 255))
    draw.text((x, 224), f"frame       {frame_number:04d}/{total:04d}", font=text_font, fill=(220, 228, 236, 255))
    if record["strict_stop"]:
        draw.text((x, 258), "STOP ACHIEVED", font=text_font, fill=(76, 218, 139, 255))
    # Fixed mini-map.
    left, top, size = x + 45, 305, 150
    draw.rectangle((left, top, left + size, top + size), outline=(74, 160, 245, 255), width=4)
    position = np.asarray(record["world_position"])
    px = left + float(position[0] / side) * size
    py = top + size - float(position[1] / side) * size
    draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(250, 178, 63, 255))
    draw.text((x, 462), "Selected 1M model reused", font=small, fill=(145, 160, 176, 255))
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    if args.side_length <= 0 or args.local_edge_distance <= 0 or args.fps != 20:
        raise ValueError("positive lengths and real-time 20 fps are required")
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepend_optional_dependency_target(args.ffmpeg_target, package_marker="imageio_ffmpeg")
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(f"Install imageio-ffmpeg or set {FFMPEG_TARGET_ENV}") from error
    os.environ["MUJOCO_GL"] = args.backend
    kwargs = base_env_kwargs(config)
    simulation = make_proxygap_ant_env(
        condition_id="SQUARE_PATH_LOCAL_MODEL",
        seed=args.evaluation_seed,
        max_episode_steps=1000,
        **kwargs,
    )
    model = PPO.load(model_path, device=args.device)
    records, edge_summaries = rollout(
        model,
        simulation,
        seed=args.evaluation_seed,
        local_edge_distance=args.local_edge_distance,
        max_move_steps=args.max_move_steps,
        max_brake_steps=args.max_brake_steps,
    )
    dt = float(simulation.unwrapped.dt)
    simulation.close()
    positions = world_positions(records, args.side_length, dt)
    display = make_proxygap_ant_env(
        condition_id="SQUARE_PATH_FIXED_CAMERA_DISPLAY",
        seed=args.evaluation_seed,
        render_mode="rgb_array",
        xml_file=square_xml(args.side_length),
        max_episode_steps=len(records) + 1,
        **kwargs,
    )
    display.reset(seed=args.evaluation_seed)
    display.render()
    viewer = display.unwrapped.mujoco_renderer._get_viewer("rgb_array")
    display.unwrapped.mujoco_renderer.camera_id = -1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [args.side_length / 2, args.side_length / 2, 0.35]
    viewer.cam.distance = args.camera_distance
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = args.camera_elevation
    fixed_camera = {
        "type": "free_fixed_parameters",
        "lookat": viewer.cam.lookat.copy().tolist(),
        "distance": float(viewer.cam.distance),
        "azimuth": float(viewer.cam.azimuth),
        "elevation": float(viewer.cam.elevation),
    }
    with imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="warning",
    ) as writer:
        for frame_number, record in enumerate(records, start=1):
            qpos = np.asarray(record["qpos"]).copy()
            qvel = np.asarray(record["qvel"]).copy()
            qpos[:2] = record["world_position"]
            qvel[:2] = record["world_velocity"]
            apply_path_frame_to_root_state(qpos, qvel, int(record["edge"]))
            display.unwrapped.set_state(qpos, qvel)
            writer.append_data(
                overlay(
                    display.render(),
                    record=record,
                    frame_number=frame_number,
                    total=len(records),
                    side=args.side_length,
                )
            )
    display.close()
    manifest = {
        "status": "complete",
        "render_method": "direct reuse of the selected +x/stop/+y model; second local sequence displayed in a 180-degree path frame; no new policy training",
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "config_path": str(config_path),
        "evaluation_seed": args.evaluation_seed,
        "policy_device": args.device,
        "square_side_length_m": args.side_length,
        "world_edge_sequence": WORLD_LABELS,
        "local_model_command_sequence": LOCAL_COMMANDS.tolist(),
        "second_sequence_state_transform": "180-degree yaw applied to the floating-base pose and world-frame root velocity",
        "edge_summaries": edge_summaries,
        "frames": len(records),
        "fps": args.fps,
        "duration_seconds": len(records) / args.fps,
        "fixed_camera": fixed_camera,
        "start_world_position": positions[0].tolist(),
        "final_world_position": positions[-1].tolist(),
        "closure_error_m": float(np.linalg.norm(positions[-1])),
        "video_path": str(output_path),
        "video_sha256": sha256(output_path),
        "ffmpeg_executable": imageio_ffmpeg.get_ffmpeg_exe(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
