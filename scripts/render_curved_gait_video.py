"""Render one trained tangent-aligned curved gait with a fixed distant camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
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

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.planar_transition import (  # noqa: E402
    quaternion_yaw_angle,
    wrapped_angle_difference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "curved_gait_tangent_v2_body_frame_20260818.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("constant_left", "constant_right", "s_curve"),
        default="constant_left",
    )
    parser.add_argument("--curvature", type=float, default=0.25)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--evaluation-seed", type=int, default=53111)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", default="glfw")
    parser.add_argument("--camera-distance", type=float, default=11.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-58.0)
    parser.add_argument("--hide-target-path", action="store_true")
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


def env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    commands = config["commands"]
    reward = config["reward"]
    preserved = config["preserved_pre_pitch_reward"]
    segment_min, segment_max = commands["segment_steps_interval"]
    return {
        "ctrl_cost_weight": float(preserved["ctrl_cost_weight"]),
        "orientation_shaping_weight": float(preserved["orientation_shaping_weight"]),
        "orientation_shaping_scale": float(preserved["orientation_shaping_scale"]),
        "orientation_shaping_function": str(preserved["orientation_shaping_function"]),
        "action_rate_shaping_weight": float(preserved["action_rate_shaping_weight"]),
        "vertical_velocity_shaping_weight": float(preserved["vertical_velocity_shaping_weight"]),
        "vertical_velocity_shaping_scale": float(preserved["vertical_velocity_shaping_scale"]),
        "roll_pitch_angular_velocity_shaping_weight": float(preserved["roll_pitch_angular_velocity_shaping_weight"]),
        "roll_pitch_angular_velocity_shaping_scale": float(preserved["roll_pitch_angular_velocity_shaping_scale"]),
        "foot_landing_height_threshold": float(preserved["foot_landing_height_threshold_m"]),
        "foot_lateral_velocity_shaping_weight": float(preserved["foot_lateral_velocity_shaping_weight_per_foot"]),
        "foot_lateral_velocity_shaping_scale": float(preserved["foot_lateral_velocity_shaping_scale_m_per_s"]),
        "foot_vertical_velocity_shaping_weight": float(preserved["foot_vertical_velocity_shaping_weight_per_foot"]),
        "foot_vertical_velocity_shaping_scale": float(preserved["foot_vertical_velocity_shaping_scale_m_per_s"]),
        "airborne_shaping_weight": float(preserved.get("airborne_shaping_weight", 0.0)),
        "foot_contact_gap_shaping_weight": float(preserved.get("foot_contact_gap_shaping_weight", 0.0)),
        "foot_contact_gap_grace_seconds": float(preserved.get("foot_contact_gap_grace_seconds", 0.5)),
        "foot_contact_gap_scale_seconds": float(preserved.get("foot_contact_gap_scale_seconds", 0.5)),
        "augment_previous_applied_action": bool(preserved["augment_previous_applied_action"]),
        "command_frame": str(commands["command_frame"]),
        "observation_frame": str(commands.get("observation_frame", "world")),
        "curvature_slew_rate": float(commands["curvature_slew_rate_per_m_per_s"]),
        "lateral_speed_slew_rate": float(
            commands.get("lateral_speed_slew_rate_m_per_s2", 0.40)
        ),
        "segment_steps_min": int(segment_min),
        "segment_steps_max": int(segment_max),
        "warmup_steps": int(commands["warmup_steps"]),
        "s_curve_period_steps": int(commands["s_curve_period_steps"]),
        "planar_tracking_weight": float(reward["planar_velocity_tracking_weight"]),
        "planar_tracking_scale": float(reward["planar_velocity_tracking_scale_m_per_s"]),
        "planar_tracking_function": str(reward["planar_velocity_tracking_function"]),
        "cross_axis_velocity_weight": float(reward["cross_axis_velocity_weight"]),
        "cross_axis_velocity_scale": float(reward["cross_axis_velocity_scale_m_per_s"]),
        "heading_alignment_weight": float(reward["heading_alignment_weight"]),
        "heading_alignment_scale": math.radians(float(reward["heading_alignment_scale_degrees"])),
        "heading_alignment_function": str(reward.get("heading_alignment_function", "pseudo_huber")),
        "yaw_rate_tracking_weight": float(reward["yaw_rate_tracking_weight"]),
        "yaw_rate_tracking_scale": float(reward["yaw_rate_tracking_scale_rad_per_s"]),
        "yaw_rate_tracking_function": str(reward.get("yaw_rate_tracking_function", "pseudo_huber")),
        "heading_tolerance": math.radians(float(reward["heading_tolerance_degrees"])),
        "heading_termination_threshold": math.radians(float(reward["heading_termination_threshold_degrees"])),
        "heading_termination_consecutive_steps": int(reward["heading_termination_consecutive_steps"]),
    }


def target_path(
    *,
    start_xy: np.ndarray,
    initial_heading: float,
    profile: str,
    curvature: float,
    speed: float,
    steps: int,
    dt: float,
    period_steps: int,
    slew_rate: float,
) -> np.ndarray:
    position = np.asarray(start_xy, dtype=np.float64).copy()
    heading = float(initial_heading)
    current = curvature if profile == "constant_left" else -curvature if profile == "constant_right" else 0.0
    points = [position.copy()]
    for step in range(steps):
        heading = wrapped_angle_difference(heading + speed * current * dt, 0.0)
        position += speed * dt * np.asarray([math.cos(heading), math.sin(heading)])
        points.append(position.copy())
        desired = (
            curvature * math.sin(2.0 * math.pi * float(step + 1) / period_steps)
            if profile == "s_curve"
            else current
        )
        current += float(np.clip(desired - current, -slew_rate * dt, slew_rate * dt))
    return np.asarray(points)


def path_xml(points: np.ndarray, *, show_target_path: bool) -> Path:
    source = (ROOT / "assets" / "ant_render_large_floor.xml").read_text(encoding="utf-8")
    markers = []
    stride = max(1, len(points) // 100)
    if show_target_path:
        for index, point in enumerate(points[::stride]):
            color = "0.20 0.88 0.48 1" if index == 0 else "0.12 0.48 0.95 0.88"
            markers.append(
                f'<geom name="curve_marker_{index}" type="cylinder" '
                f'pos="{point[0]:.6f} {point[1]:.6f} 0.014" size="0.055 0.014" '
                f'rgba="{color}" contype="0" conaffinity="0"/>'
            )
    rendered = source.replace("<worldbody>", "<worldbody>\n" + "\n".join(markers), 1)
    path = Path(tempfile.gettempdir()) / "proxygap_curved_gait_fixed_camera.xml"
    path.write_text(rendered, encoding="utf-8")
    return path


def overlay(
    frame: np.ndarray,
    *,
    step: int,
    total: int,
    profile: str,
    speed: float,
    info: dict[str, Any],
    actual_yaw: float,
    fps: int,
) -> np.ndarray:
    scene = Image.fromarray(frame).convert("RGB")
    panel = 310
    canvas = Image.new("RGB", (scene.width + panel, scene.height), "#111820")
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    x = scene.width + 20
    title = load_font(18)
    text_font = load_font(14)
    small = load_font(11)
    heading = float(info.get("proxygap_curve_tangent_heading_step", 0.0))
    error = wrapped_angle_difference(actual_yaw, heading)
    target_yaw_rate = float(info.get("proxygap_curve_yaw_rate_command_step", 0.0))
    actual_yaw_rate = float(info.get("proxygap_curve_actual_yaw_rate_step", 0.0))
    cross_velocity = float(info.get("proxygap_curve_cross_axis_velocity_step", 0.0))
    draw.text((x, 18), "CURVED GAIT / FIXED CAMERA", font=small, fill=(190, 202, 214, 255))
    draw.text((x, 45), profile.upper().replace("_", " "), font=title, fill=(242, 246, 250, 255))
    draw.rounded_rectangle((x, 82, canvas.width - 20, 124), radius=7, fill=(74, 160, 245, 255))
    draw.text((x + 12, 92), "TANGENT-ALIGNED WALK", font=text_font, fill=(12, 18, 24, 255))
    lines = (
        ("time", f"{step / fps:5.2f} s"),
        ("speed cmd", f"{speed:5.2f} m/s"),
        ("target yaw", f"{math.degrees(heading):+6.1f} deg"),
        ("body yaw", f"{math.degrees(actual_yaw):+6.1f} deg"),
        ("heading err", f"{math.degrees(error):+6.1f} deg"),
        ("yaw rate", f"{actual_yaw_rate:+5.2f} / {target_yaw_rate:+5.2f}"),
        ("cross speed", f"{cross_velocity:+5.2f} m/s"),
        ("frame", f"{step:04d}/{total:04d}"),
    )
    for row, (label, value) in enumerate(lines):
        y = 153 + row * 31
        draw.text((x, y), label, font=small, fill=(145, 160, 176, 255))
        draw.text((x + 105, y - 2), value, font=text_font, fill=(220, 228, 236, 255))
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    if args.curvature <= 0 or args.speed <= 0 or args.steps <= 0 or args.fps != 20:
        raise ValueError("positive curvature/speed/steps and real-time 20 fps are required")
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = env_kwargs(config)
    probe = make_curved_gait_env(
        condition_id="CURVED_GAIT_VIDEO_PROBE",
        seed=args.evaluation_seed,
        profile=args.profile,
        speed_min=args.speed,
        speed_max=args.speed,
        max_abs_curvature=args.curvature,
        max_episode_steps=args.steps,
        **kwargs,
    )
    probe.reset(seed=args.evaluation_seed)
    start_xy = np.asarray(probe.unwrapped.data.qpos[:2], dtype=np.float64).copy()
    initial_heading = quaternion_yaw_angle(np.asarray(probe.unwrapped.data.qpos[3:7]))
    dt = float(probe.unwrapped.dt)
    probe.close()
    points = target_path(
        start_xy=start_xy,
        initial_heading=initial_heading,
        profile=args.profile,
        curvature=args.curvature,
        speed=args.speed,
        steps=args.steps,
        dt=dt,
        period_steps=int(config["commands"]["s_curve_period_steps"]),
        slew_rate=float(config["commands"]["curvature_slew_rate_per_m_per_s"]),
    )
    prepend_optional_dependency_target(args.ffmpeg_target, package_marker="imageio_ffmpeg")
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(f"Install imageio-ffmpeg or set {FFMPEG_TARGET_ENV}") from error
    os.environ["MUJOCO_GL"] = args.backend
    env = make_curved_gait_env(
        condition_id="CURVED_GAIT_FIXED_CAMERA_VIDEO",
        seed=args.evaluation_seed,
        render_mode="rgb_array",
        xml_file=path_xml(points, show_target_path=not args.hide_target_path),
        profile=args.profile,
        speed_min=args.speed,
        speed_max=args.speed,
        max_abs_curvature=args.curvature,
        max_episode_steps=args.steps,
        **kwargs,
    )
    observation, info = env.reset(seed=args.evaluation_seed)
    model = PPO.load(model_path, device=args.device)
    env.render()
    viewer = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
    env.unwrapped.mujoco_renderer.camera_id = -1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    viewer.cam.lookat[:] = [center[0], center[1], 0.35]
    viewer.cam.distance = args.camera_distance
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = args.camera_elevation
    fixed_camera = {
        "lookat": viewer.cam.lookat.copy().tolist(),
        "distance": float(viewer.cam.distance),
        "azimuth": float(viewer.cam.azimuth),
        "elevation": float(viewer.cam.elevation),
    }
    actual_positions = [np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()]
    termination_reason = "requested_horizon"
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
        for step in range(1, args.steps + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            actual_yaw = quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
            actual_positions.append(np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy())
            writer.append_data(
                overlay(
                    env.render(),
                    step=step,
                    total=args.steps,
                    profile=args.profile,
                    speed=args.speed,
                    info=info,
                    actual_yaw=actual_yaw,
                    fps=args.fps,
                )
            )
            if terminated or truncated:
                termination_reason = "terminated" if terminated else "time_limit"
                break
    summary = env.episode_summary()
    env.close()
    actual = np.asarray(actual_positions)
    manifest = {
        "status": "complete",
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "config_path": str(config_path),
        "profile": args.profile,
        "speed_m_per_s": args.speed,
        "max_abs_curvature_per_m": args.curvature,
        "evaluation_seed": args.evaluation_seed,
        "frames": len(actual_positions) - 1,
        "fps": args.fps,
        "duration_seconds": (len(actual_positions) - 1) / args.fps,
        "termination_reason": termination_reason,
        "target_path_markers_visible": not args.hide_target_path,
        "fixed_camera": fixed_camera,
        "planned_start_xy": points[0].tolist(),
        "planned_final_xy": points[min(len(actual_positions) - 1, len(points) - 1)].tolist(),
        "actual_start_xy": actual[0].tolist(),
        "actual_final_xy": actual[-1].tolist(),
        "episode_summary": summary,
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
