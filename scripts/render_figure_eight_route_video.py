"""Drive the trained local gait around a closed-loop figure-eight route."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

try:
    from _portable_runtime import (
        FFMPEG_TARGET_ENV,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )
except ModuleNotFoundError:
    from scripts._portable_runtime import (
        FFMPEG_TARGET_ENV,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap.planar_transition import (  # noqa: E402
    quaternion_yaw_angle,
    wrapped_angle_difference,
)
from render_curved_gait_video import (  # noqa: E402
    env_kwargs,
    json_safe,
    load_font,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "curved_gait_tangent_v4_canonical_frame_20260818.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument(
        "--semi-major",
        type=float,
        default=None,
        help="Horizontal semi-axis for both ellipses; requires --semi-minor.",
    )
    parser.add_argument(
        "--semi-minor",
        type=float,
        default=None,
        help="Vertical semi-axis for both ellipses; requires --semi-major.",
    )
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--lookahead", type=float, default=0.8)
    parser.add_argument(
        "--controller-mode",
        choices=("bounded_heading", "tangent_lateral", "pure_pursuit"),
        default="bounded_heading",
    )
    parser.add_argument("--path-heading-correction-limit-degrees", type=float, default=10.0)
    parser.add_argument("--yaw-feedback-gain", type=float, default=1.0)
    parser.add_argument("--yaw-rate-limit", type=float, default=0.28)
    parser.add_argument("--bounded-heading-use-lateral", action="store_true")
    parser.add_argument("--lateral-error-gain", type=float, default=1.0)
    parser.add_argument("--lateral-speed-limit", type=float, default=0.45)
    parser.add_argument("--lateral-command-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--heading-lead-degrees", type=float, default=0.0)
    parser.add_argument("--axis-feedback-gain", type=float, default=0.0)
    parser.add_argument("--axis-feedback-limit-degrees", type=float, default=35.0)
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--evaluation-seed", type=int, default=53401)
    parser.add_argument("--playback-speed", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", default="glfw")
    parser.add_argument("--camera-distance", type=float, default=16.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-62.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg-target", type=Path, default=optional_ffmpeg_target())
    return parser.parse_args()


def _uniform_arc_phases(
    horizontal_radius: float,
    vertical_radius: float,
    *,
    points_per_loop: int,
) -> tuple[np.ndarray, float]:
    """Return approximately uniform arc-length samples for one ellipse."""
    if math.isclose(horizontal_radius, vertical_radius):
        phases = np.linspace(0.0, 2.0 * math.pi, points_per_loop + 1)
        return phases, 2.0 * math.pi * horizontal_radius
    dense_count = max(20_000, points_per_loop * 20)
    dense_phase = np.linspace(0.0, 2.0 * math.pi, dense_count + 1)
    dense_points = np.column_stack(
        (
            horizontal_radius * np.cos(dense_phase),
            vertical_radius * np.sin(dense_phase),
        )
    )
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1)),
        )
    )
    target = np.linspace(0.0, float(cumulative[-1]), points_per_loop + 1)
    return np.interp(target, cumulative, dense_phase), float(cumulative[-1])


def figure_eight_route(
    horizontal_radius: float,
    vertical_radius: float | None = None,
    *,
    points_per_loop: int = 1000,
):
    """Return two tangent circles or ellipses from their centre intersection.

    The right loop is traversed counter-clockwise and the left loop clockwise.
    Both branches have the same -Y tangent at the intersection. Samples are
    approximately uniform in arc length so controller lookahead remains metric.
    """
    vertical_radius = (
        horizontal_radius if vertical_radius is None else vertical_radius
    )
    if horizontal_radius <= 0 or vertical_radius <= 0 or points_per_loop < 4:
        raise ValueError("ellipse radii and points_per_loop must be positive")
    right_phase, perimeter = _uniform_arc_phases(
        horizontal_radius,
        vertical_radius,
        points_per_loop=points_per_loop,
    )
    left_phase = right_phase[1:]
    right = np.column_stack(
        (
            horizontal_radius - horizontal_radius * np.cos(right_phase),
            -vertical_radius * np.sin(right_phase),
        )
    )
    left = np.column_stack(
        (
            -horizontal_radius + horizontal_radius * np.cos(left_phase),
            -vertical_radius * np.sin(left_phase),
        )
    )
    positions = np.vstack((right, left))
    right_derivative = np.column_stack(
        (
            horizontal_radius * np.sin(right_phase),
            -vertical_radius * np.cos(right_phase),
        )
    )
    left_derivative = np.column_stack(
        (
            -horizontal_radius * np.sin(left_phase),
            -vertical_radius * np.cos(left_phase),
        )
    )
    right_tangent = right_derivative / np.linalg.norm(
        right_derivative, axis=1, keepdims=True
    )
    left_tangent = left_derivative / np.linalg.norm(
        left_derivative, axis=1, keepdims=True
    )
    tangents = np.vstack((right_tangent, left_tangent))
    right_denominator = (
        (horizontal_radius * np.sin(right_phase)) ** 2
        + (vertical_radius * np.cos(right_phase)) ** 2
    ) ** 1.5
    left_denominator = (
        (horizontal_radius * np.sin(left_phase)) ** 2
        + (vertical_radius * np.cos(left_phase)) ** 2
    ) ** 1.5
    curvatures = np.concatenate(
        (
            horizontal_radius * vertical_radius / right_denominator,
            -horizontal_radius * vertical_radius / left_denominator,
        )
    )
    spacing = perimeter / points_per_loop
    return positions, tangents, curvatures, spacing, points_per_loop


def route_xml(points: np.ndarray) -> Path:
    source = (ROOT / "assets" / "ant_render_large_floor.xml").read_text(encoding="utf-8")
    stride = max(1, len(points) // 260)
    markers = []
    for index, point in enumerate(points[::stride]):
        markers.append(
            f'<geom name="figure8_marker_{index}" type="cylinder" '
            f'pos="{point[0]:.6f} {point[1]:.6f} 0.014" size="0.05 0.014" '
            'rgba="0.12 0.48 0.95 0.90" contype="0" conaffinity="0"/>'
        )
    markers.append(
        '<geom name="figure8_centre" type="cylinder" pos="0 0 0.020" '
        'size="0.12 0.020" rgba="0.20 0.88 0.48 1" '
        'contype="0" conaffinity="0"/>'
    )
    rendered = source.replace("<worldbody>", "<worldbody>\n" + "\n".join(markers), 1)
    path = Path(tempfile.gettempdir()) / "proxygap_figure_eight_route.xml"
    path.write_text(rendered, encoding="utf-8")
    return path


def update_reference(
    points: np.ndarray,
    position: np.ndarray,
    reference_index: int,
    *,
    search_ahead: int,
) -> int:
    start = max(0, reference_index - 8)
    stop = min(len(points), reference_index + search_ahead + 1)
    distances = np.linalg.norm(points[start:stop] - position, axis=1)
    nearest = start + int(np.argmin(distances))
    return max(reference_index, nearest)


def exact_centre_start(env) -> np.ndarray:
    """Set root x/y to the crossing and torso yaw to the shared -Y tangent."""
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64).copy()
    qpos[:2] = 0.0
    yaw = -0.5 * math.pi
    qpos[3:7] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    qvel[:6] = 0.0
    env.unwrapped.set_state(qpos, qvel)
    env.env.metrics.reset(initial_x=0.0, initial_y=0.0)
    raw = np.asarray(env.unwrapped._get_obs())
    return env.env._augment_observation(raw)


def overlay(frame: np.ndarray, *, record: dict[str, Any], total_steps: int) -> np.ndarray:
    scene = Image.fromarray(frame).convert("RGB")
    panel = 320
    canvas = Image.new("RGB", (scene.width + panel, scene.height), "#111820")
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    x = scene.width + 20
    title = load_font(18)
    text_font = load_font(14)
    small = load_font(11)
    loop = "RIGHT LOOP" if record["loop_index"] == 1 else "LEFT LOOP"
    draw.text((x, 18), "FIGURE-EIGHT / FIXED CAMERA", font=small, fill=(190, 202, 214, 255))
    draw.text((x, 45), loop, font=title, fill=(242, 246, 250, 255))
    draw.rounded_rectangle((x, 82, canvas.width - 20, 124), radius=7, fill=(74, 160, 245, 255))
    draw.text((x + 12, 92), "CENTRE-START ROUTE", font=text_font, fill=(12, 18, 24, 255))
    lines = (
        ("sim time", f"{record['step'] * 0.05:6.2f} s"),
        ("route", f"{100.0 * record['route_fraction']:5.1f} %"),
        ("body yaw", f"{math.degrees(record['body_yaw']):+6.1f} deg"),
        ("path tangent", f"{math.degrees(record['path_tangent_heading']):+6.1f} deg"),
        ("axis error", f"{math.degrees(record['axis_error']):+6.1f} deg"),
        ("route error", f"{record['route_error_m']:5.2f} m"),
        ("speed", f"{record['planar_speed_m_per_s']:5.2f} m/s"),
        ("step", f"{record['step']:04d}/{total_steps:04d}"),
    )
    for row, (label, value) in enumerate(lines):
        y = 153 + row * 31
        draw.text((x, y), label, font=small, fill=(145, 160, 176, 255))
        draw.text((x + 113, y - 2), value, font=text_font, fill=(220, 228, 236, 255))
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    if (args.semi_major is None) != (args.semi_minor is None):
        raise ValueError("--semi-major and --semi-minor must be supplied together")
    horizontal_radius = args.radius if args.semi_major is None else args.semi_major
    vertical_radius = args.radius if args.semi_minor is None else args.semi_minor
    if (
        horizontal_radius <= 0
        or vertical_radius <= 0
        or args.speed <= 0
        or args.lookahead <= 0
        or args.max_steps <= 0
    ):
        raise ValueError("route/controller values must be positive")
    if (
        args.axis_feedback_gain < 0
        or args.axis_feedback_limit_degrees < 0
        or args.lateral_error_gain < 0
        or args.lateral_speed_limit < 0
        or args.path_heading_correction_limit_degrees < 0
        or args.yaw_feedback_gain < 0
        or args.yaw_rate_limit <= 0
    ):
        raise ValueError("controller gains and limits must be non-negative")
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = env_kwargs(config)
    points, tangents, curvatures, spacing, split_index = figure_eight_route(
        horizontal_radius,
        vertical_radius,
    )
    route_curvature = float(np.max(np.abs(curvatures)))
    if route_curvature > 0.35 + 1e-12:
        raise ValueError("route curvature exceeds the V4 training limit")
    render_mode = None if args.dry_run else "rgb_array"
    xml_file = None if args.dry_run else route_xml(points)
    if not args.dry_run:
        prepend_optional_dependency_target(args.ffmpeg_target, package_marker="imageio_ffmpeg")
        try:
            import imageio.v2 as imageio  # noqa: PLC0415
            import imageio_ffmpeg  # noqa: PLC0415
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(f"Install imageio-ffmpeg or set {FFMPEG_TARGET_ENV}") from error
        os.environ["MUJOCO_GL"] = args.backend
    env = make_curved_gait_env(
        condition_id="FIGURE_EIGHT_ROUTE_CONTROLLER",
        seed=args.evaluation_seed,
        render_mode=render_mode,
        xml_file=xml_file,
        max_episode_steps=args.max_steps,
        profile="external",
        speed_min=args.speed,
        speed_max=args.speed,
        max_abs_curvature=max(0.35, args.yaw_rate_limit / args.speed),
        heading_termination_enabled=False,
        **kwargs,
    )
    observation, _ = env.reset(seed=args.evaluation_seed)
    base_observation = exact_centre_start(env)
    initial_heading = -0.5 * math.pi
    commanded_curvature = route_curvature
    observation = env.set_external_curve_command(
        base_observation,
        target_heading=initial_heading,
        yaw_rate=args.speed * commanded_curvature,
        speed=args.speed,
    )
    model = PPO.load(model_path, device=args.device)
    writer = None
    fixed_camera = None
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env.render()
        viewer = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
        env.unwrapped.mujoco_renderer.camera_id = -1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        fixed_camera = {
            "lookat": viewer.cam.lookat.copy().tolist(),
            "distance": float(viewer.cam.distance),
            "azimuth": float(viewer.cam.azimuth),
            "elevation": float(viewer.cam.elevation),
        }
        writer = imageio.get_writer(
            output_path,
            format="FFMPEG",
            mode="I",
            fps=20,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="warning",
        )
    reference_index = 0
    lookahead_points = max(2, int(round(args.lookahead / spacing)))
    search_ahead = max(lookahead_points * 5, 80)
    records: list[dict[str, Any]] = []
    route_errors: list[float] = []
    axis_errors: list[float] = []
    right_axis_errors: list[float] = []
    left_axis_errors: list[float] = []
    cross_track_errors: list[float] = []
    lateral_speed_commands: list[float] = []
    actual_lateral_speeds: list[float] = []
    completion_reason = "maximum_steps"
    max_curvature_delta = float(config["commands"]["curvature_slew_rate_per_m_per_s"]) * float(env.unwrapped.dt)
    try:
        for step in range(1, args.max_steps + 1):
            position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
            reference_index = update_reference(
                points,
                position,
                reference_index,
                search_ahead=search_ahead,
            )
            target_index = min(reference_index + lookahead_points, len(points) - 1)
            delta = points[target_index] - position
            body_yaw_before_step = quaternion_yaw_angle(
                np.asarray(env.unwrapped.data.qpos[3:7])
            )
            current_path_tangent_heading = math.atan2(
                float(tangents[reference_index, 1]),
                float(tangents[reference_index, 0]),
            )
            current_normal = np.asarray(
                [-tangents[reference_index, 1], tangents[reference_index, 0]],
                dtype=np.float64,
            )
            cross_track_error = float(
                np.dot(position - points[reference_index], current_normal)
            )
            lateral_speed_command = float(
                np.clip(
                    args.lateral_command_sign
                    * args.lateral_error_gain
                    * cross_track_error,
                    -args.lateral_speed_limit,
                    args.lateral_speed_limit,
                )
            )
            pure_pursuit_heading = (
                math.atan2(float(delta[1]), float(delta[0]))
                if float(np.linalg.norm(delta)) > 1e-9
                else math.atan2(
                    float(tangents[target_index, 1]),
                    float(tangents[target_index, 0]),
                )
            )
            if args.controller_mode == "pure_pursuit":
                target_heading = pure_pursuit_heading
                lateral_speed_command = 0.0
            elif args.controller_mode == "bounded_heading":
                path_heading_correction = float(
                    np.clip(
                        wrapped_angle_difference(
                            pure_pursuit_heading,
                            current_path_tangent_heading,
                        ),
                        -math.radians(args.path_heading_correction_limit_degrees),
                        math.radians(args.path_heading_correction_limit_degrees),
                    )
                )
                target_heading = wrapped_angle_difference(
                    current_path_tangent_heading + path_heading_correction,
                    0.0,
                )
                if not args.bounded_heading_use_lateral:
                    lateral_speed_command = 0.0
            else:
                target_heading = current_path_tangent_heading
            current_axis_error = wrapped_angle_difference(
                body_yaw_before_step,
                current_path_tangent_heading,
            )
            axis_feedback = float(
                np.clip(
                    -args.axis_feedback_gain * current_axis_error,
                    -math.radians(args.axis_feedback_limit_degrees),
                    math.radians(args.axis_feedback_limit_degrees),
                )
            )
            target_heading = wrapped_angle_difference(
                target_heading + math.radians(args.heading_lead_degrees) + axis_feedback,
                0.0,
            )
            desired_curvature = float(curvatures[reference_index])
            commanded_curvature += float(
                np.clip(
                    desired_curvature - commanded_curvature,
                    -max_curvature_delta,
                    max_curvature_delta,
                )
            )
            yaw_rate_command = float(
                np.clip(
                    args.speed * commanded_curvature
                    + args.yaw_feedback_gain
                    * wrapped_angle_difference(target_heading, body_yaw_before_step),
                    -args.yaw_rate_limit,
                    args.yaw_rate_limit,
                )
            )
            observation = env.set_external_curve_command(
                observation,
                target_heading=target_heading,
                yaw_rate=yaw_rate_command,
                speed=args.speed,
                lateral_speed=lateral_speed_command,
            )
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
            velocity = np.asarray(env.unwrapped.data.qvel[:2], dtype=np.float64).copy()
            body_yaw = quaternion_yaw_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
            path_tangent_heading = math.atan2(
                float(tangents[reference_index, 1]),
                float(tangents[reference_index, 0]),
            )
            route_error = float(np.linalg.norm(position - points[reference_index]))
            axis_error = wrapped_angle_difference(body_yaw, path_tangent_heading)
            path_normal = np.asarray(
                [-tangents[reference_index, 1], tangents[reference_index, 0]],
                dtype=np.float64,
            )
            actual_lateral_speed = float(np.dot(velocity, path_normal))
            route_errors.append(route_error)
            axis_errors.append(axis_error)
            cross_track_errors.append(cross_track_error)
            lateral_speed_commands.append(lateral_speed_command)
            actual_lateral_speeds.append(actual_lateral_speed)
            (right_axis_errors if reference_index <= split_index else left_axis_errors).append(
                axis_error
            )
            record = {
                "step": step,
                "position_xy": position.copy(),
                "reference_xy": points[reference_index].copy(),
                "reference_index": reference_index,
                "route_fraction": reference_index / (len(points) - 1),
                "loop_index": 1 if reference_index <= split_index else 2,
                "body_yaw": body_yaw,
                "path_tangent_heading": path_tangent_heading,
                "axis_error": axis_error,
                "route_error_m": route_error,
                "planar_speed_m_per_s": float(np.linalg.norm(velocity)),
                "actual_lateral_speed_m_per_s": actual_lateral_speed,
                "commanded_curvature_per_m": commanded_curvature,
                "yaw_rate_command_rad_per_s": yaw_rate_command,
                "cross_track_error_m": cross_track_error,
                "lateral_speed_command_m_per_s": lateral_speed_command,
                "axis_feedback_degrees": math.degrees(axis_feedback),
                "heading_constraint_terminated": bool(
                    info["proxygap_heading_constraint_terminated"]
                ),
            }
            records.append(record)
            if writer is not None and (step - 1) % args.playback_speed == 0:
                writer.append_data(overlay(env.render(), record=record, total_steps=args.max_steps))
            if reference_index >= len(points) - lookahead_points - 1 and float(np.linalg.norm(position)) <= max(0.75, args.lookahead):
                completion_reason = "returned_to_centre"
                break
            if terminated:
                completion_reason = "environment_terminated"
                break
            if truncated:
                completion_reason = "time_limit"
                break
    finally:
        if writer is not None:
            writer.close()
    summary = env.episode_summary()
    env.close()
    final_position = records[-1]["position_xy"] if records else np.zeros(2)
    manifest = {
        "status": "complete",
        "route": (
            "two tangent circles; centre start; right loop then left loop"
            if math.isclose(horizontal_radius, vertical_radius)
            else "two tangent ellipses; centre start; right loop then left loop"
        ),
        "route_coordinates_enter_policy": False,
        "route_position_reward_enabled": False,
        "high_level_controller": (
            "monotone local projection plus bounded path-heading correction and yaw-rate feedback"
            if args.controller_mode == "bounded_heading"
            else (
                "monotone local projection plus tangent heading and lateral-velocity correction"
                if args.controller_mode == "tangent_lateral"
                else "monotone local projection plus pure-pursuit lookahead and optional body-axis feedback"
            )
        ),
        "start_position_xy": [0.0, 0.0],
        "start_heading_degrees": -90.0,
        "loop_order": ["right_counter_clockwise", "left_clockwise"],
        "radius_m": (
            horizontal_radius
            if math.isclose(horizontal_radius, vertical_radius)
            else None
        ),
        "semi_major_m": horizontal_radius,
        "semi_minor_m": vertical_radius,
        "route_length_m": float(
            np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        ),
        "route_curvature_abs_per_m": (
            route_curvature
            if math.isclose(horizontal_radius, vertical_radius)
            else None
        ),
        "route_curvature_abs_min_per_m": float(np.min(np.abs(curvatures))),
        "route_curvature_abs_max_per_m": route_curvature,
        "commanded_speed_m_per_s": args.speed,
        "lookahead_m": args.lookahead,
        "controller_mode": args.controller_mode,
        "path_heading_correction_limit_degrees": (
            args.path_heading_correction_limit_degrees
        ),
        "yaw_feedback_gain_per_second": args.yaw_feedback_gain,
        "yaw_rate_limit_rad_per_s": args.yaw_rate_limit,
        "bounded_heading_use_lateral": bool(args.bounded_heading_use_lateral),
        "lateral_error_gain_per_s": args.lateral_error_gain,
        "lateral_speed_limit_m_per_s": args.lateral_speed_limit,
        "lateral_command_sign": args.lateral_command_sign,
        "heading_lead_degrees": args.heading_lead_degrees,
        "axis_feedback_gain": args.axis_feedback_gain,
        "axis_feedback_limit_degrees": args.axis_feedback_limit_degrees,
        "steps": len(records),
        "simulation_seconds": len(records) * float(config["commands"]["environment_dt_seconds"]),
        "route_completion_fraction": records[-1]["route_fraction"] if records else 0.0,
        "completion_reason": completion_reason,
        "final_position_xy": final_position,
        "centre_closure_error_m": float(np.linalg.norm(final_position)),
        "route_error_mean_m": float(np.mean(route_errors)) if route_errors else None,
        "route_error_rms_m": float(np.sqrt(np.mean(np.square(route_errors)))) if route_errors else None,
        "route_error_max_m": float(np.max(route_errors)) if route_errors else None,
        "cross_track_error_mean_m": (
            float(np.mean(cross_track_errors)) if cross_track_errors else None
        ),
        "cross_track_error_rms_m": (
            float(np.sqrt(np.mean(np.square(cross_track_errors))))
            if cross_track_errors
            else None
        ),
        "lateral_speed_command_mean_m_per_s": (
            float(np.mean(lateral_speed_commands)) if lateral_speed_commands else None
        ),
        "actual_lateral_speed_mean_m_per_s": (
            float(np.mean(actual_lateral_speeds)) if actual_lateral_speeds else None
        ),
        "lateral_speed_tracking_rmse_m_per_s": (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(actual_lateral_speeds)
                            - np.asarray(lateral_speed_commands)
                        )
                    )
                )
            )
            if lateral_speed_commands
            else None
        ),
        "trace_every_100_steps": [
            json_safe(record)
            for record in records[::100]
        ],
        "body_axis_tangent_error_rms_degrees": (
            math.degrees(float(np.sqrt(np.mean(np.square(axis_errors))))) if axis_errors else None
        ),
        "body_axis_tangent_error_mean_degrees": (
            math.degrees(float(np.mean(axis_errors))) if axis_errors else None
        ),
        "right_loop_axis_error_mean_degrees": (
            math.degrees(float(np.mean(right_axis_errors))) if right_axis_errors else None
        ),
        "left_loop_axis_error_mean_degrees": (
            math.degrees(float(np.mean(left_axis_errors))) if left_axis_errors else None
        ),
        "right_loop_axis_error_rms_degrees": (
            math.degrees(float(np.sqrt(np.mean(np.square(right_axis_errors)))))
            if right_axis_errors
            else None
        ),
        "left_loop_axis_error_rms_degrees": (
            math.degrees(float(np.sqrt(np.mean(np.square(left_axis_errors)))))
            if left_axis_errors
            else None
        ),
        "body_axis_tangent_error_max_abs_degrees": (
            math.degrees(float(np.max(np.abs(axis_errors)))) if axis_errors else None
        ),
        "fall": bool(summary["fall"]),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "config_path": str(config_path),
        "evaluation_seed": args.evaluation_seed,
        "dry_run": bool(args.dry_run),
        "playback_speed": args.playback_speed,
        "video_frames": (
            math.ceil(len(records) / args.playback_speed) if not args.dry_run else 0
        ),
        "video_fps": 20 if not args.dry_run else None,
        "video_duration_seconds": (
            math.ceil(len(records) / args.playback_speed) / 20.0
            if not args.dry_run
            else None
        ),
        "fixed_camera": fixed_camera,
        "video_path": str(output_path) if not args.dry_run else None,
        "video_sha256": sha256(output_path) if not args.dry_run else None,
        "episode_summary": summary,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
