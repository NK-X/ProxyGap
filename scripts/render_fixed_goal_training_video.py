"""Render an auditable diagnostic video for a fixed-goal training checkpoint.

The video is a qualitative diagnostic, not a substitute for the recorded
episode metrics.  It uses a predeclared deterministic evaluation seed and
stores the checkpoint, map, configuration, rollout and video hashes.
"""

from __future__ import annotations

import argparse
import av
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.metrics import quaternion_tilt_angle  # noqa: E402
from run_fixed_goal_terrain_training import make_task_env  # noqa: E402


DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_training_20260818"
    / "seed_62801"
)
WIDTH = 1280
HEIGHT = 720
FPS = 20
BACKGROUND = (246, 244, 238)
INK = (24, 32, 39)
WHITE = (250, 252, 251)
BLUE = (57, 93, 169)
TEAL = (36, 138, 119)
AMBER = (232, 164, 55)
RED = (204, 67, 58)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--evaluation-seed", type=int, default=74803)
    parser.add_argument("--physical-seconds", type=float, default=45.0)
    parser.add_argument("--intro-seconds", type=float, default=1.5)
    parser.add_argument("--outro-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    candidates = (
        system_root / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        system_root / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def terrain_colours(heights: np.ndarray) -> np.ndarray:
    stops = np.asarray(
        [
            (24, 62, 82),
            (36, 102, 105),
            (91, 139, 91),
            (184, 166, 93),
            (237, 221, 181),
        ],
        dtype=np.float64,
    )
    normalised = (heights - float(heights.min())) / max(
        float(np.ptp(heights)), 1e-12
    )
    scaled = normalised * (len(stops) - 1)
    lower = np.minimum(scaled.astype(int), len(stops) - 2)
    weight = (scaled - lower)[..., None]
    colours = stops[lower] * (1.0 - weight) + stops[lower + 1] * weight
    return np.asarray(np.clip(colours, 0, 255), dtype=np.uint8)


def make_map_base(heights: np.ndarray, *, size: int = 222) -> Image.Image:
    rgb = terrain_colours(heights)
    bins = np.floor((heights - float(heights.min())) / 0.5).astype(np.int32)
    contour = np.zeros_like(heights, dtype=bool)
    contour[1:, :] |= bins[1:, :] != bins[:-1, :]
    contour[:, 1:] |= bins[:, 1:] != bins[:, :-1]
    rgb[contour] = np.asarray((242, 242, 232), dtype=np.uint8)
    return Image.fromarray(np.flipud(rgb), mode="RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )


def map_point(
    x: float,
    y: float,
    *,
    half_extent: float,
    origin: tuple[int, int],
    size: int,
) -> tuple[int, int]:
    px = origin[0] + round(
        np.clip((x + half_extent) / (2.0 * half_extent), 0.0, 1.0) * (size - 1)
    )
    py = origin[1] + round(
        np.clip((half_extent - y) / (2.0 * half_extent), 0.0, 1.0) * (size - 1)
    )
    return int(px), int(py)


def make_camera(
    *,
    position: np.ndarray,
    goal: np.ndarray,
    terrain_height: float,
    local_valley_depth: float,
    progress: float,
) -> mujoco.MjvCamera:
    direction = np.asarray(goal - position, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-9:
        direction /= norm
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    visibility_factor = float(np.clip(local_valley_depth / 1.5, 0.0, 1.0))
    camera.lookat[:] = (
        float(position[0] + 0.65 * direction[0]),
        float(position[1] + 0.65 * direction[1]),
        float(terrain_height + 0.45),
    )
    # A low chase view is useful for gait inspection on exposed ground.  In a
    # valley, the camera rises towards an overhead view so intervening
    # heightfield cells cannot hide the robot.  Heightfields have no overhangs,
    # hence a sufficiently steep overhead line of sight remains unobstructed.
    camera.distance = 7.4 + 2.2 * visibility_factor
    camera.azimuth = 220.0 + 7.0 * math.sin(2.0 * math.pi * progress)
    camera.elevation = -24.0 - 48.0 * visibility_factor
    return camera


def local_valley_depth(
    env: Any,
    position: np.ndarray,
    *,
    radius_m: float = 5.0,
) -> float:
    """Return surrounding terrain relief above the robot's local ground.

    Sampling four rings is deterministic and inexpensive relative to rendering.
    The quantity is used only for camera placement; it never enters the policy.
    """
    local_height = float(env._terrain_height(float(position[0]), float(position[1])))
    maximum_height = local_height
    for radius in np.linspace(radius_m / 4.0, radius_m, 4):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            sample_x = float(position[0] + radius * math.cos(float(angle)))
            sample_y = float(position[1] + radius * math.sin(float(angle)))
            maximum_height = max(
                maximum_height,
                float(env._terrain_height(sample_x, sample_y)),
            )
    return max(0.0, maximum_height - local_height)


def draw_overlay(
    rgb: np.ndarray,
    *,
    mode: str,
    physical_time: float,
    requested_seconds: float,
    position: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    trail: list[np.ndarray],
    map_base: Image.Image,
    half_extent: float,
    distance: float,
    best_progress: float,
    torso_tilt_degrees: float,
    support_count: int,
    maximum_contact_speed: float,
    slip_threshold: float,
    camera_valley_depth: float,
    airborne: bool,
    terminated: bool,
    evaluation_seed: int,
) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((22, 20, 570, 100), radius=13, fill=(250, 252, 251, 236))
    draw.text((42, 32), "FIXED-MAP FINAL POLICY", font=font(21, bold=True), fill=INK)
    subtitle = (
        f"DETERMINISTIC REPRESENTATIVE EVALUATION - SEED {evaluation_seed}"
        if mode == "rollout"
        else "FROZEN TERRAIN AND CHECKPOINT PROVENANCE"
    )
    draw.text((42, 68), subtitle, font=font(11, bold=True), fill=TEAL)

    map_size = map_base.width
    map_origin = (WIDTH - map_size - 26, 26)
    draw.rounded_rectangle(
        (
            map_origin[0] - 10,
            map_origin[1] - 10,
            map_origin[0] + map_size + 10,
            map_origin[1] + map_size + 38,
        ),
        radius=12,
        fill=(250, 252, 251, 238),
    )
    image.paste(map_base, map_origin)
    draw = ImageDraw.Draw(image, "RGBA")
    if len(trail) > 1:
        trail_points = [
            map_point(
                float(point[0]),
                float(point[1]),
                half_extent=half_extent,
                origin=map_origin,
                size=map_size,
            )
            for point in trail
        ]
        draw.line(trail_points, fill=(255, 255, 255, 235), width=3)
    start_marker = map_point(
        float(start[0]),
        float(start[1]),
        half_extent=half_extent,
        origin=map_origin,
        size=map_size,
    )
    goal_marker = map_point(
        float(goal[0]),
        float(goal[1]),
        half_extent=half_extent,
        origin=map_origin,
        size=map_size,
    )
    current_marker = map_point(
        float(position[0]),
        float(position[1]),
        half_extent=half_extent,
        origin=map_origin,
        size=map_size,
    )
    draw.ellipse(
        (start_marker[0] - 5, start_marker[1] - 5, start_marker[0] + 5, start_marker[1] + 5),
        fill=TEAL,
        outline=WHITE,
        width=2,
    )
    draw.regular_polygon((goal_marker[0], goal_marker[1], 8), n_sides=5, rotation=18, fill=AMBER, outline=WHITE)
    draw.ellipse(
        (
            current_marker[0] - 7,
            current_marker[1] - 7,
            current_marker[0] + 7,
            current_marker[1] + 7,
        ),
        fill=BLUE,
        outline=WHITE,
        width=2,
    )
    draw.text(
        (map_origin[0], map_origin[1] + map_size + 8),
        "True elevation contours | S start | star goal",
        font=font(9, bold=True),
        fill=INK,
    )

    panel_top = HEIGHT - 108
    draw.rectangle((0, panel_top, WIDTH, HEIGHT), fill=(17, 26, 33, 224))
    time_text = (
        f"t = {physical_time:05.2f} / {requested_seconds:.2f} s"
        if mode == "rollout"
        else "Recorded deterministic policy rollout"
    )
    draw.text((28, panel_top + 18), time_text, font=font(15, bold=True), fill=WHITE)
    draw.text(
        (28, panel_top + 54),
        f"distance {distance:6.2f} m   |   best progress {best_progress:5.2f} m   |   position ({position[0]:+.2f}, {position[1]:+.2f}) m",
        font=font(12),
        fill=(219, 227, 230),
    )

    contact_exceeded = maximum_contact_speed > slip_threshold
    status_colour = RED if terminated or airborne or contact_exceeded else TEAL
    if terminated:
        status = "TERMINATED / FALL"
    elif airborne:
        status = "FOUR-FOOT AIRBORNE"
    elif contact_exceeded:
        status = "CONTACT-SPEED EXCEEDANCE"
    else:
        status = "NO EVENT FLAG"
    draw.rounded_rectangle((760, panel_top + 14, 1252, panel_top + 88), radius=10, fill=(*status_colour, 222))
    draw.text((780, panel_top + 25), status, font=font(14, bold=True), fill=WHITE)
    draw.text(
        (780, panel_top + 57),
        f"support {support_count}/4 | tilt {torso_tilt_degrees:4.1f} deg | contact max {maximum_contact_speed:4.2f} m/s | valley {camera_valley_depth:.2f} m",
        font=font(10),
        fill=WHITE,
    )
    return image


def encode_frame(stream: Any, container: Any, image: Image.Image) -> None:
    frame = av.VideoFrame.from_ndarray(
        np.asarray(image.convert("RGB"), dtype=np.uint8), format="rgb24"
    )
    for packet in stream.encode(frame):
        container.mux(packet)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Rollout record cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(frames: list[tuple[str, Image.Image]], path: Path) -> None:
    canvas = Image.new("RGB", (1240, 744), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), "Fixed-map final policy - diagnostic frames", font=font(21, bold=True), fill=INK)
    for index, (label, frame) in enumerate(frames[:4]):
        row, column = divmod(index, 2)
        x = 20 + column * 610
        y = 58 + row * 340
        thumb = frame.resize((600, 338), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.rounded_rectangle((x + 12, y + 12, x + 224, y + 44), radius=7, fill=(250, 252, 251))
        draw.text((x + 24, y + 20), label, font=font(11, bold=True), fill=INK)
    canvas.save(path, format="PNG", optimize=True)


def validate_video(path: Path, *, expected_width: int, expected_height: int) -> dict[str, Any]:
    decoded = 0
    first_shape: tuple[int, ...] | None = None
    last_shape: tuple[int, ...] | None = None
    with av.open(str(path), mode="r") as container:
        video_stream = container.streams.video[0]
        average_rate = float(video_stream.average_rate)
        for frame in container.decode(video=0):
            shape = frame.to_ndarray(format="rgb24").shape
            if first_shape is None:
                first_shape = shape
            last_shape = shape
            decoded += 1
    if decoded <= 0 or first_shape != (expected_height, expected_width, 3):
        raise RuntimeError("Encoded video failed frame-level decoding validation")
    if last_shape != first_shape:
        raise RuntimeError("Encoded video frame geometry changed during playback")
    return {
        "decoded_frames": decoded,
        "first_frame_shape": list(first_shape),
        "last_frame_shape": list(last_shape),
        "average_frame_rate": average_rate,
        "decoded_duration_seconds": decoded / average_rate,
    }


def main() -> None:
    args = parse_args()
    if args.physical_seconds < 10.0:
        raise ValueError("physical-seconds must be at least 10 for an acceptance video")
    if args.fps != FPS:
        raise ValueError("fps must equal the 20 Hz environment control rate")
    run_root = args.run_root.expanduser().resolve()
    config_path = run_root / "frozen_run_config.json"
    execution_path = run_root / "execution_record.json"
    scene_path = run_root / "task_scenes" / "spawn_0_0.000.xml"
    for path in (config_path, execution_path, scene_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    v22_config_path = ROOT / config["base_policy"]["configuration"]
    v22_config = json.loads(v22_config_path.read_text(encoding="utf-8"))
    model_path = (
        args.model.expanduser().resolve()
        if args.model is not None
        else Path(execution["final_model"]).resolve()
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    expected_model_hash = str(execution.get("final_model_sha256", ""))
    if expected_model_hash and sha256(model_path) != expected_model_hash:
        raise ValueError("Final checkpoint SHA-256 does not match execution record")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "videos" / f"representative_seed_{args.evaluation_seed}"
    )
    filename_stem = f"fixed_map_final_policy_seed_{args.evaluation_seed}"
    video_path = output_dir / f"{filename_stem}.mp4"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    approved = config["approved_map"]
    heights_path = ROOT / approved["heights_path"]
    if sha256(heights_path) != approved["heights_sha256"]:
        raise ValueError("Frozen height map has changed")
    heights = np.load(heights_path, allow_pickle=False)
    half_extent = float(approved["map_half_extent_m"])
    start = np.asarray(approved["start_xy_m"], dtype=np.float64)
    goal = np.asarray(approved["goal_xy_m"], dtype=np.float64)
    map_base = make_map_base(heights)

    environment_dt = 1.0 / FPS
    requested_steps = round(args.physical_seconds / environment_dt)
    if not math.isclose(requested_steps * environment_dt, args.physical_seconds, abs_tol=1e-9):
        raise ValueError("physical-seconds must be divisible by 0.05 s")
    env = make_task_env(
        config,
        v22_config,
        xml_path=scene_path,
        seed=args.evaluation_seed,
        spawn_fraction=0.0,
        max_episode_steps=requested_steps,
        cruise_speed=float(config["evaluation"]["cruise_speed_m_per_s"]),
        terminate_on_success=True,
    )
    observation, info = env.reset(seed=args.evaluation_seed)
    actual_dt = float(env.unwrapped.dt)
    if not math.isclose(actual_dt, environment_dt, abs_tol=1e-12):
        env.close()
        raise ValueError(f"Expected dt={environment_dt}; observed {actual_dt}")
    policy = PPO.load(model_path, device="cpu")
    if tuple(observation.shape) != tuple(policy.observation_space.shape):
        env.close()
        raise RuntimeError("Policy and environment observation spaces do not match")

    compiled = env.unwrapped.model
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_friction = np.asarray(compiled.geom_friction[floor_id], dtype=np.float64)
    floor_condim = int(compiled.geom_condim[floor_id])
    if not np.allclose(floor_friction, approved["fixed_friction"], atol=0.0, rtol=0.0):
        env.close()
        raise RuntimeError("Compiled floor friction differs from frozen configuration")
    if floor_condim != int(approved["condim"]):
        env.close()
        raise RuntimeError("Compiled floor condim differs from frozen configuration")

    compiled.vis.global_.offwidth = WIDTH
    compiled.vis.global_.offheight = HEIGHT
    renderer = mujoco.Renderer(compiled, height=HEIGHT, width=WIDTH)
    scene_option = mujoco.MjvOption()
    # The large start/goal cylinders belong to visual site group 2.  Hide
    # them in the 3D view because an overhead valley camera can project the
    # start marker over the robot.  Start and goal remain explicit in the
    # topographic inset and this setting cannot affect MuJoCo dynamics.
    scene_option.sitegroup[2] = 0
    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}
    stream.gop_size = args.fps * 2

    intro_frames = round(args.intro_seconds * args.fps)
    outro_frames = round(args.outro_seconds * args.fps)
    initial_distance = float(np.linalg.norm(goal - np.asarray(env.unwrapped.data.qpos[:2])))
    best_progress = 0.0
    trail: list[np.ndarray] = [
        np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
    ]
    records: list[dict[str, Any]] = []
    keyframes: list[tuple[str, Image.Image]] = []
    last_frame: Image.Image | None = None

    initial_position = trail[-1]
    initial_terrain = float(env._terrain_height(float(initial_position[0]), float(initial_position[1])))
    initial_valley_depth = local_valley_depth(env, initial_position)
    for index in range(intro_frames):
        renderer.update_scene(
            env.unwrapped.data,
            camera=make_camera(
                position=initial_position,
                goal=goal,
                terrain_height=initial_terrain,
                local_valley_depth=initial_valley_depth,
                progress=index / max(1, intro_frames - 1),
            ),
            scene_option=scene_option,
        )
        frame = draw_overlay(
            np.asarray(renderer.render(), dtype=np.uint8),
            mode="intro",
            physical_time=0.0,
            requested_seconds=args.physical_seconds,
            position=initial_position,
            start=start,
            goal=goal,
            trail=trail,
            map_base=map_base,
            half_extent=half_extent,
            distance=initial_distance,
            best_progress=0.0,
            torso_tilt_degrees=math.degrees(
                quaternion_tilt_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
            ),
            support_count=0,
            maximum_contact_speed=0.0,
            slip_threshold=float(config["task_adapter"]["slip_speed_threshold_m_per_s"]),
            camera_valley_depth=initial_valley_depth,
            airborne=False,
            terminated=False,
            evaluation_seed=args.evaluation_seed,
        )
        encode_frame(stream, container, frame)
        last_frame = frame
    if last_frame is not None:
        keyframes.append(("Frozen start state", last_frame.copy()))

    termination_reason = "requested_video_horizon"
    completed_steps = 0
    for step in range(1, requested_steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        completed_steps = step
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        position = qpos[:2].copy()
        trail.append(position)
        distance = float(np.linalg.norm(goal - position))
        best_progress = max(best_progress, initial_distance - distance)
        terrain_height = float(env._terrain_height(float(position[0]), float(position[1])))
        valley_depth = local_valley_depth(env, position)
        contact_mask = np.asarray(
            info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool
        )
        contact_speeds = np.asarray(
            info.get(
                "proxygap_foot_contact_tangential_speeds_m_per_s_step", np.zeros(4)
            ),
            dtype=np.float64,
        )
        active_speeds = (
            contact_speeds[contact_mask]
            if contact_mask.shape == (4,) and contact_speeds.shape == (4,)
            else np.asarray([])
        )
        maximum_contact_speed = float(active_speeds.max()) if active_speeds.size else 0.0
        airborne = bool(
            contact_mask.shape == (4,) and not np.any(contact_mask)
        )
        torso_tilt = float(quaternion_tilt_angle(qpos[3:7]))
        records.append(
            {
                "step": step,
                "time_seconds": step * actual_dt,
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "terrain_height_m": terrain_height,
                "torso_z_m": float(qpos[2]),
                "distance_to_goal_m": distance,
                "best_progress_m": best_progress,
                "torso_tilt_degrees": math.degrees(torso_tilt),
                "support_count": int(contact_mask.sum()) if contact_mask.shape == (4,) else 0,
                "maximum_contact_tangential_speed_m_per_s": maximum_contact_speed,
                "camera_local_valley_depth_m": valley_depth,
                "contact_speed_threshold_exceeded": maximum_contact_speed
                > float(config["task_adapter"]["slip_speed_threshold_m_per_s"]),
                "airborne": airborne,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        renderer.update_scene(
            env.unwrapped.data,
            camera=make_camera(
                position=position,
                goal=goal,
                terrain_height=terrain_height,
                local_valley_depth=valley_depth,
                progress=step / requested_steps,
            ),
            scene_option=scene_option,
        )
        frame = draw_overlay(
            np.asarray(renderer.render(), dtype=np.uint8),
            mode="rollout",
            physical_time=step * actual_dt,
            requested_seconds=args.physical_seconds,
            position=position,
            start=start,
            goal=goal,
            trail=trail,
            map_base=map_base,
            half_extent=half_extent,
            distance=distance,
            best_progress=best_progress,
            torso_tilt_degrees=math.degrees(torso_tilt),
            support_count=int(contact_mask.sum()) if contact_mask.shape == (4,) else 0,
            maximum_contact_speed=maximum_contact_speed,
            slip_threshold=float(config["task_adapter"]["slip_speed_threshold_m_per_s"]),
            camera_valley_depth=valley_depth,
            airborne=airborne,
            terminated=bool(terminated),
            evaluation_seed=args.evaluation_seed,
        )
        encode_frame(stream, container, frame)
        last_frame = frame
        if step in {requested_steps // 3, 2 * requested_steps // 3, requested_steps}:
            keyframes.append((f"Physical rollout t={step * actual_dt:.1f} s", frame.copy()))
        if terminated or truncated:
            termination_reason = "terminated" if terminated else "time_limit"
            break

    if last_frame is None:
        renderer.close()
        env.close()
        container.close()
        raise RuntimeError("No rollout frame was produced")
    for _ in range(outro_frames):
        encode_frame(stream, container, last_frame)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    episode_summary = env.episode_summary()
    renderer.close()
    env.close()
    trace_path = output_dir / f"{filename_stem}_trace.csv"
    write_rows(trace_path, records)
    contact_sheet_path = output_dir / f"{filename_stem}_contact_sheet.png"
    make_contact_sheet(keyframes, contact_sheet_path)
    last_frame.save(
        output_dir / f"{filename_stem}_final_frame.png",
        format="PNG",
        optimize=True,
    )

    qa = validate_video(video_path, expected_width=WIDTH, expected_height=HEIGHT)
    total_frames = intro_frames + completed_steps + outro_frames
    if qa["decoded_frames"] != total_frames:
        raise RuntimeError(
            f"Decoded frame count {qa['decoded_frames']} differs from encoded count {total_frames}"
        )
    manifest = {
        "schema_version": "proxygap-fixed-goal-training-video-v1",
        "purpose": "qualitative diagnostic paired to quantitative fixed-map evaluation",
        "selection_rule": "predeclared representative evaluation seed; no best-video selection",
        "camera_rule": (
            "robot-following free camera; elevation adapts from -24 to -72 degrees "
            "using deterministic 5 m local valley relief, preventing heightfield "
            "occlusion while retaining a lower gait-inspection view on exposed ground; "
            "large 3D start/goal site markers are hidden to prevent visual occlusion "
            "and remain visible in the topographic inset"
        ),
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": args.fps,
            "frames": total_frames,
            "duration_seconds": total_frames / args.fps,
            "physical_rollout_seconds": completed_steps * actual_dt,
        },
        "rollout": {
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "configuration": str(config_path),
            "configuration_sha256": sha256(config_path),
            "scene": str(scene_path),
            "scene_sha256": sha256(scene_path),
            "height_array": str(heights_path),
            "height_array_sha256": sha256(heights_path),
            "evaluation_seed": args.evaluation_seed,
            "deterministic_policy": True,
            "commanded_speed_m_per_s": float(config["evaluation"]["cruise_speed_m_per_s"]),
            "completed_steps": completed_steps,
            "termination_reason": termination_reason,
            "initial_distance_m": initial_distance,
            "best_progress_m": best_progress,
            "episode_summary_at_video_end": json_safe(episode_summary),
        },
        "terrain_contact": {
            "friction": floor_friction.tolist(),
            "condim": floor_condim,
        },
        "diagnostic_boundary": (
            "Contact-speed threshold exceedance is a proxy and may include impacts or brief "
            "foot adjustment; video evidence does not replace the full five-seed evaluation."
        ),
        "files": {
            "trace_csv": str(trace_path),
            "trace_csv_sha256": sha256(trace_path),
            "contact_sheet": str(contact_sheet_path),
            "contact_sheet_sha256": sha256(contact_sheet_path),
        },
        "qa": {
            **qa,
            "duration_at_least_10_seconds": qa["decoded_duration_seconds"] >= 10.0,
            "map_hash_matches_frozen_config": sha256(heights_path)
            == approved["heights_sha256"],
            "checkpoint_hash_matches_execution_record": (
                not expected_model_hash or sha256(model_path) == expected_model_hash
            ),
            "friction_matches_frozen_config": np.array_equal(
                floor_friction, np.asarray(approved["fixed_friction"], dtype=np.float64)
            ),
        },
    }
    manifest_path = output_dir / f"{filename_stem}_video_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
