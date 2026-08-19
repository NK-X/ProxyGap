"""Render the fixed-map arrival rollout as an auditable dual-view video.

The left pane follows the robot and raises the camera in valleys.  The right
pane is a fixed oblique overview from the goal side towards the start.  Both
panes show the same deterministic MuJoCo state; the overview is not a second
simulation.  The surface trail and overview position marker are visual aids
only and cannot affect contacts, observations, actions or rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.metrics import quaternion_tilt_angle  # noqa: E402
from render_fixed_goal_training_video import (  # noqa: E402
    AMBER,
    BACKGROUND,
    BLUE,
    INK,
    RED,
    TEAL,
    WHITE,
    encode_frame,
    font,
    json_safe,
    local_valley_depth,
    make_camera,
    make_contact_sheet,
    make_map_base,
    map_point,
    sha256,
    validate_video,
    write_rows,
)
from run_fixed_goal_terrain_training import make_task_env  # noqa: E402


DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_training_20260818"
    / "seed_62801"
)
DEFAULT_EVALUATION_CONTRACT = (
    ROOT / "configs" / "fixed_map_reach_a_corrected_replication_v2_20260819.json"
)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIEW_WIDTH = FRAME_WIDTH // 2
VIEW_HEIGHT = 560
PANEL_HEIGHT = FRAME_HEIGHT - VIEW_HEIGHT
FPS = 20
MAP_SIZE = 174
TRAIL_MAX_SEGMENTS = 1600


def load_video_encoder() -> Any:
    """Load the optional PyAV dependency only for an actual render run."""

    try:
        import av  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dual-view video rendering requires PyAV; install it in the render "
            "environment before running this script."
        ) from exc
    return av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--evaluation-contract", type=Path, default=DEFAULT_EVALUATION_CONTRACT
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--condition-label",
        help="Optional auditable condition label shown in the bottom panel.",
    )
    parser.add_argument(
        "--airborne-shaping-weight",
        type=float,
        help=(
            "Optional replay-only reward configuration override. This does not "
            "change policy actions or physics, but keeps episode reward metadata "
            "consistent with support-priority checkpoints."
        ),
    )
    parser.add_argument("--evaluation-seed", type=int, default=74803)
    parser.add_argument(
        "--physical-seconds",
        type=float,
        help="Defaults to the corrected evaluation horizon recorded in the contract.",
    )
    parser.add_argument("--intro-seconds", type=float, default=1.5)
    parser.add_argument("--outro-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--render-stride",
        type=int,
        default=20,
        help="Render every Nth 20 Hz physics step while recording every step.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def audited_contract_controller(contract: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Return controller settings from either legacy or paired-audit contracts."""
    if "controller" in contract:
        condition = dict(contract["controller"])
        return condition, float(condition["cruise_speed_m_per_s"])
    conditions = contract.get("controller_conditions", [])
    if len(conditions) != 1:
        raise ValueError("dual-view replay requires exactly one controller condition")
    return dict(conditions[0]), float(contract["cruise_speed_m_per_s"])


def formal_paired_episode_evidence(
    *,
    contract: dict[str, Any],
    model_hash: str,
    evaluation_seed: int,
) -> dict[str, Any] | None:
    """Load the matching formal row when rendering the new paired contract."""
    candidates = contract.get("candidates", [])
    matched = next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("model_sha256", "")).lower() == model_hash.lower()
        ),
        None,
    )
    if matched is None or "output_root" not in contract:
        return None
    results_path = ROOT / contract["output_root"] / "episode_results.csv"
    aggregate_path = ROOT / contract["output_root"] / "aggregate_results.json"
    if not results_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError("Formal paired result files are required for this video")
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        (
            item
            for item in rows
            if item.get("candidate") == matched.get("name")
            and int(item["evaluation_seed"]) == int(evaluation_seed)
        ),
        None,
    )
    if row is None:
        raise ValueError("Formal paired result lacks the predeclared video episode")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return {
        "candidate": matched.get("name"),
        "episode_results_csv": str(results_path.resolve()),
        "episode_results_csv_sha256": sha256(results_path),
        "aggregate_results_json": str(aggregate_path.resolve()),
        "aggregate_results_json_sha256": sha256(aggregate_path),
        "formal_episode_row": row,
        "formal_candidate_aggregate": aggregate["by_candidate"][matched["name"]],
    }


def overview_camera(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    half_extent: float,
    terrain_midpoint_height: float,
) -> mujoco.MjvCamera:
    """Return a fixed camera located on the goal side of the map.

    MuJoCo's free-camera azimuth points from the camera towards ``lookat``.
    Adding 180 degrees to the start-to-goal bearing therefore places the
    camera head on the goal side and makes it look back towards the start.
    """
    direction = np.asarray(goal - start, dtype=np.float64)
    if float(np.linalg.norm(direction)) <= 1e-9:
        raise ValueError("start and goal must be distinct")
    bearing_degrees = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (
        float((start[0] + goal[0]) / 2.0),
        float((start[1] + goal[1]) / 2.0),
        float(terrain_midpoint_height),
    )
    # Keep both diagonal corners in frame.  The extra margin is important near
    # the goal-side corner because the camera is oblique rather than vertical.
    camera.distance = 3.3 * float(half_extent)
    camera.azimuth = (bearing_degrees + 180.0) % 360.0
    camera.elevation = -55.0
    return camera


def resample_surface_trail(
    points_xyz: list[np.ndarray] | np.ndarray,
    *,
    maximum_points: int,
) -> np.ndarray:
    """Arc-length resample a trail while preserving both endpoints."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface trail must have shape (n, 3)")
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two")
    if len(points) <= maximum_points:
        return points.copy()
    increments = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], maximum_points, axis=0)
    targets = np.linspace(0.0, total, maximum_points)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)]
    )


def _initialise_visual_geom(
    geom: mujoco.MjvGeom,
    *,
    geom_type: mujoco.mjtGeom,
    rgba: tuple[float, float, float, float],
) -> None:
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.emission = 0.28


def add_surface_trail(
    scene: mujoco.MjvScene,
    points_xyz: list[np.ndarray],
    *,
    maximum_segments: int = TRAIL_MAX_SEGMENTS,
) -> int:
    """Add a visual-only line trail just above the heightfield surface."""
    available = max(0, int(scene.maxgeom) - int(scene.ngeom) - 3)
    segment_limit = min(int(maximum_segments), available)
    if len(points_xyz) < 2 or segment_limit <= 0:
        return 0
    sampled = resample_surface_trail(
        points_xyz,
        maximum_points=min(len(points_xyz), segment_limit + 1),
    )
    added = 0
    for start_point, end_point in zip(sampled[:-1], sampled[1:]):
        if float(np.linalg.norm(end_point - start_point)) <= 1e-8:
            continue
        geom = scene.geoms[scene.ngeom]
        _initialise_visual_geom(
            geom,
            geom_type=mujoco.mjtGeom.mjGEOM_LINE,
            rgba=(0.98, 0.55, 0.08, 0.96),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            4.0,
            start_point,
            end_point,
        )
        scene.ngeom += 1
        added += 1
    return added


def add_overview_position_marker(
    scene: mujoco.MjvScene,
    position_xyz: np.ndarray,
) -> None:
    """Add a visual-only cyan mast above the robot in the overview pane."""
    if int(scene.ngeom) + 2 > int(scene.maxgeom):
        return
    base = np.asarray(position_xyz, dtype=np.float64).copy()
    top = base.copy()
    base[2] += 0.18
    top[2] += 1.35
    geom = scene.geoms[scene.ngeom]
    _initialise_visual_geom(
        geom,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        rgba=(0.05, 0.95, 0.87, 0.94),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        0.10,
        base,
        top,
    )
    scene.ngeom += 1
    sphere = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        sphere,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((0.24, 0.24, 0.24), dtype=np.float64),
        top,
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray((0.05, 0.95, 0.87, 0.98), dtype=np.float32),
    )
    sphere.emission = 0.35
    scene.ngeom += 1


def draw_minimap(
    image: Image.Image,
    *,
    map_base: Image.Image,
    trail: list[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    position: np.ndarray,
    half_extent: float,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    origin = (FRAME_WIDTH - MAP_SIZE - 18, 18)
    draw.rounded_rectangle(
        (
            origin[0] - 7,
            origin[1] - 7,
            origin[0] + MAP_SIZE + 7,
            origin[1] + MAP_SIZE + 25,
        ),
        radius=8,
        fill=(250, 252, 251, 232),
        outline=(24, 32, 39, 150),
        width=1,
    )
    image.paste(map_base, origin)
    draw = ImageDraw.Draw(image, "RGBA")
    if len(trail) > 1:
        trail_points = [
            map_point(
                float(point[0]),
                float(point[1]),
                half_extent=half_extent,
                origin=origin,
                size=MAP_SIZE,
            )
            for point in trail
        ]
        draw.line(trail_points, fill=(255, 255, 255, 242), width=3)
    start_marker = map_point(
        float(start[0]),
        float(start[1]),
        half_extent=half_extent,
        origin=origin,
        size=MAP_SIZE,
    )
    goal_marker = map_point(
        float(goal[0]),
        float(goal[1]),
        half_extent=half_extent,
        origin=origin,
        size=MAP_SIZE,
    )
    current_marker = map_point(
        float(position[0]),
        float(position[1]),
        half_extent=half_extent,
        origin=origin,
        size=MAP_SIZE,
    )
    draw.ellipse(
        (
            start_marker[0] - 4,
            start_marker[1] - 4,
            start_marker[0] + 4,
            start_marker[1] + 4,
        ),
        fill=TEAL,
        outline=WHITE,
        width=1,
    )
    draw.regular_polygon(
        (goal_marker[0], goal_marker[1], 7),
        n_sides=5,
        rotation=18,
        fill=AMBER,
        outline=WHITE,
    )
    draw.ellipse(
        (
            current_marker[0] - 6,
            current_marker[1] - 6,
            current_marker[0] + 6,
            current_marker[1] + 6,
        ),
        fill=BLUE,
        outline=WHITE,
        width=2,
    )
    draw.text(
        (origin[0] + 5, origin[1] + MAP_SIZE + 5),
        "PLAN VIEW",
        font=font(9, bold=True),
        fill=INK,
    )


def compose_dual_view(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    map_base: Image.Image,
    trail_xy: list[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    position: np.ndarray,
    half_extent: float,
    physical_time: float,
    requested_seconds: float,
    distance: float,
    best_progress: float,
    torso_tilt_degrees: float,
    support_count: int,
    maximum_contact_speed: float,
    slip_threshold: float,
    current_airborne: bool,
    ever_airborne: bool,
    ever_contact_speed_exceeded: bool,
    unhealthy_termination: bool,
    spatial_success: bool,
    evaluation_seed: int,
    evaluation_group_index: int | None,
    evaluation_group_count: int,
    checkpoint_name: str,
    commanded_speed: float,
    yaw_gain: float,
    maximum_curvature: float,
    floor_friction: np.ndarray,
    floor_condim: int,
    map_hash: str,
    time_limit_reached: bool = False,
) -> Image.Image:
    """Compose two rendered panes and a white provenance/status panel."""
    if left_rgb.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("left render has unexpected geometry")
    if right_rgb.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("right render has unexpected geometry")
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), BACKGROUND)
    image.paste(Image.fromarray(left_rgb, mode="RGB"), (0, 0))
    image.paste(Image.fromarray(right_rgb, mode="RGB"), (VIEW_WIDTH, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line((VIEW_WIDTH, 0, VIEW_WIDTH, VIEW_HEIGHT), fill=(255, 255, 255, 180), width=2)
    # Pane labels are drawn directly over the scene.  Deliberately avoid the
    # former large white title card so the left gait view remains unobstructed.
    draw.text(
        (16, 14),
        "FOLLOW CAMERA",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 170),
    )
    draw.text(
        (VIEW_WIDTH + 16, 14),
        "GOAL-TO-START OVERVIEW",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 170),
    )
    draw_minimap(
        image,
        map_base=map_base,
        trail=trail_xy,
        start=start,
        goal=goal,
        position=position,
        half_extent=half_extent,
    )

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, VIEW_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT), fill=(250, 249, 245, 255))
    draw.line((0, VIEW_HEIGHT, FRAME_WIDTH, VIEW_HEIGHT), fill=(77, 88, 96, 210), width=2)
    group_text = (
        f"paired evaluation {evaluation_group_index}/{evaluation_group_count}"
        if evaluation_group_index is not None
        else f"paired evaluation ?/{evaluation_group_count}"
    )
    draw.text(
        (20, VIEW_HEIGHT + 13),
        f"{group_text}  |  seed {evaluation_seed}  |  t {physical_time:06.2f}/{requested_seconds:.2f} s",
        font=font(15, bold=True),
        fill=INK,
    )
    draw.text(
        (20, VIEW_HEIGHT + 43),
        "Planner: direct-to-goal baseline (no global route)",
        font=font(12, bold=True),
        fill=RED,
    )
    draw.text(
        (20, VIEW_HEIGHT + 69),
        f"Checkpoint: {checkpoint_name}",
        font=font(11),
        fill=INK,
    )
    draw.text(
        (20, VIEW_HEIGHT + 94),
        f"Controller: speed {commanded_speed:.2f} m/s | yaw gain {yaw_gain:.2f} s^-1 | |curvature| <= {maximum_curvature:.2f} m^-1",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (20, VIEW_HEIGHT + 119),
        f"Fixed contact: friction [{floor_friction[0]:.1f}, {floor_friction[1]:.1f}, {floor_friction[2]:.1f}] | condim {floor_condim} | map {map_hash[:12]}",
        font=font(10),
        fill=INK,
    )

    middle_x = 650
    draw.text(
        (middle_x, VIEW_HEIGHT + 13),
        f"Distance {distance:6.2f} m  |  best progress {best_progress:6.2f} m",
        font=font(13, bold=True),
        fill=INK,
    )
    draw.text(
        (middle_x, VIEW_HEIGHT + 43),
        f"Position ({position[0]:+.2f}, {position[1]:+.2f}) m  |  torso tilt {torso_tilt_degrees:4.1f} deg",
        font=font(11),
        fill=INK,
    )
    draw.text(
        (middle_x, VIEW_HEIGHT + 69),
        f"Current support {support_count}/4  |  airborne {int(current_airborne)}  |  contact max {maximum_contact_speed:.2f} m/s",
        font=font(11),
        fill=INK,
    )
    draw.text(
        (middle_x, VIEW_HEIGHT + 94),
        f"Episode flags: any airborne {int(ever_airborne)} | any contact speed > {slip_threshold:.2f} m/s: {int(ever_contact_speed_exceeded)}",
        font=font(10),
        fill=INK,
    )

    safety_qualified = not (
        ever_airborne or ever_contact_speed_exceeded or unhealthy_termination
    )
    if spatial_success and safety_qualified:
        status = "SPATIAL ARRIVAL / SAFETY PASS"
        status_colour = TEAL
    elif spatial_success:
        status = "SPATIAL ARRIVAL / SAFETY FAIL"
        status_colour = RED
    elif unhealthy_termination:
        status = "TERMINATED / FALL"
        status_colour = RED
    elif time_limit_reached:
        if safety_qualified:
            status = "TIME LIMIT / NO ARRIVAL"
            status_colour = AMBER
        else:
            status = "TIME LIMIT / SAFETY FAIL"
            status_colour = RED
    elif current_airborne:
        status = "FOUR-FOOT AIRBORNE"
        status_colour = RED
    elif maximum_contact_speed > slip_threshold:
        status = "CONTACT-SPEED EXCEEDANCE"
        status_colour = AMBER
    else:
        status = "ROLLOUT IN PROGRESS"
        status_colour = TEAL
    draw.rounded_rectangle(
        (middle_x, VIEW_HEIGHT + 119, FRAME_WIDTH - 18, FRAME_HEIGHT - 12),
        radius=7,
        fill=(*status_colour, 238),
    )
    draw.text(
        (middle_x + 14, VIEW_HEIGHT + 126),
        status,
        font=font(12, bold=True),
        fill=WHITE,
    )
    return image


def render_pair(
    renderer: mujoco.Renderer,
    *,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption,
    follow_camera: mujoco.MjvCamera,
    fixed_overview_camera: mujoco.MjvCamera,
    trail_xyz: list[np.ndarray],
    overview_position_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=follow_camera, scene_option=scene_option)
    add_surface_trail(renderer.scene, trail_xyz)
    left = np.asarray(renderer.render(), dtype=np.uint8).copy()
    renderer.update_scene(
        data,
        camera=fixed_overview_camera,
        scene_option=scene_option,
    )
    add_surface_trail(renderer.scene, trail_xyz)
    add_overview_position_marker(renderer.scene, overview_position_xyz)
    right = np.asarray(renderer.render(), dtype=np.uint8).copy()
    return left, right


def main() -> None:
    args = parse_args()
    if args.fps != FPS:
        raise ValueError("fps must equal the 20 Hz environment control rate")
    if args.render_stride <= 0:
        raise ValueError("render-stride must be positive")
    if args.intro_seconds < 0.0 or args.outro_seconds < 0.0:
        raise ValueError("intro and outro durations cannot be negative")

    run_root = args.run_root.expanduser().resolve()
    contract_path = args.evaluation_contract.expanduser().resolve()
    config_path = run_root / "frozen_run_config.json"
    execution_path = run_root / "execution_record.json"
    scene_path = run_root / "task_scenes" / "spawn_0_0.000.xml"
    for path in (contract_path, config_path, execution_path, scene_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    condition, commanded_speed = audited_contract_controller(contract)
    yaw_gain = float(condition["yaw_gain_per_second"])
    maximum_curvature = float(condition["maximum_abs_curvature_per_m"])
    config["task_adapter"]["yaw_gain_per_second"] = yaw_gain
    config["task_adapter"]["maximum_abs_curvature_per_m"] = maximum_curvature
    for key in (
        "yaw_deadband_degrees",
        "slow_radius_m",
        "curvature_speed_reduction_gain",
        "minimum_turn_speed_fraction",
    ):
        if key in condition:
            config["task_adapter"][key] = condition[key]
    if "independent_success" in contract:
        success = contract["independent_success"]
        config["task_adapter"]["arrival_radius_m"] = success["arrival_radius_m"]
        config["task_adapter"]["hold_radius_m"] = success["hold_radius_m"]
        config["task_adapter"]["hold_seconds"] = success["hold_seconds"]
    reward_configuration = config["base_policy"].get(
        "configuration",
        config["base_policy"].get("reward_configuration"),
    )
    if not reward_configuration:
        raise ValueError("Run configuration does not identify its reward configuration")
    v22_config_path = ROOT / reward_configuration
    v22_config = json.loads(v22_config_path.read_text(encoding="utf-8"))
    if args.airborne_shaping_weight is not None:
        if not math.isfinite(args.airborne_shaping_weight) or args.airborne_shaping_weight < 0.0:
            raise ValueError("airborne-shaping-weight must be finite and non-negative")
        v22_config["preserved_pre_pitch_reward"]["airborne_shaping_weight"] = float(
            args.airborne_shaping_weight
        )

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
    formal_evidence = formal_paired_episode_evidence(
        contract=contract,
        model_hash=sha256(model_path),
        evaluation_seed=args.evaluation_seed,
    )

    evaluation_seeds = [int(seed) for seed in contract["evaluation_seeds"]]
    group_index = (
        evaluation_seeds.index(args.evaluation_seed) + 1
        if args.evaluation_seed in evaluation_seeds
        else None
    )
    physical_seconds = (
        float(args.physical_seconds)
        if args.physical_seconds is not None
        else int(contract["horizon_steps"]) / FPS
    )
    requested_steps = round(physical_seconds * FPS)
    if not math.isclose(requested_steps / FPS, physical_seconds, abs_tol=1e-9):
        raise ValueError("physical-seconds must be divisible by 0.05 s")
    minimum_encoded_frames = (
        round(args.intro_seconds * args.fps)
        + math.ceil(requested_steps / args.render_stride)
        + round(args.outro_seconds * args.fps)
    )
    if minimum_encoded_frames / args.fps < 10.0:
        raise ValueError("requested settings can produce a video shorter than 10 seconds")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / contract["output_root"]
        / "videos"
        / f"seed_{args.evaluation_seed}_dual_view_v1"
    )
    stem = f"fixed_map_final_policy_seed_{args.evaluation_seed}_dual_view_v1"
    video_path = output_dir / f"{stem}.mp4"
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
    map_base = make_map_base(heights, size=MAP_SIZE)

    env = make_task_env(
        config,
        v22_config,
        xml_path=scene_path,
        seed=args.evaluation_seed,
        spawn_fraction=0.0,
        max_episode_steps=requested_steps,
        cruise_speed=commanded_speed,
        terminate_on_success=True,
    )
    observation, _ = env.reset(seed=args.evaluation_seed)
    actual_dt = float(env.unwrapped.dt)
    if not math.isclose(actual_dt, 1.0 / FPS, abs_tol=1e-12):
        env.close()
        raise ValueError(f"Expected dt={1.0 / FPS}; observed {actual_dt}")
    policy = PPO.load(model_path, device="cpu")
    if tuple(observation.shape) != tuple(policy.observation_space.shape):
        env.close()
        raise RuntimeError("Policy and environment observation spaces do not match")

    compiled = env.unwrapped.model
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_friction = np.asarray(compiled.geom_friction[floor_id], dtype=np.float64)
    floor_condim = int(compiled.geom_condim[floor_id])
    if not np.array_equal(
        floor_friction, np.asarray(approved["fixed_friction"], dtype=np.float64)
    ):
        env.close()
        raise RuntimeError("Compiled floor friction differs from frozen configuration")
    if floor_condim != int(approved["condim"]):
        env.close()
        raise RuntimeError("Compiled floor condim differs from frozen configuration")

    compiled.vis.global_.offwidth = VIEW_WIDTH
    compiled.vis.global_.offheight = VIEW_HEIGHT
    renderer = mujoco.Renderer(compiled, height=VIEW_HEIGHT, width=VIEW_WIDTH)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[2] = 0
    fixed_overview_camera = overview_camera(
        start=start,
        goal=goal,
        half_extent=half_extent,
        terrain_midpoint_height=float((heights.min() + heights.max()) / 2.0),
    )

    av_module = load_video_encoder()
    container = av_module.open(
        str(video_path), mode="w", options={"movflags": "+faststart"}
    )
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = FRAME_WIDTH
    stream.height = FRAME_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}
    stream.gop_size = args.fps * 2

    intro_frames = round(args.intro_seconds * args.fps)
    outro_frames = round(args.outro_seconds * args.fps)
    initial_position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
    initial_height = float(
        env._terrain_height(float(initial_position[0]), float(initial_position[1]))
    )
    initial_distance = float(np.linalg.norm(goal - initial_position))
    initial_valley_depth = local_valley_depth(env, initial_position)
    trail_xy: list[np.ndarray] = [initial_position.copy()]
    trail_xyz: list[np.ndarray] = [
        np.asarray((initial_position[0], initial_position[1], initial_height + 0.07))
    ]
    records: list[dict[str, Any]] = []
    keyframes: list[tuple[str, Image.Image]] = []
    best_progress = 0.0
    ever_airborne = False
    ever_contact_speed_exceeded = False
    unhealthy_termination = False
    last_frame: Image.Image | None = None

    initial_follow = make_camera(
        position=initial_position,
        goal=goal,
        terrain_height=initial_height,
        local_valley_depth=initial_valley_depth,
        progress=0.0,
    )
    left, right = render_pair(
        renderer,
        data=env.unwrapped.data,
        scene_option=scene_option,
        follow_camera=initial_follow,
        fixed_overview_camera=fixed_overview_camera,
        trail_xyz=trail_xyz,
        overview_position_xyz=trail_xyz[-1],
    )
    initial_frame = compose_dual_view(
        left,
        right,
        map_base=map_base,
        trail_xy=trail_xy,
        start=start,
        goal=goal,
        position=initial_position,
        half_extent=half_extent,
        physical_time=0.0,
        requested_seconds=physical_seconds,
        distance=initial_distance,
        best_progress=0.0,
        torso_tilt_degrees=math.degrees(
            quaternion_tilt_angle(np.asarray(env.unwrapped.data.qpos[3:7]))
        ),
        support_count=0,
        maximum_contact_speed=0.0,
        slip_threshold=float(config["task_adapter"]["slip_speed_threshold_m_per_s"]),
        current_airborne=False,
        ever_airborne=False,
        ever_contact_speed_exceeded=False,
        unhealthy_termination=False,
        spatial_success=False,
        evaluation_seed=args.evaluation_seed,
        evaluation_group_index=group_index,
        evaluation_group_count=len(evaluation_seeds),
            checkpoint_name=args.condition_label or model_path.name,
        commanded_speed=commanded_speed,
        yaw_gain=yaw_gain,
        maximum_curvature=maximum_curvature,
        floor_friction=floor_friction,
        floor_condim=floor_condim,
        map_hash=approved["heights_sha256"],
    )
    for _ in range(intro_frames):
        encode_frame(stream, container, initial_frame)
    last_frame = initial_frame
    keyframes.append(("Frozen start state", initial_frame.copy()))

    termination_reason = "requested_video_horizon"
    completed_steps = 0
    rendered_rollout_frames = 0
    for step in range(1, requested_steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        completed_steps = step
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        position = qpos[:2].copy()
        terrain_height = float(env._terrain_height(float(position[0]), float(position[1])))
        trail_xy.append(position.copy())
        trail_xyz.append(
            np.asarray((position[0], position[1], terrain_height + 0.07), dtype=np.float64)
        )
        distance = float(np.linalg.norm(goal - position))
        best_progress = max(best_progress, initial_distance - distance)
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
        current_airborne = bool(contact_mask.shape == (4,) and not np.any(contact_mask))
        contact_speed_exceeded = maximum_contact_speed > float(
            config["task_adapter"]["slip_speed_threshold_m_per_s"]
        )
        ever_airborne = ever_airborne or current_airborne
        ever_contact_speed_exceeded = (
            ever_contact_speed_exceeded or contact_speed_exceeded
        )
        spatial_success = bool(info.get("proxygap_fixed_goal_success", False))
        unhealthy_termination = unhealthy_termination or bool(terminated and not spatial_success)
        torso_tilt_degrees = math.degrees(quaternion_tilt_angle(qpos[3:7]))
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
                "torso_tilt_degrees": torso_tilt_degrees,
                "support_count": int(contact_mask.sum()) if contact_mask.shape == (4,) else 0,
                "maximum_contact_tangential_speed_m_per_s": maximum_contact_speed,
                "camera_local_valley_depth_m": valley_depth,
                "contact_speed_threshold_exceeded": contact_speed_exceeded,
                "airborne": current_airborne,
                "episode_ever_airborne": ever_airborne,
                "episode_ever_contact_speed_exceeded": ever_contact_speed_exceeded,
                "reward": float(reward),
                "spatial_success": spatial_success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        should_render = (
            step == 1
            or step % args.render_stride == 0
            or terminated
            or truncated
        )
        if should_render:
            follow = make_camera(
                position=position,
                goal=goal,
                terrain_height=terrain_height,
                local_valley_depth=valley_depth,
                progress=step / requested_steps,
            )
            left, right = render_pair(
                renderer,
                data=env.unwrapped.data,
                scene_option=scene_option,
                follow_camera=follow,
                fixed_overview_camera=fixed_overview_camera,
                trail_xyz=trail_xyz,
                overview_position_xyz=trail_xyz[-1],
            )
            frame = compose_dual_view(
                left,
                right,
                map_base=map_base,
                trail_xy=trail_xy,
                start=start,
                goal=goal,
                position=position,
                half_extent=half_extent,
                physical_time=step * actual_dt,
                requested_seconds=physical_seconds,
                distance=distance,
                best_progress=best_progress,
                torso_tilt_degrees=torso_tilt_degrees,
                support_count=int(contact_mask.sum()) if contact_mask.shape == (4,) else 0,
                maximum_contact_speed=maximum_contact_speed,
                slip_threshold=float(
                    config["task_adapter"]["slip_speed_threshold_m_per_s"]
                ),
                current_airborne=current_airborne,
                ever_airborne=ever_airborne,
                ever_contact_speed_exceeded=ever_contact_speed_exceeded,
                unhealthy_termination=unhealthy_termination,
                spatial_success=spatial_success,
                evaluation_seed=args.evaluation_seed,
                evaluation_group_index=group_index,
                evaluation_group_count=len(evaluation_seeds),
                checkpoint_name=args.condition_label or model_path.name,
                commanded_speed=commanded_speed,
                yaw_gain=yaw_gain,
                maximum_curvature=maximum_curvature,
                floor_friction=floor_friction,
                floor_condim=floor_condim,
                map_hash=approved["heights_sha256"],
                time_limit_reached=bool(step >= requested_steps or truncated),
            )
            encode_frame(stream, container, frame)
            rendered_rollout_frames += 1
            last_frame = frame
            if step in {
                requested_steps // 3,
                2 * requested_steps // 3,
                requested_steps,
            } or terminated or truncated:
                keyframes.append((f"Physical rollout t={step * actual_dt:.1f} s", frame.copy()))
        if terminated or truncated:
            termination_reason = (
                "spatial_goal_contract" if spatial_success else "terminated"
            ) if terminated else "time_limit"
            break

    if last_frame is None:
        renderer.close()
        env.close()
        container.close()
        raise RuntimeError("No frame was produced")
    for _ in range(outro_frames):
        encode_frame(stream, container, last_frame)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    episode_summary = env.episode_summary()
    renderer.close()
    env.close()
    trace_path = output_dir / f"{stem}_trace.csv"
    write_rows(trace_path, records)
    contact_sheet_path = output_dir / f"{stem}_contact_sheet.png"
    make_contact_sheet(keyframes, contact_sheet_path)
    final_frame_path = output_dir / f"{stem}_final_frame.png"
    last_frame.save(final_frame_path, format="PNG", optimize=True)

    qa = validate_video(
        video_path,
        expected_width=FRAME_WIDTH,
        expected_height=FRAME_HEIGHT,
    )
    total_frames = intro_frames + rendered_rollout_frames + outro_frames
    if qa["decoded_frames"] != total_frames:
        raise RuntimeError(
            f"Decoded frame count {qa['decoded_frames']} differs from encoded count {total_frames}"
        )
    manifest = {
        "schema_version": "proxygap-fixed-goal-dual-view-video-v1",
        "purpose": "dual-view visualisation of an existing deterministic fixed-map rollout",
        "training_status": "no new training; both panes visualise the same replayed policy trajectory",
        "selection_rule": contract.get("representative_video", {}).get(
            "selection_rule",
            f"predeclared evaluation seed {args.evaluation_seed}; "
            "no best-looking video selection",
        ),
        "planner": "direct-to-goal baseline (no global route)",
        "camera_rules": {
            "left": (
                "robot-following free camera; elevation adapts from -24 to -72 degrees "
                "using deterministic 5 m local valley relief"
            ),
            "right": (
                "fixed oblique overview at elevation -55 degrees, camera head on the "
                "goal side, looking across the map towards the start; distance equals "
                "3.3 times the 40 m map half-extent"
            ),
        },
        "visual_only_geometry": {
            "surface_trail": (
                "orange four-pixel line segments arc-length resampled to at most 1600 segments "
                "and offset 0.07 m above recorded terrain height; actual trajectory only"
            ),
            "overview_position_marker": (
                "cyan 1.17 m mast and 0.24 m sphere above the current surface point; "
                "overview pane only"
            ),
            "dynamics_boundary": (
                "added after renderer.update_scene as MjvScene decorations; absent from "
                "MjModel and therefore cannot change dynamics, policy observations or rewards"
            ),
        },
        "success_semantics": {
            "goal_success": (
                "spatial contract only: first enter 1.5 m then remain within 2.0 m for 2 s"
            ),
            "safety_qualification": (
                "reported separately; any four-foot airborne or contact-speed exceedance "
                "makes this visual diagnostic fail its conservative safety display"
            ),
            "spatial_success": bool(episode_summary.get("fixed_goal_success", False)),
            "qualified_safety": bool(
                episode_summary.get("fixed_goal_qualified_no_fall_no_airborne_no_slip", False)
            ),
        },
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": args.fps,
            "frames": total_frames,
            "duration_seconds": total_frames / args.fps,
            "physical_rollout_seconds": completed_steps * actual_dt,
            "render_stride_physics_steps": args.render_stride,
            "playback_speed_factor": args.render_stride,
            "layout": {
                "left_follow_view_pixels": [0, 0, VIEW_WIDTH, VIEW_HEIGHT],
                "right_overview_pixels": [VIEW_WIDTH, 0, VIEW_WIDTH, VIEW_HEIGHT],
                "bottom_parameter_panel_pixels": [0, VIEW_HEIGHT, FRAME_WIDTH, PANEL_HEIGHT],
            },
        },
        "rollout": {
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "condition_label": args.condition_label,
            "airborne_shaping_weight": float(
                v22_config["preserved_pre_pitch_reward"]["airborne_shaping_weight"]
            ),
            "configuration": str(config_path),
            "configuration_sha256": sha256(config_path),
            "corrected_evaluation_contract": str(contract_path),
            "corrected_evaluation_contract_sha256": sha256(contract_path),
            "scene": str(scene_path),
            "scene_sha256": sha256(scene_path),
            "height_array": str(heights_path),
            "height_array_sha256": sha256(heights_path),
            "evaluation_seed": args.evaluation_seed,
            "paired_evaluation_group": [group_index, len(evaluation_seeds)],
            "deterministic_policy": True,
            "commanded_speed_m_per_s": commanded_speed,
            "yaw_gain_per_second": yaw_gain,
            "maximum_abs_curvature_per_m": maximum_curvature,
            "completed_steps": completed_steps,
            "termination_reason": termination_reason,
            "initial_distance_m": initial_distance,
            "best_progress_m": best_progress,
            "episode_summary_at_video_end": json_safe(episode_summary),
        },
        "formal_paired_evaluation_evidence": formal_evidence,
        "terrain_contact": {
            "friction": floor_friction.tolist(),
            "condim": floor_condim,
        },
        "diagnostic_boundary": (
            "Contact-speed threshold exceedance can include landing impact or brief foot "
            "adjustment and is not by itself proof of sustained physical sliding."
        ),
        "files": {
            "trace_csv": str(trace_path),
            "trace_csv_sha256": sha256(trace_path),
            "contact_sheet": str(contact_sheet_path),
            "contact_sheet_sha256": sha256(contact_sheet_path),
            "final_frame": str(final_frame_path),
            "final_frame_sha256": sha256(final_frame_path),
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
            "both_views_share_one_deterministic_rollout": True,
        },
    }
    manifest_path = output_dir / f"{stem}_video_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
