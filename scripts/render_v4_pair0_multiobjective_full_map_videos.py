"""Render three exact formal V4+PAIR0 multi-objective full-map replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_post_seal_full_map_v1 as full_map  # noqa: E402
import render_fixed_goal_dual_view_video as dual  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402
from render_fixed_goal_training_video import (  # noqa: E402
    INK,
    TEAL,
    WHITE,
    encode_frame,
    font,
    local_valley_depth,
    make_camera,
    make_contact_sheet,
    make_map_base,
    validate_video,
)

DEFAULT_CONFIG = ROOT / "configs/v4_pair0_multiobjective_full_map_video_v1_20260820.json"
FORMAL_CONFIG = ROOT / "configs/v4_pair0_multiobjective_full_map_final_v1_20260820.json"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} changed: {actual!r} != {expected!r}")


def validate_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require_equal(
        config.get("schema_version"),
        "proxygap-v4-pair0-multiobjective-full-map-video-v1",
        "schema",
    )
    require_equal(config.get("status"), "frozen_read_only_three_objective_formal_replay_video", "status")
    evidence_root = ROOT / config["formal_evidence"]["root"]
    for name, key in (("manifest.json", "manifest_sha256"), ("summary.json", "summary_sha256"), ("frozen_config.json", "frozen_config_sha256")):
        require_equal(sha256(evidence_root / name), config["formal_evidence"][key], f"formal {name}")
    require_equal(sha256(FORMAL_CONFIG), config["formal_evidence"]["frozen_config_sha256"], "live formal config")
    source = config["source"]
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("pair0_scene", "pair0_scene_sha256"),
        ("canonical_evaluation_config", "canonical_evaluation_config_sha256"),
        ("fixed_map_config", "fixed_map_config_sha256"),
        ("heights", "heights_sha256"),
    ):
        require_equal(sha256(ROOT / source[path_key]), source[hash_key], path_key)
    visual = config["visualisation"]
    require_equal(sha256(ROOT / visual["overview_profile"]), visual["overview_profile_sha256"], "overview profile")
    require_equal([visual["width"], visual["height"], visual["fps"]], [1280, 720, 20], "video dimensions")
    require_equal(int(visual["render_stride_control_steps"]), 20, "render stride")
    formal_summary = json.loads((evidence_root / "summary.json").read_text(encoding="utf-8"))
    require_equal(bool(formal_summary["all_objectives_passed"]), True, "formal all-objective gate")
    require_equal(len(config["episodes"]), 3, "episode count")
    require_equal([item["objective_id"] for item in config["episodes"]], ["time_priority", "balanced", "energy_priority"], "objective order")
    for episode in config["episodes"]:
        for path_key, hash_key in (("control_trace", "control_trace_sha256"), ("substep_trace", "substep_trace_sha256"), ("result", "result_sha256")):
            require_equal(sha256(evidence_root / episode[path_key]), episode[hash_key], f"{episode['objective_id']} {path_key}")
        result = json.loads((evidence_root / episode["result"]).read_text(encoding="utf-8"))
        require_equal(int(result["evaluation_seed"]), int(episode["evaluation_seed"]), "episode seed")
        require_equal(result["regime"], episode["contract_id"], "episode contract")
        require_equal(bool(result["safety_qualified_completion"]), True, "episode completion")
        require_equal(int(result["duration_corrected_slip_event_count"]), 0, "episode sustained slip")
    execution = config["execution"]
    require_equal(bool(execution["training_permitted"]), False, "training boundary")
    require_equal(bool(execution["checkpoint_write_permitted"]), False, "checkpoint boundary")
    require_equal(bool(execution["formal_evidence_mutation_permitted"]), False, "formal evidence boundary")
    return config


def route_points(formal_config: dict[str, Any], contract_id: str) -> np.ndarray:
    path = ROOT / formal_config["route_contracts"][contract_id]["route"]
    rows = read_csv(path)
    return np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float64)


def map_with_planned_route(heights: np.ndarray, points: np.ndarray, half_extent: float) -> Image.Image:
    image = make_map_base(heights, size=dual.MAP_SIZE)
    draw = ImageDraw.Draw(image, "RGBA")
    size = image.width
    pixels = [
        (
            int(round((float(point[0]) + half_extent) / (2.0 * half_extent) * (size - 1))),
            int(round((half_extent - float(point[1])) / (2.0 * half_extent) * (size - 1))),
        )
        for point in points
    ]
    if len(pixels) >= 2:
        draw.line(pixels, fill=(70, 220, 230, 245), width=3, joint="curve")
    return image


def compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    episode: dict[str, Any],
    result: dict[str, Any],
    map_base: Image.Image,
    trail_xy: list[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    position: np.ndarray,
    half_extent: float,
    physical_time: float,
    distance: float,
    best_progress: float,
    route_progress: float,
    cross_track: float,
    current_support: int,
    mean_support: float,
    current_full_zero: bool,
    cumulative_full_zero: int,
    completed_steps: int,
    tilt_degrees: float,
    maximum_contact_speed: float,
    floor_friction: np.ndarray,
    floor_condim: int,
    map_hash: str,
    overview_profile_id: str,
    terrain_min_height_m: float,
    terrain_max_height_m: float,
    final: bool,
) -> Image.Image:
    image = dual.compose_dual_view(
        left,
        right,
        map_base=map_base,
        trail_xy=trail_xy,
        start=start,
        goal=goal,
        position=position,
        half_extent=half_extent,
        physical_time=physical_time,
        requested_seconds=float(result["elapsed_seconds"]),
        distance=distance,
        best_progress=best_progress,
        torso_tilt_degrees=tilt_degrees,
        support_count=current_support,
        maximum_contact_speed=maximum_contact_speed,
        slip_threshold=0.2,
        current_airborne=current_full_zero,
        ever_airborne=cumulative_full_zero > 0,
        ever_contact_speed_exceeded=False,
        unhealthy_termination=False,
        spatial_success=final,
        evaluation_seed=int(episode["evaluation_seed"]),
        evaluation_group_index=None,
        evaluation_group_count=3,
        checkpoint_name="V4 checkpoint_1024000.zip + PAIR0",
        commanded_speed=0.5,
        yaw_gain=0.75,
        maximum_curvature=0.2,
        floor_friction=floor_friction,
        floor_condim=floor_condim,
        map_hash=map_hash,
        time_limit_reached=False,
        overview_profile_id=overview_profile_id,
        terrain_min_height_m=terrain_min_height_m,
        terrain_max_height_m=terrain_max_height_m,
        overview_vertical_scale=1.0,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, dual.VIEW_HEIGHT, dual.FRAME_WIDTH, dual.FRAME_HEIGHT), fill=(250, 249, 245, 255))
    draw.line((0, dual.VIEW_HEIGHT, dual.FRAME_WIDTH, dual.VIEW_HEIGHT), fill=(77, 88, 96, 220), width=2)
    weights = episode["weights_time_energy"]
    zero_fraction = cumulative_full_zero / max(1, completed_steps)
    draw.text((20, dual.VIEW_HEIGHT + 12), f"{episode['display_name']}  |  weights time {weights[0]:.1f} / energy {weights[1]:.1f}", font=font(15, bold=True), fill=INK)
    draw.text((20, dual.VIEW_HEIGHT + 41), f"Formal seed {episode['evaluation_seed']}  |  candidate-bank route: {episode['contract_id']}", font=font(11, bold=True), fill=(18, 92, 106))
    draw.text((20, dual.VIEW_HEIGHT + 67), f"t {physical_time:6.2f}/{result['elapsed_seconds']:.2f} s  |  route progress {route_progress:6.2f} m  |  cross-track {cross_track:5.2f} m", font=font(10), fill=INK)
    draw.text((20, dual.VIEW_HEIGHT + 93), "Mini-map: cyan planned route | orange actual trajectory", font=font(10, bold=True), fill=(42, 108, 116))
    draw.text((20, dual.VIEW_HEIGHT + 119), "Known frozen map | 20x playback | mechanical work is a proxy, not battery energy", font=font(9), fill=(70, 70, 70))

    x0 = 648
    draw.text((x0, dual.VIEW_HEIGHT + 12), f"Distance {distance:6.2f} m  |  best progress {best_progress:6.2f} m  |  tilt {tilt_degrees:4.1f} deg", font=font(12, bold=True), fill=INK)
    draw.text((x0, dual.VIEW_HEIGHT + 39), f"Support now {current_support}/4 | mean {mean_support:.3f}/4 | full-zero {cumulative_full_zero}/{completed_steps} ({100.0 * zero_fraction:.2f}%)", font=font(10), fill=INK)
    draw.text((x0, dual.VIEW_HEIGHT + 65), f"Final episode: time {result['elapsed_seconds']:.2f} s | positive work {result['actuator_positive_mechanical_work_total_j']:.1f} J proxy | path {result['path_length_m']:.2f} m", font=font(9), fill=INK)
    draw.text((x0, dual.VIEW_HEIGHT + 88), "Safety contract: fall 0 | torso ground 0 | sustained non-foot 0 | sustained slip events 0", font=font(9, bold=True), fill=(24, 104, 84))
    colour = TEAL if final else (49, 112, 122)
    status = "ARRIVED + 2 s HOLD | SAFETY PASS" if final else "FORMAL EXACT REPLAY IN PROGRESS"
    draw.rounded_rectangle((x0, dual.VIEW_HEIGHT + 111, dual.FRAME_WIDTH - 15, dual.FRAME_HEIGHT - 10), radius=7, fill=(*colour, 245))
    draw.text((x0 + 14, dual.VIEW_HEIGHT + 120), status, font=font(12, bold=True), fill=WHITE)
    return image


def compare_substeps(
    formal_rows: list[dict[str, str]],
    actual_rows: list[dict[str, Any]],
    step: int,
) -> int:
    mismatches = 0
    expected_rows = formal_rows[(step - 1) * 5 : step * 5]
    require_equal(len(expected_rows), 5, "formal substep slice")
    for expected, actual in zip(expected_rows, actual_rows, strict=True):
        for key in ("contact_mask", "normal_forces_n", "tangential_speeds_m_per_s"):
            if np.asarray(json.loads(expected[key])).tolist() != np.asarray(actual[key]).tolist():
                mismatches += 1
        for key in ("force_qualified_supported", "nonfoot_ground", "torso_ground"):
            if int(expected[key]) != int(actual[key]):
                mismatches += 1
    return mismatches


def render_episode(
    *,
    config: dict[str, Any],
    formal_config: dict[str, Any],
    canonical: dict[str, Any],
    fixed: dict[str, Any],
    episode: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    evidence_root = ROOT / config["formal_evidence"]["root"]
    result = json.loads((evidence_root / episode["result"]).read_text(encoding="utf-8"))
    control_rows = read_csv(evidence_root / episode["control_trace"])
    substep_rows = read_csv(evidence_root / episode["substep_trace"])
    require_equal(len(control_rows), int(result["control_steps"]), "control trace rows")
    require_equal(len(substep_rows), 5 * len(control_rows), "substep trace rows")
    source = config["source"]
    seed = int(episode["evaluation_seed"])
    condition = full_map._make_condition(canonical, fixed)
    condition["task_adapter"].update({"maximum_abs_curvature_per_m": 0.2, "yaw_gain_per_second": 0.75, "slow_radius_m": 4.0})
    policy_config = json.loads((ROOT / canonical["source"]["policy_configuration"]).read_text(encoding="utf-8"))
    env = full_map.fixed_task.make_task_env(
        condition,
        policy_config,
        xml_path=ROOT / source["pair0_scene"],
        seed=seed,
        spawn_fraction=0.0,
        max_episode_steps=12000,
        cruise_speed=0.5,
        terminate_on_success=False,
    )
    observation, _ = env.reset(seed=seed)
    require_equal(tuple(observation.shape), (135,), "replay observation")
    audit_state = l2.install_five_substep_audit(env)
    slip_cfg = canonical["duration_corrected_slip"]
    slip_tracker = l2.DurationCorrectedSlipTracker(
        dt=0.01,
        speed_threshold=float(slip_cfg["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip_cfg["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip_cfg["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip_cfg["minimum_sustained_seconds"]),
    )
    required_hold = int(math.ceil(2.0 / float(env.unwrapped.dt)))
    arrival = full_map.direct.ArrivalDwellTracker(arrival_radius_m=1.5, hold_radius_m=2.0, required_hold_steps=required_hold)

    approved = fixed["approved_map"]
    heights = np.load(ROOT / source["heights"], allow_pickle=False)
    half_extent = float(approved["map_half_extent_m"])
    start = np.asarray(approved["start_xy_m"], dtype=np.float64)
    goal = np.asarray(approved["goal_xy_m"], dtype=np.float64)
    planned = route_points(formal_config, episode["contract_id"])
    map_base = map_with_planned_route(heights, planned, half_extent)
    overview_profile, _ = dual.load_overview_profile(ROOT / config["visualisation"]["overview_profile"])
    overview_camera = dual.overview_camera(
        start=start,
        goal=goal,
        half_extent=half_extent,
        terrain_midpoint_height=float((heights.min() + heights.max()) / 2.0),
        profile=overview_profile,
    )
    compiled = env.unwrapped.model
    floor_id = int(mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    floor_friction = np.asarray(compiled.geom_friction[floor_id], dtype=np.float64)
    floor_condim = int(compiled.geom_condim[floor_id])
    require_equal(int(compiled.npair), 4, "PAIR0 count")
    require_equal(floor_friction.tolist(), [1.0, 0.5, 0.5], "floor friction")
    require_equal(floor_condim, 3, "floor condim")
    compiled.vis.global_.offwidth = dual.VIEW_WIDTH
    compiled.vis.global_.offheight = dual.VIEW_HEIGHT
    renderer = mujoco.Renderer(compiled, height=dual.VIEW_HEIGHT, width=dual.VIEW_WIDTH)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[2] = 0

    episode_root = output_root / episode["objective_id"]
    episode_root.mkdir(parents=True, exist_ok=False)
    stem = f"v4_pair0_{episode['objective_id']}_seed_{seed}_full_map_relief_v1"
    video_path = episode_root / f"{stem}.mp4"
    av_module = dual.load_video_encoder()
    container = av_module.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=20)
    stream.width = dual.FRAME_WIDTH
    stream.height = dual.FRAME_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(config["visualisation"]["crf"]), "preset": str(config["visualisation"]["preset"])}
    stream.gop_size = 40

    initial_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
    initial_distance = float(np.linalg.norm(goal - initial_xy))
    previous_xy = initial_xy.copy()
    minimum_distance = initial_distance
    path_length = 0.0
    support_sum = 0
    cumulative_full_zero = 0
    trail_xy = [initial_xy.copy()]
    initial_height = float(env._terrain_height(float(initial_xy[0]), float(initial_xy[1])))
    trail_xyz = [np.asarray((initial_xy[0], initial_xy[1], initial_height + 0.07), dtype=np.float64)]
    rendered_frames = 0
    state_mismatches = 0
    substep_mismatches = 0
    replay_rows: list[dict[str, Any]] = []
    keyframes: list[tuple[str, Image.Image]] = []
    last_frame: Image.Image | None = None

    def make_visual(step: int, row: dict[str, str] | None, support: int, mean_support: float, full_zero: bool, max_speed: float, final: bool) -> Image.Image:
        xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
        terrain_height = float(env._terrain_height(float(xy[0]), float(xy[1])))
        target = goal if row is None else np.asarray((float(row["target_x_m"]), float(row["target_y_m"])), dtype=np.float64)
        follow = make_camera(position=xy, goal=target, terrain_height=terrain_height, local_valley_depth=local_valley_depth(env, xy), progress=step / max(1, len(control_rows)))
        left, right = dual.render_pair(
            renderer,
            data=env.unwrapped.data,
            scene_option=scene_option,
            follow_camera=follow,
            fixed_overview_camera=overview_camera,
            trail_xyz=trail_xyz,
            overview_position_xyz=trail_xyz[-1],
            overview_profile=overview_profile,
        )
        distance = float(np.linalg.norm(goal - xy))
        tilt = math.degrees(full_map.direct.terrain_relative_tilt_rad(env, np.asarray(env.unwrapped.data.qpos, dtype=np.float64)))
        return compose_frame(
            left,
            right,
            episode=episode,
            result=result,
            map_base=map_base,
            trail_xy=trail_xy,
            start=start,
            goal=goal,
            position=xy,
            half_extent=half_extent,
            physical_time=step * 0.05,
            distance=distance,
            best_progress=initial_distance - minimum_distance,
            route_progress=0.0 if row is None else float(row["route_progress_m"]),
            cross_track=0.0 if row is None else float(row["route_cross_track_m"]),
            current_support=support,
            mean_support=mean_support,
            current_full_zero=full_zero,
            cumulative_full_zero=cumulative_full_zero,
            completed_steps=step,
            tilt_degrees=tilt,
            maximum_contact_speed=max_speed,
            floor_friction=floor_friction,
            floor_condim=floor_condim,
            map_hash=source["heights_sha256"],
            overview_profile_id=str(overview_profile["profile_id"]),
            terrain_min_height_m=float(heights.min()),
            terrain_max_height_m=float(heights.max()),
            final=final,
        )

    initial_frame = make_visual(0, None, 0, 0.0, False, 0.0, False)
    intro_frames = round(float(config["visualisation"]["intro_seconds"]) * 20)
    outro_frames = round(float(config["visualisation"]["outro_seconds"]) * 20)
    for _ in range(intro_frames):
        encode_frame(stream, container, initial_frame)
    keyframes.append((f"{episode['display_name']} start", initial_frame.copy()))
    last_frame = initial_frame

    for index, row in enumerate(control_rows, start=1):
        env.goal_xy = np.asarray((float(row["target_x_m"]), float(row["target_y_m"])), dtype=np.float64)
        env.set_task_speed(float(row["scheduled_speed_m_per_s"]))
        env.slow_radius = 4.0 if float(np.linalg.norm(env.goal_xy - planned[-1])) <= 1e-9 else env.arrival_radius
        action = np.asarray(json.loads(row["action"]), dtype=np.float64)
        observation, _, terminated, truncated, _ = env.step(action)
        substeps = audit_state.get("last")
        if not isinstance(substeps, list) or len(substeps) != 5:
            raise RuntimeError("Replay did not produce five physics substeps")
        actual_substeps: list[dict[str, Any]] = []
        interval_contacts = []
        maximum_contact_speed = 0.0
        for item in substeps:
            contacts = np.asarray(item["contacts"], dtype=bool)
            speeds = np.asarray(item["speeds"], dtype=np.float64)
            forces = np.asarray(item["forces"], dtype=np.float64)
            slip_tracker.update(contact_mask=contacts, tangential_speeds=speeds, normal_forces=forces)
            active = speeds[contacts]
            maximum_contact_speed = max(maximum_contact_speed, float(np.max(active)) if active.size else 0.0)
            interval_contacts.append(contacts.copy())
            actual_substeps.append({
                "contact_mask": contacts.astype(int).tolist(),
                "normal_forces_n": forces.tolist(),
                "tangential_speeds_m_per_s": speeds.tolist(),
                "force_qualified_supported": int(np.any(contacts & (forces >= 1.0))),
                "nonfoot_ground": int(bool(item["nonfoot"])),
                "torso_ground": int(bool(item["torso"])),
            })
        substep_mismatches += compare_substeps(substep_rows, actual_substeps, index)
        matrix = np.asarray(interval_contacts, dtype=bool)
        full_zero = not bool(np.any(matrix))
        cumulative_full_zero += int(full_zero)
        support_sum += int(np.sum(matrix))
        support = int(np.sum(matrix[-1]))
        xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        path_length += float(np.linalg.norm(xy - previous_xy))
        previous_xy = xy
        distance = float(np.linalg.norm(goal - xy))
        minimum_distance = min(minimum_distance, distance)
        arrival.update(step=index, distance_m=distance, stable=False)
        if float(row["x_m"]) != float(xy[0]) or float(row["y_m"]) != float(xy[1]):
            state_mismatches += 1
        if float(row["goal_distance_m"]) != distance or int(row["full_control_interval_zero_foot"]) != int(full_zero):
            state_mismatches += 1
        replay_rows.append({
            "control_step": index,
            "x_m": float(xy[0]),
            "y_m": float(xy[1]),
            "goal_distance_m": distance,
            "target_x_m": float(env.goal_xy[0]),
            "target_y_m": float(env.goal_xy[1]),
            "scheduled_speed_m_per_s": float(row["scheduled_speed_m_per_s"]),
            "full_control_interval_zero_foot": int(full_zero),
            "action": json.dumps(action.tolist(), separators=(",", ":")),
        })
        trail_xy.append(xy.copy())
        terrain_height = float(env._terrain_height(float(xy[0]), float(xy[1])))
        trail_xyz.append(np.asarray((xy[0], xy[1], terrain_height + 0.07), dtype=np.float64))
        should_render = index == 1 or index % int(config["visualisation"]["render_stride_control_steps"]) == 0 or index == len(control_rows)
        if should_render:
            frame = make_visual(index, row, support, support_sum / (index * 5), full_zero, maximum_contact_speed, index == len(control_rows))
            encode_frame(stream, container, frame)
            rendered_frames += 1
            last_frame = frame
            if index in {len(control_rows) // 3, (2 * len(control_rows)) // 3, len(control_rows)}:
                keyframes.append((f"t={index * 0.05:.1f} s", frame.copy()))
        if terminated or truncated:
            raise RuntimeError(f"Replay ended unexpectedly at control step {index}")

    if last_frame is None:
        raise RuntimeError("No video frame rendered")
    corrected = slip_tracker.finalise()
    require_equal(len(corrected["events"]), 0, "replay sustained slip events")
    require_equal(state_mismatches, 0, "formal state mismatch count")
    require_equal(substep_mismatches, 0, "formal substep mismatch count")
    require_equal(bool(arrival.spatial_success), True, "replay spatial hold")
    require_equal(path_length, float(result["path_length_m"]), "replay path length")
    require_equal(cumulative_full_zero / len(control_rows), float(result["full_control_zero_foot_fraction"]), "replay zero-foot fraction")
    episode_summary = env.env.episode_summary()
    replay_work = float(np.sum(episode_summary.get("actuator_positive_mechanical_work_j_by_actuator", [])))
    require_equal(replay_work, float(result["actuator_positive_mechanical_work_total_j"]), "replay positive work")
    for _ in range(outro_frames):
        encode_frame(stream, container, last_frame)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    renderer.close()
    env.close()

    replay_trace = episode_root / f"{stem}_replay_control_trace.csv"
    write_csv(replay_trace, replay_rows)
    final_frame = episode_root / f"{stem}_final_frame.png"
    last_frame.save(final_frame, format="PNG", optimize=True)
    contact_sheet = episode_root / f"{stem}_contact_sheet.png"
    make_contact_sheet(keyframes, contact_sheet)
    qa = validate_video(video_path, expected_width=dual.FRAME_WIDTH, expected_height=dual.FRAME_HEIGHT)
    expected_frames = intro_frames + rendered_frames + outro_frames
    require_equal(int(qa["decoded_frames"]), expected_frames, "decoded frame count")
    manifest = {
        "schema_version": "proxygap-v4-pair0-multiobjective-video-episode-v1",
        "status": "final_verified_exact_formal_replay",
        "objective": episode["objective_id"],
        "weights_time_energy": episode["weights_time_energy"],
        "contract_id": episode["contract_id"],
        "evaluation_seed": seed,
        "formal_result": result,
        "exactness": {"state_mismatch_count": state_mismatches, "substep_mismatch_count": substep_mismatches, "control_rows": len(control_rows), "physics_substep_rows": len(substep_rows)},
        "video": {"path": str(video_path), "sha256": sha256(video_path), "width": 1280, "height": 720, "fps": 20, "frames": expected_frames, "duration_seconds": expected_frames / 20.0, "playback_speed_factor": 20, "qa": qa},
        "files": {"replay_trace": str(replay_trace), "replay_trace_sha256": sha256(replay_trace), "final_frame": str(final_frame), "final_frame_sha256": sha256(final_frame), "contact_sheet": str(contact_sheet), "contact_sheet_sha256": sha256(contact_sheet)},
        "claim_boundary": config["claim_boundary"],
    }
    write_json(episode_root / "episode_manifest.json", manifest)
    return manifest


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = validate_config(config_path)
    if args.validate_only:
        print("VIDEO_CONFIG_VALIDATION_OK_NO_RENDER")
        return
    output_root = ROOT / config["execution"]["output_root"]
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    checkpoint = ROOT / config["source"]["checkpoint"]
    checkpoint_before = sha256(checkpoint)
    started = time.perf_counter()
    stage = "freeze_config"
    try:
        shutil.copy2(config_path, output_root / "frozen_video_config.json")
        formal_config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
        canonical = json.loads((ROOT / config["source"]["canonical_evaluation_config"]).read_text(encoding="utf-8"))
        fixed = json.loads((ROOT / config["source"]["fixed_map_config"]).read_text(encoding="utf-8"))
        stage = "render_three_exact_formal_replays"
        episode_manifests = [
            render_episode(config=config, formal_config=formal_config, canonical=canonical, fixed=fixed, episode=episode, output_root=output_root)
            for episode in config["episodes"]
        ]
        require_equal(sha256(checkpoint), checkpoint_before, "checkpoint before/after")
        stage = "write_root_manifest"
        root_manifest = {
            "schema_version": "proxygap-v4-pair0-multiobjective-video-root-v1",
            "status": "final_verified_three_objective_exact_formal_replays",
            "configuration_sha256": sha256(config_path),
            "formal_manifest_sha256": config["formal_evidence"]["manifest_sha256"],
            "checkpoint_sha256_before_after": checkpoint_before,
            "episode_count": 3,
            "all_formal_episodes_successful": all(item["formal_result"]["safety_qualified_completion"] for item in episode_manifests),
            "all_duration_corrected_slip_event_counts_zero": all(item["formal_result"]["duration_corrected_slip_event_count"] == 0 for item in episode_manifests),
            "all_replays_exact": all(item["exactness"]["state_mismatch_count"] == 0 and item["exactness"]["substep_mismatch_count"] == 0 for item in episode_manifests),
            "training_performed": False,
            "checkpoint_written": False,
            "formal_evidence_modified": False,
            "wall_seconds": time.perf_counter() - started,
            "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__},
            "episodes": [{"objective": item["objective"], "seed": item["evaluation_seed"], "video_sha256": item["video"]["sha256"], "episode_manifest": f"{item['objective']}/episode_manifest.json"} for item in episode_manifests],
            "claim_boundary": config["claim_boundary"],
        }
        write_json(output_root / "manifest.json", root_manifest)
        root_manifest["artifact_inventory_excludes_manifest_and_digest"] = inventory(output_root)
        write_json(output_root / "manifest.json", root_manifest)
        digest = sha256(output_root / "manifest.json")
        (output_root / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="utf-8")
        print(json.dumps({"status": root_manifest["status"], "manifest_sha256": digest, "output_root": str(output_root)}, ensure_ascii=False))
    except BaseException as exc:
        write_json(output_root / config["execution"]["failure_record_name"], {"status": "failed_closed", "stage": stage, "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc(), "training_performed": False, "scientific_interpretation_permitted": False})
        raise


if __name__ == "__main__":
    main()
