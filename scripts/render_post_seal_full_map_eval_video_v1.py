"""Render the frozen post-seal formal full-map failure as a dual-view video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_post_seal_full_map_v1 as formal  # noqa: E402
import render_fixed_goal_dual_view_video as dual  # noqa: E402
from render_fixed_goal_training_video import (  # noqa: E402
    RED,
    WHITE,
    encode_frame,
    font,
    local_valley_depth,
    make_camera,
    make_contact_sheet,
    make_map_base,
    validate_video,
)


DEFAULT_CONFIG = ROOT / "configs" / "post_seal_full_map_eval_video_v1_20260819.json"
STEM = "post_seal_full_map_seed_1763594348_dual_view_relief_v1"
FLOAT_FIELDS = (
    "time_seconds",
    "x_m",
    "y_m",
    "torso_z_m",
    "terrain_z_m",
    "torso_clearance_m",
    "goal_distance_m",
    "minimum_goal_distance_so_far_m",
    "net_progress_so_far_m",
    "path_length_so_far_m",
    "world_vx_m_per_s",
    "world_vy_m_per_s",
    "planar_speed_m_per_s",
    "terrain_relative_torso_tilt_rad",
    "mean_support_count_this_control_interval",
    "reward_step",
)
INT_FIELDS = (
    "evaluation_seed",
    "control_step",
    "endpoint_support_count",
    "full_control_interval_zero_foot",
    "force_qualified_slip_candidate_in_interval",
    "goal_entered",
    "spatial_hold_run_steps",
    "strict_stable_step",
    "strict_stable_hold_run_steps",
    "spatial_hold_success",
    "strict_stable_dwell_success",
    "finite_step",
    "environment_terminated",
    "environment_truncated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def verified_json(path: Path, expected_hash: str) -> dict[str, Any]:
    require_equal(sha256(path), expected_hash, f"SHA-256 for {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require_equal(config.get("schema_version"), "proxygap-post-seal-full-map-eval-video-v1", "schema")
    require_equal(config.get("status"), "frozen_read_only_formal_episode_video", "status")
    episode = config["episode_contract"]
    require_equal(int(episode["evaluation_seed"]), 1763594348, "seed")
    require_equal(int(episode["horizon_control_steps"]), 12000, "horizon")
    require_equal(float(episode["physical_seconds"]), 600.0, "physical seconds")
    require_equal(episode["display_outcome"], "FAILED TO REACH / HORIZON", "outcome")
    require_equal(bool(episode["formal_goal_entered"]), False, "formal goal entry")
    require_equal(bool(episode["formal_strict_stable_dwell_success"]), False, "formal dwell")
    require_equal(bool(episode["formal_safety_qualified_success"]), False, "formal qualification")
    require_equal(sha256(ROOT / episode["checkpoint"]), episode["checkpoint_sha256"], "checkpoint SHA-256")
    visual = config["visualisation"]
    require_equal(int(visual["width"]), dual.FRAME_WIDTH, "width")
    require_equal(int(visual["height"]), dual.FRAME_HEIGHT, "height")
    require_equal(int(visual["fps"]), 20, "fps")
    require_equal(int(visual["render_stride_control_steps"]), 20, "render stride")
    require_equal(int(visual["playback_speed_factor"]), 20, "playback speed")
    require_equal(sha256(ROOT / visual["overview_profile"]), visual["overview_profile_sha256"], "overview profile")
    evidence = config["formal_evidence"]
    root = ROOT / evidence["artifact_root"]
    require_equal(sha256(root / "manifest.json"), evidence["manifest_sha256"], "formal manifest")
    for path_key, hash_key in (
        ("frozen_config", "frozen_config_sha256"),
        ("execution_record", "execution_record_sha256"),
        ("episode_result", "episode_result_sha256"),
        ("control_trace", "control_trace_sha256"),
        ("physics_substep_trace", "physics_substep_trace_sha256"),
        ("pair0_scene", "pair0_scene_sha256"),
        ("contact_audit", "contact_audit_sha256"),
    ):
        require_equal(sha256(ROOT / evidence[path_key]), evidence[hash_key], path_key)
    for dependency, expected_hash in config["runtime_dependencies"].items():
        require_equal(sha256(ROOT / dependency), expected_hash, f"runtime {dependency}")
    execution = verified_json(ROOT / evidence["execution_record"], evidence["execution_record_sha256"])
    result = verified_json(ROOT / evidence["episode_result"], evidence["episode_result_sha256"])
    require_equal(execution["mode"], "formal", "formal execution mode")
    require_equal(execution["sealed_checkpoint_sha256_before"], episode["checkpoint_sha256"], "execution checkpoint")
    require_equal(int(result["evaluation_seed"]), int(episode["evaluation_seed"]), "result seed")
    require_equal(int(result["completed_control_steps"]), 12000, "result steps")
    require_equal(result["termination_reason"], episode["formal_termination_reason"], "termination")
    require_equal(bool(result["goal_entered"]), False, "result goal entry")
    require_equal(bool(result["strict_stable_dwell_success"]), False, "result stable dwell")
    require_equal(bool(result["safety_qualified_success"]), False, "result qualification")
    require_equal(float(result["best_progress_m"]), float(episode["best_progress_m"]), "best progress")
    require_equal(float(result["net_progress_m"]), float(episode["net_progress_m"]), "net progress")
    require_equal(int(result["full_control_interval_zero_foot_count"]), int(episode["full_control_interval_zero_foot_count"]), "zero-foot count")
    contact = verified_json(ROOT / evidence["contact_audit"], evidence["contact_audit_sha256"])
    require_equal(int(contact["explicit_pair_count"]), 4, "PAIR0 count")
    require_equal(contact["friction"], [1.0, 0.5, 0.5], "floor friction")
    require_equal(int(contact["condim"]), 3, "floor condim")
    execution_cfg = config["execution"]
    for key in (
        "training_permitted",
        "model_save_permitted",
        "seed_change_permitted",
        "checkpoint_change_permitted",
        "formal_artifact_mutation_permitted",
    ):
        require_equal(bool(execution_cfg[key]), False, key)
    return config, execution, result


def load_trace(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require_equal(len(rows), expected_rows, "formal control trace row count")
    require_equal(int(rows[0]["control_step"]), 1, "first trace step")
    require_equal(int(rows[-1]["control_step"]), expected_rows, "last trace step")
    return rows


def compare_row(
    formal_row: dict[str, str],
    replay_row: dict[str, Any],
    mismatch_examples: list[dict[str, Any]],
) -> int:
    mismatches = 0
    for key in FLOAT_FIELDS:
        expected = float(formal_row[key])
        actual = float(replay_row[key])
        if actual != expected:
            mismatches += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {"step": replay_row["control_step"], "field": key, "expected": expected, "actual": actual}
                )
    for key in INT_FIELDS:
        expected = int(formal_row[key])
        actual = int(replay_row[key])
        if actual != expected:
            mismatches += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {"step": replay_row["control_step"], "field": key, "expected": expected, "actual": actual}
                )
    if formal_row["mode"] != replay_row["mode"]:
        mismatches += 1
    expected_action = np.asarray(json.loads(formal_row["action"]), dtype=np.float64)
    actual_action = np.asarray(json.loads(str(replay_row["action"])), dtype=np.float64)
    if not np.array_equal(expected_action, actual_action):
        mismatches += 1
        if len(mismatch_examples) < 20:
            mismatch_examples.append(
                {"step": replay_row["control_step"], "field": "action", "expected": expected_action.tolist(), "actual": actual_action.tolist()}
            )
    return mismatches


def compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    map_base: Image.Image,
    trail_xy: list[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    position: np.ndarray,
    half_extent: float,
    physical_time: float,
    distance: float,
    best_progress: float,
    net_progress: float,
    terrain_tilt_degrees: float,
    current_support: int,
    cumulative_mean_support: float,
    maximum_contact_speed: float,
    current_airborne: bool,
    ever_airborne: bool,
    cumulative_zero_foot_count: int,
    completed_steps: int,
    floor_friction: np.ndarray,
    floor_condim: int,
    map_hash: str,
    overview_profile_id: str,
    terrain_min_height_m: float,
    terrain_max_height_m: float,
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
        requested_seconds=600.0,
        distance=distance,
        best_progress=best_progress,
        torso_tilt_degrees=terrain_tilt_degrees,
        support_count=current_support,
        maximum_contact_speed=maximum_contact_speed,
        slip_threshold=0.2,
        current_airborne=current_airborne,
        ever_airborne=ever_airborne,
        ever_contact_speed_exceeded=maximum_contact_speed > 0.2,
        unhealthy_termination=False,
        spatial_success=False,
        evaluation_seed=1763594348,
        evaluation_group_index=1,
        evaluation_group_count=1,
        checkpoint_name="PAIR0 checkpoint_2727936.zip",
        commanded_speed=0.5,
        yaw_gain=0.75,
        maximum_curvature=0.35,
        floor_friction=floor_friction,
        floor_condim=floor_condim,
        map_hash=map_hash,
        time_limit_reached=physical_time >= 600.0,
        overview_profile_id=overview_profile_id,
        terrain_min_height_m=terrain_min_height_m,
        terrain_max_height_m=terrain_max_height_m,
        overview_vertical_scale=1.0,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    x0 = dual.VIEW_WIDTH + 8
    y0 = dual.VIEW_HEIGHT + 5
    draw.rounded_rectangle(
        (x0, y0, dual.FRAME_WIDTH - 10, dual.FRAME_HEIGHT - 8),
        radius=8,
        fill=(250, 249, 245, 255),
        outline=(77, 88, 96, 210),
        width=1,
    )
    zero_fraction = cumulative_zero_foot_count / max(1, completed_steps)
    draw.text(
        (x0 + 13, y0 + 9),
        f"Progress: best {best_progress:6.2f} m | net {net_progress:6.2f} m | distance {distance:6.2f} m",
        font=font(12, bold=True),
        fill=(24, 32, 39),
    )
    draw.text(
        (x0 + 13, y0 + 37),
        f"Support: current {current_support}/4 | cumulative mean {cumulative_mean_support:.3f}/4",
        font=font(11),
        fill=(24, 32, 39),
    )
    draw.text(
        (x0 + 13, y0 + 63),
        f"Full-interval zero-foot: {cumulative_zero_foot_count}/{completed_steps} ({100.0 * zero_fraction:.2f}%)",
        font=font(11),
        fill=(24, 32, 39),
    )
    draw.rounded_rectangle(
        (x0 + 8, y0 + 91, dual.FRAME_WIDTH - 18, dual.FRAME_HEIGHT - 14),
        radius=7,
        fill=(*RED, 245),
    )
    draw.text(
        (x0 + 22, y0 + 98),
        "OUTCOME: FAILED TO REACH / HORIZON",
        font=font(12, bold=True),
        fill=WHITE,
    )
    draw.text(
        (20, dual.VIEW_HEIGHT + 43),
        "Formal replay: direct-to-goal | 20x playback | no retraining or seed change",
        font=font(11, bold=True),
        fill=RED,
    )
    return image


def write_manifest(output_root: Path, status: str) -> str:
    inventory = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}:
            inventory.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": "proxygap-post-seal-full-map-video-artifact-v1",
            "status": status,
            "inventory_count": len(inventory),
            "inventory": inventory,
        },
    )
    digest = sha256(output_root / "manifest.json")
    (output_root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8"
    )
    return digest


def render(config_path: Path) -> Path:
    config, formal_execution, formal_result = validate_config(config_path)
    output_root = ROOT / config["execution"]["output_root"]
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    stage = "freeze_video_config"
    started = time.time()
    input_hashes_before = {
        key: sha256(ROOT / value)
        for key, value in config["formal_evidence"].items()
        if key.endswith(("config", "record", "result", "trace", "scene", "audit"))
        and isinstance(value, str)
    }
    checkpoint_path = ROOT / config["episode_contract"]["checkpoint"]
    checkpoint_before = sha256(checkpoint_path)
    try:
        (output_root / "frozen_video_config.json").write_bytes(config_path.read_bytes())
        stage = "load_formal_inputs"
        evidence = config["formal_evidence"]
        formal_config = json.loads((ROOT / evidence["frozen_config"]).read_text(encoding="utf-8"))
        fixed_config = json.loads(
            (ROOT / formal_config["fixed_map"]["configuration"]).read_text(encoding="utf-8")
        )
        trace_rows = load_trace(
            ROOT / evidence["control_trace"],
            int(config["replay_exactness"]["required_control_rows"]),
        )
        scene_path = ROOT / evidence["pair0_scene"]
        model = PPO.load(checkpoint_path, device="cpu")
        model.policy.set_training_mode(False)
        require_equal(tuple(model.observation_space.shape), (135,), "model observation")
        require_equal(tuple(model.action_space.shape), (8,), "model action")
        require_equal(int(model.num_timesteps), 2727936, "model timesteps")
        policy_config = json.loads(
            (ROOT / formal_config["source"]["policy_configuration"]).read_text(encoding="utf-8")
        )
        condition = formal._make_condition(formal_config, fixed_config)
        seed = int(config["episode_contract"]["evaluation_seed"])
        horizon = int(config["episode_contract"]["horizon_control_steps"])
        env = formal.fixed_task.make_task_env(
            condition,
            policy_config,
            xml_path=scene_path,
            seed=seed,
            spawn_fraction=0.0,
            max_episode_steps=horizon,
            cruise_speed=float(formal_config["controller"]["cruise_speed_m_per_s"]),
            terminate_on_success=False,
        )
        observation, _ = env.reset(seed=seed)
        require_equal(tuple(observation.shape), (135,), "replay observation")
        audit_state = formal.l2.install_five_substep_audit(env)
        slip_cfg = formal_config["duration_corrected_slip"]
        slip_tracker = formal.l2.DurationCorrectedSlipTracker(
            dt=0.01,
            speed_threshold=float(slip_cfg["tangential_speed_threshold_m_per_s"]),
            minimum_normal_force=float(slip_cfg["minimum_normal_force_n"]),
            landing_grace_seconds=float(slip_cfg["landing_grace_seconds"]),
            minimum_sustained_seconds=float(slip_cfg["minimum_sustained_seconds"]),
        )
        arrival = formal.direct.ArrivalDwellTracker(
            arrival_radius_m=1.5,
            hold_radius_m=2.0,
            required_hold_steps=40,
        )
        approved = fixed_config["approved_map"]
        heights_path = ROOT / approved["heights_path"]
        heights = np.load(heights_path, allow_pickle=False)
        half_extent = float(approved["map_half_extent_m"])
        start = np.asarray(approved["start_xy_m"], dtype=np.float64)
        goal = np.asarray(approved["goal_xy_m"], dtype=np.float64)
        map_base = make_map_base(heights, size=dual.MAP_SIZE)
        overview_profile, overview_path = dual.load_overview_profile(
            ROOT / config["visualisation"]["overview_profile"]
        )
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
        require_equal(int(compiled.npair), 4, "compiled PAIR0 count")
        require_equal(floor_friction.tolist(), [1.0, 0.5, 0.5], "compiled friction")
        require_equal(floor_condim, 3, "compiled condim")
        compiled.vis.global_.offwidth = dual.VIEW_WIDTH
        compiled.vis.global_.offheight = dual.VIEW_HEIGHT
        renderer = mujoco.Renderer(compiled, height=dual.VIEW_HEIGHT, width=dual.VIEW_WIDTH)
        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[2] = 0
        video_path = output_root / f"{STEM}.mp4"
        av_module = dual.load_video_encoder()
        container = av_module.open(str(video_path), mode="w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=20)
        stream.width = dual.FRAME_WIDTH
        stream.height = dual.FRAME_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(config["visualisation"]["crf"]), "preset": "medium"}
        stream.gop_size = 40

        stage = "exact_replay_and_render"
        initial_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        initial_height = float(env._terrain_height(float(initial_xy[0]), float(initial_xy[1])))
        initial_distance = float(np.linalg.norm(goal - initial_xy))
        previous_xy = initial_xy.copy()
        minimum_distance = initial_distance
        path_length = 0.0
        best_progress = 0.0
        support_sum = 0
        cumulative_zero = 0
        ever_airborne = False
        trail_xy: list[np.ndarray] = [initial_xy.copy()]
        trail_xyz: list[np.ndarray] = [
            np.asarray((initial_xy[0], initial_xy[1], initial_height + 0.07), dtype=np.float64)
        ]
        replay_rows: list[dict[str, Any]] = []
        mismatch_examples: list[dict[str, Any]] = []
        mismatch_count = 0
        keyframes: list[tuple[str, Image.Image]] = []
        last_frame: Image.Image | None = None
        rendered_rollout_frames = 0
        initial_follow = make_camera(
            position=initial_xy,
            goal=goal,
            terrain_height=initial_height,
            local_valley_depth=local_valley_depth(env, initial_xy),
            progress=0.0,
        )
        left, right = dual.render_pair(
            renderer,
            data=env.unwrapped.data,
            scene_option=scene_option,
            follow_camera=initial_follow,
            fixed_overview_camera=overview_camera,
            trail_xyz=trail_xyz,
            overview_position_xyz=trail_xyz[-1],
            overview_profile=overview_profile,
        )
        initial_frame = compose_frame(
            left,
            right,
            map_base=map_base,
            trail_xy=trail_xy,
            start=start,
            goal=goal,
            position=initial_xy,
            half_extent=half_extent,
            physical_time=0.0,
            distance=initial_distance,
            best_progress=0.0,
            net_progress=0.0,
            terrain_tilt_degrees=0.0,
            current_support=0,
            cumulative_mean_support=0.0,
            maximum_contact_speed=0.0,
            current_airborne=False,
            ever_airborne=False,
            cumulative_zero_foot_count=0,
            completed_steps=0,
            floor_friction=floor_friction,
            floor_condim=floor_condim,
            map_hash=approved["heights_sha256"],
            overview_profile_id=str(overview_profile["profile_id"]),
            terrain_min_height_m=float(heights.min()),
            terrain_max_height_m=float(heights.max()),
        )
        intro_frames = round(float(config["visualisation"]["intro_seconds"]) * 20)
        outro_frames = round(float(config["visualisation"]["outro_seconds"]) * 20)
        for _ in range(intro_frames):
            encode_frame(stream, container, initial_frame)
        keyframes.append(("Formal replay start", initial_frame.copy()))
        last_frame = initial_frame

        for step in range(1, horizon + 1):
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float64)
            observation, reward, terminated, truncated, _ = env.step(action_array)
            substeps = audit_state.get("last")
            if not isinstance(substeps, list) or len(substeps) != 5:
                raise RuntimeError("Replay did not produce five physics substeps")
            interval_contacts: list[np.ndarray] = []
            interval_qualified = False
            maximum_contact_speed = 0.0
            for substep in substeps:
                contacts = np.asarray(substep["contacts"], dtype=bool)
                speeds = np.asarray(substep["speeds"], dtype=np.float64)
                forces = np.asarray(substep["forces"], dtype=np.float64)
                _, qualified = slip_tracker.update(
                    contact_mask=contacts,
                    tangential_speeds=speeds,
                    normal_forces=forces,
                )
                interval_contacts.append(contacts.copy())
                interval_qualified = interval_qualified or bool(np.any(qualified))
                active = speeds[contacts]
                maximum_contact_speed = max(
                    maximum_contact_speed,
                    float(np.max(active)) if active.size else 0.0,
                )
            interval_matrix = np.asarray(interval_contacts, dtype=bool)
            endpoint_support = int(np.sum(interval_matrix[-1]))
            full_zero = not bool(np.any(interval_matrix))
            cumulative_zero += int(full_zero)
            support_sum += int(np.sum(interval_matrix))
            ever_airborne = ever_airborne or full_zero
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64).copy()
            xy = qpos[:2]
            path_length += float(np.linalg.norm(xy - previous_xy))
            previous_xy = xy.copy()
            distance = float(np.linalg.norm(env.goal_xy - xy))
            minimum_distance = min(minimum_distance, distance)
            best_progress = max(best_progress, initial_distance - distance)
            terrain_height = float(env._terrain_height(float(xy[0]), float(xy[1])))
            terrain_tilt = formal.direct.terrain_relative_tilt_rad(env, qpos)
            finite_step = bool(
                np.all(np.isfinite(observation))
                and np.all(np.isfinite(action_array))
                and math.isfinite(float(reward))
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
                and math.isfinite(terrain_tilt)
            )
            stable = formal.direct._stable_step(
                qpos=qpos,
                qvel=qvel,
                terrain_tilt=terrain_tilt,
                support_count=endpoint_support,
                corrected_slip_candidate=interval_qualified,
                settings={
                    "require_finite_state": True,
                    "maximum_planar_speed_m_per_s": 0.2,
                    "maximum_terrain_relative_torso_tilt_degrees": 30.0,
                    "minimum_foot_support_count": 1,
                    "require_no_duration_corrected_slip_candidate": True,
                },
            )
            arrival.update(step=step, distance_m=distance, stable=stable)
            replay_row = {
                "mode": "formal",
                "evaluation_seed": seed,
                "control_step": step,
                "time_seconds": step * 0.05,
                "x_m": float(qpos[0]),
                "y_m": float(qpos[1]),
                "torso_z_m": float(qpos[2]),
                "terrain_z_m": terrain_height,
                "torso_clearance_m": float(qpos[2] - terrain_height),
                "goal_distance_m": distance,
                "minimum_goal_distance_so_far_m": minimum_distance,
                "net_progress_so_far_m": initial_distance - distance,
                "path_length_so_far_m": path_length,
                "world_vx_m_per_s": float(qvel[0]),
                "world_vy_m_per_s": float(qvel[1]),
                "planar_speed_m_per_s": float(np.linalg.norm(qvel[:2])),
                "terrain_relative_torso_tilt_rad": terrain_tilt,
                "endpoint_support_count": endpoint_support,
                "mean_support_count_this_control_interval": float(np.mean(np.sum(interval_matrix, axis=1))),
                "full_control_interval_zero_foot": int(full_zero),
                "force_qualified_slip_candidate_in_interval": int(interval_qualified),
                "goal_entered": int(arrival.goal_entered),
                "spatial_hold_run_steps": arrival.hold_run_steps,
                "strict_stable_step": int(stable),
                "strict_stable_hold_run_steps": arrival.strict_run_steps,
                "spatial_hold_success": int(arrival.spatial_success),
                "strict_stable_dwell_success": int(arrival.strict_dwell_success),
                "action": json.dumps(action_array.tolist(), separators=(",", ":")),
                "reward_step": float(reward),
                "finite_step": int(finite_step),
                "environment_terminated": int(terminated),
                "environment_truncated": int(truncated),
            }
            mismatch_count += compare_row(trace_rows[step - 1], replay_row, mismatch_examples)
            replay_rows.append(replay_row)
            trail_xy.append(xy.copy())
            trail_xyz.append(
                np.asarray((xy[0], xy[1], terrain_height + 0.07), dtype=np.float64)
            )
            if step == 1 or step % int(config["visualisation"]["render_stride_control_steps"]) == 0 or step == horizon:
                follow = make_camera(
                    position=xy,
                    goal=goal,
                    terrain_height=terrain_height,
                    local_valley_depth=local_valley_depth(env, xy),
                    progress=step / horizon,
                )
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
                frame = compose_frame(
                    left,
                    right,
                    map_base=map_base,
                    trail_xy=trail_xy,
                    start=start,
                    goal=goal,
                    position=xy,
                    half_extent=half_extent,
                    physical_time=step * 0.05,
                    distance=distance,
                    best_progress=best_progress,
                    net_progress=initial_distance - distance,
                    terrain_tilt_degrees=math.degrees(terrain_tilt),
                    current_support=endpoint_support,
                    cumulative_mean_support=support_sum / (step * 5),
                    maximum_contact_speed=maximum_contact_speed,
                    current_airborne=full_zero,
                    ever_airborne=ever_airborne,
                    cumulative_zero_foot_count=cumulative_zero,
                    completed_steps=step,
                    floor_friction=floor_friction,
                    floor_condim=floor_condim,
                    map_hash=approved["heights_sha256"],
                    overview_profile_id=str(overview_profile["profile_id"]),
                    terrain_min_height_m=float(heights.min()),
                    terrain_max_height_m=float(heights.max()),
                )
                encode_frame(stream, container, frame)
                rendered_rollout_frames += 1
                last_frame = frame
                if step in (4000, 8000, 12000):
                    keyframes.append((f"Formal replay t={step * 0.05:.0f} s", frame.copy()))
            if terminated and step < horizon:
                raise RuntimeError("Replay terminated earlier than frozen formal trace")
            if truncated and step < horizon:
                raise RuntimeError("Replay truncated earlier than frozen formal trace")

        if last_frame is None:
            raise RuntimeError("No video frame was rendered")
        corrected = slip_tracker.finalise()
        require_equal(len(corrected["events"]), int(config["episode_contract"]["duration_corrected_slip_event_count"]), "replay corrected slip events")
        require_equal(mismatch_count, 0, "formal trace replay mismatch count")
        require_equal(cumulative_zero, int(config["episode_contract"]["full_control_interval_zero_foot_count"]), "replay zero-foot count")
        require_equal(best_progress, float(config["episode_contract"]["best_progress_m"]), "replay best progress")
        require_equal(float(replay_rows[-1]["goal_distance_m"]), float(config["episode_contract"]["final_distance_m"]), "replay final distance")
        for _ in range(outro_frames):
            encode_frame(stream, container, last_frame)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        renderer.close()
        env.close()

        stage = "write_and_validate_video_evidence"
        replay_trace_path = output_root / f"{STEM}_replay_control_trace.csv"
        write_csv(replay_trace_path, replay_rows)
        exactness = {
            "schema_version": "proxygap-post-seal-formal-replay-exactness-v1",
            "status": "exact",
            "formal_trace": str((ROOT / evidence["control_trace"]).resolve()),
            "formal_trace_sha256": sha256(ROOT / evidence["control_trace"]),
            "rows_compared": len(replay_rows),
            "fields_compared_per_row": 1 + len(FLOAT_FIELDS) + len(INT_FIELDS) + 1,
            "float_fields": list(FLOAT_FIELDS),
            "integer_fields": list(INT_FIELDS),
            "other_fields": ["mode", "action"],
            "mismatch_count": mismatch_count,
            "mismatch_examples": mismatch_examples,
            "replay_trace": str(replay_trace_path),
            "replay_trace_sha256": sha256(replay_trace_path),
            "seed": seed,
            "checkpoint_sha256": sha256(checkpoint_path),
            "pair0_scene_sha256": sha256(scene_path),
        }
        exactness_path = output_root / f"{STEM}_replay_exactness.json"
        write_json(exactness_path, exactness)
        final_frame_path = output_root / f"{STEM}_final_frame.png"
        last_frame.save(final_frame_path, format="PNG", optimize=True)
        contact_sheet_path = output_root / f"{STEM}_contact_sheet.png"
        make_contact_sheet(keyframes, contact_sheet_path)
        qa = validate_video(video_path, expected_width=dual.FRAME_WIDTH, expected_height=dual.FRAME_HEIGHT)
        total_frames = intro_frames + rendered_rollout_frames + outro_frames
        require_equal(int(qa["decoded_frames"]), total_frames, "decoded frame count")
        input_hashes_after = {
            key: sha256(ROOT / value)
            for key, value in config["formal_evidence"].items()
            if key.endswith(("config", "record", "result", "trace", "scene", "audit"))
            and isinstance(value, str)
        }
        require_equal(input_hashes_after, input_hashes_before, "formal inputs before/after")
        checkpoint_after = sha256(checkpoint_path)
        require_equal(checkpoint_after, checkpoint_before, "checkpoint before/after")
        manifest = {
            "schema_version": "proxygap-post-seal-full-map-eval-video-result-v1",
            "status": "final_verified",
            "outcome_displayed": "FAILED TO REACH / HORIZON",
            "video": {
                "path": str(video_path),
                "sha256": sha256(video_path),
                "width": dual.FRAME_WIDTH,
                "height": dual.FRAME_HEIGHT,
                "fps": 20,
                "encoded_frames": total_frames,
                "duration_seconds": total_frames / 20.0,
                "physical_rollout_seconds": 600.0,
                "render_stride_control_steps": 20,
                "playback_speed_factor": 20,
                "codec": "H.264/libx264",
                "pixel_format": "yuv420p",
            },
            "formal_binding": {
                "evaluation_seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256_before": checkpoint_before,
                "checkpoint_sha256_after": checkpoint_after,
                "pair0_scene": str(scene_path.resolve()),
                "pair0_scene_sha256": sha256(scene_path),
                "formal_trace_sha256": sha256(ROOT / evidence["control_trace"]),
                "formal_result_sha256": sha256(ROOT / evidence["episode_result"]),
                "formal_manifest_sha256": sha256((ROOT / evidence["artifact_root"] / "manifest.json")),
                "input_hashes_before": input_hashes_before,
                "input_hashes_after": input_hashes_after,
                "trace_exactness": exactness,
            },
            "layout": {
                "left": "deterministic following camera",
                "right": "goal-to-start oblique relief-v2 overview at physical 1:1 vertical scale",
                "ground_trail": "actual replayed surface trajectory, visual-only",
                "mini_map": True,
                "bottom_panel": "failure outcome, progress, support and full-interval zero-foot metrics",
            },
            "formal_result": {
                "goal_entered": False,
                "strict_stable_dwell_success": False,
                "safety_qualified_success": False,
                "termination_reason": "horizon_truncated",
                "best_progress_m": formal_result["best_progress_m"],
                "net_progress_m": formal_result["net_progress_m"],
                "final_distance_m": formal_result["final_distance_m"],
                "mean_support_count_per_physics_substep": formal_result["mean_distal_support_count_per_physics_substep"],
                "full_control_interval_zero_foot_count": formal_result["full_control_interval_zero_foot_count"],
                "full_control_interval_zero_foot_fraction": formal_result["full_control_interval_zero_foot_fraction"],
                "duration_corrected_slip_event_count": formal_result["duration_corrected_slip_event_count"],
            },
            "immutability": {
                "training_performed": False,
                "seed_changed": False,
                "checkpoint_modified": False,
                "formal_artifacts_modified": False,
                "both_views_share_one_replay_state": True,
            },
            "files": {
                "replay_control_trace": str(replay_trace_path),
                "replay_control_trace_sha256": sha256(replay_trace_path),
                "replay_exactness": str(exactness_path),
                "replay_exactness_sha256": sha256(exactness_path),
                "contact_sheet": str(contact_sheet_path),
                "contact_sheet_sha256": sha256(contact_sheet_path),
                "final_frame": str(final_frame_path),
                "final_frame_sha256": sha256(final_frame_path),
                "overview_profile": str(overview_path),
                "overview_profile_sha256": sha256(overview_path),
            },
            "qa": {
                **qa,
                "decoded_every_frame": int(qa["decoded_frames"]) == total_frames,
                "formal_trace_exact": mismatch_count == 0,
                "checkpoint_before_after_equal": checkpoint_before == checkpoint_after,
                "formal_inputs_before_after_equal": input_hashes_before == input_hashes_after,
            },
            "runtime": {
                "wall_seconds": time.time() - started,
                "platform": platform.platform(),
                "python": sys.version,
                "mujoco": mujoco.__version__,
            },
            "claim_boundary": config["claim_boundary"],
        }
        manifest_path = output_root / f"{STEM}_video_manifest.json"
        write_json(manifest_path, manifest)
        write_manifest(output_root, "final_verified")
        return output_root
    except BaseException as exc:
        try:
            write_json(
                output_root / config["execution"]["failure_record_name"],
                {
                    "schema_version": "proxygap-post-seal-full-map-video-failure-v1",
                    "status": "failed_closed",
                    "stage": stage,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "training_performed": False,
                    "seed_changed": False,
                    "checkpoint_modified": sha256(checkpoint_path) != checkpoint_before,
                    "scientific_interpretation_permitted": False,
                },
            )
            write_manifest(output_root, "failed_closed")
        finally:
            raise


def main() -> None:
    args = parse_args()
    output_root = render(args.config.resolve())
    print(str(output_root), flush=True)


if __name__ == "__main__":
    main()
