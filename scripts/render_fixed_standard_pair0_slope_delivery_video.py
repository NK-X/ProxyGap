"""Render two fail-closed PAIR0 standard-slope diagnostic videos.

The renderer never trains or writes a checkpoint.  It first reproduces each
selected formal episode through the frozen evaluator, then replays the same
episode while rendering.  The metrics recomputed from the rendered replay must
match every field of the archived formal CSV row exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_fixed_standard_pair0_slope_capability_boundary as slope_eval  # noqa: E402
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402
from render_fixed_goal_dual_view_video import (  # noqa: E402
    add_overview_position_marker,
    add_surface_trail,
)
from render_fixed_goal_training_video import (  # noqa: E402
    AMBER,
    BACKGROUND,
    INK,
    RED,
    TEAL,
    WHITE,
    encode_frame,
    font,
)
from run_fixed_standard_support_curriculum import (  # noqa: E402
    FOOT_NAMES,
    contact_masks_from_data,
    make_standard_env,
    quaternion_tilt_relative_to_normal,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "fixed_standard_pair0_slope_delivery_video_v1_20260819.json"
)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIEW_WIDTH = 640
VIEW_HEIGHT = 540
FPS = 20
EXPECTED_EPISODES = (
    ("uphill_12deg", 12.0, 94153),
    ("downhill_16deg", -16.0, 94137),
)
BOOL_FIELDS = {
    "finite",
    "fall",
    "outer_terrain_fall",
    "inner_absolute_z_fall",
    "fixed_goal_success",
    "torso_ground_any",
    "sustained_nonfoot_contact",
}
INT_FIELDS = {
    "checkpoint_additional_timesteps",
    "checkpoint_timesteps",
    "evaluation_seed",
    "control_steps",
    "physics_substeps",
    "full_interval_zero_foot_count",
    "support_count_sum_physics_substeps",
    "supported_physics_substep_count",
    "force_qualified_supported_physics_substep_count",
    "qualified_slip_physics_substep_count",
    "corrected_sustained_slip_physics_substep_count",
    "corrected_slip_event_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty trace")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen delivery contract changed: {label}")


def verify_path(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != str(record[hash_key]):
        raise ValueError(f"Source hash changed: {path_key}")
    return path


def parse_formal_value(field: str, value: str) -> Any:
    if field in BOOL_FIELDS:
        lowered = value.strip().lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"Invalid archived boolean for {field}")
        return lowered == "true"
    if field in INT_FIELDS:
        return int(value)
    if field in {"condition_id", "scene_name"}:
        return value
    return float(value)


def typed_formal_row(row: dict[str, str]) -> dict[str, Any]:
    return {field: parse_formal_value(field, value) for field, value in row.items()}


def compare_episode_rows(
    archived: dict[str, Any], replayed: dict[str, Any]
) -> dict[str, Any]:
    if list(archived) != list(replayed):
        raise RuntimeError("Archived and replayed episode schemas differ")
    comparisons: dict[str, Any] = {}
    mismatches: list[str] = []
    for field in archived:
        expected = archived[field]
        observed = replayed[field]
        exact = bool(observed == expected)
        entry: dict[str, Any] = {
            "archived": expected,
            "replayed": observed,
            "exact_match": exact,
        }
        if isinstance(expected, float) and isinstance(observed, float):
            entry["absolute_difference"] = abs(observed - expected)
        comparisons[field] = entry
        if not exact:
            mismatches.append(field)
    result = {
        "field_count": len(archived),
        "all_fields_exact_match": not mismatches,
        "mismatched_fields": mismatches,
        "fields": comparisons,
    }
    if mismatches:
        raise RuntimeError(f"Rendered replay differs from formal row: {mismatches}")
    return result


def formal_row_for(
    rows: list[dict[str, str]], scene_name: str, evaluation_seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["condition_id"] == l2.PAIR0_ID
        and row["scene_name"] == scene_name
        and int(row["evaluation_seed"]) == evaluation_seed
    ]
    if len(matches) != 1:
        raise ValueError("Selected formal episode row is not unique")
    return typed_formal_row(matches[0])


def validate_selection(
    config: dict[str, Any], rows: list[dict[str, str]], formal_config: dict[str, Any]
) -> None:
    observed = [
        (
            str(item["scene_name"]),
            float(item["signed_slope_degrees"]),
            int(item["evaluation_seed"]),
        )
        for item in config["representative_episodes"]
    ]
    _require_equal(observed, list(EXPECTED_EPISODES), "representative episodes")
    zero_limit = float(formal_config["gates"]["maximum_full_interval_zero_foot_fraction"])
    for scene_name, _, selected_seed in EXPECTED_EPISODES:
        scene_rows = [
            row
            for row in rows
            if row["condition_id"] == l2.PAIR0_ID and row["scene_name"] == scene_name
        ]
        if len(scene_rows) != 5:
            raise ValueError(f"Formal scene matrix is incomplete: {scene_name}")
        ranked = sorted(
            scene_rows,
            key=lambda row: (
                float(row["fixed_goal_best_progress_m"]),
                int(row["evaluation_seed"]),
            ),
        )
        median_seed = int(ranked[2]["evaluation_seed"])
        _require_equal(median_seed, selected_seed, f"median-progress seed {scene_name}")
        selected = formal_row_for(rows, scene_name, selected_seed)
        if int(selected["full_interval_zero_foot_count"]) / 600.0 > zero_limit:
            raise ValueError(f"Selected episode violates zero-foot safety: {scene_name}")
        for field in (
            "finite",
            "fall",
            "torso_ground_any",
            "sustained_nonfoot_contact",
        ):
            expected = True if field == "finite" else False
            _require_equal(selected[field], expected, f"selected safety {scene_name}.{field}")
        for field in (
            "corrected_sustained_slip_physics_substep_count",
            "corrected_slip_event_count",
        ):
            _require_equal(selected[field], 0, f"selected safety {scene_name}.{field}")


def validate_config(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    _require_equal(
        config.get("schema_version"),
        "proxygap-pair0-standard-slope-delivery-video-v1",
        "schema_version",
    )
    _require_equal(
        config.get("config_id"),
        "fixed_standard_pair0_slope_delivery_video_v1_20260819",
        "config_id",
    )
    _require_equal(config.get("status"), "frozen_read_only_visual_replay", "status")
    source = config["source"]
    formal_manifest_path = verify_path(source, "formal_manifest", "formal_manifest_sha256")
    formal_config_path = verify_path(
        source, "formal_frozen_config", "formal_frozen_config_sha256"
    )
    metrics_path = verify_path(
        source, "formal_episode_metrics", "formal_episode_metrics_sha256"
    )
    scenes_path = verify_path(
        source, "formal_prepared_scenes", "formal_prepared_scenes_sha256"
    )
    verify_path(source, "formal_summary", "formal_summary_sha256")
    checkpoint_path = verify_path(source, "checkpoint", "checkpoint_sha256")
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    if bool(formal_manifest.get("training_performed", True)):
        raise ValueError("Slope source unexpectedly reports training")
    _require_equal(formal_manifest.get("video_rendered"), False, "formal video boundary")
    _require_equal(
        formal_manifest.get("checkpoint_sha256"),
        source["checkpoint_sha256"],
        "formal checkpoint hash",
    )
    formal_config = json.loads(formal_config_path.read_text(encoding="utf-8"))
    protocol, reward, validated_checkpoint = slope_eval.validate_config(formal_config)
    if validated_checkpoint.resolve() != checkpoint_path.resolve():
        raise ValueError("Slope evaluator resolved a different checkpoint")
    rows = read_csv(metrics_path)
    if len(rows) != 55:
        raise ValueError("Formal slope matrix must contain 55 rows")
    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    validate_selection(config, rows, formal_config)

    replay = config["replay"]
    for key, expected in (
        ("max_episode_steps", 600),
        ("physics_substeps_per_control_step", 5),
    ):
        _require_equal(int(replay[key]), expected, f"replay.{key}")
    for key, expected in (
        ("cruise_speed_m_per_s", 0.55),
        ("physics_timestep_seconds", 0.01),
        ("control_timestep_seconds", 0.05),
    ):
        _require_equal(float(replay[key]), expected, f"replay.{key}")
    for key in (
        "deterministic_policy",
        "require_formal_episode_row_fieldwise_exact_match",
        "require_all_600_trace_rows",
    ):
        _require_equal(replay[key], True, f"replay.{key}")
    for key in (
        "training",
        "checkpoint_write",
        "reward_change",
        "friction_change",
        "energy_formula_change",
    ):
        _require_equal(replay[key], False, f"replay.{key}")
    render = config["render"]
    for key, expected in (
        ("width_px", FRAME_WIDTH),
        ("height_px", FRAME_HEIGHT),
        ("view_height_px", VIEW_HEIGHT),
        ("fps", FPS),
    ):
        _require_equal(int(render[key]), expected, f"render.{key}")
    _require_equal(render["render_every_control_step"], True, "render cadence")
    _require_equal(render["surface_trajectory_in_both_views"], True, "trail")
    _require_equal(
        config["energy"]["status"],
        "measurement_only_not_reward_or_gate",
        "energy status",
    )
    _require_equal(
        config["output"]["root"],
        "artifacts/dev/fixed_standard_pair0_slope_delivery_video_v1_20260819/attempt_0",
        "output root",
    )
    runtime: dict[str, str] = {}
    for relative_path, expected_hash in config["runtime_dependencies"].items():
        path = ROOT / relative_path
        if sha256(path) != expected_hash:
            raise ValueError(f"Runtime dependency changed: {relative_path}")
        runtime[relative_path] = expected_hash
    for scene_name, _, _ in EXPECTED_EPISODES:
        scene = scenes[scene_name]
        for key in ("xml", "heights", "hfield", "texture"):
            if sha256(scene[f"{key}_path"]) != scene[f"{key}_sha256"]:
                raise ValueError(f"Prepared scene asset changed: {scene_name}.{key}")
        _require_equal(scene["condition_id"], l2.PAIR0_ID, "scene condition")
    return protocol, reward, checkpoint_path, formal_config, rows, scenes, runtime


def follow_camera(
    *, position: np.ndarray, terrain_height: float, signed_slope_degrees: float
) -> mujoco.MjvCamera:
    """Track from above and slightly behind; plane slopes cannot occlude the torso."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (
        float(position[0] + 0.35),
        float(position[1]),
        float(max(position[2] - 0.10, terrain_height + 0.42)),
    )
    camera.distance = 5.6
    camera.azimuth = 132.0 if signed_slope_degrees > 0.0 else 48.0
    camera.elevation = -27.0
    return camera


def fixed_relief_camera(*, height_midpoint: float) -> mujoco.MjvCamera:
    """Fixed oblique side view with unchanged physical x/y/z geometry."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (0.0, 0.0, float(height_midpoint + 0.1))
    camera.distance = 28.5
    camera.azimuth = 82.0
    camera.elevation = -24.0
    return camera


def render_pair(
    renderer: mujoco.Renderer,
    *,
    data: mujoco.MjData,
    follow: mujoco.MjvCamera,
    overview: mujoco.MjvCamera,
    trail_xyz: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=follow)
    add_surface_trail(renderer.scene, trail_xyz, maximum_segments=600)
    left = np.asarray(renderer.render(), dtype=np.uint8).copy()
    renderer.update_scene(data, camera=overview)
    add_surface_trail(renderer.scene, trail_xyz, maximum_segments=600)
    add_overview_position_marker(renderer.scene, trail_xyz[-1])
    right = np.asarray(renderer.render(), dtype=np.uint8).copy()
    return left, right


def compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scene_name: str,
    seed: int,
    slope_degrees: float,
    step: int,
    checkpoint_name: str,
    best_progress_m: float,
    endpoint_support_count: int,
    mean_support_count: float,
    zerofoot_count: int,
    formal_row: dict[str, Any],
    positive_work_j: float,
    squared_action: float,
    start_height_m: float,
    goal_height_m: float,
    terminated: bool,
    truncated: bool,
) -> Image.Image:
    if left.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("Left render geometry changed")
    if right.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("Right render geometry changed")
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), BACKGROUND)
    image.paste(Image.fromarray(left, mode="RGB"), (0, 0))
    image.paste(Image.fromarray(right, mode="RGB"), (VIEW_WIDTH, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line((VIEW_WIDTH, 0, VIEW_WIDTH, VIEW_HEIGHT), fill=(255, 255, 255, 210), width=2)
    draw.text(
        (16, 14),
        "ADAPTIVE FOLLOW / ROBOT VISIBILITY",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    draw.text(
        (VIEW_WIDTH + 16, 14),
        "FIXED PHYSICAL RELIEF / 1:1 VERTICAL SCALE",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    draw.rounded_rectangle(
        (VIEW_WIDTH + 16, 44, FRAME_WIDTH - 16, 104),
        radius=7,
        fill=(9, 17, 23, 205),
        outline=(230, 238, 235, 120),
        width=1,
    )
    direction = "UPHILL" if slope_degrees > 0.0 else "DOWNHILL"
    draw.text(
        (VIEW_WIDTH + 28, 51),
        f"{direction} -> x+ | signed angle {slope_degrees:+.1f} deg",
        font=font(11, bold=True),
        fill=WHITE,
    )
    draw.text(
        (VIEW_WIDTH + 28, 78),
        f"start z {start_height_m:+.2f} m | goal z {goal_height_m:+.2f} m | delta {goal_height_m-start_height_m:+.2f} m",
        font=font(10),
        fill=(224, 235, 232),
    )

    draw.rectangle((0, VIEW_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT), fill=(250, 249, 245, 255))
    draw.line((0, VIEW_HEIGHT, FRAME_WIDTH, VIEW_HEIGHT), fill=(77, 88, 96, 210), width=2)
    draw.text(
        (18, VIEW_HEIGHT + 10),
        "STANDARD-SLOPE DIAGNOSTIC / NOT FIXED-MAP / NOT GENERALISATION",
        font=font(12, bold=True),
        fill=AMBER,
    )
    draw.text(
        (18, VIEW_HEIGHT + 38),
        f"condition PAIR0_ADAPT | scene {scene_name} | seed {seed} | angle {slope_degrees:+.1f} deg | t {step*0.05:05.2f}/30.00 s",
        font=font(10, bold=True),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 64),
        f"checkpoint {checkpoint_name} | deterministic | 600 controls | 5 physics substeps/control",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 90),
        f"progress {best_progress_m:5.2f} m (formal final {formal_row['fixed_goal_best_progress_m']:.2f} m) | support now {endpoint_support_count}/4 | mean {mean_support_count:.2f}/4",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 116),
        f"zero-foot intervals {zerofoot_count}/{max(step, 1)} (formal final {formal_row['full_interval_zero_foot_count']}/600 = {formal_row['full_interval_zero_foot_count']/600.0:.3f})",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 142),
        f"ENERGY MEASUREMENT-ONLY: sum(a^2) {squared_action:.2f} | positive mechanical work {positive_work_j:.1f} J | not battery energy",
        font=font(10, bold=True),
        fill=(68, 77, 83),
    )

    safety_pass = bool(
        formal_row["finite"]
        and not formal_row["fall"]
        and not formal_row["torso_ground_any"]
        and not formal_row["sustained_nonfoot_contact"]
        and formal_row["corrected_sustained_slip_physics_substep_count"] == 0
        and formal_row["corrected_slip_event_count"] == 0
    )
    colour = TEAL if safety_pass else RED
    if terminated and step < 600:
        status = "EARLY TERMINATION"
        colour = RED
    elif truncated:
        status = "FORMAL 600-STEP HORIZON COMPLETE / SAFETY PASS"
    else:
        status = "FORMAL EPISODE SAFETY PASS / REPLAY IN PROGRESS"
    draw.rounded_rectangle(
        (760, VIEW_HEIGHT + 106, FRAME_WIDTH - 18, FRAME_HEIGHT - 14),
        radius=7,
        fill=(*colour, 238),
    )
    draw.text(
        (774, VIEW_HEIGHT + 120), status, font=font(10, bold=True), fill=WHITE
    )
    return image


def make_contact_sheet(
    frames: list[tuple[str, Image.Image]], path: Path, *, title: str
) -> None:
    if len(frames) != 4:
        raise ValueError("Contact sheet requires four predeclared frames")
    canvas = Image.new("RGB", (1240, 744), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), title, font=font(20, bold=True), fill=INK)
    for index, (label, frame) in enumerate(frames):
        row, column = divmod(index, 2)
        x = 20 + column * 610
        y = 58 + row * 340
        thumb = frame.resize((600, 338), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.rounded_rectangle(
            (x + 12, y + 12, x + 190, y + 44), radius=7, fill=(250, 252, 251)
        )
        draw.text((x + 24, y + 20), label, font=font(11, bold=True), fill=INK)
    canvas.save(path, format="PNG", optimize=True)


def validate_video_exhaustive(
    path: Path, *, expected_width: int, expected_height: int, expected_fps: int
) -> dict[str, Any]:
    """Decode every frame and fail on any geometry or frame-rate deviation."""
    import av

    decoded = 0
    expected_shape = (expected_height, expected_width, 3)
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise RuntimeError("Delivery file must contain exactly one video stream")
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate)
        if average_rate != float(expected_fps):
            raise RuntimeError("Delivery video frame rate changed")
        for frame in container.decode(video=0):
            shape = frame.to_ndarray(format="rgb24").shape
            if shape != expected_shape:
                raise RuntimeError(
                    f"Decoded frame {decoded} has unexpected geometry {shape}"
                )
            decoded += 1
    if decoded <= 0:
        raise RuntimeError("Delivery video decoded no frames")
    return {
        "decoded_frames": decoded,
        "every_frame_decoded": True,
        "all_decoded_frame_shapes_equal_expected": True,
        "expected_frame_shape": list(expected_shape),
        "average_frame_rate": average_rate,
        "decoded_duration_seconds": decoded / average_rate,
    }


def replay_row_from_render(
    *,
    model: PPO,
    formal_config: dict[str, Any],
    scene: dict[str, Any],
    seed: int,
    control_step: int,
    finite: bool,
    contact_rows: list[np.ndarray],
    force_qualified_support_rows: list[bool],
    nonfoot_rows: list[bool],
    torso_rows: list[bool],
    control_fullzero: list[bool],
    corrected: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    contacts = np.asarray(contact_rows, dtype=bool)
    supported = np.any(contacts, axis=1)
    force_supported = np.asarray(force_qualified_support_rows, dtype=bool)
    candidate = np.asarray(corrected["candidate"], dtype=bool)
    sustained = np.asarray(corrected["sustained"], dtype=bool)
    nonfoot_longest = l2._longest_true_run(nonfoot_rows)
    dt = float(formal_config["evaluation"]["physics_timestep_seconds"])
    row = {
        "condition_id": l2.PAIR0_ID,
        "checkpoint_additional_timesteps": 65_536,
        "checkpoint_timesteps": int(model.num_timesteps),
        "scene_name": scene["scene_name"],
        "evaluation_seed": seed,
        "control_steps": control_step,
        "physics_substeps": int(contacts.shape[0]),
        "finite": finite,
        "fall": bool(
            summary.get("fall", False)
            or summary.get("inner_absolute_z_fall", False)
        ),
        "outer_terrain_fall": bool(summary.get("fall", False)),
        "inner_absolute_z_fall": bool(summary.get("inner_absolute_z_fall", False)),
        "fixed_goal_success": bool(summary["fixed_goal_success"]),
        "fixed_goal_best_progress_m": float(summary["fixed_goal_initial_distance_m"])
        - float(summary["fixed_goal_minimum_distance_m"]),
        "fixed_goal_net_progress_m": float(summary["fixed_goal_net_progress_m"]),
        "full_interval_zero_foot_count": int(np.sum(control_fullzero)),
        "support_count_sum_physics_substeps": int(np.sum(contacts)),
        "supported_physics_substep_count": int(np.sum(supported)),
        "force_qualified_supported_physics_substep_count": int(
            np.sum(force_supported)
        ),
        "qualified_slip_physics_substep_count": int(
            np.sum(np.any(candidate, axis=1))
        ),
        "corrected_sustained_slip_physics_substep_count": int(
            np.sum(np.any(sustained, axis=1))
        ),
        "corrected_slip_event_count": len(corrected["events"]),
        "torso_ground_any": bool(np.any(torso_rows)),
        "nonfoot_ground_longest_run_seconds": nonfoot_longest * dt,
        "sustained_nonfoot_contact": bool(
            nonfoot_longest * dt
            >= float(
                formal_config["checkpoint_early_stopping"][
                    "nonfoot_contact_minimum_sustained_seconds"
                ]
            )
        ),
        "cumulative_squared_action": float(
            summary.get("cumulative_squared_action", 0.0)
        ),
        "actuator_abs_torque_time_integral_total_n_m_s": l2._vector_sum(
            summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator"
        ),
        "actuator_positive_mechanical_work_total_j": l2._vector_sum(
            summary, "actuator_positive_mechanical_work_j_by_actuator"
        ),
        "actuator_abs_mechanical_work_total_j": l2._vector_sum(
            summary, "actuator_abs_mechanical_work_j_by_actuator"
        ),
    }
    return row


def render_episode(
    *,
    config: dict[str, Any],
    formal_config: dict[str, Any],
    protocol: dict[str, Any],
    reward: dict[str, Any],
    model: PPO,
    checkpoint: Path,
    formal_rows: list[dict[str, str]],
    scene: dict[str, Any],
    signed_slope_degrees: float,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    scene_name = str(scene["scene_name"])
    formal_row = formal_row_for(formal_rows, scene_name, seed)
    evaluator_row, _, _ = l2.evaluate_episode(
        model,
        formal_config,
        protocol,
        reward,
        scene,
        condition_id=l2.PAIR0_ID,
        seed=seed,
        checkpoint_additional_timesteps=65_536,
        max_episode_steps=600,
        retain_substeps=False,
    )
    pre_render_comparison = compare_episode_rows(formal_row, evaluator_row)

    slug = f"{scene_name}_seed_{seed}"
    episode_dir = output_root / slug
    episode_dir.mkdir(parents=True)
    stem = f"pair0_{scene_name}_seed_{seed}_standard_slope_diagnostic"
    video_path = episode_dir / f"{stem}.mp4"
    trace_path = episode_dir / f"{stem}_trace_600_steps.csv"
    metric_path = episode_dir / f"{stem}_replay_episode_metrics.json"
    comparison_path = episode_dir / f"{stem}_fieldwise_metric_comparison.json"
    final_frame_path = episode_dir / f"{stem}_final_frame.png"
    contact_sheet_path = episode_dir / f"{stem}_contact_sheet.png"

    env = make_standard_env(
        protocol,
        reward,
        scene,
        condition_id=l2.PAIR0_ID,
        seed=seed,
        max_episode_steps=600,
        cruise_speed=0.55,
    )
    l2.compiled_contract_audit(
        env.unwrapped.model,
        scene,
        l2.PAIR0_ID,
        formal_config,
        construction_seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    audit_state = l2.install_five_substep_audit(env)
    compiled = env.unwrapped.model
    compiled.vis.global_.offwidth = VIEW_WIDTH
    compiled.vis.global_.offheight = VIEW_HEIGHT
    renderer = mujoco.Renderer(compiled, height=VIEW_HEIGHT, width=VIEW_WIDTH)
    heights = np.load(scene["heights_path"], allow_pickle=False)
    overview = fixed_relief_camera(
        height_midpoint=float((np.min(heights) + np.max(heights)) / 2.0)
    )
    start = np.asarray(scene["start_xy_m"], dtype=np.float64)
    goal = np.asarray(scene["goal_xy_m"], dtype=np.float64)
    initial_distance = float(np.linalg.norm(goal - start))
    start_height = float(env._terrain_height(float(start[0]), float(start[1])))
    goal_height = float(env._terrain_height(float(goal[0]), float(goal[1])))
    foot_ids = tuple(
        int(mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in FOOT_NAMES
    )
    slip = formal_config["evaluation"]["corrected_slip"]
    tracker = l2.DurationCorrectedSlipTracker(
        dt=0.01,
        speed_threshold=float(slip["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip["minimum_sustained_seconds"]),
    )

    import av

    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = FRAME_WIDTH
    stream.height = FRAME_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(config["render"]["crf"]), "preset": "medium"}
    stream.gop_size = FPS * 2

    trace_rows: list[dict[str, Any]] = []
    contact_rows: list[np.ndarray] = []
    force_supported_rows: list[bool] = []
    nonfoot_rows: list[bool] = []
    torso_rows: list[bool] = []
    control_fullzero: list[bool] = []
    trail_xyz: list[np.ndarray] = []
    finite = True
    terminated = truncated = False
    step = 0
    best_progress = 0.0
    contact_frames: list[tuple[str, Image.Image]] = []
    last_frame: Image.Image | None = None

    def current_frame(
        *, endpoint_support: int, squared_action: float, positive_work: float
    ) -> Image.Image:
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        terrain_height = float(env._terrain_height(float(qpos[0]), float(qpos[1])))
        follow = follow_camera(
            position=qpos[:3],
            terrain_height=terrain_height,
            signed_slope_degrees=signed_slope_degrees,
        )
        left, right = render_pair(
            renderer,
            data=env.unwrapped.data,
            follow=follow,
            overview=overview,
            trail_xyz=trail_xyz,
        )
        mean_support = (
            float(sum(int(np.sum(row)) for row in contact_rows)) / len(contact_rows)
            if contact_rows
            else 0.0
        )
        return compose_frame(
            left,
            right,
            scene_name=scene_name,
            seed=seed,
            slope_degrees=signed_slope_degrees,
            step=step,
            checkpoint_name=checkpoint.name,
            best_progress_m=best_progress,
            endpoint_support_count=endpoint_support,
            mean_support_count=mean_support,
            zerofoot_count=int(np.sum(control_fullzero)),
            formal_row=formal_row,
            positive_work_j=positive_work,
            squared_action=squared_action,
            start_height_m=start_height,
            goal_height_m=goal_height,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    try:
        qpos0 = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        trail_xyz.append(
            np.asarray(
                (
                    qpos0[0],
                    qpos0[1],
                    env._terrain_height(float(qpos0[0]), float(qpos0[1])) + 0.05,
                )
            )
        )
        last_frame = current_frame(
            endpoint_support=0, squared_action=0.0, positive_work=0.0
        )
        contact_frames.append(("t = 0 s", last_frame.copy()))
        intro_frames = round(float(config["render"]["intro_seconds"]) * FPS)
        outro_frames = round(float(config["render"]["outro_seconds"]) * FPS)
        for _ in range(intro_frames):
            encode_frame(stream, container, last_frame)

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward_value, terminated, truncated, info = env.step(action)
            step += 1
            substeps = audit_state.get("last")
            if not isinstance(substeps, list) or len(substeps) != 5:
                raise RuntimeError("Rendered replay did not expose five physics substeps")
            interval_contacts: list[np.ndarray] = []
            for substep in substeps:
                contacts = np.asarray(substep["contacts"], dtype=bool)
                speeds = np.asarray(substep["speeds"], dtype=np.float64)
                forces = np.asarray(substep["forces"], dtype=np.float64)
                tracker.update(
                    contact_mask=contacts,
                    tangential_speeds=speeds,
                    normal_forces=forces,
                )
                contact_rows.append(contacts.copy())
                interval_contacts.append(contacts.copy())
                force_supported_rows.append(
                    bool(
                        np.any(
                            contacts
                            & (forces >= float(slip["minimum_normal_force_n"]))
                        )
                    )
                )
                nonfoot_rows.append(bool(substep["nonfoot"]))
                torso_rows.append(bool(substep["torso"]))
            control_fullzero.append(
                not np.any(np.asarray(interval_contacts, dtype=bool))
            )
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
            x, y = float(qpos[0]), float(qpos[1])
            terrain_height = float(env._terrain_height(x, y))
            trail_xyz.append(np.asarray((x, y, terrain_height + 0.05)))
            distance = float(np.linalg.norm(goal - qpos[:2]))
            best_progress = max(best_progress, initial_distance - distance)
            finite = finite and bool(
                np.all(np.isfinite(observation))
                and np.all(np.isfinite(action))
                and np.isfinite(reward_value)
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
            )
            endpoint_contacts = interval_contacts[-1]
            endpoint_feet, endpoint_nonfoot, endpoint_torso = contact_masks_from_data(
                compiled, env.unwrapped.data, foot_ids
            )
            if not np.array_equal(endpoint_contacts, endpoint_feet):
                raise RuntimeError("Endpoint contact mask disagrees with substep audit")
            summary_now = env.episode_summary()
            squared_action = float(summary_now.get("cumulative_squared_action", 0.0))
            positive_work = l2._vector_sum(
                summary_now, "actuator_positive_mechanical_work_j_by_actuator"
            )
            normal = env._terrain_normal(x, y)
            relative_tilt = quaternion_tilt_relative_to_normal(qpos[3:7], normal)
            last_substep = substeps[-1]
            trace_rows.append(
                {
                    "condition_id": l2.PAIR0_ID,
                    "scene_name": scene_name,
                    "evaluation_seed": seed,
                    "control_step": step,
                    "time_seconds": step * 0.05,
                    "x_m": x,
                    "y_m": y,
                    "terrain_height_m": terrain_height,
                    "torso_z_m": float(qpos[2]),
                    "distance_to_goal_m": distance,
                    "best_progress_m": best_progress,
                    "endpoint_support_count": int(np.sum(endpoint_contacts)),
                    "full_interval_zero_foot": bool(control_fullzero[-1]),
                    "cumulative_full_interval_zero_foot_count": int(
                        np.sum(control_fullzero)
                    ),
                    "five_substep_support_counts": json.dumps(
                        [int(np.sum(row)) for row in interval_contacts], separators=(",", ":")
                    ),
                    "endpoint_foot_contact_mask": json.dumps(
                        endpoint_contacts.astype(int).tolist(), separators=(",", ":")
                    ),
                    "endpoint_tangential_speeds_m_per_s": json.dumps(
                        np.asarray(last_substep["speeds"], dtype=np.float64).tolist(),
                        separators=(",", ":"),
                    ),
                    "endpoint_normal_forces_n": json.dumps(
                        np.asarray(last_substep["forces"], dtype=np.float64).tolist(),
                        separators=(",", ":"),
                    ),
                    "endpoint_nonfoot_robot_ground": bool(endpoint_nonfoot),
                    "endpoint_torso_ground": bool(endpoint_torso),
                    "terrain_relative_torso_tilt_rad": float(relative_tilt),
                    "cumulative_squared_action": squared_action,
                    "cumulative_positive_mechanical_work_j": positive_work,
                    "applied_action": json.dumps(
                        np.asarray(info.get("proxygap_applied_action", action)).tolist(),
                        separators=(",", ":"),
                    ),
                    "reward": float(reward_value),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            last_frame = current_frame(
                endpoint_support=int(np.sum(endpoint_contacts)),
                squared_action=squared_action,
                positive_work=positive_work,
            )
            encode_frame(stream, container, last_frame)
            if step in {200, 400, 600}:
                contact_frames.append((f"t = {step * 0.05:.0f} s", last_frame.copy()))
            if step > 600:
                raise RuntimeError("Rendered replay exceeded the frozen horizon")

        if step != 600 or len(trace_rows) != 600:
            raise RuntimeError("Rendered replay did not retain exactly 600 control steps")
        if last_frame is None:
            raise RuntimeError("Rendered replay produced no final frame")
        for _ in range(outro_frames):
            encode_frame(stream, container, last_frame)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        corrected = tracker.finalise()
        final_summary = env.episode_summary()
        replay_row = replay_row_from_render(
            model=model,
            formal_config=formal_config,
            scene=scene,
            seed=seed,
            control_step=step,
            finite=finite,
            contact_rows=contact_rows,
            force_qualified_support_rows=force_supported_rows,
            nonfoot_rows=nonfoot_rows,
            torso_rows=torso_rows,
            control_fullzero=control_fullzero,
            corrected=corrected,
            summary=final_summary,
        )
        render_comparison = compare_episode_rows(formal_row, replay_row)
        write_rows(trace_path, trace_rows)
        write_json(metric_path, replay_row)
        write_json(
            comparison_path,
            {
                "pre_render_frozen_evaluator": pre_render_comparison,
                "rendered_replay": render_comparison,
            },
        )
        last_frame.save(final_frame_path, format="PNG", optimize=True)
        make_contact_sheet(
            contact_frames,
            contact_sheet_path,
            title=f"PAIR0 standard-slope diagnostic | {scene_name} | seed {seed}",
        )
    finally:
        try:
            if container is not None and not getattr(container, "closed", False):
                container.close()
        except Exception:
            pass
        renderer.close()
        env.close()

    total_frames = (
        round(float(config["render"]["intro_seconds"]) * FPS)
        + 600
        + round(float(config["render"]["outro_seconds"]) * FPS)
    )
    qa = validate_video_exhaustive(
        video_path,
        expected_width=FRAME_WIDTH,
        expected_height=FRAME_HEIGHT,
        expected_fps=FPS,
    )
    if int(qa["decoded_frames"]) != total_frames:
        raise RuntimeError("Encoded and fully decoded frame counts differ")
    if float(qa["average_frame_rate"]) != FPS:
        raise RuntimeError("Decoded frame rate changed")
    if float(qa["decoded_duration_seconds"]) < float(
        config["render"]["minimum_decoded_duration_seconds"]
    ):
        raise RuntimeError("Delivery video is shorter than ten seconds")

    selection = next(
        item
        for item in config["representative_episodes"]
        if item["scene_name"] == scene_name
    )
    episode_manifest = {
        "schema_version": "proxygap-pair0-standard-slope-delivery-episode-v1",
        "status": "complete_fieldwise_exact_visual_replay",
        "claim_label": config["render"]["claim_label"],
        "condition_id": l2.PAIR0_ID,
        "scene_name": scene_name,
        "signed_slope_degrees": signed_slope_degrees,
        "evaluation_seed": seed,
        "selection_rule": selection["selection_rule"],
        "selection_note": selection["selection_note"],
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "formal_episode_row": formal_row,
        "fieldwise_metric_match": True,
        "field_count_matched": len(formal_row),
        "trace": {
            "path": str(trace_path),
            "sha256": sha256(trace_path),
            "control_rows": len(trace_rows),
        },
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "width_px": FRAME_WIDTH,
            "height_px": FRAME_HEIGHT,
            "fps": FPS,
            "frames": total_frames,
            "duration_seconds": total_frames / FPS,
            "physical_rollout_seconds": 30.0,
            "playback_speed_relative_to_physical_time": 1.0,
        },
        "replay_episode_metrics": {
            "path": str(metric_path),
            "sha256": sha256(metric_path),
        },
        "fieldwise_metric_comparison": {
            "path": str(comparison_path),
            "sha256": sha256(comparison_path),
        },
        "final_frame": {
            "path": str(final_frame_path),
            "sha256": sha256(final_frame_path),
        },
        "contact_sheet": {
            "path": str(contact_sheet_path),
            "sha256": sha256(contact_sheet_path),
        },
        "scene": {
            "xml_sha256": scene["xml_sha256"],
            "heights_sha256": scene["heights_sha256"],
            "minimum_height_m": scene["minimum_height_m"],
            "maximum_height_m": scene["maximum_height_m"],
            "height_range_m": scene["height_range_m"],
            "friction": scene["fixed_friction"],
            "condim": scene["condim"],
            "vertical_exaggeration": 1.0,
        },
        "energy_status": "measurement_only_not_reward_or_gate",
        "qa": qa,
        "training_performed": False,
        "checkpoint_modified": False,
        "fixed_map_evaluated": False,
        "generalisation_claimed": False,
    }
    episode_manifest_path = episode_dir / f"{stem}_manifest.json"
    write_json(episode_manifest_path, episode_manifest)
    return {
        "scene_name": scene_name,
        "seed": seed,
        "manifest_path": str(episode_manifest_path),
        "manifest_sha256": sha256(episode_manifest_path),
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "final_frame_path": str(final_frame_path),
        "contact_sheet_path": str(contact_sheet_path),
        "qa": qa,
    }


def artifact_inventory(output_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def run(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    (
        protocol,
        reward,
        checkpoint,
        formal_config,
        formal_rows,
        scenes,
        runtime_before,
    ) = validate_config(config)
    output_root = ROOT / config["output"]["root"]
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite delivery root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    stage = "freeze_config"
    try:
        shutil.copy2(config_path, output_root / "frozen_config.json")
        if sha256(output_root / "frozen_config.json") != sha256(config_path):
            raise RuntimeError("Frozen delivery configuration changed while copying")
        stage = "snapshot_renderer"
        snapshot_dir = output_root / "runtime_snapshot" / "scripts"
        snapshot_dir.mkdir(parents=True)
        renderer_source = Path(__file__).resolve()
        shutil.copy2(renderer_source, snapshot_dir / renderer_source.name)
        renderer_hash = sha256(renderer_source)
        if sha256(snapshot_dir / renderer_source.name) != renderer_hash:
            raise RuntimeError("Renderer snapshot changed")
        stage = "load_checkpoint_read_only"
        checkpoint_hash_before = sha256(checkpoint)
        formal_runtime_before = slope_eval.validate_runtime_dependencies(formal_config)
        model = PPO.load(checkpoint, device="cpu")
        if int(model.num_timesteps) != 2_727_936:
            raise RuntimeError("Checkpoint timestep metadata changed")
        results: list[dict[str, Any]] = []
        stage = "render_uphill_12deg"
        for scene_name, slope_degrees, seed in EXPECTED_EPISODES:
            stage = f"render_{scene_name}_seed_{seed}"
            results.append(
                render_episode(
                    config=config,
                    formal_config=formal_config,
                    protocol=protocol,
                    reward=reward,
                    model=model,
                    checkpoint=checkpoint,
                    formal_rows=formal_rows,
                    scene=scenes[scene_name],
                    signed_slope_degrees=slope_degrees,
                    seed=seed,
                    output_root=output_root,
                )
            )
        stage = "post_render_provenance"
        runtime_after: dict[str, str] = {}
        for relative_path, expected_hash in config["runtime_dependencies"].items():
            observed = sha256(ROOT / relative_path)
            if observed != expected_hash:
                raise RuntimeError(f"Runtime changed during rendering: {relative_path}")
            runtime_after[relative_path] = observed
        if runtime_after != runtime_before:
            raise RuntimeError("Runtime dependency map changed during rendering")
        formal_runtime_after = slope_eval.validate_runtime_dependencies(formal_config)
        if formal_runtime_after != formal_runtime_before:
            raise RuntimeError("Formal slope runtime closure changed during rendering")
        if sha256(checkpoint) != checkpoint_hash_before:
            raise RuntimeError("Checkpoint changed during read-only rendering")
        stage = "write_report"
        report_path = output_root / "REPORT.md"
        report_path.write_text(
            "# Final PAIR0 standard-slope diagnostic videos\n\n"
            "Two deterministic, read-only visual replays were rendered from the frozen "
            "2,727,936-step PAIR0 checkpoint. Each rendered replay reproduced all 28 "
            "fields of its archived formal episode row exactly.\n\n"
            "- uphill 12 degrees: held-out seed 94153, selected by exact scene-median progress; 33/600 = 0.055 zero-foot intervals, below the frozen 0.0580555556 limit.\n"
            "- downhill 16 degrees: held-out seed 94137, selected by exact scene-median progress.\n\n"
            "The videos are standard-slope diagnostics, not fixed-map arrival or random-map generalisation evidence. Energy is displayed as measurement-only mechanical/control proxies and is neither a reward nor battery-energy estimate.\n",
            encoding="utf-8",
        )
        stage = "write_manifest"
        manifest = {
            "schema_version": "proxygap-pair0-standard-slope-delivery-video-artifact-v1",
            "status": "complete_two_episode_fieldwise_exact_visual_delivery",
            "claim_boundary": config["claim_boundary"],
            "configuration": {
                "path": str(config_path),
                "sha256": sha256(config_path),
                "frozen_path": str(output_root / "frozen_config.json"),
                "frozen_sha256": sha256(output_root / "frozen_config.json"),
            },
            "renderer": {
                "path": str(renderer_source),
                "sha256": renderer_hash,
                "snapshot_path": str(snapshot_dir / renderer_source.name),
                "snapshot_sha256": sha256(snapshot_dir / renderer_source.name),
            },
            "source_formal_manifest_sha256": config["source"][
                "formal_manifest_sha256"
            ],
            "source_episode_metrics_sha256": config["source"][
                "formal_episode_metrics_sha256"
            ],
            "checkpoint": {
                "path": str(checkpoint),
                "sha256_before": checkpoint_hash_before,
                "sha256_after": sha256(checkpoint),
                "timesteps": int(model.num_timesteps),
            },
            "runtime_dependency_sha256_before": runtime_before,
            "runtime_dependency_sha256_after": runtime_after,
            "formal_runtime_dependency_sha256_before": formal_runtime_before,
            "formal_runtime_dependency_sha256_after": formal_runtime_after,
            "episodes": results,
            "all_episode_rows_fieldwise_exact": True,
            "all_videos_fully_decoded": True,
            "training_performed": False,
            "checkpoint_modified": False,
            "reward_changed": False,
            "friction_changed": False,
            "energy_formula_changed": False,
            "energy_status": "measurement_only_not_reward_or_gate",
            "fixed_map_evaluated": False,
            "generalisation_claimed": False,
            "candidate_promoted": False,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_inventory_excludes_root_manifest": artifact_inventory(output_root),
        }
        write_json(output_root / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        failure = {
            "schema_version": "proxygap-pair0-standard-slope-delivery-video-failure-v1",
            "status": "failed_closed_do_not_deliver",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "training_performed": False,
            "checkpoint_write_performed": False,
            "retry_permitted_without_protocol_review": False,
        }
        write_json(output_root / "FAILURE_RECORD.json", failure)
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("Only the canonical delivery configuration may be executed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print("VALIDATION_OK")
        return
    manifest = run(config_path, config)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "episodes": manifest["episodes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
