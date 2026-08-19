"""Render the frozen four-episode V5 turn-balance video archive.

This runner is read-only with respect to policy checkpoints and formal evidence.
Every rendered replay must reproduce its complete archived formal CSV row in the
same field order and with exactly equal values before any video is accepted.
"""

from __future__ import annotations

import argparse
import copy
import csv
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

import evaluate_fixed_standard_pair0_flat_turn_diagnostic as turn  # noqa: E402
import render_fixed_standard_pair0_slope_delivery_video as visual  # noqa: E402
import run_fixed_standard_pair0_turn_balance_continuation_v2 as v2  # noqa: E402
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


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_v5_video_archive_v1_20260819.json"
)
EXPECTED_EPISODES = (
    ("C0_STRAIGHT_CONTINUE", "curve_left_020", 96131),
    ("C0_STRAIGHT_CONTINUE", "curve_right_020", 96131),
    ("C1_BALANCED_TURN", "curve_left_020", 96131),
    ("C1_BALANCED_TURN", "curve_right_020", 96131),
)
EXPECTED_SEEDS = (96131, 96137, 96149, 96153, 96177)
BRANCH_DIRS = {
    "C0_STRAIGHT_CONTINUE": "c0_straight_continue",
    "C1_BALANCED_TURN": "c1_balanced_turn",
}
STRING_FIELDS = {
    "condition_id",
    "condition_name",
    "condition_kind",
    "turn_effectiveness_decision",
    "branch_id",
}
NULLABLE_FIELDS = {
    "terrain_relative_first_fall_control_step",
    "terrain_relative_fall_reason",
}
BOOL_FIELDS = {
    "finite",
    "fall",
    "terrain_relative_fall",
    "torso_ground_any",
    "sustained_nonfoot_contact",
    "yaw_change_same_sign_as_target",
    "out_of_training_command_envelope",
    "fixed_goal_success",
}
INT_FIELDS = {
    "evaluation_seed",
    "checkpoint_additional_timesteps",
    "checkpoint_timesteps",
    "control_steps",
    "physics_substeps",
    "terrain_relative_maximum_unhealthy_run_steps",
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen V5 video archive contract changed: {label}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def typed_formal_row(row: dict[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for field, value in row.items():
        if field in STRING_FIELDS:
            typed[field] = value
        elif field in NULLABLE_FIELDS:
            if value == "":
                typed[field] = None
            elif field.endswith("control_step"):
                typed[field] = int(value)
            else:
                typed[field] = value
        elif field in BOOL_FIELDS:
            lowered = value.strip().lower()
            if lowered not in {"true", "false"}:
                raise ValueError(f"Invalid formal boolean: {field}={value!r}")
            typed[field] = lowered == "true"
        elif field in INT_FIELDS:
            typed[field] = int(value)
        elif value == "":
            typed[field] = None
        else:
            typed[field] = float(value)
    return typed


def compare_episode_rows(
    archived: dict[str, Any], replayed: dict[str, Any]
) -> dict[str, Any]:
    if list(archived) != list(replayed):
        raise RuntimeError("Formal and replayed turn-row field order differs")
    mismatches: list[str] = []
    fields: dict[str, Any] = {}
    for key in archived:
        expected = archived[key]
        observed = replayed[key]
        exact = bool(observed == expected)
        row: dict[str, Any] = {
            "formal": expected,
            "replayed": observed,
            "exact_match": exact,
        }
        if isinstance(expected, float) and isinstance(observed, float):
            row["absolute_difference"] = abs(observed - expected)
        fields[key] = row
        if not exact:
            mismatches.append(key)
    result = {
        "field_count": len(archived),
        "field_order_exact": True,
        "all_fields_exact_match": not mismatches,
        "mismatched_fields": mismatches,
        "fields": fields,
    }
    if mismatches:
        raise RuntimeError(f"Rendered replay differs from formal CSV: {mismatches}")
    return result


def verify_path(
    record: dict[str, Any], path_key: str, hash_key: str
) -> tuple[Path, str]:
    path = ROOT / str(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = visual.sha256(path)
    require_equal(observed, str(record[hash_key]), hash_key)
    return path, observed


def formal_row_for(
    rows: list[dict[str, str]], branch_id: str, condition_name: str, seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["branch_id"] == branch_id
        and row["condition_name"] == condition_name
        and int(row["evaluation_seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Formal episode row is not unique: {branch_id}/{condition_name}/{seed}"
        )
    return typed_formal_row(matches[0])


def validate_formal_matrix(rows: list[dict[str, str]], branch_id: str) -> None:
    conditions = [item["condition_name"] for item in v2.expected_turn_conditions()]
    expected = {
        (condition, seed) for condition in conditions for seed in EXPECTED_SEEDS
    }
    observed = {
        (row["condition_name"], int(row["evaluation_seed"])) for row in rows
    }
    require_equal(len(rows), 45, f"{branch_id} row count")
    require_equal(observed, expected, f"{branch_id} condition-seed matrix")
    require_equal(len(observed), 45, f"{branch_id} unique matrix")
    for row in rows:
        require_equal(row["branch_id"], branch_id, f"{branch_id} row branch")
        require_equal(int(row["control_steps"]), 600, f"{branch_id} controls")
        require_equal(int(row["physics_substeps"]), 3000, f"{branch_id} substeps")
        require_equal(
            int(row["checkpoint_additional_timesteps"]),
            65_536,
            f"{branch_id} additional timesteps",
        )
        require_equal(
            int(row["checkpoint_timesteps"]),
            2_793_472,
            f"{branch_id} checkpoint timesteps",
        )


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    require_equal(
        config.get("schema_version"),
        "proxygap-pair0-turn-balance-v5-video-archive-v1",
        "schema version",
    )
    require_equal(
        config.get("config_id"),
        "fixed_standard_pair0_turn_balance_v5_video_archive_v1_20260819",
        "config id",
    )
    require_equal(
        config.get("status"),
        "frozen_read_only_post_gate_visual_archive",
        "status",
    )
    source = config["source"]
    verified_inputs: dict[str, str] = {}

    def source_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
        path, digest = verify_path(record, path_key, hash_key)
        verified_inputs[path.relative_to(ROOT).as_posix()] = digest
        return path

    manifest_path = source_file(source, "formal_manifest", "formal_manifest_sha256")
    final_gate_path = source_file(
        source, "formal_final_gate", "formal_final_gate_sha256"
    )
    hard_stop_path = source_file(
        source, "formal_hard_stop", "formal_hard_stop_sha256"
    )
    v2_config_path = source_file(
        source, "formal_frozen_v2_config", "formal_frozen_v2_config_sha256"
    )
    v1_config_path = source_file(
        source, "formal_v1_config", "formal_v1_config_sha256"
    )
    scene_path = source_file(source, "training_scene", "training_scene_sha256")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_gate = json.loads(final_gate_path.read_text(encoding="utf-8"))
    hard_stop = json.loads(hard_stop_path.read_text(encoding="utf-8"))
    require_equal(
        manifest["status"],
        "formal_final_optimisation_round_complete_hard_stopped",
        "formal status",
    )
    require_equal(manifest["mode"], "formal", "formal mode")
    require_equal(bool(manifest["scientifically_evaluable"]), True, "evaluability")
    require_equal(bool(manifest["hard_stop"]), True, "manifest hard stop")
    require_equal(bool(manifest["fixed_map_evaluated"]), False, "fixed-map evaluation")
    require_equal(bool(manifest["video_rendered"]), False, "pre-archive video state")
    require_equal(
        final_gate["decision"],
        "both_fail_turning_HOLD_retain_source_PAIR0",
        "final decision",
    )
    require_equal(final_gate["branch_pass"], {
        "C0_STRAIGHT_CONTINUE": False,
        "C1_BALANCED_TURN": False,
    }, "combined branch pass")
    require_equal(bool(final_gate["C1_passed_both_turn_and_slope"]), False, "C1 gate")
    require_equal(bool(final_gate["fixed_map_authorised"]), False, "fixed-map gate")
    require_equal(bool(hard_stop["hard_stop"]), True, "hard-stop marker")
    require_equal(
        bool(hard_stop["further_optimisation_authorised"]),
        False,
        "further optimisation",
    )

    v2_config = json.loads(v2_config_path.read_text(encoding="utf-8"))
    v1, protocol, reward, source_checkpoint = v2.validate_config(v2_config)
    require_equal(visual.sha256(v1_config_path), source["formal_v1_config_sha256"], "V1")
    require_equal(
        visual.sha256(source_checkpoint),
        manifest["source_checkpoint_sha256"],
        "source checkpoint",
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    require_equal(scene["condition_id"], turn.PAIR0_ID, "scene condition")
    require_equal(scene["scene_name"], "turn_balance_flat", "scene name")
    for key in ("xml", "heights", "hfield", "texture"):
        asset = Path(scene[f"{key}_path"])
        require_equal(visual.sha256(asset), scene[f"{key}_sha256"], f"scene {key}")
        verified_inputs[asset.relative_to(ROOT).as_posix()] = scene[f"{key}_sha256"]

    branches: dict[str, Any] = {}
    for branch_id in BRANCH_DIRS:
        row = source[branch_id]
        checkpoint = source_file(row, "checkpoint", "checkpoint_sha256")
        metrics = source_file(
            row, "turn_episode_metrics", "turn_episode_metrics_sha256"
        )
        branch_gate_path = source_file(row, "branch_gate", "branch_gate_sha256")
        rows = read_csv(metrics)
        validate_formal_matrix(rows, branch_id)
        branch_gate = json.loads(branch_gate_path.read_text(encoding="utf-8"))
        require_equal(bool(branch_gate["turn_gate"]["passed"]), False, f"{branch_id} turn")
        require_equal(bool(branch_gate["slope_gate"]["passed"]), True, f"{branch_id} slope")
        require_equal(
            branch_gate["final_checkpoint_sha256"],
            row["checkpoint_sha256"],
            f"{branch_id} checkpoint gate",
        )
        branches[branch_id] = {
            "checkpoint": checkpoint,
            "metrics": metrics,
            "rows": rows,
            "branch_gate": branch_gate,
        }

    observed_episodes = tuple(
        (
            str(row["branch_id"]),
            str(row["condition_name"]),
            int(row["evaluation_seed"]),
        )
        for row in config["episodes"]
    )
    require_equal(observed_episodes, EXPECTED_EPISODES, "four episode selection")
    for branch_id, condition_name, seed in EXPECTED_EPISODES:
        formal_row_for(branches[branch_id]["rows"], branch_id, condition_name, seed)

    labels = config["frozen_gate_labels"]
    require_equal(labels["turn_gate"], "FAIL", "turn label")
    require_equal(labels["slope_continuity_gate"], "PASS", "slope label")
    require_equal(labels["fixed_map"], "NOT AUTHORISED", "fixed-map label")
    require_equal(labels["stage_B"], "HOLD", "Stage B label")
    require_equal(labels["final_decision"], final_gate["decision"], "decision label")

    replay = config["replay"]
    for key, expected in (
        ("max_episode_steps", 600),
        ("physics_substeps_per_control_step", 5),
    ):
        require_equal(int(replay[key]), expected, f"replay {key}")
    for key, expected in (
        ("physics_timestep_seconds", 0.01),
        ("control_timestep_seconds", 0.05),
    ):
        require_equal(float(replay[key]), expected, f"replay {key}")
    for key in (
        "deterministic_policy",
        "require_formal_csv_field_order_and_value_exact_match",
        "require_all_600_trace_rows",
        "require_all_3000_physics_substeps",
    ):
        require_equal(bool(replay[key]), True, f"replay {key}")
    for key in (
        "training",
        "checkpoint_write",
        "reward_change",
        "friction_change",
        "energy_formula_change",
        "checkpoint_or_seed_selection",
    ):
        require_equal(bool(replay[key]), False, f"replay {key}")

    render = config["render"]
    for key, expected in (
        ("width_px", visual.FRAME_WIDTH),
        ("height_px", visual.FRAME_HEIGHT),
        ("view_height_px", visual.VIEW_HEIGHT),
        ("fps", visual.FPS),
    ):
        require_equal(int(render[key]), expected, f"render {key}")
    require_equal(bool(render["render_every_control_step"]), True, "render cadence")
    require_equal(
        render["mandatory_label"],
        "TURN GATE: FAIL | SLOPE CONTINUITY: PASS | FIXED-MAP: NOT AUTHORISED",
        "mandatory visual label",
    )
    output = config["output"]
    require_equal(output["root"], "artifacts/dev/tb_v5_video_archive_20260819/a0", "root")
    require_equal(bool(output["fail_if_exists"]), True, "fresh root")
    require_equal(bool(output["retry_same_root"]), False, "no retry")

    runtime: dict[str, str] = {}
    for relative, expected in config["runtime_dependencies"].items():
        path = ROOT / relative
        observed = visual.sha256(path)
        require_equal(observed, expected, f"runtime {relative}")
        runtime[relative] = observed
    return {
        "v2_config": v2_config,
        "v1": v1,
        "protocol": protocol,
        "reward": reward,
        "scene": scene,
        "branches": branches,
        "runtime": runtime,
        "verified_inputs": verified_inputs,
        "final_gate": final_gate,
        "hard_stop": hard_stop,
    }


def follow_camera(position: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (float(position[0]), float(position[1]), float(position[2] - 0.12))
    camera.distance = 5.5
    camera.azimuth = 135.0
    camera.elevation = -28.0
    return camera


def turn_overview_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (3.0, 0.0, 0.0)
    camera.distance = 24.0
    camera.azimuth = 90.0
    camera.elevation = -72.0
    return camera


def compose_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    branch_id: str,
    condition: dict[str, Any],
    seed: int,
    step: int,
    formal_row: dict[str, Any],
    actual_yaw_change: float,
    current_support: int,
    cumulative_zero_foot: int,
    squared_action: float,
    positive_work_j: float,
) -> Image.Image:
    if left.shape != (visual.VIEW_HEIGHT, visual.VIEW_WIDTH, 3):
        raise ValueError("Left render geometry changed")
    if right.shape != (visual.VIEW_HEIGHT, visual.VIEW_WIDTH, 3):
        raise ValueError("Right render geometry changed")
    image = Image.new("RGB", (visual.FRAME_WIDTH, visual.FRAME_HEIGHT), BACKGROUND)
    image.paste(Image.fromarray(left, mode="RGB"), (0, 0))
    image.paste(Image.fromarray(right, mode="RGB"), (visual.VIEW_WIDTH, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line(
        (visual.VIEW_WIDTH, 0, visual.VIEW_WIDTH, visual.VIEW_HEIGHT),
        fill=(255, 255, 255, 210),
        width=2,
    )
    draw.text(
        (16, 14),
        "ADAPTIVE FOLLOW / ROBOT",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    draw.text(
        (visual.VIEW_WIDTH + 16, 14),
        "FIXED TURN TRAJECTORY / PHYSICAL SCALE",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    draw.rectangle(
        (0, visual.VIEW_HEIGHT, visual.FRAME_WIDTH, visual.FRAME_HEIGHT),
        fill=(250, 249, 245, 255),
    )
    draw.rectangle(
        (0, visual.VIEW_HEIGHT, visual.FRAME_WIDTH, visual.VIEW_HEIGHT + 34),
        fill=(*RED, 246),
    )
    draw.text(
        (18, visual.VIEW_HEIGHT + 8),
        "TURN GATE: FAIL | SLOPE CONTINUITY: PASS | FIXED-MAP: NOT AUTHORISED",
        font=font(12, bold=True),
        fill=WHITE,
    )
    target_yaw = float(condition["target_yaw_rate_rad_per_s"]) * step * 0.05
    draw.text(
        (18, visual.VIEW_HEIGHT + 44),
        f"{branch_id} | {condition['condition_name']} | seed {seed} | t {step*0.05:05.2f}/30.00 s",
        font=font(10, bold=True),
        fill=INK,
    )
    draw.text(
        (18, visual.VIEW_HEIGHT + 70),
        f"target curvature {condition['target_curvature_per_m']:+.2f} 1/m | yaw-rate {condition['target_yaw_rate_rad_per_s']:+.3f} rad/s | target yaw {target_yaw:+.3f} rad",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, visual.VIEW_HEIGHT + 96),
        f"actual yaw {actual_yaw_change:+.3f} rad | formal final {formal_row['actual_cumulative_yaw_change_rad']:+.3f} rad | ratio {formal_row['yaw_change_target_ratio']:+.3f}",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, visual.VIEW_HEIGHT + 122),
        f"support now {current_support}/4 | zero-foot intervals {cumulative_zero_foot}/{max(step, 1)} | formal final {formal_row['full_interval_zero_foot_count']}/600",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, visual.VIEW_HEIGHT + 148),
        f"ENERGY MEASUREMENT-ONLY: sum(a^2) {squared_action:.2f} | positive mechanical work {positive_work_j:.1f} J | not battery energy",
        font=font(10, bold=True),
        fill=(68, 77, 83),
    )
    draw.rounded_rectangle(
        (990, visual.VIEW_HEIGHT + 50, visual.FRAME_WIDTH - 18, visual.FRAME_HEIGHT - 20),
        radius=8,
        fill=(*AMBER, 235),
    )
    draw.text(
        (1012, visual.VIEW_HEIGHT + 72),
        "STAGE B: HOLD",
        font=font(12, bold=True),
        fill=WHITE,
    )
    draw.text(
        (1012, visual.VIEW_HEIGHT + 104),
        "READ-ONLY ARCHIVE",
        font=font(10, bold=True),
        fill=WHITE,
    )
    draw.text(
        (1012, visual.VIEW_HEIGHT + 132),
        "NO SELECTION",
        font=font(10, bold=True),
        fill=WHITE,
    )
    return image


def replay_row_from_render(
    *,
    model: PPO,
    adapter: dict[str, Any],
    branch_id: str,
    condition: dict[str, Any],
    seed: int,
    control_step: int,
    finite: bool,
    terrain_health: dict[str, Any],
    actual_yaw_change: float,
    yaw_rate_error_squared_sum: float,
    path_length: float,
    maximum_displacement: float,
    maximum_reference_error: float,
    maximum_signed_progress: float,
    initial_xy: np.ndarray,
    initial_yaw: float,
    final_xy: np.ndarray,
    contacts: np.ndarray,
    force_support_rows: list[bool],
    nonfoot_rows: list[bool],
    torso_rows: list[bool],
    control_fullzero: list[bool],
    corrected: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    speed = float(condition["speed_m_per_s"])
    target_yaw_rate = float(condition["target_yaw_rate_rad_per_s"])
    target_curvature = float(condition["target_curvature_per_m"])
    control_dt = float(adapter["evaluation"]["control_timestep_seconds"])
    dt = float(adapter["evaluation"]["physics_timestep_seconds"])
    candidate = np.asarray(corrected["candidate"], dtype=bool)
    sustained = np.asarray(corrected["sustained"], dtype=bool)
    supported = np.any(contacts, axis=1)
    force_supported = np.asarray(force_support_rows, dtype=bool)
    target_yaw_change = target_yaw_rate * control_step * control_dt
    ratio = (
        actual_yaw_change / target_yaw_change
        if abs(target_yaw_change) > 1e-12
        else None
    )
    same_sign = (
        bool(actual_yaw_change * target_yaw_change > 0.0)
        if ratio is not None
        else None
    )
    actual_curvature = actual_yaw_change / path_length if path_length > 1e-12 else None
    final_reference = turn.reference_xy(
        initial_xy,
        initial_yaw,
        speed,
        target_yaw_rate,
        control_step * control_dt,
    )
    row = {
        "condition_id": turn.PAIR0_ID,
        "condition_name": condition["condition_name"],
        "condition_kind": condition["kind"],
        "evaluation_seed": seed,
        "checkpoint_additional_timesteps": 65_536,
        "checkpoint_timesteps": int(model.num_timesteps),
        "control_steps": control_step,
        "physics_substeps": int(contacts.shape[0]),
        "finite": finite,
        "fall": bool(
            summary.get("fall", False)
            or summary.get("inner_absolute_z_fall", False)
            or terrain_health["terrain_relative_fall"]
        ),
        "terrain_relative_fall": bool(terrain_health["terrain_relative_fall"]),
        "terrain_relative_first_fall_control_step": terrain_health[
            "first_fall_control_step"
        ],
        "terrain_relative_fall_reason": terrain_health["fall_reason"],
        "terrain_relative_maximum_unhealthy_run_steps": int(
            terrain_health["maximum_unhealthy_run_steps"]
        ),
        "terrain_relative_minimum_torso_clearance_m": float(
            terrain_health["minimum_torso_clearance_m"]
        ),
        "terrain_relative_maximum_torso_clearance_m": float(
            terrain_health["maximum_torso_clearance_m"]
        ),
        "terrain_relative_maximum_torso_tilt_rad": float(
            terrain_health["maximum_torso_tilt_rad"]
        ),
        "torso_ground_any": bool(np.any(torso_rows)),
        "nonfoot_ground_longest_run_seconds": turn._longest_true_run(nonfoot_rows)
        * dt,
        "sustained_nonfoot_contact": turn._longest_true_run(nonfoot_rows) * dt
        >= float(
            adapter["checkpoint_early_stopping"][
                "nonfoot_contact_minimum_sustained_seconds"
            ]
        ),
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
        "target_speed_m_per_s": speed,
        "target_yaw_rate_rad_per_s": target_yaw_rate,
        "target_curvature_per_m": target_curvature,
        "target_cumulative_yaw_change_rad": target_yaw_change,
        "actual_cumulative_yaw_change_rad": actual_yaw_change,
        "yaw_change_target_ratio": ratio,
        "yaw_change_same_sign_as_target": same_sign,
        "cumulative_yaw_error_rad": actual_yaw_change - target_yaw_change,
        "yaw_rate_rmse_rad_per_s": math.sqrt(
            yaw_rate_error_squared_sum / max(1, control_step)
        ),
        "actual_path_integrated_curvature_per_m": actual_curvature,
        "curvature_error_per_m": (
            actual_curvature - target_curvature
            if actual_curvature is not None
            else None
        ),
        "planar_path_length_m": path_length,
        "signed_initial_heading_progress_m": float(
            np.dot(
                final_xy - initial_xy,
                np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)]),
            )
        ),
        "maximum_signed_initial_heading_progress_m": maximum_signed_progress,
        "final_com_displacement_m": float(np.linalg.norm(final_xy - initial_xy)),
        "maximum_com_displacement_m": maximum_displacement,
        "final_com_reference_error_m": float(
            np.linalg.norm(final_xy - final_reference)
        ),
        "maximum_com_reference_error_m": maximum_reference_error,
        "out_of_training_command_envelope": bool(
            condition["out_of_training_command_envelope"]
        ),
        "turn_effectiveness_decision": "descriptive_only_no_pass_fail",
        "fixed_goal_success": False,
        "fixed_goal_best_progress_m": maximum_signed_progress,
        "fixed_goal_net_progress_m": float(
            np.dot(
                final_xy - initial_xy,
                np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)]),
            )
        ),
        "cumulative_squared_action": float(
            summary.get("cumulative_squared_action", 0.0)
        ),
        "actuator_abs_torque_time_integral_total_n_m_s": turn._vector_sum(
            summary, "actuator_abs_torque_time_integral_n_m_s_by_actuator"
        ),
        "actuator_positive_mechanical_work_total_j": turn._vector_sum(
            summary, "actuator_positive_mechanical_work_j_by_actuator"
        ),
        "actuator_abs_mechanical_work_total_j": turn._vector_sum(
            summary, "actuator_abs_mechanical_work_j_by_actuator"
        ),
        "branch_id": branch_id,
    }
    return row


def render_episode(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    branch_id: str,
    condition_name: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    branch = validated["branches"][branch_id]
    checkpoint: Path = branch["checkpoint"]
    formal_row = formal_row_for(branch["rows"], branch_id, condition_name, seed)
    model = PPO.load(checkpoint, device="cpu")
    require_equal(int(model.num_timesteps), 2_793_472, "loaded final timestep")
    condition = next(
        row for row in v2.expected_turn_conditions() if row["condition_name"] == condition_name
    )
    adapter = v2.turn_adapter(validated["v1"], validated["v2_config"])
    evaluator_row, _ = turn.evaluate_episode(
        model,
        adapter,
        validated["protocol"],
        validated["reward"],
        validated["scene"],
        condition,
        seed,
    )
    evaluator_row["branch_id"] = branch_id
    pre_render_comparison = compare_episode_rows(formal_row, evaluator_row)

    slug = f"{BRANCH_DIRS[branch_id]}_{condition_name}_seed_{seed}"
    episode_root = output_root / slug
    episode_root.mkdir(parents=True)
    stem = f"{slug}_turn_gate_fail_archive"
    video_path = episode_root / f"{stem}.mp4"
    trace_path = episode_root / f"{stem}_trace_600_steps.csv"
    metrics_path = episode_root / f"{stem}_replay_episode_metrics.json"
    comparison_path = episode_root / f"{stem}_fieldwise_comparison.json"
    final_frame_path = episode_root / f"{stem}_final_frame.png"
    contact_sheet_path = episode_root / f"{stem}_contact_sheet.png"

    local_protocol = copy.deepcopy(validated["protocol"])
    local_protocol["task_adapter"]["maximum_abs_curvature_per_m"] = float(
        adapter["evaluation"]["diagnostic_command_adapter_maximum_abs_curvature_per_m"]
    )
    speed = float(condition["speed_m_per_s"])
    target_yaw_rate = float(condition["target_yaw_rate_rad_per_s"])
    horizon = 600
    control_dt = 0.05
    env = turn.slope.l2.make_standard_env(
        local_protocol,
        validated["reward"],
        validated["scene"],
        condition_id=turn.PAIR0_ID,
        seed=seed,
        max_episode_steps=horizon,
        cruise_speed=speed,
    )
    turn.slope.l2.compiled_contract_audit(
        env.unwrapped.model,
        validated["scene"],
        turn.PAIR0_ID,
        adapter,
        construction_seed=seed,
    )
    reset_observation, _ = env.reset(seed=seed)
    audit_state = turn.slope.l2.install_five_substep_audit(env)
    initial_xy = turn.whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
    initial_yaw = turn.quaternion_yaw_angle(
        np.asarray(env.unwrapped.data.qpos[3:7])
    )
    observation = turn.commanded_observation(
        env,
        reset_observation[:122],
        target_heading=initial_yaw,
        yaw_rate=target_yaw_rate,
        speed=speed,
    )
    slip = adapter["evaluation"]["corrected_slip"]
    tracker = turn.slope.l2.DurationCorrectedSlipTracker(
        dt=0.01,
        speed_threshold=float(slip["tangential_speed_threshold_m_per_s"]),
        minimum_normal_force=float(slip["minimum_normal_force_n"]),
        landing_grace_seconds=float(slip["landing_grace_seconds"]),
        minimum_sustained_seconds=float(slip["minimum_sustained_seconds"]),
    )

    compiled = env.unwrapped.model
    compiled.vis.global_.offwidth = visual.VIEW_WIDTH
    compiled.vis.global_.offheight = visual.VIEW_HEIGHT
    renderer = mujoco.Renderer(
        compiled, height=visual.VIEW_HEIGHT, width=visual.VIEW_WIDTH
    )
    overview = turn_overview_camera()
    import av

    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=visual.FPS)
    stream.width = visual.FRAME_WIDTH
    stream.height = visual.FRAME_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(config["render"]["crf"]), "preset": "medium"}
    stream.gop_size = visual.FPS * 2

    contact_rows: list[np.ndarray] = []
    force_support_rows: list[bool] = []
    nonfoot_rows: list[bool] = []
    torso_rows: list[bool] = []
    control_fullzero: list[bool] = []
    trace_rows: list[dict[str, Any]] = []
    trail_xyz: list[np.ndarray] = []
    contact_frames: list[tuple[str, Image.Image]] = []
    actual_yaw_change = 0.0
    yaw_rate_error_squared_sum = 0.0
    previous_yaw = initial_yaw
    previous_xy = initial_xy.copy()
    path_length = 0.0
    maximum_displacement = 0.0
    maximum_reference_error = 0.0
    maximum_signed_progress = 0.0
    terrain_health = turn.new_terrain_health_audit()
    finite = True
    terminated = truncated = False
    step = 0
    last_frame: Image.Image | None = None

    def make_frame(current_support: int) -> Image.Image:
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        follow = follow_camera(qpos[:3])
        left, right = visual.render_pair(
            renderer,
            data=env.unwrapped.data,
            follow=follow,
            overview=overview,
            trail_xyz=trail_xyz,
        )
        summary_now = env.env.episode_summary()
        return compose_frame(
            left,
            right,
            branch_id=branch_id,
            condition=condition,
            seed=seed,
            step=step,
            formal_row=formal_row,
            actual_yaw_change=actual_yaw_change,
            current_support=current_support,
            cumulative_zero_foot=int(np.sum(control_fullzero)),
            squared_action=float(summary_now.get("cumulative_squared_action", 0.0)),
            positive_work_j=turn._vector_sum(
                summary_now, "actuator_positive_mechanical_work_j_by_actuator"
            ),
        )

    try:
        qpos0 = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        terrain0 = float(env._terrain_height(float(qpos0[0]), float(qpos0[1])))
        trail_xyz.append(np.asarray((qpos0[0], qpos0[1], terrain0 + 0.05)))
        last_frame = make_frame(0)
        contact_frames.append(("t = 0 s", last_frame.copy()))
        intro_frames = round(float(config["render"]["intro_seconds"]) * visual.FPS)
        outro_frames = round(float(config["render"]["outro_seconds"]) * visual.FPS)
        for _ in range(intro_frames):
            encode_frame(stream, container, last_frame)

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            raw_observation, reward_value, terminated, truncated, _ = env.env.step(
                action
            )
            step += 1
            rows = audit_state.get("last")
            if not isinstance(rows, list) or len(rows) != 5:
                raise RuntimeError("Rendered turn replay did not expose five substeps")
            local_contacts: list[np.ndarray] = []
            for substep in rows:
                contacts = np.asarray(substep["contacts"], dtype=bool)
                speeds = np.asarray(substep["speeds"], dtype=np.float64)
                forces = np.asarray(substep["forces"], dtype=np.float64)
                if contacts.shape != (4,) or speeds.shape != (4,) or forces.shape != (4,):
                    raise RuntimeError("Rendered substep foot-vector shape changed")
                tracker.update(
                    contact_mask=contacts,
                    tangential_speeds=speeds,
                    normal_forces=forces,
                )
                contact_rows.append(contacts.copy())
                force_support_rows.append(
                    bool(
                        np.any(
                            contacts
                            & (forces >= float(slip["minimum_normal_force_n"]))
                        )
                    )
                )
                nonfoot_rows.append(bool(substep["nonfoot"]))
                torso_rows.append(bool(substep["torso"]))
                local_contacts.append(contacts)
            control_fullzero.append(
                not np.any(np.asarray(local_contacts, dtype=bool))
            )
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
            xy = turn.whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
            yaw = turn.quaternion_yaw_angle(qpos[3:7])
            yaw_delta = turn.wrapped_angle_difference(yaw, previous_yaw)
            actual_yaw_change += yaw_delta
            yaw_rate_error_squared_sum += (
                yaw_delta / control_dt - target_yaw_rate
            ) ** 2
            path_length += float(np.linalg.norm(xy - previous_xy))
            displacement = float(np.linalg.norm(xy - initial_xy))
            maximum_displacement = max(maximum_displacement, displacement)
            elapsed = step * control_dt
            reference_error = float(
                np.linalg.norm(
                    xy
                    - turn.reference_xy(
                        initial_xy, initial_yaw, speed, target_yaw_rate, elapsed
                    )
                )
            )
            maximum_reference_error = max(maximum_reference_error, reference_error)
            signed_progress = float(
                np.dot(
                    xy - initial_xy,
                    np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)]),
                )
            )
            maximum_signed_progress = max(maximum_signed_progress, signed_progress)
            terrain_height = float(env._terrain_height(float(qpos[0]), float(qpos[1])))
            turn.update_terrain_health_audit(
                terrain_health,
                qpos=qpos,
                qvel=qvel,
                terrain_height_m=terrain_height,
                map_half_extent_m=float(env.map_half_extent_m),
                healthy_clearance_m=tuple(float(value) for value in env.healthy_clearance),
                maximum_healthy_tilt_rad=float(env.maximum_healthy_tilt),
                unhealthy_grace_steps=int(env.unhealthy_grace_steps),
                control_step=step,
            )
            finite = finite and bool(
                np.all(np.isfinite(raw_observation))
                and np.all(np.isfinite(action))
                and np.isfinite(reward_value)
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
            )
            previous_xy = xy
            previous_yaw = yaw
            if not (terminated or truncated):
                target_heading = initial_yaw + target_yaw_rate * elapsed
                observation = turn.commanded_observation(
                    env,
                    raw_observation,
                    target_heading=target_heading,
                    yaw_rate=target_yaw_rate,
                    speed=speed,
                )
            trail_xyz.append(np.asarray((xy[0], xy[1], terrain_height + 0.05)))
            trace_rows.append(
                {
                    "branch_id": branch_id,
                    "condition_name": condition_name,
                    "evaluation_seed": seed,
                    "control_step": step,
                    "time_seconds": elapsed,
                    "x_m": float(xy[0]),
                    "y_m": float(xy[1]),
                    "actual_cumulative_yaw_change_rad": actual_yaw_change,
                    "target_cumulative_yaw_change_rad": target_yaw_rate * elapsed,
                    "endpoint_support_count": int(np.sum(local_contacts[-1])),
                    "full_interval_zero_foot": bool(control_fullzero[-1]),
                    "cumulative_full_interval_zero_foot_count": int(
                        np.sum(control_fullzero)
                    ),
                    "five_substep_support_counts": json.dumps(
                        [int(np.sum(row)) for row in local_contacts], separators=(",", ":")
                    ),
                    "reward": float(reward_value),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            last_frame = make_frame(int(np.sum(local_contacts[-1])))
            encode_frame(stream, container, last_frame)
            if step in {200, 400, 600}:
                contact_frames.append((f"t = {step * 0.05:.0f} s", last_frame.copy()))
            if step > horizon:
                raise RuntimeError("Rendered replay exceeded frozen horizon")

        if step != 600 or len(trace_rows) != 600 or len(contact_rows) != 3000:
            raise RuntimeError("Rendered replay extent is not exactly 600/3000")
        if last_frame is None:
            raise RuntimeError("Rendered replay produced no final frame")
        for _ in range(outro_frames):
            encode_frame(stream, container, last_frame)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None

        corrected = tracker.finalise()
        contacts = np.asarray(contact_rows, dtype=bool)
        final_summary = env.env.episode_summary()
        final_xy = turn.whole_robot_com_xy(env.unwrapped.model, env.unwrapped.data)
        replay_row = replay_row_from_render(
            model=model,
            adapter=adapter,
            branch_id=branch_id,
            condition=condition,
            seed=seed,
            control_step=step,
            finite=finite,
            terrain_health=terrain_health,
            actual_yaw_change=actual_yaw_change,
            yaw_rate_error_squared_sum=yaw_rate_error_squared_sum,
            path_length=path_length,
            maximum_displacement=maximum_displacement,
            maximum_reference_error=maximum_reference_error,
            maximum_signed_progress=maximum_signed_progress,
            initial_xy=initial_xy,
            initial_yaw=initial_yaw,
            final_xy=final_xy,
            contacts=contacts,
            force_support_rows=force_support_rows,
            nonfoot_rows=nonfoot_rows,
            torso_rows=torso_rows,
            control_fullzero=control_fullzero,
            corrected=corrected,
            summary=final_summary,
        )
        rendered_comparison = compare_episode_rows(formal_row, replay_row)
        visual.write_rows(trace_path, trace_rows)
        visual.write_json(metrics_path, replay_row)
        visual.write_json(
            comparison_path,
            {
                "pre_render_frozen_evaluator": pre_render_comparison,
                "rendered_replay": rendered_comparison,
            },
        )
        last_frame.save(final_frame_path, format="PNG", optimize=True)
        visual.make_contact_sheet(
            contact_frames,
            contact_sheet_path,
            title=f"V5 turn archive | {branch_id} | {condition_name} | seed {seed}",
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
        round(float(config["render"]["intro_seconds"]) * visual.FPS)
        + 600
        + round(float(config["render"]["outro_seconds"]) * visual.FPS)
    )
    qa = visual.validate_video_exhaustive(
        video_path,
        expected_width=visual.FRAME_WIDTH,
        expected_height=visual.FRAME_HEIGHT,
        expected_fps=visual.FPS,
    )
    require_equal(int(qa["decoded_frames"]), total_frames, "decoded frame count")
    if float(qa["decoded_duration_seconds"]) < float(
        config["render"]["minimum_decoded_duration_seconds"]
    ):
        raise RuntimeError("Decoded archive video is too short")
    episode_manifest = {
        "schema_version": "proxygap-pair0-turn-balance-v5-video-episode-v1",
        "status": "complete_fieldwise_exact_read_only_visual_replay",
        "branch_id": branch_id,
        "condition_name": condition_name,
        "evaluation_seed": seed,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": visual.sha256(checkpoint),
            "timesteps": int(model.num_timesteps),
        },
        "frozen_gate_labels": config["frozen_gate_labels"],
        "formal_episode_row": formal_row,
        "field_order_and_all_values_exact": True,
        "formal_field_count": len(formal_row),
        "trace": {
            "path": str(trace_path),
            "sha256": visual.sha256(trace_path),
            "control_rows": 600,
            "physics_substeps_replayed": 3000,
        },
        "video": {
            "path": str(video_path),
            "sha256": visual.sha256(video_path),
            "width_px": visual.FRAME_WIDTH,
            "height_px": visual.FRAME_HEIGHT,
            "fps": visual.FPS,
            "frames": total_frames,
            "duration_seconds": total_frames / visual.FPS,
            "physical_rollout_seconds": 30.0,
        },
        "replay_metrics": {
            "path": str(metrics_path),
            "sha256": visual.sha256(metrics_path),
        },
        "fieldwise_comparison": {
            "path": str(comparison_path),
            "sha256": visual.sha256(comparison_path),
        },
        "final_frame": {
            "path": str(final_frame_path),
            "sha256": visual.sha256(final_frame_path),
        },
        "contact_sheet": {
            "path": str(contact_sheet_path),
            "sha256": visual.sha256(contact_sheet_path),
        },
        "qa": qa,
        "turn_gate_passed": False,
        "slope_continuity_gate_passed": True,
        "fixed_map_authorised": False,
        "training_performed": False,
        "checkpoint_modified": False,
        "video_participated_in_gate": False,
    }
    episode_manifest_path = episode_root / f"{stem}_manifest.json"
    visual.write_json(episode_manifest_path, episode_manifest)
    return {
        "branch_id": branch_id,
        "condition_name": condition_name,
        "evaluation_seed": seed,
        "video_path": str(video_path),
        "video_sha256": visual.sha256(video_path),
        "episode_manifest_path": str(episode_manifest_path),
        "episode_manifest_sha256": visual.sha256(episode_manifest_path),
        "final_frame_path": str(final_frame_path),
        "final_frame_sha256": visual.sha256(final_frame_path),
        "contact_sheet_path": str(contact_sheet_path),
        "contact_sheet_sha256": visual.sha256(contact_sheet_path),
        "qa": qa,
    }


def run(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    validated = validate_config(config)
    output_root = ROOT / config["output"]["root"]
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite unique archive root: {output_root}")
    output_root.mkdir(parents=True)
    stage = "freeze_configuration"
    started = time.perf_counter()
    checkpoint_before = {
        branch_id: visual.sha256(validated["branches"][branch_id]["checkpoint"])
        for branch_id in BRANCH_DIRS
    }
    inputs_before = dict(validated["verified_inputs"])
    runtime_before = dict(validated["runtime"])
    try:
        frozen = output_root / "frozen_config.json"
        shutil.copy2(config_path, frozen)
        require_equal(visual.sha256(frozen), visual.sha256(config_path), "frozen config")
        stage = "snapshot_renderer"
        snapshot = output_root / "runtime_snapshot" / "scripts" / Path(__file__).name
        snapshot.parent.mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve(), snapshot)
        renderer_hash = visual.sha256(Path(__file__).resolve())
        require_equal(visual.sha256(snapshot), renderer_hash, "renderer snapshot")

        results: list[dict[str, Any]] = []
        final_frames: list[tuple[str, Image.Image]] = []
        for branch_id, condition_name, seed in EXPECTED_EPISODES:
            stage = f"render_{branch_id}_{condition_name}_seed_{seed}"
            result = render_episode(
                config=config,
                validated=validated,
                branch_id=branch_id,
                condition_name=condition_name,
                seed=seed,
                output_root=output_root,
            )
            results.append(result)
            with Image.open(result["final_frame_path"]) as frame:
                final_frames.append(
                    (f"{branch_id.split('_')[0]} {condition_name.replace('curve_', '')}", frame.convert("RGB"))
                )

        stage = "root_contact_sheet"
        root_sheet = output_root / "V5_TURN_ARCHIVE_FOUR_EPISODE_CONTACT_SHEET.png"
        visual.make_contact_sheet(
            final_frames,
            root_sheet,
            title="V5 final turn archive | TURN FAIL | SLOPE PASS | FIXED-MAP NOT AUTHORISED",
        )
        stage = "post_render_immutability"
        validated_after = validate_config(config)
        require_equal(validated_after["verified_inputs"], inputs_before, "formal inputs after")
        require_equal(validated_after["runtime"], runtime_before, "runtime after")
        checkpoint_after = {
            branch_id: visual.sha256(validated["branches"][branch_id]["checkpoint"])
            for branch_id in BRANCH_DIRS
        }
        require_equal(checkpoint_after, checkpoint_before, "checkpoint before/after")
        stage = "write_report"
        report = output_root / "REPORT.md"
        report.write_text(
            "# V5 turn-balance read-only video archive\n\n"
            "Four deterministic visual replays were rendered after the numeric gates were frozen: "
            "C0 and C1 at seed 96131 for curve_left_020 and curve_right_020. Each replay "
            "reproduced all fields of its archived 600-control/3,000-substep formal CSV row exactly.\n\n"
            "Both final checkpoints passed standard-slope continuity and failed the turn gate. "
            "Consequently Stage B remains HOLD and fixed-map evaluation is not authorised. "
            "The videos are qualitative archives, not additional experiments or selection evidence.\n",
            encoding="utf-8",
        )
        stage = "write_manifest"
        manifest = {
            "schema_version": "proxygap-pair0-turn-balance-v5-video-archive-result-v1",
            "status": "complete_four_episode_fieldwise_exact_read_only_archive",
            "configuration": {
                "path": str(config_path),
                "sha256": visual.sha256(config_path),
                "frozen_path": str(frozen),
                "frozen_sha256": visual.sha256(frozen),
            },
            "renderer": {
                "path": str(Path(__file__).resolve()),
                "sha256": renderer_hash,
                "snapshot_path": str(snapshot),
                "snapshot_sha256": visual.sha256(snapshot),
            },
            "source_formal_manifest_sha256": config["source"]["formal_manifest_sha256"],
            "source_final_gate_sha256": config["source"]["formal_final_gate_sha256"],
            "frozen_gate_labels": config["frozen_gate_labels"],
            "episodes": results,
            "root_contact_sheet": {
                "path": str(root_sheet),
                "sha256": visual.sha256(root_sheet),
            },
            "all_four_formal_rows_field_order_and_values_exact": True,
            "all_four_videos_fully_decoded": True,
            "all_episode_extents_exact_600_controls_3000_substeps": True,
            "formal_inputs_sha256_before": inputs_before,
            "formal_inputs_sha256_after": validated_after["verified_inputs"],
            "runtime_sha256_before": runtime_before,
            "runtime_sha256_after": validated_after["runtime"],
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "training_performed": False,
            "checkpoint_modified": False,
            "reward_changed": False,
            "friction_changed": False,
            "energy_formula_changed": False,
            "video_participated_in_gate": False,
            "turn_gate_passed": False,
            "slope_continuity_gate_passed": True,
            "fixed_map_authorised": False,
            "stage_B": "HOLD",
            "claim_boundary": config["claim_boundary"],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_inventory_excludes_root_manifest": visual.artifact_inventory(
                output_root
            ),
        }
        visual.write_json(output_root / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        visual.write_json(
            output_root / "FAILURE_RECORD.json",
            {
                "schema_version": "proxygap-pair0-turn-balance-v5-video-archive-failure-v1",
                "status": "failed_closed_do_not_retry_same_root",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "training_performed": False,
                "checkpoint_write_performed": False,
                "retry_same_root_permitted": False,
                "scientific_result_changed": False,
            },
        )
        raise


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("Only the canonical V5 archive configuration is permitted")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validated = validate_config(config)
    if args.validate_only:
        output_root = ROOT / config["output"]["root"]
        print(
            json.dumps(
                {
                    "status": "VALIDATION_OK_READ_ONLY_ARCHIVE_NOT_STARTED",
                    "episodes": list(EXPECTED_EPISODES),
                    "formal_decision": validated["final_gate"]["decision"],
                    "stage_B": "HOLD",
                    "output_root_exists": output_root.exists(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    manifest = run(config_path, config)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "episode_count": len(manifest["episodes"]),
                "stage_B": manifest["stage_B"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
