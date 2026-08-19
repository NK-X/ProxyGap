"""Render auditable paired videos for the fixed standard-slope support pilot.

The renderer replays the predeclared paired seed and requires the regenerated
trace CSV to be byte-identical to the archived evaluation trace.  Both panes
share one MuJoCo state.  Cameras, trail and text are visual-only.
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
    sha256,
    validate_video,
)
from run_fixed_standard_support_curriculum import (  # noqa: E402
    FOOT_NAMES,
    contact_masks_from_data,
    make_standard_env,
    quaternion_tilt_relative_to_normal,
    reward_config_with_contact_gap_weight,
    write_rows,
)


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIEW_WIDTH = FRAME_WIDTH // 2
VIEW_HEIGHT = 540
FPS = 20
DEFAULT_PILOT = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_standard_support_curriculum_v1_20260819"
    / "paired_bound6_seed_62804"
)
CONDITION_WEIGHTS = {
    "MATCHED_CONTACT_GAP_W0_CONTROL": 0.0,
    "CONTACT_GAP_W1_INTERVENTION": 1.0,
}
SCENE_SLOPES = {"uphill_8deg": 8.0, "downhill_8deg": -8.0}
CONDITION_SLUGS = {
    "MATCHED_CONTACT_GAP_W0_CONTROL": "w0",
    "CONTACT_GAP_W1_INTERVENTION": "w1",
}
SCENE_SLUGS = {"uphill_8deg": "up8", "downhill_8deg": "down8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--condition", choices=tuple(CONDITION_WEIGHTS), required=True)
    parser.add_argument("--scene", choices=tuple(SCENE_SLOPES), required=True)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--intro-seconds", type=float, default=1.5)
    parser.add_argument("--outro-seconds", type=float, default=1.5)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def side_overview_camera(*, height_midpoint: float) -> mujoco.MjvCamera:
    """Return a physical 1:1 side view exposing the signed x-slope."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (0.0, 0.0, float(height_midpoint + 0.15))
    camera.distance = 28.0
    camera.azimuth = 90.0
    camera.elevation = -22.0
    return camera


def follow_camera(*, position: np.ndarray, terrain_height: float) -> mujoco.MjvCamera:
    """Return a low three-quarter view for support and foot-contact inspection."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (
        float(position[0] + 0.45),
        float(position[1]),
        float(terrain_height + 0.38),
    )
    camera.distance = 4.8
    camera.azimuth = 132.0
    camera.elevation = -18.0
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
    condition_id: str,
    gate_passed: bool,
    scene_name: str,
    slope_degrees: float,
    evaluation_seed: int,
    physical_time: float,
    horizon_seconds: float,
    position: np.ndarray,
    support_count: int,
    airborne: bool,
    ever_airborne: bool,
    best_progress_m: float,
    distance_to_goal_m: float,
    relative_tilt_degrees: float,
    maximum_contact_speed: float,
    height_min_m: float,
    height_max_m: float,
    checkpoint_name: str,
    friction: list[float],
    map_hash: str,
    terminated: bool,
    truncated: bool,
) -> Image.Image:
    if left.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("left render shape changed")
    if right.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
        raise ValueError("right render shape changed")
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), BACKGROUND)
    image.paste(Image.fromarray(left, mode="RGB"), (0, 0))
    image.paste(Image.fromarray(right, mode="RGB"), (VIEW_WIDTH, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line((VIEW_WIDTH, 0, VIEW_WIDTH, VIEW_HEIGHT), fill=(255, 255, 255, 190), width=2)
    draw.text(
        (16, 14),
        "SUPPORT FOLLOW CAMERA",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    draw.text(
        (VIEW_WIDTH + 16, 14),
        "PHYSICAL SIDE RELIEF (1:1 Z)",
        font=font(12, bold=True),
        fill=WHITE,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )
    direction = "UPHILL ->" if slope_degrees > 0.0 else "DOWNHILL ->"
    draw.rounded_rectangle(
        (VIEW_WIDTH + 16, 44, VIEW_WIDTH + 298, 98),
        radius=7,
        fill=(9, 17, 23, 202),
        outline=(230, 238, 235, 110),
        width=1,
    )
    draw.text(
        (VIEW_WIDTH + 28, 51),
        f"{direction}  signed slope {slope_degrees:+.1f} deg",
        font=font(10, bold=True),
        fill=WHITE,
    )
    draw.text(
        (VIEW_WIDTH + 28, 74),
        f"z {height_min_m:+.2f} to {height_max_m:+.2f} m | relief {height_max_m-height_min_m:.2f} m",
        font=font(10),
        fill=(224, 235, 232),
    )

    draw.rectangle((0, VIEW_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT), fill=(250, 249, 245, 255))
    draw.line((0, VIEW_HEIGHT, FRAME_WIDTH, VIEW_HEIGHT), fill=(77, 88, 96, 210), width=2)
    if condition_id == "MATCHED_CONTACT_GAP_W0_CONTROL":
        decision_text = "MATCHED CONTINUATION CONTROL / NOT PROMOTED"
        decision_colour = AMBER
    else:
        decision_text = "REJECTED INTERVENTION / NOT PROMOTED"
        decision_colour = RED
    draw.text(
        (18, VIEW_HEIGHT + 12),
        f"{decision_text} | {condition_id}",
        font=font(13, bold=True),
        fill=decision_colour,
    )
    draw.text(
        (18, VIEW_HEIGHT + 40),
        f"scene {scene_name} | paired seed {evaluation_seed} | slope {slope_degrees:+.1f} deg | t {physical_time:05.2f}/{horizon_seconds:.2f} s",
        font=font(11, bold=True),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 68),
        f"checkpoint {checkpoint_name} | W1-vs-W0 gate {'PASS' if gate_passed else 'FAIL'} | promoted model NONE",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 94),
        f"fixed contact friction [{friction[0]:.1f}, {friction[1]:.1f}, {friction[2]:.1f}] | condim 3 | map {map_hash[:12]}",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (18, VIEW_HEIGHT + 122),
        "Cameras/trail only: physics, policy, slope and vertical scale unchanged",
        font=font(10, bold=True),
        fill=(69, 76, 82),
    )

    x = 650
    draw.text(
        (x, VIEW_HEIGHT + 12),
        f"Progress best {best_progress_m:5.2f} m | distance {distance_to_goal_m:5.2f} m",
        font=font(13, bold=True),
        fill=INK,
    )
    draw.text(
        (x, VIEW_HEIGHT + 40),
        f"position ({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) m | terrain-relative tilt {relative_tilt_degrees:4.1f} deg",
        font=font(10),
        fill=INK,
    )
    draw.text(
        (x, VIEW_HEIGHT + 68),
        f"support {support_count}/4 | airborne now {int(airborne)} | any airborne {int(ever_airborne)} | contact max {maximum_contact_speed:.2f} m/s",
        font=font(10),
        fill=INK,
    )
    if terminated:
        status = "TERMINATED"
        colour = RED
    elif truncated:
        status = "TIME LIMIT"
        colour = AMBER
    elif airborne:
        status = "FOUR-FOOT AIRBORNE"
        colour = RED
    else:
        status = "ROLLOUT IN PROGRESS"
        colour = TEAL
    draw.rounded_rectangle(
        (x, VIEW_HEIGHT + 99, FRAME_WIDTH - 18, FRAME_HEIGHT - 13),
        radius=7,
        fill=(*colour, 238),
    )
    draw.text((x + 14, VIEW_HEIGHT + 111), status, font=font(12, bold=True), fill=WHITE)
    return image


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if args.render_stride <= 0:
        raise ValueError("render-stride must be positive")
    pilot_root = args.pilot_root.expanduser().resolve()
    manifest_path = pilot_root / "manifest.json"
    comparison_path = pilot_root / "comparison_summary.json"
    config_path = pilot_root / "frozen_run_config.json"
    scene_manifest_path = pilot_root / "standard_scene_manifest.json"
    for path in (manifest_path, comparison_path, config_path, scene_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    representative_seed = int(manifest["representative_video_inputs"]["trace_seed"])
    seed = representative_seed if args.evaluation_seed is None else int(args.evaluation_seed)
    if seed != representative_seed:
        raise ValueError("only the predeclared representative trace seed may be rendered")
    trace_record = next(
        item
        for item in manifest["representative_video_inputs"]["traces"]
        if item["condition_id"] == args.condition
        and item["scene_name"] == args.scene
        and int(item["evaluation_seed"]) == seed
    )
    archived_trace = Path(trace_record["path"])
    if sha256(archived_trace) != trace_record["sha256"]:
        raise ValueError("archived representative trace hash changed")
    checkpoint_record = next(
        item for item in manifest["checkpoints"] if item["condition_id"] == args.condition
    )
    checkpoint = Path(checkpoint_record["path"])
    if sha256(checkpoint) != checkpoint_record["sha256"]:
        raise ValueError("checkpoint hash changed")
    scene = scene_manifest["scenes"][args.scene]
    for key in ("xml", "heights"):
        path = Path(scene[f"{key}_path"])
        if sha256(path) != scene[f"{key}_sha256"]:
            raise ValueError(f"standard-scene {key} hash changed")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pilot_root
        / "videos"
        / CONDITION_SLUGS[args.condition]
        / f"{SCENE_SLUGS[args.scene]}_s{seed}_v2"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{CONDITION_SLUGS[args.condition]}_{SCENE_SLUGS[args.scene]}_s{seed}_v2"
    video_path = output_dir / f"{stem}.mp4"
    replay_trace_path = output_dir / f"{stem}_trace.csv"

    base_reward_path = ROOT / config["frozen_sources"]["reward_configuration"]
    if sha256(base_reward_path) != config["frozen_sources"]["reward_configuration_sha256"]:
        raise ValueError("reward configuration changed")
    base_reward = json.loads(base_reward_path.read_text(encoding="utf-8"))
    reward_config = reward_config_with_contact_gap_weight(
        base_reward, CONDITION_WEIGHTS[args.condition]
    )
    max_steps = int(config["paired_evaluation"]["max_episode_steps"])
    cruise_speed = float(config["paired_evaluation"]["cruise_speed_m_per_s"])
    env = make_standard_env(
        config,
        reward_config,
        scene,
        condition_id=args.condition,
        seed=seed,
        max_episode_steps=max_steps,
        cruise_speed=cruise_speed,
    )
    observation, _ = env.reset(seed=seed)
    model = PPO.load(checkpoint, device="cpu")
    if tuple(model.observation_space.shape) != tuple(observation.shape):
        env.close()
        raise RuntimeError("checkpoint observation interface changed")
    compiled = env.unwrapped.model
    compiled.vis.global_.offwidth = VIEW_WIDTH
    compiled.vis.global_.offheight = VIEW_HEIGHT
    renderer = mujoco.Renderer(compiled, height=VIEW_HEIGHT, width=VIEW_WIDTH)
    heights = np.load(scene["heights_path"], allow_pickle=False)
    overview = side_overview_camera(
        height_midpoint=float((np.min(heights) + np.max(heights)) / 2.0)
    )

    import av

    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = FRAME_WIDTH
    stream.height = FRAME_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}
    stream.gop_size = FPS * 2

    start = np.asarray(scene["start_xy_m"], dtype=np.float64)
    goal = np.asarray(scene["goal_xy_m"], dtype=np.float64)
    initial_distance = float(np.linalg.norm(goal - start))
    dt = float(env.unwrapped.dt)
    horizon_seconds = max_steps * dt
    trace_rows: list[dict[str, Any]] = []
    trail_xyz: list[np.ndarray] = []
    best_progress = 0.0
    ever_airborne = False
    last_frame: Image.Image | None = None
    rendered = 0
    gate_passed = bool(comparison["retention_gate"]["passed"])
    slope = SCENE_SLOPES[args.scene]
    foot_ids = tuple(
        int(mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in FOOT_NAMES
    )

    def build_frame(
        *,
        step: int,
        support_count: int,
        airborne: bool,
        distance: float,
        relative_tilt: float,
        maximum_contact_speed: float,
        terminated: bool,
        truncated: bool,
    ) -> Image.Image:
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        x, y = float(qpos[0]), float(qpos[1])
        terrain_height = float(env._terrain_height(x, y))
        follow = follow_camera(position=qpos[:2], terrain_height=terrain_height)
        left, right = render_pair(
            renderer,
            data=env.unwrapped.data,
            follow=follow,
            overview=overview,
            trail_xyz=trail_xyz,
        )
        return compose_frame(
            left,
            right,
            condition_id=args.condition,
            gate_passed=gate_passed,
            scene_name=args.scene,
            slope_degrees=slope,
            evaluation_seed=seed,
            physical_time=step * dt,
            horizon_seconds=horizon_seconds,
            position=qpos[:3],
            support_count=support_count,
            airborne=airborne,
            ever_airborne=ever_airborne,
            best_progress_m=best_progress,
            distance_to_goal_m=distance,
            relative_tilt_degrees=math.degrees(relative_tilt),
            maximum_contact_speed=maximum_contact_speed,
            height_min_m=float(np.min(heights)),
            height_max_m=float(np.max(heights)),
            checkpoint_name=checkpoint.name,
            friction=list(scene["fixed_friction"]),
            map_hash=scene["heights_sha256"],
            terminated=terminated,
            truncated=truncated,
        )

    initial_qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    initial_height = float(env._terrain_height(float(initial_qpos[0]), float(initial_qpos[1])))
    trail_xyz.append(np.asarray((initial_qpos[0], initial_qpos[1], initial_height + 0.05)))
    initial_normal = env._terrain_normal(float(initial_qpos[0]), float(initial_qpos[1]))
    initial_tilt = quaternion_tilt_relative_to_normal(initial_qpos[3:7], initial_normal)
    last_frame = build_frame(
        step=0,
        support_count=0,
        airborne=False,
        distance=initial_distance,
        relative_tilt=initial_tilt,
        maximum_contact_speed=0.0,
        terminated=False,
        truncated=False,
    )
    intro_frames = round(args.intro_seconds * FPS)
    outro_frames = round(args.outro_seconds * FPS)
    for _ in range(intro_frames):
        encode_frame(stream, container, last_frame)

    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        x, y = float(qpos[0]), float(qpos[1])
        terrain_height = float(env._terrain_height(x, y))
        trail_xyz.append(np.asarray((x, y, terrain_height + 0.05)))
        distance = float(np.linalg.norm(goal - qpos[:2]))
        best_progress = max(best_progress, initial_distance - distance)
        contact_mask = np.asarray(
            info.get("proxygap_foot_contact_mask_step", np.zeros(4)), dtype=bool
        )
        contact_speeds = np.asarray(
            info.get("proxygap_foot_contact_tangential_speeds_m_per_s_step", np.zeros(4)),
            dtype=np.float64,
        )
        active_speeds = contact_speeds[contact_mask]
        maximum_contact_speed = float(active_speeds.max()) if active_speeds.size else 0.0
        support_count = int(contact_mask.sum())
        airborne = bool(not np.any(contact_mask))
        ever_airborne = ever_airborne or airborne
        normal = env._terrain_normal(x, y)
        relative_tilt = quaternion_tilt_relative_to_normal(qpos[3:7], normal)
        endpoint_feet, endpoint_nonfoot, endpoint_torso = contact_masks_from_data(
            compiled, env.unwrapped.data, foot_ids
        )
        if not np.array_equal(endpoint_feet, contact_mask):
            raise RuntimeError("independent contact mask disagrees during video replay")
        trace_rows.append(
            {
                "condition_id": args.condition,
                "scene_name": args.scene,
                "evaluation_seed": seed,
                "step": step,
                "time_seconds": step * dt,
                "x_m": x,
                "y_m": y,
                "terrain_height_m": terrain_height,
                "torso_z_m": float(qpos[2]),
                "distance_to_goal_m": distance,
                "support_count": support_count,
                "foot_contact_mask": json.dumps(contact_mask.astype(int).tolist()),
                "airborne_endpoint": airborne,
                "relative_torso_tilt_rad": relative_tilt,
                "maximum_contact_tangential_speed_m_per_s": maximum_contact_speed,
                "contact_speed_threshold_exceeded": bool(
                    maximum_contact_speed
                    > float(config["task_adapter"]["slip_speed_threshold_m_per_s"])
                ),
                "endpoint_nonfoot_robot_ground": endpoint_nonfoot,
                "endpoint_torso_ground": endpoint_torso,
                "applied_action": json.dumps(
                    np.asarray(info.get("proxygap_applied_action", action)).tolist(),
                    separators=(",", ":"),
                ),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if step == 1 or step % args.render_stride == 0 or terminated or truncated:
            last_frame = build_frame(
                step=step,
                support_count=support_count,
                airborne=airborne,
                distance=distance,
                relative_tilt=relative_tilt,
                maximum_contact_speed=maximum_contact_speed,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            encode_frame(stream, container, last_frame)
            rendered += 1

    if last_frame is None:
        raise RuntimeError("video replay produced no frame")
    for _ in range(outro_frames):
        encode_frame(stream, container, last_frame)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    renderer.close()
    env.close()

    write_rows(replay_trace_path, trace_rows)
    replay_hash = sha256(replay_trace_path)
    if replay_hash != trace_record["sha256"]:
        raise RuntimeError("video replay trace is not byte-identical to paired evaluation")
    final_frame_path = output_dir / f"{stem}_final_frame.png"
    last_frame.save(final_frame_path, format="PNG", optimize=True)
    total_frames = intro_frames + rendered + outro_frames
    qa = validate_video(video_path, expected_width=FRAME_WIDTH, expected_height=FRAME_HEIGHT)
    if qa["decoded_frames"] != total_frames:
        raise RuntimeError("encoded and decoded frame counts differ")
    if qa["decoded_duration_seconds"] < 10.0:
        raise RuntimeError("standard-support video is shorter than 10 seconds")
    diagnosis_path = pilot_root / "diagnosis_summary.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    paired_rows = load_csv_rows(pilot_root / "logs" / "paired_evaluation_episodes.csv")
    condition_rows = [row for row in paired_rows if row["condition_id"] == args.condition]
    condition_success_count = sum(
        str(row.get("fixed_goal_success", "")).strip().lower() in {"1", "true"}
        for row in condition_rows
    )
    source_progress = float(diagnosis["aggregate"]["fixed_goal_best_progress_m_mean"])
    condition_progress = float(
        comparison["condition_aggregates"][args.condition][
            "fixed_goal_best_progress_m_mean"
        ]
    )
    decision = (
        "matched_continuation_control_not_promoted"
        if args.condition == "MATCHED_CONTACT_GAP_W0_CONTROL"
        else "rejected_intervention_not_promoted"
    )
    video_manifest = {
        "schema_version": "proxygap-standard-support-video-v2",
        "selection_rule": manifest["representative_video_inputs"]["selection_rule"],
        "condition_id": args.condition,
        "condition_decision": decision,
        "retention_gate_passed": gate_passed,
        "local_w1_vs_w0_gate_reference": comparison["retention_gate"][
            "retained_condition"
        ],
        "local_gate_reference_is_not_promotion": True,
        "incumbent_condition": "SOURCE_STAGE1_STANDARD_DIAGNOSIS",
        "promoted_condition": None,
        "source_stage1_mean_best_progress_m": source_progress,
        "condition_mean_best_progress_m": condition_progress,
        "condition_progress_ratio_to_source": condition_progress / source_progress,
        "condition_task_success_count": condition_success_count,
        "condition_task_episode_count": len(condition_rows),
        "scene_name": args.scene,
        "signed_slope_degrees": slope,
        "evaluation_seed": seed,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "archived_trace": {"path": str(archived_trace), "sha256": trace_record["sha256"]},
        "replay_trace": {"path": str(replay_trace_path), "sha256": replay_hash},
        "trace_byte_identical": True,
        "scene": {
            "xml": str(scene["xml_path"]),
            "xml_sha256": scene["xml_sha256"],
            "heights": str(scene["heights_path"]),
            "heights_sha256": scene["heights_sha256"],
            "minimum_height_m": scene["minimum_height_m"],
            "maximum_height_m": scene["maximum_height_m"],
            "fixed_friction": scene["fixed_friction"],
            "condim": scene["condim"],
        },
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FPS,
            "frames": total_frames,
            "duration_seconds": total_frames / FPS,
            "physical_rollout_seconds": step * dt,
            "render_stride": args.render_stride,
        },
        "final_frame": {"path": str(final_frame_path), "sha256": sha256(final_frame_path)},
        "camera_boundary": (
            "follow and fixed side cameras, trail and annotations are MjvScene/PIL only; "
            "physical terrain and z scale remain 1:1"
        ),
        "qa": qa,
    }
    manifest_out = output_dir / f"{stem}_video_manifest.json"
    manifest_out.write_text(
        json.dumps(video_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(video_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
