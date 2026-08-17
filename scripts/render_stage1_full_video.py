"""Render one complete stage-one Ant trajectory at simulated real time.

The video is qualitative evidence only.  Candidate selection must use the
predeclared numerical screen before this script is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

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
except ModuleNotFoundError:  # Support module-style execution from the repository root.
    from scripts._portable_runtime import (
        FFMPEG_TARGET_ENV,
        LATIN_FONT_NAMES,
        iter_font_files,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap import make_proxygap_ant_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition_id", required=True)
    parser.add_argument("--ctrl_cost_weight", type=float, required=True)
    parser.add_argument("--orientation_shaping_weight", type=float, default=0.0)
    parser.add_argument("--orientation_shaping_scale", type=float, default=1.0)
    parser.add_argument("--lateral_drift_shaping_weight", type=float, default=0.0)
    parser.add_argument("--lateral_drift_shaping_scale", type=float, default=1.0)
    parser.add_argument(
        "--lateral_shaping_signal",
        choices=("offset_tanh", "velocity_tanh_squared"),
        default="offset_tanh",
    )
    parser.add_argument("--lateral_velocity_target", type=float, default=0.0)
    parser.add_argument(
        "--orientation_shaping_function",
        choices=("tanh", "cosine"),
        default="tanh",
    )
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--evaluation_seed", type=int, required=True)
    parser.add_argument("--target_timesteps", type=int, default=300_000)
    parser.add_argument("--max_steps", type=int, default=1_000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--augment_previous_applied_action", action="store_true")
    parser.add_argument("--action_slew_l2_limit", type=float)
    parser.add_argument("--replace_forward_reward_with_tracking", action="store_true")
    parser.add_argument("--forward_velocity_target", type=float, default=1.0)
    parser.add_argument("--forward_velocity_tracking_scale", type=float, default=0.5)
    parser.add_argument("--forward_velocity_tracking_weight", type=float, default=1.0)
    parser.add_argument("--action_rate_shaping_weight", type=float, default=0.0)
    parser.add_argument("--vertical_velocity_shaping_weight", type=float, default=0.0)
    parser.add_argument("--vertical_velocity_shaping_scale", type=float, default=1.0)
    parser.add_argument(
        "--roll_pitch_angular_velocity_shaping_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--roll_pitch_angular_velocity_shaping_scale", type=float, default=1.0
    )
    parser.add_argument("--foot_landing_height_threshold", type=float, default=0.03)
    parser.add_argument(
        "--foot_lateral_velocity_shaping_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--foot_lateral_velocity_shaping_scale", type=float, default=1.0
    )
    parser.add_argument(
        "--foot_vertical_velocity_shaping_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--foot_vertical_velocity_shaping_scale", type=float, default=1.0
    )
    parser.add_argument("--pitch_balance_shaping_weight", type=float, default=0.0)
    parser.add_argument(
        "--foot_geom_names",
        nargs=4,
        default=(
            "left_ankle_geom",
            "right_ankle_geom",
            "third_ankle_geom",
            "fourth_ankle_geom",
        ),
    )
    parser.add_argument("--xml_file")
    parser.add_argument("--pad_to_horizon", action="store_true")
    parser.add_argument("--backend", default="glfw")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--ffmpeg_target",
        type=Path,
        default=optional_ffmpeg_target(),
        help=(
            "Optional pip-target containing imageio-ffmpeg. Defaults to "
            f"{FFMPEG_TARGET_ENV}, then the active Python environment."
        ),
    )
    parser.add_argument("--output", required=True)
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
    if isinstance(value, (list, tuple)):
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
    for name in LATIN_FONT_NAMES:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_frame(
    frame: np.ndarray,
    *,
    condition_id: str,
    ctrl_cost_weight: float,
    orientation_shaping_weight: float,
    lateral_drift_shaping_weight: float,
    orientation_shaping_function: str,
    action_slew_l2_limit: float | None,
    replace_forward_reward_with_tracking: bool,
    forward_velocity_target: float,
    action_rate_shaping_weight: float,
    pitch_balance_shaping_weight: float,
    training_seed: int,
    evaluation_seed: int,
    step: int,
    max_steps: int,
    dt: float,
    info: dict[str, Any],
    episode_ended: bool = False,
) -> np.ndarray:
    scene = Image.fromarray(frame).convert("RGB")
    sidebar_width = 380
    image = Image.new("RGB", (scene.width + sidebar_width, scene.height), "#12181f")
    image.paste(scene, (0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(
        (scene.width, 0, image.width, image.height), fill=(18, 24, 31, 255)
    )
    draw.line(
        (scene.width, 0, scene.width, image.height), fill=(244, 194, 68, 255), width=3
    )
    font = load_font(18)
    small = load_font(14)
    compact = load_font(11)
    tiny = load_font(10)
    label = load_font(13)

    x = scene.width + 22
    draw.text(
        (x, 20),
        "PROXYGAP  /  FULL TRAJECTORY",
        fill=(244, 194, 68, 255),
        font=label,
    )
    draw.text((x, 50), condition_id, fill=(250, 250, 250, 255), font=font)
    draw.text(
        (x, 82),
        f"control-cost weight   {ctrl_cost_weight:g}",
        fill=(224, 231, 238, 255),
        font=small,
    )
    draw.text(
        (x, 109),
        f"posture/lateral lambda {orientation_shaping_weight:g} / {lateral_drift_shaping_weight:g}",
        fill=(224, 231, 238, 255),
        font=small,
    )
    draw.text(
        (x, 136),
        f"posture function      {orientation_shaping_function}",
        fill=(224, 231, 238, 255),
        font=small,
    )
    draw.text(
        (x, 163),
        f"forward objective     {'target tracking' if replace_forward_reward_with_tracking else 'linear velocity'}",
        fill=(224, 231, 238, 255),
        font=small,
    )
    draw.text(
        (x, 188),
        f"target/rate/pitch    {forward_velocity_target:g} / {action_rate_shaping_weight:g} / {pitch_balance_shaping_weight:g}",
        fill=(224, 231, 238, 255),
        font=small,
    )
    draw.text(
        (x, 213),
        f"slew / train / eval   {action_slew_l2_limit if action_slew_l2_limit is not None else 'none'} / {training_seed} / {evaluation_seed}",
        fill=(224, 231, 238, 255),
        font=compact,
    )
    draw.line((x, 238, image.width - 22, 238), fill=(75, 87, 99, 255), width=1)

    line_2 = (
        f"step       {step:04d} / {max_steps:04d}\n"
        f"time       {step * dt:05.2f} s\n"
        f"x          {float(info.get('x_position', float('nan'))):+.2f} m\n"
        f"y          {float(info.get('y_position', float('nan'))):+.2f} m\n"
        f"torso z    {float(info.get('proxygap_torso_height_step', float('nan'))):.2f} m"
    )
    tilt_deg = math.degrees(
        float(info.get("proxygap_torso_tilt_step", float("nan")))
    )
    pitch_deg = math.degrees(
        float(info.get("proxygap_torso_pitch_step", float("nan")))
    )
    pitch_positive_time = (
        int(info.get("proxygap_pitch_balance_event_positive_steps_step", 0)) * dt
    )
    pitch_negative_time = (
        int(info.get("proxygap_pitch_balance_event_negative_steps_step", 0)) * dt
    )
    line_3 = (
        f"shaped return    {float(info.get('proxygap_proxy_return', 0.0)):+.1f}\n"
        f"base proxy       {float(info.get('proxygap_base_proxy_return', 0.0)):+.1f}\n"
        f"posture penalty  {float(info.get('proxygap_orientation_penalty_step', 0.0)):.3f}\n"
        f"net progress    {float(info.get('proxygap_net_forward_progress', 0.0)):+.2f} m\n"
        f"mean velocity   {float(info.get('proxygap_mean_forward_velocity', 0.0)):+.2f} m/s\n"
        f"torso tilt      {tilt_deg:.1f} deg\n"
        f"signed pitch   {pitch_deg:+.1f} deg\n"
        f"action delta    {float(info.get('proxygap_applied_action_change_l2_step', 0.0)):.2f}\n"
        f"rate penalty   {float(info.get('proxygap_action_rate_penalty_step', 0.0)):.3f}\n"
        f"feet grounded  {int(info.get('proxygap_foot_landing_active_count_step', 0))} / 4\n"
        f"foot vy/vz r   {float(info.get('reward_foot_lateral_velocity_shaping', 0.0)):+.3f} / {float(info.get('reward_foot_vertical_velocity_shaping', 0.0)):+.3f}\n"
        f"pitch event    {int(info.get('proxygap_pitch_balance_event_landed_count_step', 0))} / 4;  +{pitch_positive_time:.2f}/-{pitch_negative_time:.2f} s\n"
        f"pitch score/r  {float(info.get('proxygap_pitch_balance_event_score_step', 0.0)):.2f} / {float(info.get('reward_pitch_balance_shaping', 0.0)):+.3f}"
    )
    draw.multiline_text(
        (x, 245), line_2, fill=(224, 231, 238, 255), font=compact, spacing=0
    )
    draw.multiline_text(
        (x, 312), line_3, fill=(244, 194, 68, 255), font=tiny, spacing=0
    )
    if episode_ended:
        draw.rectangle((18, 18, scene.width - 18, 68), fill=(150, 25, 35, 220))
        draw.text(
            (36, 31),
            "EPISODE ENDED - FINAL FRAME HELD TO 50 s",
            fill=(255, 255, 255, 255),
            font=font,
        )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.fps <= 0:
        raise ValueError("max_steps and fps must be positive")
    expected_fps = int(round(1.0 / 0.05))
    if args.fps != expected_fps:
        raise ValueError(
            f"Stage-one real-time rendering requires {expected_fps} fps for dt=0.05"
        )

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepend_optional_dependency_target(
        args.ffmpeg_target,
        package_marker="imageio_ffmpeg",
    )
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install imageio and imageio-ffmpeg in the active environment, or pass "
            f"--ffmpeg_target/set {FFMPEG_TARGET_ENV}."
        ) from exc

    os.environ["MUJOCO_GL"] = args.backend
    env = make_proxygap_ant_env(
        ctrl_cost_weight=args.ctrl_cost_weight,
        condition_id=args.condition_id,
        seed=args.evaluation_seed,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        xml_file=args.xml_file,
        orientation_shaping_weight=args.orientation_shaping_weight,
        orientation_shaping_scale=args.orientation_shaping_scale,
        orientation_shaping_function=args.orientation_shaping_function,
        lateral_drift_shaping_weight=args.lateral_drift_shaping_weight,
        lateral_drift_shaping_scale=args.lateral_drift_shaping_scale,
        lateral_shaping_signal=args.lateral_shaping_signal,
        lateral_velocity_target=args.lateral_velocity_target,
        replace_forward_reward_with_tracking=args.replace_forward_reward_with_tracking,
        forward_velocity_target=args.forward_velocity_target,
        forward_velocity_tracking_scale=args.forward_velocity_tracking_scale,
        forward_velocity_tracking_weight=args.forward_velocity_tracking_weight,
        action_rate_shaping_weight=args.action_rate_shaping_weight,
        vertical_velocity_shaping_weight=args.vertical_velocity_shaping_weight,
        vertical_velocity_shaping_scale=args.vertical_velocity_shaping_scale,
        roll_pitch_angular_velocity_shaping_weight=(
            args.roll_pitch_angular_velocity_shaping_weight
        ),
        roll_pitch_angular_velocity_shaping_scale=(
            args.roll_pitch_angular_velocity_shaping_scale
        ),
        foot_landing_height_threshold=args.foot_landing_height_threshold,
        foot_lateral_velocity_shaping_weight=(
            args.foot_lateral_velocity_shaping_weight
        ),
        foot_lateral_velocity_shaping_scale=(
            args.foot_lateral_velocity_shaping_scale
        ),
        foot_vertical_velocity_shaping_weight=(
            args.foot_vertical_velocity_shaping_weight
        ),
        foot_vertical_velocity_shaping_scale=(
            args.foot_vertical_velocity_shaping_scale
        ),
        pitch_balance_shaping_weight=args.pitch_balance_shaping_weight,
        foot_geom_names=tuple(args.foot_geom_names),
        augment_previous_applied_action=args.augment_previous_applied_action,
        action_slew_l2_limit=args.action_slew_l2_limit,
    )
    model = PPO.load(model_path, device=args.device)
    observation, _ = env.reset(seed=args.evaluation_seed)
    frames_written = 0
    trajectory_frames = 0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
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
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action)
            frame = env.render()
            writer.append_data(
                overlay_frame(
                    frame,
                    condition_id=args.condition_id,
                    ctrl_cost_weight=args.ctrl_cost_weight,
                    orientation_shaping_weight=args.orientation_shaping_weight,
                    lateral_drift_shaping_weight=args.lateral_drift_shaping_weight,
                    orientation_shaping_function=args.orientation_shaping_function,
                    action_slew_l2_limit=args.action_slew_l2_limit,
                    replace_forward_reward_with_tracking=args.replace_forward_reward_with_tracking,
                    forward_velocity_target=args.forward_velocity_target,
                    action_rate_shaping_weight=args.action_rate_shaping_weight,
                    pitch_balance_shaping_weight=args.pitch_balance_shaping_weight,
                    training_seed=args.training_seed,
                    evaluation_seed=args.evaluation_seed,
                    step=frames_written + 1,
                    max_steps=args.max_steps,
                    dt=float(env.unwrapped.dt),
                    info=final_info,
                )
            )
            frames_written += 1
            trajectory_frames += 1
        if args.pad_to_horizon and frames_written < args.max_steps:
            held_frame = overlay_frame(
                frame,
                condition_id=args.condition_id,
                ctrl_cost_weight=args.ctrl_cost_weight,
                orientation_shaping_weight=args.orientation_shaping_weight,
                lateral_drift_shaping_weight=args.lateral_drift_shaping_weight,
                orientation_shaping_function=args.orientation_shaping_function,
                action_slew_l2_limit=args.action_slew_l2_limit,
                replace_forward_reward_with_tracking=args.replace_forward_reward_with_tracking,
                forward_velocity_target=args.forward_velocity_target,
                action_rate_shaping_weight=args.action_rate_shaping_weight,
                pitch_balance_shaping_weight=args.pitch_balance_shaping_weight,
                training_seed=args.training_seed,
                evaluation_seed=args.evaluation_seed,
                step=trajectory_frames,
                max_steps=args.max_steps,
                dt=float(env.unwrapped.dt),
                info=final_info,
                episode_ended=True,
            )
            while frames_written < args.max_steps:
                writer.append_data(held_frame)
                frames_written += 1

    summary = env.episode_summary()
    dt = float(env.unwrapped.dt)
    env.close()
    expected_duration = frames_written / args.fps
    episode_simulated_duration = trajectory_frames * dt
    if not math.isclose(
        trajectory_frames / args.fps,
        episode_simulated_duration,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Trajectory playback speed does not match simulation time")
    if trajectory_frames != int(summary["episode_length"]):
        raise RuntimeError("Trajectory frame count does not match the episode length")
    if args.pad_to_horizon and frames_written != args.max_steps:
        raise RuntimeError("Padded video does not cover the requested horizon")
    if not (bool(summary["terminated"]) or bool(summary["truncated"])):
        raise RuntimeError("The rendered trajectory did not reach an episode boundary")

    manifest = {
        "status": "complete_trajectory_video_rendered",
        "research_stage": "development_qualitative_check_only",
        "condition_id": args.condition_id,
        "ctrl_cost_weight": args.ctrl_cost_weight,
        "orientation_shaping_weight": args.orientation_shaping_weight,
        "orientation_shaping_scale": args.orientation_shaping_scale,
        "orientation_shaping_function": args.orientation_shaping_function,
        "lateral_drift_shaping_weight": args.lateral_drift_shaping_weight,
        "lateral_drift_shaping_scale": args.lateral_drift_shaping_scale,
        "lateral_shaping_signal": args.lateral_shaping_signal,
        "lateral_velocity_target": args.lateral_velocity_target,
        "action_observation_augmented": args.augment_previous_applied_action,
        "action_slew_l2_limit": args.action_slew_l2_limit,
        "replace_forward_reward_with_tracking": args.replace_forward_reward_with_tracking,
        "forward_velocity_target": args.forward_velocity_target,
        "forward_velocity_tracking_scale": args.forward_velocity_tracking_scale,
        "forward_velocity_tracking_weight": args.forward_velocity_tracking_weight,
        "action_rate_shaping_weight": args.action_rate_shaping_weight,
        "vertical_velocity_shaping_weight": args.vertical_velocity_shaping_weight,
        "vertical_velocity_shaping_scale": args.vertical_velocity_shaping_scale,
        "roll_pitch_angular_velocity_shaping_weight": (
            args.roll_pitch_angular_velocity_shaping_weight
        ),
        "roll_pitch_angular_velocity_shaping_scale": (
            args.roll_pitch_angular_velocity_shaping_scale
        ),
        "foot_landing_height_threshold": args.foot_landing_height_threshold,
        "foot_lateral_velocity_shaping_weight": (
            args.foot_lateral_velocity_shaping_weight
        ),
        "foot_lateral_velocity_shaping_scale": (
            args.foot_lateral_velocity_shaping_scale
        ),
        "foot_vertical_velocity_shaping_weight": (
            args.foot_vertical_velocity_shaping_weight
        ),
        "foot_vertical_velocity_shaping_scale": (
            args.foot_vertical_velocity_shaping_scale
        ),
        "pitch_balance_shaping_weight": args.pitch_balance_shaping_weight,
        "foot_geom_names": list(args.foot_geom_names),
        "policy_device": args.device,
        "render_xml_file": str(Path(args.xml_file).resolve()) if args.xml_file else None,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "target_timesteps": args.target_timesteps,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "video_path": str(output_path),
        "video_sha256": sha256(output_path),
        "frames": frames_written,
        "trajectory_frames": trajectory_frames,
        "padded_frames": frames_written - trajectory_frames,
        "fps": args.fps,
        "simulator_dt_seconds": dt,
        "video_duration_seconds": expected_duration,
        "episode_simulated_duration_seconds": episode_simulated_duration,
        "video_timeline_duration_seconds": expected_duration,
        "playback_speed_ratio": 1.0,
        "ffmpeg_executable": imageio_ffmpeg.get_ffmpeg_exe(),
        "episode_summary": summary,
        "claim_boundary": (
            "This complete trajectory supports qualitative interpretation only. "
            "It is not used to select the coefficient or to estimate uncertainty."
        ),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
