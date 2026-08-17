"""Render the selected forward -> stop -> positive-y policy only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.planar_transition import make_planar_transition_env  # noqa: E402
from run_planar_translation_transition import environment_kwargs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "planar_translation_transition_v3_20260818.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--target-timesteps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", default="glfw")
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


def overlay(frame: np.ndarray, *, step: int, dt: float, info: dict[str, Any]) -> np.ndarray:
    scene = Image.fromarray(frame).convert("RGB")
    sidebar = 330
    canvas = Image.new("RGB", (scene.width + sidebar, scene.height), "#111820")
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((scene.width, 0, canvas.width, canvas.height), fill=(17, 24, 32, 255))
    phase = str(info.get("proxygap_command_phase_step", "unknown"))
    colors = {
        "forward": (74, 210, 130, 255),
        "brake": (250, 178, 63, 255),
        "lateral": (74, 160, 245, 255),
    }
    accent = colors.get(phase, (220, 220, 220, 255))
    x = scene.width + 22
    title = load_font(18)
    label = load_font(14)
    value = load_font(22)
    small = load_font(12)
    draw.text((x, 20), "PLANAR COMMAND TRANSITION", font=label, fill=(238, 242, 247, 255))
    draw.rounded_rectangle((x, 53, canvas.width - 22, 103), radius=8, fill=accent)
    draw.text((x + 14, 64), phase.upper(), font=value, fill=(10, 15, 20, 255))
    command = np.asarray(info.get("proxygap_command_xy_step", [np.nan, np.nan]))
    velocity = np.asarray(info.get("proxygap_planar_velocity_step", [np.nan, np.nan]))
    yaw_deg = math.degrees(float(info.get("proxygap_yaw_error_step", float("nan"))))
    stop = bool(info.get("proxygap_stop_achieved", False))
    forced = bool(info.get("proxygap_stop_transition_forced", False))
    draw.text((x, 126), "COMMAND  (m/s)", font=small, fill=(145, 160, 176, 255))
    draw.text((x, 145), f"vx {command[0]:+4.1f}   vy {command[1]:+4.1f}", font=title, fill=(238, 242, 247, 255))
    draw.text((x, 190), "MEASURED VELOCITY", font=small, fill=(145, 160, 176, 255))
    draw.text((x, 209), f"vx {velocity[0]:+5.2f}   vy {velocity[1]:+5.2f}", font=title, fill=(238, 242, 247, 255))
    draw.line((x, 253, canvas.width - 22, 253), fill=(68, 80, 94, 255), width=1)
    draw.text((x, 273), f"time                 {step * dt:5.2f} s", font=label, fill=(218, 226, 235, 255))
    draw.text((x, 303), f"planar speed         {np.linalg.norm(velocity):5.2f} m/s", font=label, fill=(218, 226, 235, 255))
    draw.text((x, 333), f"yaw error            {yaw_deg:+5.1f} deg", font=label, fill=(218, 226, 235, 255))
    status = "STOP ACHIEVED" if stop else "BRAKING" if phase == "brake" else "WAITING"
    if forced:
        status = "BRAKE WINDOW ENDED"
    draw.text((x, 373), status, font=label, fill=accent)
    draw.text((x, 418), "No torso-yaw steering command", font=small, fill=(145, 160, 176, 255))
    draw.text((x, 439), "Motor order: 6,7,0,1,2,3,4,5", font=small, fill=(145, 160, 176, 255))
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.fps <= 0:
        raise ValueError("max-steps and fps must be positive")
    if args.fps != 20:
        raise ValueError("Ant dt=0.05 requires 20 fps for real-time playback")
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError("config or model is missing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepend_optional_dependency_target(args.ffmpeg_target, package_marker="imageio_ffmpeg")
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Install imageio-ffmpeg or set {FFMPEG_TARGET_ENV}"
        ) from error
    os.environ["MUJOCO_GL"] = args.backend
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # MuJoCo on Windows cannot open this XML through the repository's Chinese
    # path, so render from an ASCII-only temporary copy of the audited asset.
    xml_path = Path(tempfile.gettempdir()) / "proxygap_ant_render_large_floor.xml"
    shutil.copy2(ROOT / "assets" / "ant_render_large_floor.xml", xml_path)
    kwargs = environment_kwargs(config, evaluation=True)
    env = make_planar_transition_env(
        condition_id="T1__STOP_TO_POSITIVE_Y_VIDEO",
        seed=args.evaluation_seed,
        render_mode="rgb_array",
        xml_file=xml_path,
        max_episode_steps=args.max_steps,
        **kwargs,
    )
    model = PPO.load(model_path, device=args.device)
    observation, _ = env.reset(seed=args.evaluation_seed)
    dt = float(env.unwrapped.dt)
    final_info: dict[str, Any] = {}
    frames = 0
    terminated = False
    truncated = False
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
            frames += 1
            writer.append_data(overlay(env.render(), step=frames, dt=dt, info=final_info))
    summary = env.episode_summary()
    env.close()
    if frames != args.max_steps or not truncated:
        raise RuntimeError("render did not cover the requested fixed horizon")
    manifest = {
        "status": "complete",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "target_timesteps": args.target_timesteps,
        "video_path": str(output_path),
        "video_sha256": sha256(output_path),
        "frames": frames,
        "fps": args.fps,
        "duration_seconds": frames / args.fps,
        "simulator_dt_seconds": dt,
        "policy_device": args.device,
        "ffmpeg_executable": imageio_ffmpeg.get_ffmpeg_exe(),
        "episode_summary": summary,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
