"""Render a real Ant policy rollout on a large, contour-textured MuJoCo terrain.

The terrain is a separate showcase profile.  It deliberately has a global
peak-to-valley range greater than five nominal Ant heights while preserving a
broad, gently undulating central corridor for the existing flat-trained policy.
No policy actions are scripted: every locomotion action comes from the supplied
PPO checkpoint and is stepped through MuJoCo at the environment control rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from run_curved_gait_training import common_env_kwargs  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "curved_gait_tangent_v22_contact_observation_pilot_20260818.json"
DEFAULT_MODEL = (
    ROOT
    / "artifacts"
    / "dev"
    / "curved_gait_tangent_v22_contact_observation_pilot_20260818"
    / "runs"
    / "seed_43812"
    / "models"
    / "checkpoint_2203648.zip"
)
DEFAULT_BASE_XML = (
    ROOT
    / "artifacts"
    / "terrain_pilot_v1"
    / "bundles"
    / "mixed_validation_0"
    / "ant_terrain.xml"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "enriched_terrain_walk_v1"

WIDTH = 1280
HEIGHT = 720
FPS = 20
EXPECTED_FRICTION = np.asarray([1.0, 0.5, 0.5], dtype=np.float64)
EXPECTED_CONDIM = 3

INK = (23, 32, 40)
MID = (84, 99, 108)
WHITE = (249, 251, 251)
TEAL = (54, 126, 124)
AMBER = (213, 142, 77)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--base-xml", type=Path, default=DEFAULT_BASE_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-seed", type=int, default=62021)
    parser.add_argument("--walk-seconds", type=float, default=12.0)
    parser.add_argument("--intro-seconds", type=float, default=1.5)
    parser.add_argument("--outro-seconds", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--half-extent", type=float, default=20.0)
    parser.add_argument("--grid", type=int, default=513)
    parser.add_argument("--contour-interval", type=float, default=0.5)
    parser.add_argument("--minimum-height-ratio", type=float, default=5.5)
    parser.add_argument("--minimum-terrain-range", type=float, default=6.0)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
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
        system_root / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
        system_root / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_env(
    config: dict,
    *,
    seed: int,
    steps: int,
    speed: float,
    xml_file: Path | None = None,
    condition_id: str,
):
    return make_curved_gait_env(
        condition_id=condition_id,
        seed=seed,
        xml_file=xml_file,
        max_episode_steps=steps,
        profile="straight",
        speed_min=speed,
        speed_max=speed,
        max_abs_curvature=0.0,
        max_abs_lateral_speed=0.0,
        fixed_lateral_speed=0.0,
        heading_termination_enabled=False,
        **common_env_kwargs(config),
    )


def probe_robot_height(config: dict, *, seed: int, speed: float) -> dict[str, float]:
    env = make_env(
        config,
        seed=seed,
        steps=20,
        speed=speed,
        condition_id="ENRICHED_TERRAIN_HEIGHT_PROBE",
    )
    observation, _ = env.reset(seed=seed)
    model = env.unwrapped.model
    data = env.unwrapped.data
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
    if torso_id < 0:
        env.close()
        raise RuntimeError("Ant XML does not contain torso_geom")
    ground_to_torso_top = float(data.geom_xpos[torso_id, 2] + model.geom_size[torso_id, 0])
    result = {
        "observation_dimension": int(observation.shape[0]),
        "environment_dt_seconds": float(env.unwrapped.dt),
        "torso_center_height_m": float(data.geom_xpos[torso_id, 2]),
        "torso_radius_m": float(model.geom_size[torso_id, 0]),
        "nominal_robot_height_m": ground_to_torso_top,
    }
    env.close()
    return result


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def gaussian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def build_showcase_heights(
    *,
    grid: int,
    half_extent: float,
    target_range_m: float,
) -> tuple[np.ndarray, dict]:
    coordinates = np.linspace(-half_extent, half_extent, grid, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)

    landscape = (
        gaussian(x, y, cx=9.5, cy=9.0, sx=5.4, sy=4.5, amplitude=1.00)
        + gaussian(x, y, cx=-10.0, cy=11.5, sx=4.8, sy=5.8, amplitude=0.58)
        + gaussian(x, y, cx=13.5, cy=-7.5, sx=3.9, sy=5.0, amplitude=0.42)
        + gaussian(x, y, cx=-10.5, cy=-9.5, sx=6.2, sy=5.1, amplitude=-1.08)
        + gaussian(x, y, cx=2.0, cy=-14.0, sx=7.0, sy=3.5, amplitude=-0.45)
    )
    landscape += 0.16 * np.sin(0.26 * x + 0.14 * y) * np.cos(0.18 * y - 0.08 * x)
    landscape += 0.09 * np.sin(0.46 * x - 0.31 * y)

    # The existing policy was trained on flat ground.  A wide central corridor
    # keeps this showcase an honest policy rollout rather than a guaranteed fall,
    # while the surrounding continuous surface carries the requested extremes.
    corridor_blend = smoothstep((np.abs(y) - 3.2) / 4.8)
    gentle_corridor = 0.018 * np.sin(0.35 * x) + 0.010 * np.sin(0.19 * x + 0.4)
    heights = corridor_blend * landscape + (1.0 - corridor_blend) * gentle_corridor

    radius = np.hypot(x, y)
    spawn_blend = smoothstep((radius - 1.1) / 1.4)
    heights *= spawn_blend
    heights -= heights[grid // 2, grid // 2]
    heights *= target_range_m / float(np.ptp(heights))

    dx = 2.0 * half_extent / (grid - 1)
    dz_dy, dz_dx = np.gradient(heights, dx, dx)
    slope = np.hypot(dz_dx, dz_dy)
    metadata = {
        "half_extent_x_m": half_extent,
        "half_extent_y_m": half_extent,
        "rows": grid,
        "cols": grid,
        "cell_spacing_m": dx,
        "minimum_height_m": float(heights.min()),
        "maximum_height_m": float(heights.max()),
        "height_range_m": float(np.ptp(heights)),
        "maximum_gradient": float(slope.max()),
        "maximum_gradient_degrees": float(math.degrees(math.atan(float(slope.max())))),
        "central_corridor_half_width_m": 3.2,
        "central_transition_outer_half_width_m": 8.0,
        "spawn_flat_radius_m": 1.1,
    }
    return heights, metadata


def colourise_heights(heights: np.ndarray) -> np.ndarray:
    span = float(np.ptp(heights))
    normalised = (heights - float(heights.min())) / span
    stops = np.asarray(
        [
            (0.10, 0.25, 0.34),
            (0.18, 0.43, 0.50),
            (0.38, 0.66, 0.62),
            (0.73, 0.80, 0.65),
            (0.82, 0.64, 0.39),
            (0.66, 0.38, 0.27),
        ],
        dtype=np.float64,
    )
    scaled = normalised * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    return stops[lower] * (1.0 - fraction) + stops[upper] * fraction


def contour_mask(heights: np.ndarray, *, interval_m: float) -> np.ndarray:
    origin = math.floor(float(heights.min()) / interval_m) * interval_m
    bins = np.floor((heights - origin) / interval_m).astype(np.int32)
    edges = np.zeros_like(bins, dtype=bool)
    edges[1:, :] |= bins[1:, :] != bins[:-1, :]
    edges[:-1, :] |= bins[:-1, :] != bins[1:, :]
    edges[:, 1:] |= bins[:, 1:] != bins[:, :-1]
    edges[:, :-1] |= bins[:, :-1] != bins[:, 1:]
    expanded = edges.copy()
    expanded[1:, :] |= edges[:-1, :]
    expanded[:-1, :] |= edges[1:, :]
    expanded[:, 1:] |= edges[:, :-1]
    expanded[:, :-1] |= edges[:, 1:]
    return expanded


def write_scene_assets(
    *,
    scene_dir: Path,
    base_xml: Path,
    heights: np.ndarray,
    geometry: dict,
    contour_interval_m: float,
) -> dict[str, Path]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    heights_path = scene_dir / "heights_m.npy"
    hfield_path = scene_dir / "terrain.hfield"
    texture_path = scene_dir / "terrain_contours.png"
    xml_path = scene_dir / "ant_enriched_terrain.xml"
    np.save(heights_path, np.asarray(heights, dtype=np.float64), allow_pickle=False)
    payload = struct.pack("<ii", heights.shape[0], heights.shape[1])
    payload += np.asarray(heights, dtype="<f4", order="C").tobytes(order="C")
    hfield_path.write_bytes(payload)

    rgb = colourise_heights(heights)
    lines = contour_mask(heights, interval_m=contour_interval_m)
    rgb[lines] = np.asarray((0.96, 0.98, 0.96), dtype=np.float64)
    texture = np.asarray(np.clip(np.flipud(rgb) * 255.0, 0, 255), dtype=np.uint8)
    Image.fromarray(texture, mode="RGB").save(texture_path, format="PNG", optimize=True)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("base XML lacks asset or worldbody")
    hfields = [item for item in asset.findall("hfield") if item.get("name") == "terrain"]
    if len(hfields) != 1:
        raise ValueError("base XML must contain one terrain hfield")
    elevation_range = float(np.ptp(heights))
    hfields[0].set("file", hfield_path.name)
    hfields[0].set(
        "size",
        f"{geometry['half_extent_x_m']:.9g} {geometry['half_extent_y_m']:.9g} "
        f"{elevation_range:.9g} 1.0",
    )

    textures = [item for item in asset.findall("texture") if item.get("name") == "texplane"]
    if len(textures) != 1:
        raise ValueError("base XML must contain texplane")
    texture_node = textures[0]
    for attribute in (
        "builtin",
        "height",
        "width",
        "rgb1",
        "rgb2",
        "mark",
        "markrgb",
        "random",
    ):
        texture_node.attrib.pop(attribute, None)
    texture_node.set("type", "2d")
    texture_node.set("file", texture_path.name)
    for item in asset.findall("texture"):
        if item.get("type") == "skybox":
            item.set("rgb1", "0.95 0.97 0.98")
            item.set("rgb2", "0.62 0.72 0.76")
    for material in asset.findall("material"):
        if material.get("name") == "MatPlane":
            material.set("texrepeat", "1 1")
            material.set("reflectance", "0.08")
            material.set("shininess", "0.12")
            material.set("specular", "0.10")

    floor = worldbody.find("./geom[@name='floor']")
    if floor is None:
        raise ValueError("base XML lacks floor geom")
    floor.set("pos", f"0 0 {float(heights.min()):.12g}")
    floor.set("friction", "1 0.5 0.5")
    floor.set("condim", "3")
    floor.set("rgba", "0.40 0.60 0.57 1")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for child in list(visual):
        if child.tag in {"global", "quality", "headlight", "rgba"}:
            visual.remove(child)
    ET.SubElement(visual, "global", {"offwidth": str(WIDTH), "offheight": str(HEIGHT)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.25 0.27 0.28",
            "diffuse": "0.34 0.36 0.37",
            "specular": "0.06 0.06 0.06",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.86 0.91 0.93 1"})

    for light in worldbody.findall("light"):
        worldbody.remove(light)
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "true",
            "pos": "-10 -14 16",
            "dir": "0.58 0.42 -0.62",
            "diffuse": "0.86 0.84 0.78",
            "specular": "0.12 0.12 0.11",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "false",
            "pos": "12 4 11",
            "dir": "-0.60 -0.15 -0.78",
            "diffuse": "0.22 0.29 0.34",
            "specular": "0.03 0.03 0.03",
        },
    )
    for geom in worldbody.iter("geom"):
        name = geom.get("name", "")
        if name == "floor":
            continue
        if name == "torso_geom":
            geom.set("rgba", "0.08 0.13 0.18 1")
        elif "ankle" in name:
            geom.set("rgba", "0.91 0.57 0.28 1")
        else:
            geom.set("rgba", "0.75 0.42 0.22 1")

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return {
        "xml": xml_path,
        "hfield": hfield_path,
        "heights": heights_path,
        "texture": texture_path,
    }


def height_at(heights: np.ndarray, *, x: float, y: float, half_extent: float) -> float:
    rows, cols = heights.shape
    col_f = np.clip((x + half_extent) / (2.0 * half_extent) * (cols - 1), 0, cols - 1)
    row_f = np.clip((y + half_extent) / (2.0 * half_extent) * (rows - 1), 0, rows - 1)
    col0 = min(int(math.floor(col_f)), cols - 2)
    row0 = min(int(math.floor(row_f)), rows - 2)
    tx = col_f - col0
    ty = row_f - row0
    z00 = heights[row0, col0]
    z10 = heights[row0, col0 + 1]
    z01 = heights[row0 + 1, col0]
    z11 = heights[row0 + 1, col0 + 1]
    return float((1 - ty) * ((1 - tx) * z00 + tx * z10) + ty * ((1 - tx) * z01 + tx * z11))


def make_map_base(heights: np.ndarray, *, contour_interval_m: float, size: int = 210) -> Image.Image:
    rgb = colourise_heights(heights)
    lines = contour_mask(heights, interval_m=contour_interval_m)
    rgb[lines] = np.asarray((0.97, 0.98, 0.96))
    array = np.asarray(np.clip(np.flipud(rgb) * 255.0, 0, 255), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def map_point(x: float, y: float, *, half_extent: float, origin: tuple[int, int], size: int) -> tuple[int, int]:
    px = origin[0] + int(np.clip((x + half_extent) / (2.0 * half_extent), 0.0, 1.0) * (size - 1))
    py = origin[1] + int(np.clip((half_extent - y) / (2.0 * half_extent), 0.0, 1.0) * (size - 1))
    return px, py


def overlay_frame(
    rgb: np.ndarray,
    *,
    mode: str,
    time_seconds: float,
    walk_seconds: float,
    position_xy: np.ndarray,
    trail: list[np.ndarray],
    map_base: Image.Image,
    half_extent: float,
    terrain_range: float,
    height_ratio: float,
    contour_interval_m: float,
) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((24, 22, 492, 104), radius=13, fill=(249, 251, 251, 232))
    draw.text((44, 35), "ENRICHED CONTINUOUS TERRAIN", font=font(20, bold=True), fill=INK)
    subtitle = "AERIAL OVERVIEW" if mode == "intro" else "REAL MUJOCO POLICY ROLLOUT"
    draw.text((44, 70), subtitle, font=font(12, bold=True), fill=TEAL)

    map_size = map_base.width
    map_origin = (WIDTH - map_size - 28, 28)
    draw.rounded_rectangle(
        (map_origin[0] - 10, map_origin[1] - 10, WIDTH - 18, map_origin[1] + map_size + 31),
        radius=12,
        fill=(249, 251, 251, 235),
    )
    image.paste(map_base, map_origin)
    draw = ImageDraw.Draw(image, "RGBA")
    if len(trail) > 1:
        points = [
            map_point(float(point[0]), float(point[1]), half_extent=half_extent, origin=map_origin, size=map_size)
            for point in trail
        ]
        draw.line(points, fill=(255, 255, 255, 230), width=3)
    marker = map_point(
        float(position_xy[0]),
        float(position_xy[1]),
        half_extent=half_extent,
        origin=map_origin,
        size=map_size,
    )
    draw.ellipse((marker[0] - 7, marker[1] - 7, marker[0] + 7, marker[1] + 7), fill=AMBER, outline=WHITE, width=2)
    draw.text((map_origin[0], map_origin[1] + map_size + 7), "True elevation contours", font=font(10, bold=True), fill=INK)

    draw.rectangle((0, HEIGHT - 62, WIDTH, HEIGHT), fill=(17, 27, 34, 218))
    if mode == "intro":
        left_text = f"3D overview | peak-to-valley {terrain_range:.2f} m"
    elif mode == "outro":
        left_text = f"Physical rollout complete | {walk_seconds:.1f} s walking"
    else:
        left_text = f"t = {time_seconds:05.2f} / {walk_seconds:.2f} s | x = {position_xy[0]:+.2f} m | y = {position_xy[1]:+.2f} m"
    draw.text((28, HEIGHT - 46), left_text, font=font(14, bold=True), fill=(245, 248, 249))
    right_text = (
        f"Delta h = {terrain_range:.2f} m = {height_ratio:.2f} x Ant height  |  "
        f"contours {contour_interval_m:.1f} m  |  friction [1.0, 0.5, 0.5]"
    )
    draw.text((WIDTH - 28, HEIGHT - 43), right_text, font=font(11), fill=(210, 220, 224), anchor="ra")
    return image


def camera_intro(progress: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.distance = 31.0 - 2.0 * math.sin(math.pi * progress)
    camera.azimuth = 205.0 + 30.0 * progress
    camera.elevation = -28.0
    return camera


def camera_walk(
    *,
    progress: float,
    position_xy: np.ndarray,
    terrain_z: float,
) -> mujoco.MjvCamera:
    ease = smoothstep(np.asarray(min(1.0, progress / 0.22))).item()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (
        float(position_xy[0] + 1.3 * ease),
        float(position_xy[1]),
        float(terrain_z + 0.35),
    )
    camera.distance = 17.0 * (1.0 - ease) + 10.5 * ease
    camera.azimuth = 205.0 + 7.0 * math.sin(2.0 * math.pi * progress)
    camera.elevation = -24.0 * (1.0 - ease) - 18.0 * ease
    return camera


def encode(stream, container, image: Image.Image) -> None:
    frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB"), dtype=np.uint8), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def make_contact_sheet(frames: list[tuple[str, Image.Image]], output_path: Path) -> None:
    thumb_w, thumb_h = 600, 338
    sheet = Image.new("RGB", (1240, 744), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "Enriched terrain walk - acceptance frames", font=font(20, bold=True), fill=INK)
    for index, (label, frame) in enumerate(frames):
        row, col = divmod(index, 2)
        x = 20 + col * 610
        y = 54 + row * 344
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.rounded_rectangle((x + 12, y + 12, x + 228, y + 43), radius=7, fill=(249, 251, 251))
        draw.text((x + 24, y + 19), label, font=font(12, bold=True), fill=INK)
    sheet.save(output_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    if args.walk_seconds < 10.0:
        raise ValueError("walk-seconds must be at least 10")
    if args.grid < 257 or args.grid % 2 == 0:
        raise ValueError("grid must be an odd integer of at least 257")
    if args.contour_interval <= 0 or args.minimum_height_ratio <= 5.0:
        raise ValueError("contour interval must be positive and height ratio must exceed five")

    config_path = args.config.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    base_xml = args.base_xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (config_path, model_path, base_xml):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = output_dir / "scene"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    robot = probe_robot_height(config, seed=args.evaluation_seed, speed=args.speed)
    robot_height = float(robot["nominal_robot_height_m"])
    target_range = max(args.minimum_terrain_range, args.minimum_height_ratio * robot_height)
    heights, geometry = build_showcase_heights(
        grid=args.grid,
        half_extent=args.half_extent,
        target_range_m=target_range,
    )
    height_ratio = float(geometry["height_range_m"] / robot_height)
    if height_ratio <= 5.0:
        raise RuntimeError(f"terrain range ratio did not exceed five: {height_ratio}")
    scene_files = write_scene_assets(
        scene_dir=scene_dir,
        base_xml=base_xml,
        heights=heights,
        geometry=geometry,
        contour_interval_m=args.contour_interval,
    )

    steps = round(args.walk_seconds / float(robot["environment_dt_seconds"]))
    if not math.isclose(steps * float(robot["environment_dt_seconds"]), args.walk_seconds, abs_tol=1e-9):
        raise ValueError("walk-seconds must be divisible by the environment dt")
    env = make_env(
        config,
        seed=args.evaluation_seed,
        steps=steps,
        speed=args.speed,
        xml_file=scene_files["xml"],
        condition_id="ENRICHED_TERRAIN_REAL_WALK_VIDEO",
    )
    observation, _ = env.reset(seed=args.evaluation_seed)
    policy = PPO.load(model_path, device="cpu")
    if tuple(observation.shape) != tuple(policy.observation_space.shape):
        env.close()
        raise RuntimeError(
            f"observation mismatch: env={observation.shape}, policy={policy.observation_space.shape}"
        )
    compiled = env.unwrapped.model
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    compiled_friction = np.asarray(compiled.geom_friction[floor_id], dtype=np.float64)
    compiled_condim = int(compiled.geom_condim[floor_id])
    if not np.array_equal(compiled_friction, EXPECTED_FRICTION) or compiled_condim != EXPECTED_CONDIM:
        env.close()
        raise RuntimeError("compiled ground friction or condim changed")

    # Gymnasium's MuJoCo base class applies its default 480 px render size after
    # XML compilation.  Restore the requested offscreen framebuffer on the
    # already compiled model before constructing the independent renderer.
    compiled.vis.global_.offwidth = WIDTH
    compiled.vis.global_.offheight = HEIGHT
    renderer = mujoco.Renderer(compiled, height=HEIGHT, width=WIDTH)
    option = mujoco.MjvOption()
    map_base = make_map_base(heights, contour_interval_m=args.contour_interval)
    video_path = output_dir / "enriched_terrain_robot_walk.mp4"
    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "slow"}
    stream.gop_size = FPS * 2

    intro_frames = round(args.intro_seconds * FPS)
    outro_frames = round(args.outro_seconds * FPS)
    trail: list[np.ndarray] = [np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()]
    keyframes: list[tuple[str, Image.Image]] = []
    start_position = trail[0].copy()

    for index in range(intro_frames):
        progress = index / max(1, intro_frames - 1)
        renderer.update_scene(env.unwrapped.data, camera=camera_intro(progress), scene_option=option)
        frame = overlay_frame(
            np.asarray(renderer.render(), dtype=np.uint8),
            mode="intro",
            time_seconds=0.0,
            walk_seconds=args.walk_seconds,
            position_xy=trail[-1],
            trail=trail,
            map_base=map_base,
            half_extent=args.half_extent,
            terrain_range=float(geometry["height_range_m"]),
            height_ratio=height_ratio,
            contour_interval_m=args.contour_interval,
        )
        encode(stream, container, frame)
        if index == intro_frames // 2:
            keyframes.append(("3D aerial overview", frame.copy()))

    termination_reason = "requested_horizon"
    completed_steps = 0
    for step in range(1, steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        completed_steps = step
        position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64).copy()
        trail.append(position)
        local_z = height_at(
            heights,
            x=float(position[0]),
            y=float(position[1]),
            half_extent=args.half_extent,
        )
        progress = step / steps
        renderer.update_scene(
            env.unwrapped.data,
            camera=camera_walk(progress=progress, position_xy=position, terrain_z=local_z),
            scene_option=option,
        )
        frame = overlay_frame(
            np.asarray(renderer.render(), dtype=np.uint8),
            mode="walk",
            time_seconds=step * float(robot["environment_dt_seconds"]),
            walk_seconds=args.walk_seconds,
            position_xy=position,
            trail=trail,
            map_base=map_base,
            half_extent=args.half_extent,
            terrain_range=float(geometry["height_range_m"]),
            height_ratio=height_ratio,
            contour_interval_m=args.contour_interval,
        )
        encode(stream, container, frame)
        if step in {steps // 3, 2 * steps // 3, steps}:
            keyframes.append((f"Physical rollout t={step * float(robot['environment_dt_seconds']):.1f} s", frame.copy()))
        if terminated or truncated:
            termination_reason = "terminated" if terminated else "time_limit"
            break

    actual_walk_seconds = completed_steps * float(robot["environment_dt_seconds"])
    if actual_walk_seconds < 10.0:
        renderer.close()
        env.close()
        container.close()
        raise RuntimeError(f"policy walked only {actual_walk_seconds:.2f} s")

    final_frame = keyframes[-1][1].copy()
    for _ in range(outro_frames):
        outro = overlay_frame(
            np.asarray(final_frame, dtype=np.uint8),
            mode="outro",
            time_seconds=actual_walk_seconds,
            walk_seconds=actual_walk_seconds,
            position_xy=trail[-1],
            trail=trail,
            map_base=map_base,
            half_extent=args.half_extent,
            terrain_range=float(geometry["height_range_m"]),
            height_ratio=height_ratio,
            contour_interval_m=args.contour_interval,
        )
        encode(stream, container, outro)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    summary = env.episode_summary()
    renderer.close()
    env.close()
    contact_sheet_path = output_dir / "enriched_terrain_walk_contact_sheet.png"
    make_contact_sheet(keyframes[:4], contact_sheet_path)
    keyframes[-1][1].save(output_dir / "enriched_terrain_walk_hero.png", format="PNG", optimize=True)

    trail_array = np.asarray(trail, dtype=np.float64)
    planar_distance = float(np.linalg.norm(np.diff(trail_array, axis=0), axis=1).sum())
    net_displacement = float(np.linalg.norm(trail_array[-1] - trail_array[0]))
    total_frames = intro_frames + completed_steps + outro_frames
    manifest = {
        "schema_version": "proxygap-enriched-terrain-walk-v1",
        "scope": "separate visual showcase terrain with real MuJoCo PPO actions",
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frames": total_frames,
            "duration_seconds": total_frames / FPS,
            "physical_walk_seconds": actual_walk_seconds,
        },
        "robot": robot,
        "terrain": {
            **geometry,
            "terrain_to_robot_height_ratio": height_ratio,
            "ratio_requirement": "> 5.0",
            "ratio_requirement_passed": height_ratio > 5.0,
            "contour_interval_m": args.contour_interval,
            "contours_derived_from_same_height_array": True,
            "ground_friction": compiled_friction.tolist(),
            "ground_condim": compiled_condim,
            "profile_boundary": "global extremes surround a broad central policy-safe corridor",
        },
        "rollout": {
            "checkpoint": str(model_path),
            "checkpoint_sha256": sha256(model_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "evaluation_seed": args.evaluation_seed,
            "commanded_speed_m_per_s": args.speed,
            "completed_steps": completed_steps,
            "termination_reason": termination_reason,
            "start_xy_m": start_position.tolist(),
            "final_xy_m": trail_array[-1].tolist(),
            "net_displacement_m": net_displacement,
            "planar_path_length_m": planar_distance,
            "actions_from_policy_checkpoint": True,
            "scripted_joint_animation": False,
            "episode_summary": json_safe(summary),
        },
        "scene_files": {name: str(path) for name, path in scene_files.items()},
        "scene_sha256": {name: sha256(path) for name, path in scene_files.items()},
        "qa": {
            "video_duration_at_least_10_seconds": total_frames / FPS >= 10.0,
            "physical_walk_at_least_10_seconds": actual_walk_seconds >= 10.0,
            "height_range_exceeds_five_robot_heights": height_ratio > 5.0,
            "fixed_friction_verified_in_compiled_model": bool(
                np.array_equal(compiled_friction, EXPECTED_FRICTION)
                and compiled_condim == EXPECTED_CONDIM
            ),
            "true_height_contours": True,
        },
    }
    manifest_path = output_dir / "enriched_terrain_walk_qa.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
