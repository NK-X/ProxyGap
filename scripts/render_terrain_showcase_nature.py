"""Render a provenance-preserving terrain showcase from validated MuJoCo bundles.

This script changes presentation only.  It copies each source XML and hfield to a
temporary directory, applies a restrained render style to that temporary XML,
and verifies the source asset hashes before and after rendering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import av
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLES_ROOT = PROJECT_ROOT / "artifacts" / "terrain_pilot_v1" / "bundles"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "terrain_showcase_nature_v1"
EXPECTED_FRICTION = (1.0, 0.5, 0.5)
EXPECTED_CONDIM = 3

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
RENDER_WIDTH = 960
PANEL_WIDTH = CANVAS_WIDTH - RENDER_WIDTH

WHITE = (250, 251, 251)
INK = (24, 34, 42)
MID = (89, 103, 111)
LIGHT = (221, 227, 228)
TEAL = (56, 114, 116)
TEAL_PALE = (218, 234, 231)
BLUE = (15, 77, 146)

SCENE_ORDER = ("slope", "dome", "bowl", "mixed")
SCENE_LABELS = {
    "slope": ("a", "Planar slope", "A controlled directional gradient"),
    "dome": ("b", "Convex surface", "A smooth positive-curvature region"),
    "bowl": ("c", "Concave surface", "A smooth negative-curvature region"),
    "mixed": ("d", "Mixed terrain", "Slope and broad curved features combined"),
}


@dataclass(frozen=True)
class TerrainScene:
    kind: str
    bundle_dir: Path
    xml_path: Path
    hfield_path: Path
    heights_path: Path
    manifest_path: Path
    manifest: dict
    heights: np.ndarray
    source_hashes: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-root", type=Path, default=DEFAULT_BUNDLES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-scene", type=float, default=3.4)
    parser.add_argument("--intro-seconds", type=float, default=1.8)
    parser.add_argument("--outro-seconds", type=float, default=1.4)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_file_hash(manifest: dict, name: str) -> str:
    expected = manifest.get("files_sha256", {}).get(name)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing manifest SHA-256 for {name}")
    return expected


def discover_scenes(bundles_root: Path) -> list[TerrainScene]:
    scenes_by_kind: dict[str, TerrainScene] = {}
    for manifest_path in sorted(bundles_root.glob("*/manifest.json")):
        bundle_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = str(manifest["seed"]["terrain_kind"])
        if kind not in SCENE_ORDER or kind in scenes_by_kind:
            continue
        xml_path = bundle_dir / "ant_terrain.xml"
        hfield_path = bundle_dir / "terrain.hfield"
        heights_path = bundle_dir / "heights_m.npy"
        for path in (xml_path, hfield_path, heights_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        contact = manifest.get("contact", {})
        friction = tuple(float(value) for value in contact.get("ground_friction", ()))
        condim = int(contact.get("ground_condim", -1))
        if friction != EXPECTED_FRICTION or condim != EXPECTED_CONDIM:
            raise ValueError(
                f"{bundle_dir.name}: expected friction={EXPECTED_FRICTION}, "
                f"condim={EXPECTED_CONDIM}; got friction={friction}, condim={condim}"
            )

        source_hashes = {
            "ant_terrain.xml": sha256(xml_path),
            "terrain.hfield": sha256(hfield_path),
            "heights_m.npy": sha256(heights_path),
        }
        for name, actual in source_hashes.items():
            expected = _manifest_file_hash(manifest, name)
            if actual != expected:
                raise ValueError(
                    f"{bundle_dir.name}: {name} SHA-256 mismatch: {actual} != {expected}"
                )
        heights = np.asarray(np.load(heights_path), dtype=np.float64)
        if heights.ndim != 2 or heights.shape != (513, 513):
            raise ValueError(f"{bundle_dir.name}: unexpected height shape {heights.shape}")

        scenes_by_kind[kind] = TerrainScene(
            kind=kind,
            bundle_dir=bundle_dir,
            xml_path=xml_path,
            hfield_path=hfield_path,
            heights_path=heights_path,
            manifest_path=manifest_path,
            manifest=manifest,
            heights=heights,
            source_hashes=source_hashes,
        )
    missing = [kind for kind in SCENE_ORDER if kind not in scenes_by_kind]
    if missing:
        raise FileNotFoundError(f"Missing validation bundles for: {', '.join(missing)}")
    return [scenes_by_kind[kind] for kind in SCENE_ORDER]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    candidates = (
        (system_root / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf")),
        (system_root / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf")),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def write_elevation_texture(scene: TerrainScene, target_path: Path) -> None:
    height = scene.heights
    span = float(np.ptp(height))
    normalised = np.zeros_like(height) if span <= 1e-12 else (height - float(height.min())) / span
    stops = np.asarray(
        [
            (0.91, 0.95, 0.94),
            (0.69, 0.82, 0.79),
            (0.34, 0.59, 0.57),
            (0.14, 0.30, 0.34),
        ],
        dtype=np.float64,
    )
    scaled = normalised * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    texture = np.asarray(np.clip(np.flipud(rgb) * 255.0, 0, 255), dtype=np.uint8)
    Image.fromarray(texture, mode="RGB").save(target_path, format="PNG", optimize=True)


def style_xml(scene: TerrainScene, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene.hfield_path, target_dir / "terrain.hfield")
    write_elevation_texture(scene, target_dir / "elevation_texture.png")
    tree = ET.parse(scene.xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        raise ValueError("MuJoCo XML has no asset section")
    for texture in asset.findall("texture"):
        if texture.get("name") == "texplane":
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
                texture.attrib.pop(attribute, None)
            texture.set("type", "2d")
            texture.set("file", "elevation_texture.png")
        elif texture.get("type") == "skybox":
            texture.set("rgb1", "0.97 0.98 0.985")
            texture.set("rgb2", "0.72 0.79 0.81")
    for material in asset.findall("material"):
        if material.get("name") == "MatPlane":
            material.set("reflectance", "0.08")
            material.set("shininess", "0.18")
            material.set("specular", "0.12")
            material.set("texrepeat", "1 1")
        elif material.get("name") == "geom":
            material.set("reflectance", "0.06")
            material.set("shininess", "0.12")
            material.set("specular", "0.10")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for child in list(visual):
        if child.tag in {"global", "quality", "headlight", "rgba"}:
            visual.remove(child)
    ET.SubElement(
        visual,
        "global",
        {"offwidth": str(RENDER_WIDTH), "offheight": str(CANVAS_HEIGHT)},
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.28 0.30 0.31",
            "diffuse": "0.30 0.32 0.33",
            "specular": "0.05 0.05 0.05",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.90 0.93 0.94 1"})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML has no worldbody")
    for light in worldbody.findall("light"):
        worldbody.remove(light)
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "true",
            "pos": "-8 -5 6",
            "dir": "0.78 0.38 -0.48",
            "diffuse": "0.78 0.77 0.72",
            "specular": "0.14 0.14 0.14",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "false",
            "pos": "7 2 8",
            "dir": "-0.60 -0.12 -1",
            "diffuse": "0.18 0.24 0.28",
            "specular": "0.04 0.04 0.04",
        },
    )

    for geom in worldbody.iter("geom"):
        name = geom.get("name", "")
        if name == "floor":
            geom.set("rgba", "0.34 0.50 0.47 1")
        elif name == "torso_geom":
            geom.set("rgba", "0.07 0.12 0.18 1")
        elif "ankle" in name:
            geom.set("rgba", "0.30 0.52 0.54 1")
        else:
            geom.set("rgba", "0.18 0.32 0.38 1")

    styled_xml = target_dir / "terrain_showcase.xml"
    tree.write(styled_xml, encoding="utf-8", xml_declaration=True)
    return styled_xml


def make_topography(scene: TerrainScene, size: int = 252) -> Image.Image:
    height = scene.heights
    half_x = float(scene.manifest["geometry"]["half_extent_x_m"])
    half_y = float(scene.manifest["geometry"]["half_extent_y_m"])
    extent = (-half_x, half_x, -half_y, half_y)
    cmap = LinearSegmentedColormap.from_list(
        "terrain_teal",
        ["#edf2f1", "#b8d2ce", "#5b9691", "#244d56"],
    )
    fig = plt.figure(figsize=(size / 300, size / 300), dpi=300, facecolor="white")
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.imshow(height, origin="lower", extent=extent, cmap=cmap, interpolation="bilinear")
    levels = np.linspace(float(height.min()), float(height.max()), 7)
    if np.ptp(height) > 1e-9:
        ax.contour(
            np.linspace(-half_x, half_x, height.shape[1]),
            np.linspace(-half_y, half_y, height.shape[0]),
            height,
            levels=levels,
            colors="#ffffff",
            linewidths=0.35,
            alpha=0.72,
        )
    spawn_radius = float(scene.manifest["configuration"]["terrain"]["spawn_flat_radius_m"])
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            spawn_radius,
            fill=False,
            edgecolor="#ffffff",
            linewidth=1.15,
            linestyle=(0, (3, 2)),
        )
    )
    ax.scatter([0], [0], s=12, c="#0f4d92", edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal")
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def save_overview_plate(scenes: list[TerrainScene], output_dir: Path) -> dict[str, str]:
    """Export the same source arrays as an editable four-panel scientific plate."""
    cmap = LinearSegmentedColormap.from_list(
        "terrain_teal_plate",
        ["#edf2f1", "#b8d2ce", "#5b9691", "#244d56"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for panel_index, (ax, scene) in enumerate(zip(axes.flat, scenes, strict=True)):
        geometry = scene.manifest["geometry"]
        half_x = float(geometry["half_extent_x_m"])
        half_y = float(geometry["half_extent_y_m"])
        height = scene.heights
        image = ax.imshow(
            height,
            origin="lower",
            extent=(-half_x, half_x, -half_y, half_y),
            cmap=cmap,
            interpolation="bilinear",
        )
        if np.ptp(height) > 1e-9:
            ax.contour(
                np.linspace(-half_x, half_x, height.shape[1]),
                np.linspace(-half_y, half_y, height.shape[0]),
                height,
                levels=np.linspace(float(height.min()), float(height.max()), 7),
                colors="white",
                linewidths=0.4,
                alpha=0.72,
            )
        spawn_radius = float(scene.manifest["configuration"]["terrain"]["spawn_flat_radius_m"])
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                spawn_radius,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                linestyle=(0, (3, 2)),
            )
        )
        ax.scatter([0], [0], s=12, c="#0f4d92", edgecolors="white", linewidths=0.5)
        panel_letter, title, _ = SCENE_LABELS[scene.kind]
        ax.text(
            -0.08,
            1.04,
            panel_letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("Height (m)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6, length=2)
        ax.text(
            0.01,
            -0.22,
            f"Range {float(height.min()):+.3f} to {float(height.max()):+.3f} m; "
            f"max grade {_metric(scene, 'max_triangular_slope'):.3f}",
            transform=ax.transAxes,
            fontsize=6,
            color="#59676f",
        )
    fig.suptitle(
        "Validated continuous terrain library",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.98,
        0.985,
        "513 x 513 heightfields | fixed friction [1.0, 0.5, 0.5] | dashed: flat spawn zone",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#59676f",
    )
    prefix = output_dir / "terrain_overview_plate"
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "svg": str(prefix.with_suffix(".svg")),
        "pdf": str(prefix.with_suffix(".pdf")),
        "tiff": str(prefix.with_suffix(".tiff")),
        "png": str(prefix.with_suffix(".png")),
    }


def _metric(scene: TerrainScene, key: str, fallback: float = float("nan")) -> float:
    recipe = scene.manifest.get("recipe", {})
    scaled = recipe.get("design_metrics_scaled", {})
    value = scaled.get(key, fallback)
    return float(value)


def render_panel(scene: TerrainScene, topo: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (PANEL_WIDTH, CANVAS_HEIGHT), WHITE)
    draw = ImageDraw.Draw(panel)
    label, title, subtitle = SCENE_LABELS[scene.kind]
    draw.text((30, 27), label, font=_font(20, bold=True), fill=BLUE)
    draw.text((62, 24), title, font=_font(25, bold=True), fill=INK)
    draw.text((30, 68), subtitle, font=_font(14), fill=MID)
    draw.line((30, 102, PANEL_WIDTH - 28, 102), fill=LIGHT, width=2)

    panel.paste(topo, (34, 122))
    draw.text((34, 383), "Elevation map", font=_font(13, bold=True), fill=INK)
    draw.text((34, 405), "Colour: elevation; dashed: flat spawn", font=_font(11), fill=MID)

    geometry = scene.manifest["geometry"]
    min_h = float(geometry["minimum_height_m"])
    max_h = float(geometry["maximum_height_m"])
    max_slope = _metric(scene, "max_triangular_slope")
    slope_deg = math.degrees(math.atan(max_slope))
    terrain_id = str(scene.manifest["seed"]["terrain_id"])

    y = 457
    rows = (
        ("Height range", f"{min_h:+.3f} to {max_h:+.3f} m"),
        ("Max local grade", f"{max_slope:.3f}  ({slope_deg:.1f} deg)"),
        ("Height grid", f"{scene.heights.shape[0]} x {scene.heights.shape[1]}"),
        ("Ground friction", "[1.0, 0.5, 0.5]"),
        ("Contact dimension", "3"),
    )
    for key, value in rows:
        draw.text((34, y), key, font=_font(11), fill=MID)
        draw.text((166, y - 1), value, font=_font(12, bold=True), fill=INK)
        y += 31

    draw.rounded_rectangle((30, 628, PANEL_WIDTH - 28, 669), radius=9, fill=TEAL_PALE)
    draw.text((47, 639), "VALIDATED GEOMETRY", font=_font(12, bold=True), fill=TEAL)
    draw.text((30, 685), terrain_id, font=_font(9), fill=MID)
    draw.text((30, 703), "Preview only - no locomotion claim", font=_font(9), fill=MID)
    return panel


def camera_for_scene(scene: TerrainScene, progress: float) -> mujoco.MjvCamera:
    starts = {"slope": 205.0, "dome": 135.0, "bowl": 42.0, "mixed": 225.0}
    overview_elevations = {"slope": -20.0, "dome": -23.0, "bowl": -21.0, "mixed": -25.0}
    overview_distances = {"slope": 15.6, "dome": 14.8, "bowl": 14.8, "mixed": 15.2}
    close_distances = {"slope": 9.6, "dome": 8.6, "bowl": 8.8, "mixed": 9.0}
    features = scene.manifest.get("recipe", {}).get("features", [])
    if isinstance(features, dict):
        features = [features]
    target_x = 0.0
    target_y = 0.0
    if features:
        strongest = max(features, key=lambda feature: abs(float(feature.get("amplitude_m", 0.0))))
        target_x = float(strongest.get("centre_x_m", 0.0))
        target_y = float(strongest.get("centre_y_m", 0.0))
    ease = 0.5 - 0.5 * math.cos(math.pi * progress)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = (0.82 * target_x * ease, 0.82 * target_y * ease, 0.03)
    camera.distance = (
        overview_distances[scene.kind] * (1.0 - ease) + close_distances[scene.kind] * ease
    )
    camera.azimuth = starts[scene.kind] + 30.0 * progress
    camera.elevation = overview_elevations[scene.kind] * (1.0 - ease) - 10.0 * ease
    return camera


def add_render_labels(frame: Image.Image, scene: TerrainScene) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((24, 24, 305, 72), radius=10, fill=(248, 250, 250, 235))
    draw.text((41, 36), "MUJOCO HEIGHTFIELD", font=_font(15, bold=True), fill=INK)
    draw.text(
        (26, 677),
        "16 m x 16 m  |  colour = elevation  |  Ant = scale",
        font=_font(11),
        fill=(238, 242, 242),
    )


def fade_to_white(frame: Image.Image, alpha: float) -> Image.Image:
    if alpha <= 0.0:
        return frame
    return Image.blend(frame, Image.new("RGB", frame.size, WHITE), min(1.0, alpha))


def make_intro(scenes: list[TerrainScene], topographies: dict[str, Image.Image]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((64, 60), "CONTINUOUS TERRAIN LIBRARY", font=_font(34, bold=True), fill=INK)
    draw.text(
        (64, 111),
        "MuJoCo heightfield preview - validated geometry, presentation-only restyling",
        font=_font(17),
        fill=MID,
    )
    draw.line((64, 155, CANVAS_WIDTH - 64, 155), fill=LIGHT, width=2)
    card_w = 268
    gap = 28
    x0 = 64
    for index, scene in enumerate(scenes):
        x = x0 + index * (card_w + gap)
        draw.rounded_rectangle((x, 192, x + card_w, 553), radius=14, fill=(244, 247, 247), outline=LIGHT, width=2)
        topo = topographies[scene.kind].resize((224, 224), Image.Resampling.LANCZOS)
        image.paste(topo, (x + 22, 214))
        label, title, _ = SCENE_LABELS[scene.kind]
        draw.text((x + 22, 455), label, font=_font(15, bold=True), fill=BLUE)
        draw.text((x + 49, 452), title, font=_font(17, bold=True), fill=INK)
        geometry = scene.manifest["geometry"]
        draw.text(
            (x + 22, 490),
            f"{float(geometry['minimum_height_m']):+.2f} to "
            f"{float(geometry['maximum_height_m']):+.2f} m",
            font=_font(12),
            fill=MID,
        )
    draw.rounded_rectangle((64, 607, 417, 653), radius=10, fill=TEAL_PALE)
    draw.text((84, 620), "FIXED FRICTION  [1.0, 0.5, 0.5]", font=_font(13, bold=True), fill=TEAL)
    draw.text((450, 620), "Grid 513 x 513  |  Validation split only", font=_font(13), fill=MID)
    return image


def make_outro() -> Image.Image:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((84, 105), "RENDER REVIEW COMPLETE", font=_font(36, bold=True), fill=INK)
    draw.text(
        (84, 168),
        "The visual layer was refined without changing the scientific environment.",
        font=_font(19),
        fill=MID,
    )
    rules = (
        "Height arrays and terrain seeds unchanged",
        "MuJoCo collision assets unchanged",
        "Ground friction fixed at [1.0, 0.5, 0.5]",
        "No reward, policy, energy or termination logic used",
    )
    y = 270
    for rule in rules:
        draw.ellipse((88, y + 5, 100, y + 17), fill=TEAL)
        draw.text((124, y), rule, font=_font(18), fill=INK)
        y += 60
    draw.line((84, 556, CANVAS_WIDTH - 84, 556), fill=LIGHT, width=2)
    draw.text((84, 592), "Acceptance preview - terrain appearance only", font=_font(14, bold=True), fill=BLUE)
    draw.text((84, 625), "Locomotion performance requires a separate controlled evaluation.", font=_font(13), fill=MID)
    return image


def encode_frame(stream: av.video.stream.VideoStream, container: av.container.OutputContainer, image: Image.Image) -> None:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def make_contact_sheet(keyframes: list[tuple[str, Image.Image]], output: Path) -> None:
    thumb_w, thumb_h = 600, 338
    sheet = Image.new("RGB", (1240, 744), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "Terrain render acceptance sheet", font=_font(20, bold=True), fill=INK)
    for index, (kind, frame) in enumerate(keyframes):
        row, col = divmod(index, 2)
        x = 20 + col * 610
        y = 54 + row * 344
        thumb = frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        label, title, _ = SCENE_LABELS[kind]
        draw.rounded_rectangle((x + 12, y + 12, x + 175, y + 43), radius=7, fill=(250, 251, 251))
        draw.text((x + 23, y + 19), f"{label}  {title}", font=_font(12, bold=True), fill=INK)
    sheet.save(output, format="PNG", optimize=True)


def render_showcase(args: argparse.Namespace) -> dict:
    if args.fps <= 0 or args.seconds_per_scene <= 0:
        raise ValueError("fps and seconds-per-scene must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = discover_scenes(args.bundles_root.expanduser().resolve())
    overview_plate = save_overview_plate(scenes, output_dir)
    topographies = {scene.kind: make_topography(scene) for scene in scenes}
    panels = {scene.kind: render_panel(scene, topographies[scene.kind]) for scene in scenes}

    video_path = output_dir / "terrain_showcase_nature_v1.mp4"
    intro = make_intro(scenes, topographies)
    outro = make_outro()
    intro.save(output_dir / "terrain_showcase_intro.png", format="PNG", optimize=True)

    container = av.open(str(video_path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = CANVAS_WIDTH
    stream.height = CANVAS_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "slow"}
    stream.gop_size = args.fps * 2

    intro_frames = max(1, round(args.intro_seconds * args.fps))
    outro_frames = max(1, round(args.outro_seconds * args.fps))
    scene_frames = max(2, round(args.seconds_per_scene * args.fps))
    fade_frames = min(max(4, round(0.30 * args.fps)), scene_frames // 3)
    keyframes: list[tuple[str, Image.Image]] = []

    for _ in range(intro_frames):
        encode_frame(stream, container, intro)

    with tempfile.TemporaryDirectory(prefix="terrain_showcase_", dir=output_dir) as temp_text:
        temp_root = Path(temp_text)
        for scene_index, scene in enumerate(scenes):
            scene_dir = temp_root / scene.kind
            styled_xml = style_xml(scene, scene_dir)
            model = mujoco.MjModel.from_xml_path(str(styled_xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            renderer = mujoco.Renderer(model, height=CANVAS_HEIGHT, width=RENDER_WIDTH)
            option = mujoco.MjvOption()
            panel = panels[scene.kind]

            for frame_index in range(scene_frames):
                progress = frame_index / (scene_frames - 1)
                camera = camera_for_scene(scene, progress)
                renderer.update_scene(data, camera=camera, scene_option=option)
                render_array = np.asarray(renderer.render(), dtype=np.uint8)
                frame = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), WHITE)
                render_image = Image.fromarray(render_array, mode="RGB")
                frame.paste(render_image, (0, 0))
                frame.paste(panel, (RENDER_WIDTH, 0))
                add_render_labels(frame, scene)
                fade_alpha = 0.0
                if frame_index < fade_frames:
                    fade_alpha = 1.0 - frame_index / fade_frames
                elif frame_index >= scene_frames - fade_frames:
                    fade_alpha = (frame_index - (scene_frames - fade_frames)) / fade_frames
                frame = fade_to_white(frame, fade_alpha)
                encode_frame(stream, container, frame)
                if frame_index == round(0.72 * (scene_frames - 1)):
                    keyframes.append((scene.kind, frame.copy()))
                    if scene.kind == "mixed":
                        frame.save(output_dir / "terrain_showcase_hero.png", format="PNG", optimize=True)
            renderer.close()

    for _ in range(outro_frames):
        encode_frame(stream, container, outro)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    contact_sheet_path = output_dir / "terrain_showcase_contact_sheet.png"
    make_contact_sheet(keyframes, contact_sheet_path)

    post_hashes: dict[str, dict[str, str]] = {}
    scene_records: list[dict] = []
    for scene in scenes:
        hashes = {
            "ant_terrain.xml": sha256(scene.xml_path),
            "terrain.hfield": sha256(scene.hfield_path),
            "heights_m.npy": sha256(scene.heights_path),
        }
        post_hashes[scene.kind] = hashes
        if hashes != scene.source_hashes:
            raise RuntimeError(f"Source bundle changed while rendering: {scene.bundle_dir}")
        scene_records.append(
            {
                "kind": scene.kind,
                "terrain_id": scene.manifest["seed"]["terrain_id"],
                "source_bundle": str(scene.bundle_dir),
                "source_hashes_before": scene.source_hashes,
                "source_hashes_after": hashes,
                "source_unchanged": True,
                "ground_friction": list(EXPECTED_FRICTION),
                "ground_condim": EXPECTED_CONDIM,
                "height_shape": list(scene.heights.shape),
                "height_min_m": float(scene.heights.min()),
                "height_max_m": float(scene.heights.max()),
                "max_triangular_slope": _metric(scene, "max_triangular_slope"),
            }
        )

    expected_frames = intro_frames + len(scenes) * scene_frames + outro_frames
    report = {
        "schema_version": "proxygap-terrain-showcase-qa-v1",
        "scope": "rendering-only; no terrain, contact, policy, reward, energy, or termination changes",
        "figure_contract": {
            "core_conclusion": "The validated library contains four visually distinct continuous terrain classes at fixed contact settings.",
            "archetype": "asymmetric mixed-modality image plate",
            "hero_evidence": "MuJoCo low-angle whole-terrain render",
            "supporting_evidence": "true height-array map and manifest-derived geometry metadata",
            "review_risk": "camera and shading can exaggerate relief; numerical ranges are therefore displayed alongside every render",
        },
        "video": {
            "path": str(video_path),
            "sha256": sha256(video_path),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "fps": args.fps,
            "frames": expected_frames,
            "duration_seconds": expected_frames / args.fps,
        },
        "outputs": {
            "hero_png": str(output_dir / "terrain_showcase_hero.png"),
            "contact_sheet_png": str(contact_sheet_path),
            "intro_png": str(output_dir / "terrain_showcase_intro.png"),
            "overview_plate": overview_plate,
        },
        "scenes": scene_records,
        "qa": {
            "python_only_visual_backend": True,
            "fixed_friction_verified": True,
            "source_hashes_verified_before_and_after": True,
            "no_source_bundle_writes": True,
            "locomotion_claim": False,
        },
    }
    report_path = output_dir / "terrain_showcase_qa.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = render_showcase(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
