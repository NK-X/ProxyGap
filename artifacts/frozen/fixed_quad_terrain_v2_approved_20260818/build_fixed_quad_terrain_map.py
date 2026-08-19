"""Build and review fixed 80 m by 80 m four-quadrant MuJoCo terrain V2.

This script only constructs and renders the candidate map.  It does not run a
policy rollout.  The start and goal lie on the lower-left to upper-right
diagonal, and the four equal-area quadrants use deliberately different
analytic terrain families.  Unlike the rejected V1 candidate, V2 has no gentle
diagonal corridor: the direct route deliberately crosses convex, concave,
saddle, longitudinal-slope and cross-slope features.  The resulting height
array is the single source for MuJoCo collision geometry, surface colour and
contour lines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import textwrap
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_XML = (
    ROOT
    / "artifacts"
    / "terrain_pilot_v1"
    / "bundles"
    / "mixed_validation_0"
    / "ant_terrain.xml"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fixed_quad_terrain_v2_review"

MAP_SIDE_M = 80.0
HALF_EXTENT_M = MAP_SIDE_M / 2.0
GRID = 1025
CONTOUR_INTERVAL_M = 0.5
TARGET_HEIGHT_RANGE_M = 6.0
START_XY = np.asarray([-34.0, -34.0], dtype=np.float64)
GOAL_XY = np.asarray([34.0, 34.0], dtype=np.float64)
EXPECTED_FRICTION = np.asarray([1.0, 0.5, 0.5], dtype=np.float64)
EXPECTED_CONDIM = 3
WIDTH = 1280
HEIGHT = 720

INK = (23, 32, 40)
MID = (85, 99, 108)
WHITE = (249, 251, 251)
TEAL = (49, 124, 120)
GREEN = (53, 168, 105)
AMBER = (222, 137, 62)
RED = (198, 73, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-xml", type=Path, default=DEFAULT_BASE_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


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


def build_fixed_heights() -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    coordinates = np.linspace(-HALF_EXTENT_M, HALF_EXTENT_M, GRID, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)

    # Q1, south-west: broad rolling terraces with alternating low hills.
    sw = (
        0.62 * np.sin(0.19 * x + 0.04 * y)
        + 0.34 * np.cos(0.15 * y - 0.03 * x)
        + gaussian(x, y, cx=-19.0, cy=-28.0, sx=7.0, sy=5.0, amplitude=0.82)
        - gaussian(x, y, cx=-29.0, cy=-15.0, sx=5.0, sy=8.0, amplitude=0.58)
    )

    # Q2, south-east: oblique ridge segments separated by a saddle.
    q2_u = ((x - 20.0) + (y + 20.0)) / math.sqrt(2.0)
    q2_v = (-(x - 20.0) + (y + 20.0)) / math.sqrt(2.0)
    se = (
        1.12 * np.exp(-0.5 * (q2_v / 4.8) ** 2) * (0.72 + 0.28 * np.cos(0.20 * q2_u))
        - 0.76 * np.exp(-0.5 * ((q2_v - 11.0) / 5.5) ** 2)
        + 0.32 * np.tanh(0.010 * q2_u * q2_v)
        + 0.12 * np.sin(0.32 * q2_u)
    )

    # Q3, north-west: a concave basin enclosed by a broken annular rim.
    q3_r = np.hypot(x + 21.0, y - 21.0)
    q3_theta = np.arctan2(y - 21.0, x + 21.0)
    nw = (
        1.08 * np.exp(-0.5 * ((q3_r - 12.0) / 3.4) ** 2) * (0.84 + 0.16 * np.cos(3.0 * q3_theta))
        - 1.02 * np.exp(-0.5 * (q3_r / 6.4) ** 2)
        + 0.14 * np.sin(0.22 * x - 0.17 * y)
    )

    # Q4, north-east: two unequal domes divided by a curved mountain pass.
    ne = (
        gaussian(x, y, cx=16.0, cy=25.0, sx=6.2, sy=8.0, amplitude=1.05)
        + gaussian(x, y, cx=29.0, cy=15.0, sx=5.0, sy=6.0, amplitude=0.83)
        - gaussian(x, y, cx=24.0, cy=22.0, sx=4.3, sy=4.6, amplitude=0.74)
        + 0.18 * np.sin(0.18 * x + 0.24 * y)
    )

    # Smooth quadrant gates make the four tiles continuous at x=0 and y=0.
    east = smoothstep((x + 3.0) / 6.0)
    north = smoothstep((y + 3.0) / 6.0)
    masks = {
        "south_west": (1.0 - east) * (1.0 - north),
        "south_east": east * (1.0 - north),
        "north_west": (1.0 - east) * north,
        "north_east": east * north,
    }
    landscape = (
        masks["south_west"] * sw
        + masks["south_east"] * se
        + masks["north_west"] * nw
        + masks["north_east"] * ne
    )
    landscape += 0.10 * np.sin(0.08 * x + 0.11 * y) * np.cos(0.06 * y)

    # Scale the base four-quadrant landscape before adding route-crossing
    # features.  This is not a corridor: no band is flattened or replaced.
    landscape -= 0.5 * (float(landscape.max()) + float(landscape.min()))
    landscape *= 5.2 / float(np.ptp(landscape))

    # Deliberately place a sequence of different surface forms across the
    # direct start-to-goal line.  The terms are added to the existing terrain,
    # rather than blended with a simplified path surface.  A robot following
    # the diagonal therefore has to climb, descend and tolerate changing
    # lateral gradients.
    diagonal_coordinate = (x + y) / math.sqrt(2.0)
    cross_coordinate = (y - x) / math.sqrt(2.0)

    def route_gaussian(centre: float, along_width: float, cross_width: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(
            -0.5
            * (
                ((diagonal_coordinate - centre) / along_width) ** 2
                + (cross_coordinate / cross_width) ** 2
            )
        )

    challenge_features = (
        route_gaussian(-34.0, 6.8, 10.0, 0.92)  # convex uphill and crest
        + route_gaussian(-21.0, 5.6, 8.5, -0.88)  # concave depression
        + route_gaussian(-7.0, 7.5, 7.5, 0.58)  # longitudinal saddle approach
        + route_gaussian(4.0, 5.2, 8.0, -0.48)  # saddle exit dip
        + route_gaussian(22.0, 7.0, 9.0, 0.94)  # broad ridge ascent/descent
        + route_gaussian(36.0, 5.8, 8.0, -0.62)  # concave goal approach
    )
    # The signed term changes cross-slope direction around the centre.  Its
    # centre-line elevation is zero, but the Ant's left and right feet see
    # different local heights and the sign reverses during travel.
    cross_slope_sequence = (
        0.52
        * np.tanh(cross_coordinate / 4.8)
        * (
            np.exp(-0.5 * ((diagonal_coordinate + 9.0) / 8.5) ** 2)
            - np.exp(-0.5 * ((diagonal_coordinate - 11.0) / 8.0) ** 2)
        )
    )
    local_roughness = (
        0.13
        * np.sin(0.41 * diagonal_coordinate + 0.18 * cross_coordinate)
        * np.exp(-0.5 * (cross_coordinate / 10.0) ** 2)
    )
    heights = landscape + challenge_features + cross_slope_sequence + local_roughness

    # Flatten only the immediate start and goal pads.  There is no connecting
    # safe strip between them.  Both points remain 6 m inside the map boundary.
    for point in (START_XY, GOAL_XY):
        radius = np.hypot(x - point[0], y - point[1])
        pad_blend = smoothstep((radius - 2.0) / 1.5)
        pad_height = height_at(heights, float(point[0]), float(point[1]))
        heights = pad_blend * heights + (1.0 - pad_blend) * pad_height

    # Ease the outermost six metres towards a stable boundary height.  The
    # start and goal lie exactly at the inner edge of this band, so their pads
    # remain unchanged while the map avoids an artificial perimeter cliff.
    distance_to_edge = np.minimum(HALF_EXTENT_M - np.abs(x), HALF_EXTENT_M - np.abs(y))
    boundary_blend = smoothstep(distance_to_edge / 6.0)
    heights *= boundary_blend

    # Preserve an exact 6.0 m peak-to-valley range for continuity with V1.
    heights -= 0.5 * (float(heights.max()) + float(heights.min()))
    heights *= TARGET_HEIGHT_RANGE_M / float(np.ptp(heights))

    spacing = MAP_SIDE_M / (GRID - 1)
    dz_dy, dz_dx = np.gradient(heights, spacing, spacing)
    gradient = np.hypot(dz_dx, dz_dy)
    route_neighbourhood = np.abs(cross_coordinate) <= 2.0
    metadata = {
        "terrain_id": "fixed_quad_terrain_v2_complex_diagonal",
        "deterministic_analytic_map": True,
        "replaces_rejected_candidate": "fixed_quad_terrain_v1",
        "artificial_safe_diagonal_corridor": False,
        "map_width_m": MAP_SIDE_M,
        "map_length_m": MAP_SIDE_M,
        "map_area_m2": MAP_SIDE_M * MAP_SIDE_M,
        "previous_map_area_m2": 40.0 * 40.0,
        "area_ratio_to_previous": 4.0,
        "quadrant_side_m": 40.0,
        "rows": GRID,
        "cols": GRID,
        "cell_spacing_m": spacing,
        "minimum_height_m": float(heights.min()),
        "maximum_height_m": float(heights.max()),
        "height_range_m": float(np.ptp(heights)),
        "maximum_gradient": float(gradient.max()),
        "maximum_gradient_degrees": float(math.degrees(math.atan(float(gradient.max())))),
        "direct_route_neighbourhood_half_width_m": 2.0,
        "maximum_gradient_within_direct_route_neighbourhood": float(gradient[route_neighbourhood].max()),
        "maximum_gradient_within_direct_route_neighbourhood_degrees": float(
            math.degrees(math.atan(float(gradient[route_neighbourhood].max())))
        ),
        "direct_route_feature_sequence": [
            "start pad",
            "rolling incline",
            "convex hill and crest",
            "concave depression",
            "saddle with cross-slope reversal",
            "broad ridge ascent and descent",
            "concave goal approach",
            "goal pad",
        ],
        "start_xy_m": START_XY.tolist(),
        "goal_xy_m": GOAL_XY.tolist(),
        "straight_line_distance_m": float(np.linalg.norm(GOAL_XY - START_XY)),
        "start_boundary_clearance_m": float(HALF_EXTENT_M - max(abs(START_XY))),
        "goal_boundary_clearance_m": float(HALF_EXTENT_M - max(abs(GOAL_XY))),
        "contour_interval_m": CONTOUR_INTERVAL_M,
        "fixed_friction": EXPECTED_FRICTION.tolist(),
        "condim": EXPECTED_CONDIM,
    }
    components = {
        "south_west": sw,
        "south_east": se,
        "north_west": nw,
        "north_east": ne,
        "x": x,
        "y": y,
        "gradient": gradient,
        "dz_dx": dz_dx,
        "dz_dy": dz_dy,
    }
    return heights, metadata, components


def height_at(heights: np.ndarray, x: float, y: float) -> float:
    col_f = np.clip((x + HALF_EXTENT_M) / MAP_SIDE_M * (heights.shape[1] - 1), 0, heights.shape[1] - 1)
    row_f = np.clip((y + HALF_EXTENT_M) / MAP_SIDE_M * (heights.shape[0] - 1), 0, heights.shape[0] - 1)
    col0 = min(int(math.floor(col_f)), heights.shape[1] - 2)
    row0 = min(int(math.floor(row_f)), heights.shape[0] - 2)
    tx = col_f - col0
    ty = row_f - row0
    z00 = heights[row0, col0]
    z10 = heights[row0, col0 + 1]
    z01 = heights[row0 + 1, col0]
    z11 = heights[row0 + 1, col0 + 1]
    return float((1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11))


def contour_mask(heights: np.ndarray) -> np.ndarray:
    origin = math.floor(float(heights.min()) / CONTOUR_INTERVAL_M) * CONTOUR_INTERVAL_M
    bins = np.floor((heights - origin) / CONTOUR_INTERVAL_M).astype(np.int32)
    edges = np.zeros_like(bins, dtype=bool)
    edges[1:, :] |= bins[1:, :] != bins[:-1, :]
    edges[:-1, :] |= bins[:-1, :] != bins[1:, :]
    edges[:, 1:] |= bins[:, 1:] != bins[:, :-1]
    edges[:, :-1] |= bins[:, :-1] != bins[:, 1:]
    return edges


def colourise_heights(heights: np.ndarray) -> np.ndarray:
    normalised = (heights - float(heights.min())) / float(np.ptp(heights))
    stops = np.asarray(
        [
            (0.08, 0.22, 0.34),
            (0.14, 0.40, 0.52),
            (0.32, 0.62, 0.60),
            (0.70, 0.79, 0.62),
            (0.82, 0.62, 0.36),
            (0.66, 0.34, 0.24),
        ],
        dtype=np.float64,
    )
    scaled = normalised * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    return stops[lower] * (1.0 - fraction) + stops[upper] * fraction


def map_pixel(x: float, y: float, size: int) -> tuple[int, int]:
    px = int(np.clip((x + HALF_EXTENT_M) / MAP_SIDE_M, 0.0, 1.0) * (size - 1))
    py = int(np.clip((HALF_EXTENT_M - y) / MAP_SIDE_M, 0.0, 1.0) * (size - 1))
    return px, py


def topographic_map(heights: np.ndarray, *, size: int = 1000) -> Image.Image:
    rgb = colourise_heights(heights)
    lines = contour_mask(heights)
    rgb[lines] = np.asarray((0.97, 0.98, 0.96), dtype=np.float64)
    array = np.asarray(np.clip(np.flipud(rgb) * 255.0, 0, 255), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")

    # Quadrant borders and distinct structure labels.
    centre = size // 2
    draw.line((centre, 0, centre, size), fill=(255, 255, 255, 155), width=3)
    draw.line((0, centre, size, centre), fill=(255, 255, 255, 155), width=3)
    labels = (
        ("Q3  RING BASIN", (24, 24)),
        ("Q4  TWIN PEAKS + PASS", (centre + 24, 24)),
        ("Q1  ROLLING TERRACES", (24, centre + 24)),
        ("Q2  RIDGES + SADDLE", (centre + 24, centre + 24)),
    )
    for label, origin in labels:
        box = draw.textbbox(origin, label, font=font(19, bold=True))
        draw.rounded_rectangle((box[0] - 9, box[1] - 7, box[2] + 9, box[3] + 7), radius=8, fill=(16, 27, 34, 198))
        draw.text(origin, label, font=font(19, bold=True), fill=WHITE)

    start = map_pixel(float(START_XY[0]), float(START_XY[1]), size)
    goal = map_pixel(float(GOAL_XY[0]), float(GOAL_XY[1]), size)
    dash_count = 34
    for index in range(dash_count):
        t0 = index / dash_count
        t1 = min(1.0, t0 + 0.55 / dash_count)
        p0 = (int(start[0] + (goal[0] - start[0]) * t0), int(start[1] + (goal[1] - start[1]) * t0))
        p1 = (int(start[0] + (goal[0] - start[0]) * t1), int(start[1] + (goal[1] - start[1]) * t1))
        draw.line((*p0, *p1), fill=(255, 255, 255, 220), width=5)
    challenge_stations = (
        (-34.0, "1"),
        (-21.0, "2"),
        (-7.0, "3"),
        (4.0, "4"),
        (22.0, "5"),
        (36.0, "6"),
    )
    for along_coordinate, label in challenge_stations:
        coordinate = along_coordinate / math.sqrt(2.0)
        point = map_pixel(coordinate, coordinate, size)
        draw.ellipse(
            (point[0] - 11, point[1] - 11, point[0] + 11, point[1] + 11),
            fill=AMBER,
            outline=WHITE,
            width=3,
        )
        draw.text(point, label, font=font(13, bold=True), fill=INK, anchor="mm")
    for point, colour, label, anchor in (
        (start, GREEN, "START  (-34, -34)", "ls"),
        (goal, RED, "GOAL  (+34, +34)", "rs"),
    ):
        draw.ellipse((point[0] - 13, point[1] - 13, point[0] + 13, point[1] + 13), fill=colour, outline=WHITE, width=4)
        offset = 20 if anchor == "ls" else -20
        draw.text((point[0] + offset, point[1] - 2), label, font=font(18, bold=True), fill=WHITE, anchor=anchor, stroke_width=2, stroke_fill=INK)
    legend = "1 convex crest   2 concave basin   3 saddle entry   4 cross-slope reversal   5 broad ridge   6 goal dip"
    draw.rounded_rectangle((68, size - 54, size - 68, size - 14), radius=8, fill=(16, 27, 34, 205))
    draw.text((size // 2, size - 34), legend, font=font(13, bold=True), fill=WHITE, anchor="mm")
    return image


def write_scene(output_dir: Path, base_xml: Path, heights: np.ndarray) -> dict[str, Path]:
    scene_dir = output_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    heights_path = scene_dir / "heights_m.npy"
    hfield_path = scene_dir / "terrain.hfield"
    texture_path = scene_dir / "terrain_contours.png"
    xml_path = scene_dir / "ant_fixed_quad_terrain.xml"

    np.save(heights_path, heights.astype(np.float64), allow_pickle=False)
    hfield_path.write_bytes(
        struct.pack("<ii", heights.shape[0], heights.shape[1])
        + np.asarray(heights, dtype="<f4", order="C").tobytes(order="C")
    )
    texture_rgb = colourise_heights(heights)
    lines = contour_mask(heights)
    texture_rgb[lines] = np.asarray((0.97, 0.98, 0.96), dtype=np.float64)
    Image.fromarray(
        np.asarray(np.clip(np.flipud(texture_rgb) * 255.0, 0, 255), dtype=np.uint8), mode="RGB"
    ).save(texture_path, format="PNG", optimize=True)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Base XML lacks asset or worldbody")
    hfield = asset.find("./hfield[@name='terrain']")
    floor = worldbody.find("./geom[@name='floor']")
    texture = asset.find("./texture[@name='texplane']")
    if hfield is None or floor is None or texture is None:
        raise ValueError("Base XML lacks terrain hfield, floor or texplane")
    hfield.set("file", hfield_path.name)
    hfield.set("size", f"{HALF_EXTENT_M:g} {HALF_EXTENT_M:g} {float(np.ptp(heights)):.9g} 1")
    floor.set("pos", f"0 0 {float(heights.min()):.12g}")
    floor.set("friction", "1 0.5 0.5")
    floor.set("condim", "3")
    floor.set("rgba", "0.40 0.60 0.57 1")
    for attribute in ("builtin", "height", "width", "rgb1", "rgb2", "mark", "markrgb", "random"):
        texture.attrib.pop(attribute, None)
    texture.set("type", "2d")
    texture.set("file", texture_path.name)
    for material in asset.findall("material"):
        if material.get("name") == "MatPlane":
            material.set("texrepeat", "1 1")
            material.set("reflectance", "0.06")
            material.set("shininess", "0.10")
            material.set("specular", "0.08")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for child in list(visual):
        if child.tag in {"global", "quality", "headlight", "rgba"}:
            visual.remove(child)
    ET.SubElement(visual, "global", {"offwidth": str(WIDTH), "offheight": str(HEIGHT)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(visual, "headlight", {"ambient": "0.26 0.28 0.29", "diffuse": "0.34 0.36 0.37", "specular": "0.05 0.05 0.05"})
    ET.SubElement(visual, "rgba", {"haze": "0.87 0.92 0.94 1"})

    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    ET.SubElement(worldbody, "light", {"directional": "true", "castshadow": "true", "pos": "-18 -22 28", "dir": "0.55 0.43 -0.71", "diffuse": "0.86 0.84 0.78", "specular": "0.10 0.10 0.09"})
    ET.SubElement(worldbody, "light", {"directional": "true", "castshadow": "false", "pos": "22 10 20", "dir": "-0.60 -0.20 -0.78", "diffuse": "0.22 0.29 0.34", "specular": "0.02 0.02 0.02"})

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

    for name, xy, rgba in (
        ("fixed_start_marker", START_XY, "0.20 0.78 0.42 0.85"),
        ("fixed_goal_marker", GOAL_XY, "0.88 0.25 0.22 0.85"),
    ):
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": name,
                "type": "cylinder",
                "pos": f"{xy[0]:g} {xy[1]:g} {height_at(heights, float(xy[0]), float(xy[1])) + 0.04:.9g}",
                "size": "1.15 0.035",
                "rgba": rgba,
                "group": "2",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return {"xml": xml_path, "heights": heights_path, "hfield": hfield_path, "texture": texture_path}


def configure_robot_at_start(model: mujoco.MjModel, data: mujoco.MjData, heights: np.ndarray) -> None:
    data.qpos[0] = START_XY[0]
    data.qpos[1] = START_XY[1]
    data.qpos[2] = height_at(heights, float(START_XY[0]), float(START_XY[1])) + 0.78
    yaw = math.pi / 4.0
    data.qpos[3:7] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def camera(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    result.fixedcamid = -1
    result.trackbodyid = -1
    result.lookat[:] = lookat
    result.distance = distance
    result.azimuth = azimuth
    result.elevation = elevation
    return result


def render_views(xml_path: Path, heights: np.ndarray, output_dir: Path) -> dict[str, Path]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    configure_robot_at_start(model, data, heights)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    views = {
        "overview": camera((0.0, 0.0, 0.0), 92.0, 222.0, -48.0),
        "diagonal": camera((0.0, 0.0, 0.0), 74.0, 225.0, -31.0),
        "start": camera((-26.0, -26.0, height_at(heights, -26.0, -26.0)), 27.0, 225.0, -25.0),
    }
    paths: dict[str, Path] = {}
    for name, view_camera in views.items():
        renderer.update_scene(data, camera=view_camera)
        array = renderer.render()
        path = output_dir / f"fixed_quad_terrain_{name}.png"
        Image.fromarray(array, mode="RGB").save(path, format="PNG", optimize=True)
        paths[name] = path
    renderer.close()
    return paths


def quadrant_metrics(heights: np.ndarray) -> tuple[dict, float]:
    mid = heights.shape[0] // 2
    # Remove the three-cell centre overlap so every sample belongs to one tile.
    quadrants = {
        "Q1_south_west_rolling_terraces": heights[:mid, :mid],
        "Q2_south_east_ridges_saddle": heights[:mid, mid + 1 :],
        "Q3_north_west_ring_basin": heights[mid + 1 :, :mid],
        "Q4_north_east_twin_peaks_pass": heights[mid + 1 :, mid + 1 :],
    }
    metrics: dict[str, dict] = {}
    standardised: dict[str, np.ndarray] = {}
    for name, tile in quadrants.items():
        dz_dy, dz_dx = np.gradient(tile)
        laplacian = np.gradient(dz_dx, axis=1) + np.gradient(dz_dy, axis=0)
        metrics[name] = {
            "sha256_float64": hashlib.sha256(np.asarray(tile, dtype="<f8").tobytes()).hexdigest(),
            "minimum_height_m": float(tile.min()),
            "maximum_height_m": float(tile.max()),
            "height_standard_deviation_m": float(tile.std()),
            "gradient_rms_per_cell": float(np.sqrt(np.mean(dz_dx**2 + dz_dy**2))),
            "curvature_rms_per_cell2": float(np.sqrt(np.mean(laplacian**2))),
        }
        standardised[name] = ((tile - tile.mean()) / max(float(tile.std()), 1e-12)).ravel()
    correlations: dict[str, float] = {}
    names = list(standardised)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            correlations[f"{left}__vs__{right}"] = float(
                np.mean(standardised[left] * standardised[right])
            )
    metrics["pairwise_normalised_correlations"] = correlations
    max_abs = max(abs(value) for value in correlations.values())
    return metrics, max_abs


def diagonal_profile(heights: np.ndarray, output_path: Path, *, samples: int = 801) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.linspace(0.0, 1.0, samples)
    points = START_XY[None, :] + t[:, None] * (GOAL_XY - START_XY)[None, :]
    distances = t * float(np.linalg.norm(GOAL_XY - START_XY))
    elevations = np.asarray([height_at(heights, float(point[0]), float(point[1])) for point in points])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("distance_m", "x_m", "y_m", "height_m"))
        for distance, point, elevation in zip(distances, points, elevations, strict=True):
            writer.writerow((f"{distance:.6f}", f"{point[0]:.6f}", f"{point[1]:.6f}", f"{elevation:.9f}"))
    slope = np.gradient(elevations, distances)
    return distances, elevations, float(np.max(np.abs(slope)))


def profile_panel(distances: np.ndarray, elevations: np.ndarray, *, width: int = 1040, height: int = 310) -> Image.Image:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 40, 52
    x0, x1 = margin_left, width - margin_right
    y0, y1 = margin_top, height - margin_bottom
    minimum = float(elevations.min()) - 0.08
    maximum = float(elevations.max()) + 0.08
    points = []
    for distance, elevation in zip(distances, elevations, strict=True):
        px = x0 + (distance / distances[-1]) * (x1 - x0)
        py = y1 - (elevation - minimum) / (maximum - minimum) * (y1 - y0)
        points.append((px, py))
    draw.rectangle((x0, y0, x1, y1), outline=(182, 193, 198), width=2)
    draw.line(points, fill=TEAL, width=4, joint="curve")
    draw.text((20, 10), "Direct start-to-goal elevation profile", font=font(20, bold=True), fill=INK)
    draw.text((x0, y1 + 14), "START", font=font(14, bold=True), fill=GREEN)
    draw.text((x1, y1 + 14), "GOAL", font=font(14, bold=True), fill=RED, anchor="ra")
    draw.text((width // 2, y1 + 14), f"distance {distances[-1]:.2f} m", font=font(14), fill=MID, anchor="ma")
    draw.text((14, (y0 + y1) // 2), "height (m)", font=font(13), fill=MID)
    draw.text((x0 + 10, y0 + 10), f"range on direct route: {float(np.ptp(elevations)):.2f} m", font=font(13, bold=True), fill=INK)
    return image


def label_render(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result, "RGBA")
    draw.rounded_rectangle((24, 22, 590, 105), radius=13, fill=(249, 251, 251, 232))
    draw.text((44, 35), title, font=font(22, bold=True), fill=INK)
    draw.text((44, 72), subtitle, font=font(13, bold=True), fill=TEAL)
    draw.rectangle((0, HEIGHT - 52, WIDTH, HEIGHT), fill=(17, 27, 34, 218))
    draw.text((28, HEIGHT - 35), "80 m x 80 m | four distinct 40 m x 40 m quadrants | contours 0.5 m", font=font(13, bold=True), fill=WHITE)
    draw.text((WIDTH - 28, HEIGHT - 35), "START (-34,-34)  ->  GOAL (+34,+34)", font=font(13, bold=True), fill=WHITE, anchor="ra")
    return result


def make_review_sheet(
    overview: Image.Image,
    diagonal: Image.Image,
    topo: Image.Image,
    profile: Image.Image,
    metadata: dict,
    output_path: Path,
) -> None:
    sheet = Image.new("RGB", (1800, 1590), (238, 243, 244))
    draw = ImageDraw.Draw(sheet, "RGBA")
    draw.text((35, 22), "FIXED QUADRANT TERRAIN V2 - COMPLEX ROUTE REVIEW", font=font(28, bold=True), fill=INK)
    draw.text((35, 62), "Collision heightfield, surface colour and contours share the same deterministic 1025 x 1025 height array.", font=font(16), fill=MID)

    panel_w, panel_h = 850, 478
    sheet.paste(overview.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (35, 100))
    sheet.paste(diagonal.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (915, 100))
    sheet.paste(topo.resize((650, 650), Image.Resampling.LANCZOS), (35, 610))
    sheet.paste(profile.resize((1040, 310), Image.Resampling.LANCZOS), (725, 610))

    info_y = 960
    draw.rounded_rectangle((725, info_y, 1765, 1285), radius=16, fill=(249, 251, 251, 238))
    draw.text((755, info_y + 22), "Fixed-map acceptance facts", font=font(22, bold=True), fill=INK)
    facts = (
        f"Map area: {metadata['map_area_m2']:.0f} m2 (exactly {metadata['area_ratio_to_previous']:.0f} x V1)",
        f"Grid: {metadata['rows']} x {metadata['cols']} | spacing {metadata['cell_spacing_m']:.5f} m",
        f"Peak-to-valley: {metadata['height_range_m']:.2f} m | global max slope {metadata['maximum_gradient_degrees']:.1f} deg",
        f"Direct diagonal: {metadata['straight_line_distance_m']:.2f} m | max longitudinal grade {metadata['direct_diagonal_profile']['maximum_absolute_grade_degrees']:.1f} deg",
        f"Route challenge: height range {metadata['direct_diagonal_profile']['height_range_m']:.2f} m | grade reversals {metadata['direct_diagonal_profile']['meaningful_uphill_downhill_reversals']} | cross-slope reversals {metadata['direct_diagonal_profile']['meaningful_cross_slope_sign_reversals']}",
        f"Fixed friction: {metadata['fixed_friction']} | condim={metadata['condim']}",
        f"Quadrant similarity ceiling: |r|={metadata['maximum_pairwise_quadrant_correlation']:.3f}",
    )
    for index, fact in enumerate(facts):
        draw.text((760, info_y + 70 + index * 34), fact, font=font(17), fill=INK)

    draw.rounded_rectangle((35, 1300, 1765, 1550), radius=16, fill=(249, 251, 251, 238))
    draw.text((65, 1322), "Approval boundary", font=font(21, bold=True), fill=INK)
    body = (
        "No robot traversal has been run on this V2 candidate. There is no artificial safe corridor between the pads: "
        "numbered stations mark the convex crest, concave basin, saddle, cross-slope reversal, broad ridge and goal dip. "
        "Approval freezes this exact height-array hash before the completion, fall, airborne and slip tests begin."
    )
    wrapped = "\n".join(textwrap.wrap(body, width=145))
    draw.multiline_text((65, 1364), wrapped, font=font(17), fill=MID, spacing=9)
    sheet.save(output_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heights, metadata, components = build_fixed_heights()
    assets = write_scene(args.output_dir, args.base_xml, heights)

    metrics, max_correlation = quadrant_metrics(heights)
    metadata["quadrant_structures"] = {
        "Q1_south_west": "rolling terraces",
        "Q2_south_east": "oblique ridges and saddle",
        "Q3_north_west": "ring basin with broken rim",
        "Q4_north_east": "unequal twin peaks and curved pass",
    }
    metadata["quadrant_metrics"] = metrics
    metadata["maximum_pairwise_quadrant_correlation"] = max_correlation

    compiled = mujoco.MjModel.from_xml_path(str(assets["xml"]))
    floor_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    compiled_friction = compiled.geom_friction[floor_id].copy()
    compiled_condim = int(compiled.geom_condim[floor_id])
    if not np.allclose(compiled_friction, EXPECTED_FRICTION, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Compiled friction changed: {compiled_friction}")
    if compiled_condim != EXPECTED_CONDIM:
        raise RuntimeError(f"Compiled condim changed: {compiled_condim}")
    if max_correlation >= 0.90:
        raise RuntimeError(f"Quadrants are too structurally similar: max |r|={max_correlation:.3f}")

    profile_path = args.output_dir / "fixed_quad_diagonal_profile.csv"
    distances, elevations, max_profile_slope = diagonal_profile(heights, profile_path)
    longitudinal_grade = np.gradient(elevations, distances)
    smoothing_window = 31
    smoothed_grade = np.convolve(
        longitudinal_grade,
        np.ones(smoothing_window, dtype=np.float64) / smoothing_window,
        mode="same",
    )

    def count_meaningful_sign_reversals(values: np.ndarray, threshold: float) -> int:
        states = np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))
        compressed: list[int] = []
        for state in states:
            state_int = int(state)
            if state_int != 0 and (not compressed or compressed[-1] != state_int):
                compressed.append(state_int)
        return max(0, len(compressed) - 1)

    direct_points_t = np.linspace(0.0, 1.0, len(distances))
    direct_points = START_XY[None, :] + direct_points_t[:, None] * (GOAL_XY - START_XY)[None, :]
    cross_grade_field = (components["dz_dy"] - components["dz_dx"]) / math.sqrt(2.0)
    cross_grades = np.asarray(
        [height_at(cross_grade_field, float(point[0]), float(point[1])) for point in direct_points],
        dtype=np.float64,
    )
    smoothed_cross_grade = np.convolve(
        cross_grades,
        np.ones(smoothing_window, dtype=np.float64) / smoothing_window,
        mode="same",
    )
    longitudinal_reversals = count_meaningful_sign_reversals(smoothed_grade, 0.015)
    cross_slope_reversals = count_meaningful_sign_reversals(smoothed_cross_grade, 0.010)
    metadata["direct_diagonal_profile"] = {
        "sample_count": int(len(distances)),
        "minimum_height_m": float(elevations.min()),
        "maximum_height_m": float(elevations.max()),
        "height_range_m": float(np.ptp(elevations)),
        "maximum_absolute_grade": max_profile_slope,
        "maximum_absolute_grade_degrees": float(math.degrees(math.atan(max_profile_slope))),
        "meaningful_uphill_downhill_reversals": longitudinal_reversals,
        "uphill_sample_fraction_above_0_015_grade": float(np.mean(smoothed_grade > 0.015)),
        "downhill_sample_fraction_below_minus_0_015_grade": float(np.mean(smoothed_grade < -0.015)),
        "maximum_absolute_cross_grade": float(np.max(np.abs(cross_grades))),
        "maximum_absolute_cross_grade_degrees": float(
            math.degrees(math.atan(float(np.max(np.abs(cross_grades)))))
        ),
        "meaningful_cross_slope_sign_reversals": cross_slope_reversals,
    }

    topo = topographic_map(heights)
    topo_path = args.output_dir / "fixed_quad_terrain_topdown.png"
    topo.save(topo_path, format="PNG", optimize=True)
    views = render_views(assets["xml"], heights, args.output_dir)
    overview = label_render(Image.open(views["overview"]).convert("RGB"), "FIXED 80 m x 80 m TERRAIN", "AERIAL OVERVIEW - NO POLICY TEST YET")
    diagonal = label_render(Image.open(views["diagonal"]).convert("RGB"), "CORNER-TO-CORNER TASK", "START AND GOAL FORM THE MAP DIAGONAL")
    overview.save(views["overview"], format="PNG", optimize=True)
    diagonal.save(views["diagonal"], format="PNG", optimize=True)
    profile = profile_panel(distances, elevations)
    profile_path_png = args.output_dir / "fixed_quad_diagonal_profile.png"
    profile.save(profile_path_png, format="PNG", optimize=True)

    review_path = args.output_dir / "fixed_quad_terrain_review_sheet.png"
    make_review_sheet(overview, diagonal, topo, profile, metadata, review_path)

    files = {
        "height_array": assets["heights"],
        "hfield": assets["hfield"],
        "contour_texture": assets["texture"],
        "mujoco_xml": assets["xml"],
        "topographic_review": topo_path,
        "overview_review": views["overview"],
        "diagonal_review": views["diagonal"],
        "start_review": views["start"],
        "diagonal_profile_csv": profile_path,
        "diagonal_profile_png": profile_path_png,
        "review_sheet": review_path,
    }
    manifest = {
        "status": "candidate_fixed_map_v2_awaiting_user_approval",
        "policy_rollout_executed": False,
        "geometry": metadata,
        "compiled_contact": {
            "floor_friction": compiled_friction.tolist(),
            "floor_condim": compiled_condim,
        },
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "qa": {
            "area_exactly_four_times_previous": math.isclose(metadata["area_ratio_to_previous"], 4.0),
            "side_length_doubled": math.isclose(metadata["map_width_m"], 80.0),
            "spatial_resolution_preserved": math.isclose(metadata["cell_spacing_m"], 0.078125),
            "four_unique_quadrant_hashes": len(
                {
                    value["sha256_float64"]
                    for key, value in metrics.items()
                    if key != "pairwise_normalised_correlations"
                }
            )
            == 4,
            "quadrant_correlation_below_0_90": max_correlation < 0.90,
            "start_and_goal_form_diagonal": bool(np.allclose(START_XY, -GOAL_XY)),
            "no_artificial_safe_diagonal_corridor": not metadata["artificial_safe_diagonal_corridor"],
            "direct_route_height_range_exceeds_1_m": float(np.ptp(elevations)) > 1.0,
            "direct_route_has_multiple_grade_reversals": longitudinal_reversals >= 4,
            "direct_route_has_cross_slope_reversal": cross_slope_reversals >= 1,
            "fixed_friction_verified_in_compiled_model": bool(np.array_equal(compiled_friction, EXPECTED_FRICTION)),
            "true_contours_derived_from_height_array": True,
            "policy_test_deferred_until_user_approval": True,
        },
    }
    manifest_path = args.output_dir / "fixed_quad_terrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
