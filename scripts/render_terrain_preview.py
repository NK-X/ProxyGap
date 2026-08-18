"""Render three development-only terrain previews at 300 dpi."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402

from terrain_generator import (  # noqa: E402
    TerrainConfig,
    generate_terrain,
    load_config,
    save_terrain_bundle,
    seed_for,
)
from terrain_validation import assert_terrain_valid, save_validation_result  # noqa: E402


def preview_configurations(base: TerrainConfig) -> dict[str, TerrainConfig]:
    return {
        "single_slope": replace(
            base,
            terrain_seed=seed_for("development", 101),
            split="development",
            hill_count=0,
            pit_count=0,
            random_fourier_terms=0,
            global_slope_x=0.03,
            global_slope_y=0.0,
        ),
        "single_hill": replace(
            base,
            terrain_seed=seed_for("development", 102),
            split="development",
            hill_count=1,
            pit_count=0,
            random_fourier_terms=0,
            global_slope_x=0.0,
            global_slope_y=0.0,
        ),
        "random_mixed": replace(
            base,
            terrain_seed=seed_for("development", 103),
            split="development",
        ),
    }


def render_preview(name: str, terrain, output_path: Path) -> None:
    x_mesh, y_mesh = np.meshgrid(
        terrain.x_coordinates_m, terrain.y_coordinates_m, indexing="xy"
    )
    dy = float(terrain.y_coordinates_m[1] - terrain.y_coordinates_m[0])
    dx = float(terrain.x_coordinates_m[1] - terrain.x_coordinates_m[0])
    dh_dy, dh_dx = np.gradient(terrain.height_m, dy, dx, edge_order=2)
    slope = np.hypot(dh_dx, dh_dy)
    stride = max(1, terrain.config.nrow // 128)

    figure = plt.figure(figsize=(12.0, 5.2), constrained_layout=True)
    surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
    surface = surface_axis.plot_surface(
        x_mesh[::stride, ::stride],
        y_mesh[::stride, ::stride],
        terrain.height_m[::stride, ::stride],
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    surface_axis.set_xlabel("x position (m)")
    surface_axis.set_ylabel("y position (m)")
    surface_axis.set_zlabel("Height (m)")
    surface_axis.set_title("Physical heightfield")
    surface_axis.view_init(elev=28.0, azim=-130.0)
    figure.colorbar(surface, ax=surface_axis, shrink=0.65, pad=0.08, label="Height (m)")

    map_axis = figure.add_subplot(1, 2, 2)
    image = map_axis.pcolormesh(
        terrain.x_coordinates_m,
        terrain.y_coordinates_m,
        slope,
        shading="auto",
        cmap="cividis",
    )
    for region, colour, label in (
        (terrain.config.start_safe_region, "#CC3311", "start-safe core"),
        (terrain.config.goal_safe_region, "#0077BB", "goal-safe core"),
    ):
        map_axis.add_patch(
            Circle(
                (region.centre_x_m, region.centre_y_m),
                region.radius_m,
                fill=False,
                edgecolor=colour,
                linewidth=1.8,
                label=label,
            )
        )
    map_axis.set_aspect("equal")
    map_axis.set_xlabel("x position (m)")
    map_axis.set_ylabel("y position (m)")
    map_axis.set_title("Slope magnitude and usable safe cores")
    map_axis.legend(loc="upper right", frameon=True)
    figure.colorbar(image, ax=map_axis, label="Slope magnitude (rise/run)")
    figure.suptitle(name.replace("_", " ").title())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "terrain_development.json",
    )
    args = parser.parse_args()
    base = load_config(args.config)
    preview_directory = ROOT / "outputs" / "previews"
    manifest_directory = ROOT / "outputs" / "manifests"
    results: dict[str, dict[str, str]] = {}
    for name, config in preview_configurations(base).items():
        terrain = generate_terrain(config)
        validation = assert_terrain_valid(terrain)
        image_path = preview_directory / f"{name}.png"
        render_preview(name, terrain, image_path)
        bundle = save_terrain_bundle(terrain, manifest_directory, name)
        validation_path = manifest_directory / f"{name}_validation.json"
        save_validation_result(validation, validation_path)
        results[name] = {
            "preview": str(image_path.resolve()),
            "manifest": str(bundle["manifest"].resolve()),
            "validation": str(validation_path.resolve()),
            "height_sha256": terrain.height_sha256,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
