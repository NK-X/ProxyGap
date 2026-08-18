"""Deterministic generation of bounded, smooth MuJoCo heightfields.

The generator deliberately supports only continuous single-valued surfaces.  All
comments use British Academic English because this module is intended to form a
traceable component of a research codebase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


GENERATOR_VERSION = "1.1.0"
ALLOWED_RESOLUTIONS = frozenset({257, 513, 1025})
MINIMUM_ANT_FEATURE_WIDTH_M = 0.16
SEED_NAMESPACE_SIZE = 1_000_000
SEED_NAMESPACES: dict[str, tuple[int, int]] = {
    "development": (0, 999_999),
    "train": (1_000_000, 1_999_999),
    "validation": (2_000_000, 2_999_999),
    "test": (3_000_000, 3_999_999),
}


def seed_for(split: str, index: int) -> int:
    """Return a seed in a disjoint, named terrain namespace."""

    if split not in SEED_NAMESPACES:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(SEED_NAMESPACES)}")
    if not 0 <= int(index) < SEED_NAMESPACE_SIZE:
        raise ValueError(f"seed index must be in [0, {SEED_NAMESPACE_SIZE})")
    return SEED_NAMESPACES[split][0] + int(index)


def split_for_seed(seed: int) -> str:
    """Return the unique namespace containing ``seed``."""

    for split, (lower, upper) in SEED_NAMESPACES.items():
        if lower <= int(seed) <= upper:
            return split
    raise ValueError(f"terrain seed {seed} is outside the declared namespaces")


@dataclass(frozen=True)
class SafeRegionConfig:
    """A circular usable core joined to the surrounding surface by a C2 blend."""

    centre_x_m: float
    centre_y_m: float
    radius_m: float
    blend_width_m: float
    maximum_absolute_slope: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SafeRegionConfig":
        return cls(
            centre_x_m=float(value["centre_x_m"]),
            centre_y_m=float(value["centre_y_m"]),
            radius_m=float(value["radius_m"]),
            blend_width_m=float(value["blend_width_m"]),
            maximum_absolute_slope=float(value["maximum_absolute_slope"]),
        )


@dataclass(frozen=True)
class TerrainConfig:
    """Complete, serialisable configuration for one terrain realisation."""

    terrain_length_m: float = 16.0
    terrain_width_m: float = 16.0
    nrow: int = 513
    ncol: int = 513
    maximum_height_m: float = 0.45
    maximum_absolute_slope: float = 0.22
    maximum_curvature: float = 0.30
    hill_count: int = 3
    pit_count: int = 2
    minimum_feature_width_m: float = 1.0
    global_slope_x: float = 0.015
    global_slope_y: float = 0.010
    smoothing_strength: float = 0.20
    friction: tuple[float, float, float] = (1.0, 0.5, 0.5)
    terrain_seed: int = 42
    split: str = "development"
    start_safe_region: SafeRegionConfig = field(
        default_factory=lambda: SafeRegionConfig(0.0, 0.0, 1.0, 1.2, 0.035)
    )
    goal_safe_region: SafeRegionConfig = field(
        default_factory=lambda: SafeRegionConfig(5.0, 0.0, 0.9, 1.1, 0.040)
    )
    random_fourier_terms: int = 8
    hill_amplitude_fraction: float = 0.55
    pit_amplitude_fraction: float = 0.45
    fourier_amplitude_fraction: float = 0.25
    heightfield_base_depth_m: float = 0.10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TerrainConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown terrain configuration fields: {unknown}")
        converted = dict(value)
        if "friction" in converted:
            converted["friction"] = tuple(float(item) for item in converted["friction"])
        for name in ("start_safe_region", "goal_safe_region"):
            if name in converted and not isinstance(converted[name], SafeRegionConfig):
                converted[name] = SafeRegionConfig.from_mapping(converted[name])
        config = cls(**converted)
        validate_config(config)
        return config

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["friction"] = list(self.friction)
        return result


@dataclass
class TerrainData:
    """A physical heightfield and the deterministic metadata used to create it."""

    config: TerrainConfig
    x_coordinates_m: np.ndarray
    y_coordinates_m: np.ndarray
    height_m: np.ndarray
    normalised_height: np.ndarray
    metadata: dict[str, Any]

    @property
    def height_sha256(self) -> str:
        return array_sha256(self.height_m)

    @property
    def normalised_sha256(self) -> str:
        return array_sha256(self.normalised_height)


def validate_config(config: TerrainConfig) -> None:
    """Reject configurations that cannot satisfy the declared geometry contract."""

    integer_fields = {
        "nrow": config.nrow,
        "ncol": config.ncol,
        "hill_count": config.hill_count,
        "pit_count": config.pit_count,
        "random_fourier_terms": config.random_fourier_terms,
        "terrain_seed": config.terrain_seed,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    finite_scalars = {
        "terrain_length_m": config.terrain_length_m,
        "terrain_width_m": config.terrain_width_m,
        "maximum_height_m": config.maximum_height_m,
        "maximum_absolute_slope": config.maximum_absolute_slope,
        "maximum_curvature": config.maximum_curvature,
        "minimum_feature_width_m": config.minimum_feature_width_m,
        "global_slope_x": config.global_slope_x,
        "global_slope_y": config.global_slope_y,
        "smoothing_strength": config.smoothing_strength,
        "hill_amplitude_fraction": config.hill_amplitude_fraction,
        "pit_amplitude_fraction": config.pit_amplitude_fraction,
        "fourier_amplitude_fraction": config.fourier_amplitude_fraction,
        "heightfield_base_depth_m": config.heightfield_base_depth_m,
    }
    for region_name, region in (
        ("start_safe_region", config.start_safe_region),
        ("goal_safe_region", config.goal_safe_region),
    ):
        finite_scalars.update(
            {
                f"{region_name}.centre_x_m": region.centre_x_m,
                f"{region_name}.centre_y_m": region.centre_y_m,
                f"{region_name}.radius_m": region.radius_m,
                f"{region_name}.blend_width_m": region.blend_width_m,
                f"{region_name}.maximum_absolute_slope": region.maximum_absolute_slope,
            }
        )
    finite_scalars.update({f"friction[{index}]": value for index, value in enumerate(config.friction)})
    for name, value in finite_scalars.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if config.nrow not in ALLOWED_RESOLUTIONS or config.ncol not in ALLOWED_RESOLUTIONS:
        raise ValueError(f"nrow and ncol must each be one of {sorted(ALLOWED_RESOLUTIONS)}")
    if config.terrain_length_m <= 0.0 or config.terrain_width_m <= 0.0:
        raise ValueError("terrain physical dimensions must be positive")
    if config.maximum_height_m < 0.0:
        raise ValueError("maximum_height_m must be non-negative")
    if config.maximum_absolute_slope <= 0.0:
        raise ValueError("maximum_absolute_slope must be positive")
    if config.maximum_curvature <= 0.0:
        raise ValueError("maximum_curvature must be positive")
    if config.hill_count < 0 or config.pit_count < 0 or config.random_fourier_terms < 0:
        raise ValueError("component counts must be non-negative")
    if config.smoothing_strength < 0.0:
        raise ValueError("smoothing_strength is a Gaussian sigma in metres and cannot be negative")
    if len(config.friction) != 3 or any(value <= 0.0 for value in config.friction):
        raise ValueError("friction must contain three fixed positive MuJoCo coefficients")
    if config.heightfield_base_depth_m <= 0.0:
        raise ValueError("heightfield_base_depth_m must be positive")
    if config.split not in SEED_NAMESPACES:
        raise ValueError(f"split must be one of {tuple(SEED_NAMESPACES)}")
    actual_split = split_for_seed(config.terrain_seed)
    if actual_split != config.split:
        raise ValueError(
            f"terrain_seed {config.terrain_seed} belongs to {actual_split!r}, not {config.split!r}"
        )

    dx = config.terrain_length_m / (config.ncol - 1)
    dy = config.terrain_width_m / (config.nrow - 1)
    resolved_minimum = max(MINIMUM_ANT_FEATURE_WIDTH_M, 8.0 * max(dx, dy))
    if config.minimum_feature_width_m + 1e-12 < resolved_minimum:
        raise ValueError(
            "minimum_feature_width_m must be at least the inspected 0.16 m Ant foot "
            "diameter and span at least eight grid intervals; "
            f"grid intervals; this configuration requires >= {resolved_minimum:.6g} m"
        )
    if min(config.terrain_length_m, config.terrain_width_m) < 6.0 * config.minimum_feature_width_m:
        raise ValueError("terrain dimensions must span at least six minimum feature widths")
    if math.hypot(config.global_slope_x, config.global_slope_y) > config.maximum_absolute_slope:
        raise ValueError("the requested global slope already exceeds maximum_absolute_slope")
    plane_corner_bound = (
        abs(config.global_slope_x) * 0.5 * config.terrain_length_m
        + abs(config.global_slope_y) * 0.5 * config.terrain_width_m
    )
    if plane_corner_bound > config.maximum_height_m + 1e-12:
        raise ValueError(
            "the requested global slope exceeds maximum_height_m at a terrain corner; "
            "adjust the formal parameters rather than relying on clipping or silent rescaling"
        )
    for value, name in (
        (config.hill_amplitude_fraction, "hill_amplitude_fraction"),
        (config.pit_amplitude_fraction, "pit_amplitude_fraction"),
        (config.fourier_amplitude_fraction, "fourier_amplitude_fraction"),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if (
        config.maximum_height_m == 0.0
        and (
            config.hill_count
            or config.pit_count
            or config.random_fourier_terms
            or config.global_slope_x
            or config.global_slope_y
        )
    ):
        raise ValueError("non-flat components require maximum_height_m > 0")

    half_length = 0.5 * config.terrain_length_m
    half_width = 0.5 * config.terrain_width_m
    outer_regions: list[tuple[float, float, float, str]] = []
    for name, region in (
        ("start_safe_region", config.start_safe_region),
        ("goal_safe_region", config.goal_safe_region),
    ):
        if region.radius_m <= 0.0:
            raise ValueError(f"{name}.radius_m must be positive")
        if region.blend_width_m < config.minimum_feature_width_m:
            raise ValueError(f"{name}.blend_width_m must be >= minimum_feature_width_m")
        if not 0.0 < region.maximum_absolute_slope <= config.maximum_absolute_slope:
            raise ValueError(
                f"{name}.maximum_absolute_slope must be positive and no greater than the global limit"
            )
        grid_guard_m = 0.25 * config.minimum_feature_width_m
        outer = region.radius_m + region.blend_width_m + grid_guard_m
        if abs(region.centre_x_m) + outer > half_length:
            raise ValueError(f"{name} extends beyond the terrain length")
        if abs(region.centre_y_m) + outer > half_width:
            raise ValueError(f"{name} extends beyond the terrain width")
        outer_regions.append((region.centre_x_m, region.centre_y_m, outer, name))
    first, second = outer_regions
    separation = math.hypot(first[0] - second[0], first[1] - second[1])
    if separation <= first[2] + second[2]:
        raise ValueError("start and goal safe-region blends must not overlap")


def _canonical_config_json(config: TerrainConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(config: TerrainConfig) -> str:
    return hashlib.sha256(_canonical_config_json(config).encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash a canonical little-endian float64 representation, including its shape."""

    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"ant-random-terrain-array-v1\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gaussian_kernel(sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 1e-12:
        return np.ones(1, dtype=np.float64)
    radius = max(1, int(math.ceil(4.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    return kernel / np.sum(kernel)


def _convolve_reflect(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size == 1:
        return array.copy()
    radius = kernel.size // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(array, padding, mode="reflect")
    return np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="valid"), axis, padded)


def _smooth_surface(height: np.ndarray, sigma_m: float, dx: float, dy: float) -> np.ndarray:
    """Apply separable Gaussian smoothing without an optional SciPy dependency."""

    if sigma_m <= 1e-12:
        return height.copy()
    along_x = _convolve_reflect(height, _gaussian_kernel(sigma_m / dx), axis=1)
    return _convolve_reflect(along_x, _gaussian_kernel(sigma_m / dy), axis=0)


def _bilinear_from_grid(
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    values: np.ndarray,
    x_m: float,
    y_m: float,
) -> float:
    x_fraction = (x_m - x_coordinates_m[0]) / (x_coordinates_m[-1] - x_coordinates_m[0])
    y_fraction = (y_m - y_coordinates_m[0]) / (y_coordinates_m[-1] - y_coordinates_m[0])
    column = float(np.clip(x_fraction, 0.0, 1.0) * (x_coordinates_m.size - 1))
    row = float(np.clip(y_fraction, 0.0, 1.0) * (y_coordinates_m.size - 1))
    c0 = min(int(math.floor(column)), x_coordinates_m.size - 2)
    r0 = min(int(math.floor(row)), y_coordinates_m.size - 2)
    tx = column - c0
    ty = row - r0
    return float(
        (1.0 - tx) * (1.0 - ty) * values[r0, c0]
        + tx * (1.0 - ty) * values[r0, c0 + 1]
        + (1.0 - tx) * ty * values[r0 + 1, c0]
        + tx * ty * values[r0 + 1, c0 + 1]
    )


def _smootherstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def _apply_safe_region(
    height: np.ndarray,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    region: SafeRegionConfig,
    grid_guard_m: float,
) -> np.ndarray:
    """Blend a constant core into the surface with zero first/second edge derivatives."""

    target_height = _bilinear_from_grid(
        x_coordinates_m,
        y_coordinates_m,
        height,
        region.centre_x_m,
        region.centre_y_m,
    )
    distance = np.hypot(x_mesh - region.centre_x_m, y_mesh - region.centre_y_m)
    plateau_radius = region.radius_m + grid_guard_m
    transition = (distance - plateau_radius) / region.blend_width_m
    retain_original = _smootherstep(transition)
    return target_height + retain_original * (height - target_height)


def differential_metrics(height_m: np.ndarray, dx_m: float, dy_m: float) -> dict[str, Any]:
    """Return finite-difference slope and Hessian spectral-curvature fields."""

    dh_dy, dh_dx = np.gradient(height_m, dy_m, dx_m, edge_order=2)
    d_dy_dx, d_dx_dx = np.gradient(dh_dx, dy_m, dx_m, edge_order=2)
    d_dy_dy, d_dx_dy = np.gradient(dh_dy, dy_m, dx_m, edge_order=2)
    d2h_dxdy = 0.5 * (d_dy_dx + d_dx_dy)
    trace = d_dx_dx + d_dy_dy
    discriminant = np.sqrt(np.maximum((d_dx_dx - d_dy_dy) ** 2 + 4.0 * d2h_dxdy**2, 0.0))
    eigenvalue_high = 0.5 * (trace + discriminant)
    eigenvalue_low = 0.5 * (trace - discriminant)
    curvature = np.maximum(np.abs(eigenvalue_high), np.abs(eigenvalue_low))
    slope = np.hypot(dh_dx, dh_dy)
    h00 = height_m[:-1, :-1]
    h01 = height_m[:-1, 1:]
    h10 = height_m[1:, :-1]
    h11 = height_m[1:, 1:]
    gx_bottom = (h01 - h00) / dx_m
    gx_top = (h11 - h10) / dx_m
    gy_left = (h10 - h00) / dy_m
    gy_right = (h11 - h01) / dy_m
    maximum_triangle_slope = float(
        max(
            np.max(np.hypot(gx_bottom, gy_left)),
            np.max(np.hypot(gx_top, gy_right)),
            np.max(np.hypot(gx_top, gy_left)),
            np.max(np.hypot(gx_bottom, gy_right)),
        )
    )
    return {
        "dh_dx": dh_dx,
        "dh_dy": dh_dy,
        "slope": slope,
        "curvature": curvature,
        "maximum_absolute_slope": float(np.max(slope)),
        "maximum_triangle_slope": maximum_triangle_slope,
        "maximum_curvature": float(np.max(curvature)),
        "maximum_absolute_height_m": float(np.max(np.abs(height_m))),
    }


def _random_gaussians(
    rng: np.random.Generator,
    count: int,
    sign: float,
    amplitude_fraction: float,
    config: TerrainConfig,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros_like(x_mesh, dtype=np.float64)
    components: list[dict[str, Any]] = []
    if count == 0 or amplitude_fraction == 0.0:
        return result, components
    total_components = max(1, config.hill_count + config.pit_count)
    base_amplitude = config.maximum_height_m * amplitude_fraction / math.sqrt(total_components)
    maximum_sigma = max(
        config.minimum_feature_width_m,
        min(
            2.0 * config.minimum_feature_width_m,
            config.terrain_length_m / 7.0,
            config.terrain_width_m / 7.0,
        ),
    )
    for index in range(count):
        sigma_x = float(rng.uniform(config.minimum_feature_width_m, maximum_sigma))
        sigma_y = float(rng.uniform(config.minimum_feature_width_m, maximum_sigma))
        margin_x = 2.0 * sigma_x
        margin_y = 2.0 * sigma_y
        centre_x = float(rng.uniform(-0.5 * config.terrain_length_m + margin_x, 0.5 * config.terrain_length_m - margin_x))
        centre_y = float(rng.uniform(-0.5 * config.terrain_width_m + margin_y, 0.5 * config.terrain_width_m - margin_y))
        angle = float(rng.uniform(0.0, math.pi))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x_local = cosine * (x_mesh - centre_x) + sine * (y_mesh - centre_y)
        y_local = -sine * (x_mesh - centre_x) + cosine * (y_mesh - centre_y)
        amplitude = sign * base_amplitude * float(rng.uniform(0.55, 1.0))
        gaussian = amplitude * np.exp(-0.5 * ((x_local / sigma_x) ** 2 + (y_local / sigma_y) ** 2))
        result += gaussian
        components.append(
            {
                "kind": "gaussian_hill" if sign > 0 else "gaussian_pit",
                "index": index,
                "centre_x_m": centre_x,
                "centre_y_m": centre_y,
                "sigma_x_m": sigma_x,
                "sigma_y_m": sigma_y,
                "orientation_rad": angle,
                "requested_amplitude_m": amplitude,
                "feature_width_m": min(sigma_x, sigma_y),
            }
        )
    return result, components


def _eligible_fourier_modes(config: TerrainConfig) -> list[tuple[int, int, float]]:
    maximum_x_mode = max(1, int(config.terrain_length_m // (4.0 * config.minimum_feature_width_m)))
    maximum_y_mode = max(1, int(config.terrain_width_m // (4.0 * config.minimum_feature_width_m)))
    modes: list[tuple[int, int, float]] = []
    for mode_x in range(0, maximum_x_mode + 1):
        for mode_y in range(-maximum_y_mode, maximum_y_mode + 1):
            if mode_x == 0 and mode_y <= 0:
                continue
            wave_number = math.hypot(
                2.0 * math.pi * mode_x / config.terrain_length_m,
                2.0 * math.pi * mode_y / config.terrain_width_m,
            )
            quarter_wavelength = math.pi / (2.0 * wave_number)
            if quarter_wavelength + 1e-12 >= config.minimum_feature_width_m:
                modes.append((mode_x, mode_y, quarter_wavelength))
    return modes


def _random_fourier_surface(
    rng: np.random.Generator,
    config: TerrainConfig,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros_like(x_mesh, dtype=np.float64)
    components: list[dict[str, Any]] = []
    count = config.random_fourier_terms
    if count == 0 or config.fourier_amplitude_fraction == 0.0:
        return result, components
    modes = _eligible_fourier_modes(config)
    if count > len(modes):
        raise ValueError(
            f"random_fourier_terms={count} exceeds {len(modes)} eligible low-frequency modes"
        )
    selected = rng.choice(len(modes), size=count, replace=False)
    amplitude_scale = config.maximum_height_m * config.fourier_amplitude_fraction / math.sqrt(count)
    for index, selected_index in enumerate(np.asarray(selected, dtype=int)):
        mode_x, mode_y, quarter_wavelength = modes[int(selected_index)]
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        signed_amplitude = amplitude_scale * float(rng.uniform(-1.0, 1.0))
        argument = (
            2.0 * math.pi * mode_x * x_mesh / config.terrain_length_m
            + 2.0 * math.pi * mode_y * y_mesh / config.terrain_width_m
            + phase
        )
        result += signed_amplitude * np.sin(argument)
        components.append(
            {
                "kind": "fourier",
                "index": index,
                "mode_x": mode_x,
                "mode_y": mode_y,
                "phase_rad": phase,
                "requested_amplitude_m": signed_amplitude,
                "feature_width_m": quarter_wavelength,
            }
        )
    return result, components


def generate_terrain(
    config: TerrainConfig,
    *,
    stochastic_residual_scale_cap: float | None = None,
) -> TerrainData:
    """Generate one deterministic field and enforce every numerical upper bound.

    ``stochastic_residual_scale_cap`` can impose a common conservative amplitude
    across otherwise identical resolution variants.  It affects only the
    stochastic residual; the global-slope base and safe-region construction are
    unchanged.
    """

    validate_config(config)
    scale_cap: float | None = None
    if stochastic_residual_scale_cap is not None:
        if isinstance(stochastic_residual_scale_cap, (bool, np.bool_)):
            raise ValueError(
                "stochastic_residual_scale_cap must be a finite number in [0, 1]"
            )
        scale_cap = float(stochastic_residual_scale_cap)
        if not math.isfinite(scale_cap) or not 0.0 <= scale_cap <= 1.0:
            raise ValueError("stochastic_residual_scale_cap must be finite and in [0, 1]")
    x_coordinates = np.linspace(
        -0.5 * config.terrain_length_m,
        0.5 * config.terrain_length_m,
        config.ncol,
        dtype=np.float64,
    )
    y_coordinates = np.linspace(
        -0.5 * config.terrain_width_m,
        0.5 * config.terrain_width_m,
        config.nrow,
        dtype=np.float64,
    )
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    dx = float(x_coordinates[1] - x_coordinates[0])
    dy = float(y_coordinates[1] - y_coordinates[0])

    seed_sequence = np.random.SeedSequence(config.terrain_seed)
    hill_seed, pit_seed, fourier_seed = seed_sequence.spawn(3)
    hills, hill_components = _random_gaussians(
        np.random.Generator(np.random.PCG64(hill_seed)),
        config.hill_count,
        +1.0,
        config.hill_amplitude_fraction,
        config,
        x_mesh,
        y_mesh,
    )
    pits, pit_components = _random_gaussians(
        np.random.Generator(np.random.PCG64(pit_seed)),
        config.pit_count,
        -1.0,
        config.pit_amplitude_fraction,
        config,
        x_mesh,
        y_mesh,
    )
    fourier, fourier_components = _random_fourier_surface(
        np.random.Generator(np.random.PCG64(fourier_seed)), config, x_mesh, y_mesh
    )
    stochastic_surface = _smooth_surface(hills + pits + fourier, config.smoothing_strength, dx, dy)
    base = config.global_slope_x * x_mesh + config.global_slope_y * y_mesh
    residual = stochastic_surface
    # A fixed physical guard makes the safe-core geometry resolution-independent.
    # The eight-interval feature rule guarantees at least two coarse-grid cells.
    grid_guard = 0.25 * config.minimum_feature_width_m
    for region in (config.start_safe_region, config.goal_safe_region):
        base = _apply_safe_region(
            base, x_mesh, y_mesh, x_coordinates, y_coordinates, region, grid_guard
        )
        residual = _apply_safe_region(
            residual, x_mesh, y_mesh, x_coordinates, y_coordinates, region, grid_guard
        )
    for surface_name, surface in (("base", base), ("residual", residual)):
        start_value = _bilinear_from_grid(
            x_coordinates,
            y_coordinates,
            surface,
            config.start_safe_region.centre_x_m,
            config.start_safe_region.centre_y_m,
        )
        if surface_name == "base":
            base = surface - start_value
        else:
            residual = surface - start_value

    base_metrics = differential_metrics(base, dx, dy)
    residual_metrics = differential_metrics(residual, dx, dy)
    base_slope_bound = max(
        base_metrics["maximum_absolute_slope"], base_metrics["maximum_triangle_slope"]
    )
    residual_slope_bound = max(
        residual_metrics["maximum_absolute_slope"],
        residual_metrics["maximum_triangle_slope"],
    )
    base_bounds = {
        "height": base_metrics["maximum_absolute_height_m"],
        "slope": base_slope_bound,
        "curvature": base_metrics["maximum_curvature"],
    }
    residual_bounds = {
        "height": residual_metrics["maximum_absolute_height_m"],
        "slope": residual_slope_bound,
        "curvature": residual_metrics["maximum_curvature"],
    }
    declared_limits = {
        "height": config.maximum_height_m,
        "slope": config.maximum_absolute_slope,
        "curvature": config.maximum_curvature,
    }
    ratios = [1.0]
    for name, limit in declared_limits.items():
        if base_bounds[name] > limit + 1e-10:
            raise ValueError(
                f"global slope plus safe-region blending violates the {name} limit before "
                "random relief is added; widen the blend or revise the configured bound"
            )
        if residual_bounds[name] > 0.0:
            ratios.append(max(0.0, (limit - base_bounds[name]) / residual_bounds[name]))
    native_constraint_scale = max(0.0, min(ratios))
    if native_constraint_scale < 1.0:
        native_constraint_scale *= 1.0 - 64.0 * np.finfo(np.float64).eps
    constraint_scale = (
        native_constraint_scale
        if scale_cap is None
        else min(native_constraint_scale, scale_cap)
    )
    unscaled = differential_metrics(base + residual, dx, dy)
    final_metrics: dict[str, Any] | None = None
    minimum_height = 0.0
    physical_range = 0.0
    normalised = np.zeros_like(base)
    height = np.zeros_like(base)
    for quantisation_attempt in range(8):
        unquantised_height = np.ascontiguousarray(
            base + constraint_scale * residual, dtype=np.float64
        )
        raw_minimum = float(np.min(unquantised_height))
        raw_maximum = float(np.max(unquantised_height))
        physical_range = raw_maximum - raw_minimum
        if physical_range > 0.0:
            unquantised_normalised = np.clip(
                (unquantised_height - raw_minimum) / physical_range, 0.0, 1.0
            )
            normalised = np.asarray(unquantised_normalised, dtype=np.float32).astype(np.float64)
            decoded = raw_minimum + normalised * physical_range
        else:
            normalised = np.zeros_like(unquantised_height)
            decoded = np.zeros_like(unquantised_height)
        decoded_start = _bilinear_from_grid(
            x_coordinates,
            y_coordinates,
            decoded,
            config.start_safe_region.centre_x_m,
            config.start_safe_region.centre_y_m,
        )
        minimum_height = raw_minimum - decoded_start
        normalised = np.ascontiguousarray(normalised, dtype=np.float64)
        height = np.ascontiguousarray(
            minimum_height + normalised * physical_range, dtype=np.float64
        )
        final_metrics = differential_metrics(height, dx, dy)
        observed_limits = {
            "height": final_metrics["maximum_absolute_height_m"],
            "slope": max(
                final_metrics["maximum_absolute_slope"],
                final_metrics["maximum_triangle_slope"],
            ),
            "curvature": final_metrics["maximum_curvature"],
        }
        if all(
            observed_limits[name] <= declared_limits[name] + 1e-12
            for name in declared_limits
        ):
            break
        if constraint_scale <= 0.0:
            raise ValueError(
                "the global-slope base violates a declared bound after MuJoCo float32 "
                "heightfield quantisation; increase the numerical margin"
            )
        correction = min(
            declared_limits[name] / observed_limits[name]
            for name in declared_limits
            if observed_limits[name] > declared_limits[name]
        )
        constraint_scale *= max(0.0, correction * (1.0 - 1e-6))
    else:
        raise RuntimeError("could not certify the float32 MuJoCo heightfield after eight attempts")
    assert final_metrics is not None

    components = hill_components + pit_components + fourier_components
    constructed_widths = [float(item["feature_width_m"]) for item in components]
    constructed_widths.extend(
        [config.start_safe_region.blend_width_m, config.goal_safe_region.blend_width_m]
    )
    metadata: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "config_sha256": config_sha256(config),
        "seed_namespace": config.split,
        "terrain_seed": config.terrain_seed,
        "coordinate_convention": "height_m[row_y, column_x], with x and y increasing by index",
        "grid_spacing_m": {"dx": dx, "dy": dy},
        "normalisation": {
            "formula": "normalised=(height_m-offset_m)/scale_m",
            "physical_offset_m": minimum_height,
            "physical_scale_m": physical_range,
            "mujoco_vertical_scale_m": max(physical_range, 1e-6),
            "mujoco_storage_dtype": "float32",
            "physical_array_decoded_from_mujoco_values": True,
        },
        "constraint_scale": constraint_scale,
        "native_constraint_scale": native_constraint_scale,
        "applied_constraint_scale": constraint_scale,
        "stochastic_residual_scale_cap": scale_cap,
        "constraint_scale_scope": "stochastic residual only; the global-slope base is preserved",
        "base_bounds": base_bounds,
        "residual_bounds": residual_bounds,
        "metrics_before_constraint_scale": {
            key: value for key, value in unscaled.items() if isinstance(value, float)
        },
        "metrics": {key: value for key, value in final_metrics.items() if isinstance(value, float)},
        "components": components,
        "constructed_feature_widths_m": constructed_widths,
        "minimum_constructed_feature_width_m": min(constructed_widths),
        "minimum_feature_width_definition": (
            "conservative construction scale: Gaussian sigma, Fourier quarter-wavelength, "
            "or safe-region transition width"
        ),
        "safe_region_blend": {
            "method": "radial quintic smootherstep (C2 at both transition edges)",
            "grid_guard_m": grid_guard,
        },
        "friction_randomised": False,
        "float32_quantisation_attempts": quantisation_attempt + 1,
    }
    terrain = TerrainData(config, x_coordinates, y_coordinates, height, normalised, metadata)
    metadata["height_sha256"] = terrain.height_sha256
    metadata["normalised_sha256"] = terrain.normalised_sha256
    return terrain


def load_config(path: str | Path) -> TerrainConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TerrainConfig.from_mapping(json.load(handle))


def save_terrain_bundle(
    terrain: TerrainData,
    output_directory: str | Path,
    stem: str,
) -> dict[str, Path]:
    """Save arrays and an auditable manifest without altering deterministic data."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    height_path = output / f"{stem}_height_m.npy"
    normalised_path = output / f"{stem}_normalised.npy"
    manifest_path = output / f"{stem}_manifest.json"
    np.save(height_path, terrain.height_m, allow_pickle=False)
    np.save(normalised_path, terrain.normalised_height, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": terrain.config.to_dict(),
        "config_sha256": config_sha256(terrain.config),
        "split": terrain.config.split,
        "terrain_seed": terrain.config.terrain_seed,
        "height_array": {
            "path": height_path.name,
            "shape": list(terrain.height_m.shape),
            "dtype": str(terrain.height_m.dtype),
            "canonical_array_sha256": terrain.height_sha256,
            "file_sha256": file_sha256(height_path),
        },
        "normalised_array": {
            "path": normalised_path.name,
            "shape": list(terrain.normalised_height.shape),
            "dtype": str(terrain.normalised_height.dtype),
            "canonical_array_sha256": terrain.normalised_sha256,
            "file_sha256": file_sha256(normalised_path),
        },
        "metadata": terrain.metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "height": height_path,
        "normalised": normalised_path,
        "manifest": manifest_path,
    }
