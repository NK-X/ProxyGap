"""Physical-coordinate queries for generated Ant terrain."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from terrain_generator import SafeRegionConfig, TerrainData


class TerrainQueries:
    """Query MuJoCo's piecewise triangular surface convention.

    Each cell is split along its lower-left to upper-right diagonal.  The
    stored nodes are decoded from the same float32 values written to MuJoCo, so
    the result agrees up to ordinary floating-point arithmetic.  The underlying
    sampled recipe is smooth, while the collision representation is necessarily
    faceted at finite resolution.
    """

    def __init__(self, terrain: TerrainData, spawn_footprint_radius_m: float = 0.65):
        self.terrain = terrain
        self.spawn_footprint_radius_m = float(spawn_footprint_radius_m)
        if self.spawn_footprint_radius_m <= 0.0:
            raise ValueError("spawn_footprint_radius_m must be positive")
        self._x = terrain.x_coordinates_m
        self._y = terrain.y_coordinates_m
        self._height = terrain.height_m
        self._dx = float(self._x[1] - self._x[0])
        self._dy = float(self._y[1] - self._y[0])

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int, float, float]:
        x = float(x_m)
        y = float(y_m)
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, abs(self._x[0]), abs(self._x[-1]), abs(self._y[0]), abs(self._y[-1])
        )
        if x < self._x[0] - tolerance or x > self._x[-1] + tolerance:
            raise ValueError(f"x={x} m is outside [{self._x[0]}, {self._x[-1]}] m")
        if y < self._y[0] - tolerance or y > self._y[-1] + tolerance:
            raise ValueError(f"y={y} m is outside [{self._y[0]}, {self._y[-1]}] m")
        x = float(np.clip(x, self._x[0], self._x[-1]))
        y = float(np.clip(y, self._y[0], self._y[-1]))
        column_float = (x - self._x[0]) / self._dx
        row_float = (y - self._y[0]) / self._dy
        column = min(int(math.floor(column_float)), self._x.size - 2)
        row = min(int(math.floor(row_float)), self._y.size - 2)
        return row, column, column_float - column, row_float - row

    def height(self, x_m: float, y_m: float) -> float:
        """Return MuJoCo-compatible piecewise triangular height in metres."""

        row, column, tx, ty = self._cell(x_m, y_m)
        h00 = self._height[row, column]
        h01 = self._height[row, column + 1]
        h10 = self._height[row + 1, column]
        h11 = self._height[row + 1, column + 1]
        if ty <= tx:
            return float(h00 * (1.0 - tx) + h01 * (tx - ty) + h11 * ty)
        return float(h00 * (1.0 - ty) + h11 * tx + h10 * (ty - tx))

    def gradient(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return ``(dh/dx, dh/dy)`` for the local MuJoCo triangle."""

        row, column, tx, ty = self._cell(x_m, y_m)
        h00 = self._height[row, column]
        h01 = self._height[row, column + 1]
        h10 = self._height[row + 1, column]
        h11 = self._height[row + 1, column + 1]
        if ty <= tx:
            dh_dx = (h01 - h00) / self._dx
            dh_dy = (h11 - h01) / self._dy
        else:
            dh_dx = (h11 - h10) / self._dx
            dh_dy = (h10 - h00) / self._dy
        return float(dh_dx), float(dh_dy)

    def normal(self, x_m: float, y_m: float) -> tuple[float, float, float]:
        """Return the upward unit normal ``(-dh/dx, -dh/dy, 1)``."""

        dh_dx, dh_dy = self.gradient(x_m, y_m)
        length = math.sqrt(1.0 + dh_dx * dh_dx + dh_dy * dh_dy)
        return -dh_dx / length, -dh_dy / length, 1.0 / length

    def slope_along(self, x_m: float, y_m: float, vx: float, vy: float) -> float:
        """Return signed rise/run in the normalised horizontal travel direction."""

        speed = math.hypot(float(vx), float(vy))
        if speed <= 1e-15:
            raise ValueError("slope_along requires a non-zero horizontal direction")
        dh_dx, dh_dy = self.gradient(x_m, y_m)
        return float(dh_dx * float(vx) / speed + dh_dy * float(vy) / speed)

    @staticmethod
    def _disc_offsets(radius_m: float) -> Iterable[tuple[float, float]]:
        yield 0.0, 0.0
        for fraction in (0.5, 1.0):
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                yield fraction * radius_m * math.cos(float(angle)), fraction * radius_m * math.sin(float(angle))

    def _is_safe_disc(
        self,
        x_m: float,
        y_m: float,
        region: SafeRegionConfig,
        footprint_radius_m: float,
    ) -> bool:
        centre_distance = math.hypot(x_m - region.centre_x_m, y_m - region.centre_y_m)
        if centre_distance + footprint_radius_m > region.radius_m + 1e-12:
            return False
        for offset_x, offset_y in self._disc_offsets(footprint_radius_m):
            query_x = x_m + offset_x
            query_y = y_m + offset_y
            if not (self._x[0] <= query_x <= self._x[-1] and self._y[0] <= query_y <= self._y[-1]):
                return False
            dh_dx, dh_dy = self.gradient(query_x, query_y)
            if math.hypot(dh_dx, dh_dy) > region.maximum_absolute_slope + 1e-10:
                return False
        return True

    def is_safe_spawn(self, x_m: float, y_m: float) -> bool:
        """Check a footprint disc, rather than only the proposed torso centre."""

        return self._is_safe_disc(
            float(x_m),
            float(y_m),
            self.terrain.config.start_safe_region,
            self.spawn_footprint_radius_m,
        )

    def sample_safe_goal(self, seed: int) -> tuple[float, float]:
        """Sample a goal without consuming or changing the terrain RNG stream."""

        region = self.terrain.config.goal_safe_region
        usable_radius = region.radius_m - self.spawn_footprint_radius_m
        if usable_radius <= 0.0:
            raise RuntimeError("goal safe region is smaller than the configured footprint")
        seed_material = (
            f"ant-terrain-goal-v1|{self.terrain.height_sha256}|{int(seed)}".encode("ascii")
        )
        entropy = np.frombuffer(hashlib.sha256(seed_material).digest(), dtype="<u4")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        for _ in range(1024):
            radius = usable_radius * math.sqrt(float(rng.random()))
            angle = 2.0 * math.pi * float(rng.random())
            candidate_x = region.centre_x_m + radius * math.cos(angle)
            candidate_y = region.centre_y_m + radius * math.sin(angle)
            if self._is_safe_disc(
                candidate_x,
                candidate_y,
                region,
                self.spawn_footprint_radius_m,
            ):
                return float(candidate_x), float(candidate_y)
        raise RuntimeError("no safe goal candidate was found after 1024 deterministic attempts")


def height(terrain: TerrainData, x_m: float, y_m: float) -> float:
    return TerrainQueries(terrain).height(x_m, y_m)


def gradient(terrain: TerrainData, x_m: float, y_m: float) -> tuple[float, float]:
    return TerrainQueries(terrain).gradient(x_m, y_m)


def normal(terrain: TerrainData, x_m: float, y_m: float) -> tuple[float, float, float]:
    return TerrainQueries(terrain).normal(x_m, y_m)


def slope_along(
    terrain: TerrainData,
    x_m: float,
    y_m: float,
    vx: float,
    vy: float,
) -> float:
    return TerrainQueries(terrain).slope_along(x_m, y_m, vx, vy)


def is_safe_spawn(terrain: TerrainData, x_m: float, y_m: float) -> bool:
    return TerrainQueries(terrain).is_safe_spawn(x_m, y_m)


def sample_safe_goal(terrain: TerrainData, seed: int) -> tuple[float, float]:
    return TerrainQueries(terrain).sample_safe_goal(seed)
