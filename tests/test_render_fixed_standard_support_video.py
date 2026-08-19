from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_fixed_standard_support_video import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    compose_frame,
    side_overview_camera,
)


def test_side_overview_exposes_physical_x_slope() -> None:
    camera = side_overview_camera(height_midpoint=0.0)
    assert np.isclose(camera.azimuth, 90.0)
    assert np.isclose(camera.elevation, -22.0)
    assert np.isclose(camera.distance, 28.0)
    assert np.allclose(camera.lookat, np.asarray((0.0, 0.0, 0.15)))


def test_standard_support_composition_retains_delivery_geometry() -> None:
    left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
    right = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), 24, dtype=np.uint8)
    frame = compose_frame(
        left,
        right,
        condition_id="MATCHED_CONTACT_GAP_W0_CONTROL",
        gate_passed=False,
        scene_name="uphill_8deg",
        slope_degrees=8.0,
        evaluation_seed=76802,
        physical_time=1.0,
        horizon_seconds=30.0,
        position=np.asarray((-5.5, 0.0, 0.0)),
        support_count=2,
        airborne=False,
        ever_airborne=True,
        best_progress_m=0.5,
        distance_to_goal_m=11.5,
        relative_tilt_degrees=3.0,
        maximum_contact_speed=0.1,
        height_min_m=-1.405,
        height_max_m=1.405,
        checkpoint_name="checkpoint_2662400.zip",
        friction=[1.0, 0.5, 0.5],
        map_hash="5602c376a6af0b6834196fcfc2a33969",
        terminated=False,
        truncated=False,
    )
    assert frame.size == (FRAME_WIDTH, FRAME_HEIGHT)
    pixels = np.asarray(frame)
    assert np.all(pixels[-1, 0] == np.asarray((250, 249, 245)))
