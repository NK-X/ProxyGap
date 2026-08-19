from __future__ import annotations

from pathlib import Path
import builtins
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_fixed_goal_dual_view_video import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAP_SIZE,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    audited_contract_controller,
    compose_dual_view,
    load_video_encoder as load_dual_view_encoder,
    overview_camera,
    resample_surface_trail,
)
import render_fixed_goal_training_video as training_video  # noqa: E402


def test_video_modules_import_without_pyav_and_fail_only_on_video_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_av(name: str, *args: object, **kwargs: object) -> object:
        if name == "av":
            raise ModuleNotFoundError("No module named 'av'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_av)
    monkeypatch.setattr(training_video, "_AV_MODULE", None)
    with pytest.raises(RuntimeError, match="requires? PyAV"):
        load_dual_view_encoder()
    with pytest.raises(RuntimeError, match="requires? PyAV"):
        training_video.load_video_encoder()


def test_paired_audit_contract_controller_includes_cruise_speed() -> None:
    condition, speed = audited_contract_controller(
        {
            "controller": {
                "cruise_speed_m_per_s": 0.5,
                "yaw_gain_per_second": 0.75,
                "maximum_abs_curvature_per_m": 0.35,
                "slow_radius_m": 4.0,
            }
        }
    )
    assert speed == 0.5
    assert condition["slow_radius_m"] == 4.0


def test_overview_camera_is_on_goal_side_for_diagonal_task() -> None:
    camera = overview_camera(
        start=np.asarray((-34.0, -34.0)),
        goal=np.asarray((34.0, 34.0)),
        half_extent=40.0,
        terrain_midpoint_height=0.0,
    )
    assert np.isclose(camera.azimuth, 225.0)
    assert np.isclose(camera.elevation, -55.0)
    assert np.isclose(camera.distance, 132.0)
    assert np.allclose(camera.lookat, np.zeros(3))


def test_surface_trail_resampling_caps_geometry_and_preserves_endpoints() -> None:
    parameter = np.linspace(0.0, 1.0, 1001)
    points = np.column_stack((parameter, parameter**2, np.sin(parameter)))
    sampled = resample_surface_trail(points, maximum_points=51)
    assert sampled.shape == (51, 3)
    assert np.allclose(sampled[0], points[0])
    assert np.allclose(sampled[-1], points[-1])


def test_dual_view_composition_has_requested_geometry_and_white_panel() -> None:
    left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
    right = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), 32, dtype=np.uint8)
    map_base = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (80, 100, 120))
    frame = compose_dual_view(
        left,
        right,
        map_base=map_base,
        trail_xy=[np.asarray((-34.0, -34.0)), np.asarray((-33.0, -33.0))],
        start=np.asarray((-34.0, -34.0)),
        goal=np.asarray((34.0, 34.0)),
        position=np.asarray((-33.0, -33.0)),
        half_extent=40.0,
        physical_time=1.0,
        requested_seconds=600.0,
        distance=94.0,
        best_progress=2.0,
        torso_tilt_degrees=4.0,
        support_count=2,
        maximum_contact_speed=0.1,
        slip_threshold=0.2,
        current_airborne=False,
        ever_airborne=False,
        ever_contact_speed_exceeded=False,
        unhealthy_termination=False,
        spatial_success=False,
        evaluation_seed=74803,
        evaluation_group_index=3,
        evaluation_group_count=5,
        checkpoint_name="checkpoint_2465792.zip",
        commanded_speed=0.5,
        yaw_gain=0.75,
        maximum_curvature=0.35,
        floor_friction=np.asarray((1.0, 0.5, 0.5)),
        floor_condim=3,
        map_hash="59e60ddd91d799f44f84aa74a2ecff1",
    )
    assert frame.size == (FRAME_WIDTH, FRAME_HEIGHT)
    pixels = np.asarray(frame)
    assert np.all(pixels[FRAME_HEIGHT - 1, 0] == np.asarray((250, 249, 245)))
