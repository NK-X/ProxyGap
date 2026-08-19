from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_fixed_standard_pair0_slope_delivery_video import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_EPISODES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    compare_episode_rows,
    compose_frame,
    fixed_relief_camera,
    follow_camera,
    formal_row_for,
    read_csv,
    validate_config,
)
import run_fixed_standard_pair0_adaptation_l2_pilot as l2  # noqa: E402


def load_config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_canonical_contract_validates_and_freezes_corrected_median_seeds() -> None:
    config = load_config()
    validate_config(config)
    observed = tuple(
        (
            row["scene_name"],
            float(row["signed_slope_degrees"]),
            int(row["evaluation_seed"]),
        )
        for row in config["representative_episodes"]
    )
    assert observed == EXPECTED_EPISODES
    assert observed[0] == ("uphill_12deg", 12.0, 94153)
    assert "33/600 = 0.055" in config["representative_episodes"][0]["selection_note"]


def test_selected_rows_are_exact_scene_medians_and_pass_safety() -> None:
    config = load_config()
    rows = read_csv(ROOT / config["source"]["formal_episode_metrics"])
    for scene_name, _, seed in EXPECTED_EPISODES:
        scene_rows = [row for row in rows if row["scene_name"] == scene_name]
        ranked = sorted(scene_rows, key=lambda row: float(row["fixed_goal_best_progress_m"]))
        assert int(ranked[2]["evaluation_seed"]) == seed
        selected = formal_row_for(rows, scene_name, seed)
        assert selected["finite"] is True
        assert selected["fall"] is False
        assert selected["torso_ground_any"] is False
        assert selected["sustained_nonfoot_contact"] is False
        assert selected["corrected_sustained_slip_physics_substep_count"] == 0
        assert selected["corrected_slip_event_count"] == 0
        assert selected["full_interval_zero_foot_count"] / 600 <= 0.0580555556


def test_both_selected_episodes_reproduce_all_28_formal_fields_before_render() -> None:
    config = load_config()
    protocol, reward, checkpoint, formal_config, rows, scenes, _ = validate_config(
        config
    )
    model = PPO.load(checkpoint, device="cpu")
    for scene_name, _, seed in EXPECTED_EPISODES:
        replayed, _, _ = l2.evaluate_episode(
            model,
            formal_config,
            protocol,
            reward,
            scenes[scene_name],
            condition_id=l2.PAIR0_ID,
            seed=seed,
            checkpoint_additional_timesteps=65_536,
            max_episode_steps=600,
            retain_substeps=False,
        )
        archived = formal_row_for(rows, scene_name, seed)
        comparison = compare_episode_rows(archived, replayed)
        assert comparison["field_count"] == 28
        assert comparison["all_fields_exact_match"] is True


def test_fieldwise_metric_comparison_is_strict_and_exhaustive() -> None:
    archived = {
        "condition_id": "PAIR0_ADAPT",
        "control_steps": 600,
        "finite": True,
        "progress": 7.25,
    }
    result = compare_episode_rows(archived, dict(archived))
    assert result["field_count"] == 4
    assert result["all_fields_exact_match"] is True
    changed = dict(archived)
    changed["progress"] = np.nextafter(7.25, 8.0)
    with pytest.raises(RuntimeError, match="progress"):
        compare_episode_rows(archived, changed)
    with pytest.raises(RuntimeError, match="schemas differ"):
        compare_episode_rows(archived, {"control_steps": 600})


def test_cameras_preserve_physical_geometry_and_keep_robot_elevated_in_view() -> None:
    overview = fixed_relief_camera(height_midpoint=0.0)
    assert np.isclose(overview.azimuth, 82.0)
    assert np.isclose(overview.elevation, -24.0)
    assert np.isclose(overview.distance, 28.5)
    assert np.allclose(overview.lookat, (0.0, 0.0, 0.1))
    uphill = follow_camera(
        position=np.asarray((-2.0, 0.0, 1.2)),
        terrain_height=0.5,
        signed_slope_degrees=12.0,
    )
    downhill = follow_camera(
        position=np.asarray((-2.0, 0.0, 1.2)),
        terrain_height=0.5,
        signed_slope_degrees=-16.0,
    )
    assert uphill.elevation == downhill.elevation == -27.0
    assert uphill.azimuth == 132.0
    assert downhill.azimuth == 48.0
    assert uphill.lookat[2] >= 0.92


def test_composition_is_1280x720_and_has_white_bottom_bar() -> None:
    pane = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
    row = {
        "fixed_goal_best_progress_m": 7.372009207381211,
        "full_interval_zero_foot_count": 33,
        "finite": True,
        "fall": False,
        "torso_ground_any": False,
        "sustained_nonfoot_contact": False,
        "corrected_sustained_slip_physics_substep_count": 0,
        "corrected_slip_event_count": 0,
    }
    frame = compose_frame(
        pane,
        pane,
        scene_name="uphill_12deg",
        seed=94153,
        slope_degrees=12.0,
        step=600,
        checkpoint_name="checkpoint_2727936.zip",
        best_progress_m=7.372009207381211,
        endpoint_support_count=2,
        mean_support_count=1.2,
        zerofoot_count=33,
        formal_row=row,
        positive_work_j=4466.0,
        squared_action=166.0,
        start_height_m=-1.275,
        goal_height_m=1.275,
        terminated=False,
        truncated=True,
    )
    assert frame.size == (FRAME_WIDTH, FRAME_HEIGHT)
    array = np.asarray(frame)
    assert float(array[VIEW_HEIGHT + 5 :, :, :].mean()) > 150.0


@pytest.mark.parametrize(
    "path,value",
    [
        (("replay", "training"), True),
        (("replay", "friction_change"), True),
        (("replay", "energy_formula_change"), True),
        (("render", "fps"), 30),
        (("representative_episodes", 0, "evaluation_seed"), 94137),
    ],
)
def test_contract_mutations_fail_closed(path: tuple, value: object) -> None:
    config = load_config()
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate_config(config)
