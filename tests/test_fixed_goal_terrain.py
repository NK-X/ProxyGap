from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.fixed_goal_terrain import FixedGoalTerrainWrapper, file_sha256
from run_fixed_goal_terrain_training import (
    make_task_env,
    prepare_task_scenes,
    terrain_value,
)


CONFIG_PATH = ROOT / "configs" / "fixed_quad_terrain_v2_training_20260818.json"


def load_configs() -> tuple[dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v22 = json.loads((ROOT / config["base_policy"]["configuration"]).read_text(encoding="utf-8"))
    return config, v22


def test_approved_height_array_matches_frozen_identity() -> None:
    config, _ = load_configs()
    path = ROOT / config["approved_map"]["heights_path"]
    assert file_sha256(path) == config["approved_map"]["heights_sha256"]
    heights = np.load(path, allow_pickle=False)
    assert heights.shape == (1025, 1025)
    assert np.ptp(heights) == 6.0


def test_task_scene_and_reset_use_lower_left_spawn_without_absolute_z_failure(
    tmp_path: Path,
) -> None:
    config, v22 = load_configs()
    output = tmp_path / "fixed_goal_task"
    output.mkdir()
    scenes, metadata = prepare_task_scenes(config, output, [0.0])
    assert len(scenes) == 1
    assert metadata[0]["spawn_fraction"] == 0.0

    env = make_task_env(
        config,
        v22,
        xml_path=scenes[0],
        seed=901,
        spawn_fraction=0.0,
        max_episode_steps=20,
        cruise_speed=0.55,
        terminate_on_success=False,
    )
    observation, info = env.reset(seed=901)
    assert observation.shape == (122,)
    position = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)
    assert np.linalg.norm(position - np.asarray([-34.0, -34.0])) < 0.25
    assert info["proxygap_fixed_goal_distance_m"] > 95.0
    assert env.unwrapped._terminate_when_unhealthy is False

    expected_airborne_steps = 0
    for _ in range(10):
        _, _, terminated, truncated, step_info = env.step(
            np.zeros(8, dtype=np.float64)
        )
        contact_mask = np.asarray(
            step_info["proxygap_foot_contact_mask_step"], dtype=bool
        )
        expected_airborne_steps += int(not np.any(contact_mask))
        assert terminated is False
        assert truncated is False
        assert step_info["proxygap_fixed_goal_distance_m"] > 95.0
    summary = env.episode_summary()
    assert summary["task_airborne_step_count"] == expected_airborne_steps
    assert np.isclose(
        summary["task_airborne_step_fraction"],
        expected_airborne_steps / 10,
    )
    env.close()


def test_terrain_value_recovers_fixed_start_and_goal_heights() -> None:
    config, _ = load_configs()
    approved = config["approved_map"]
    heights = np.load(ROOT / approved["heights_path"], allow_pickle=False)
    extent = float(approved["map_half_extent_m"])
    start_height = terrain_value(heights, -34.0, -34.0, extent)
    goal_height = terrain_value(heights, 34.0, 34.0, extent)
    assert np.isclose(start_height, -1.9348310229819923, atol=1e-10)
    assert np.isclose(goal_height, 0.2951027006509664, atol=1e-10)


def test_optional_local_terrain_preview_adds_only_thirteen_finite_values(
    tmp_path: Path,
) -> None:
    config, v22 = load_configs()
    preview_config = copy.deepcopy(config)
    preview_config["task_adapter"].update(
        {
            "augment_local_terrain_observation": True,
            "terrain_preview_longitudinal_m": [0.5, 1.0, 1.5],
            "terrain_preview_lateral_m": [-0.4, 0.0, 0.4],
        }
    )
    output = tmp_path / "terrain_preview_task"
    output.mkdir()
    scenes, _ = prepare_task_scenes(preview_config, output, [0.0])
    env = make_task_env(
        preview_config,
        v22,
        xml_path=scenes[0],
        seed=902,
        spawn_fraction=0.0,
        max_episode_steps=5,
        cruise_speed=0.55,
        terminate_on_success=False,
    )
    observation, info = env.reset(seed=902)
    assert observation.shape == (135,)
    assert np.all(np.isfinite(observation[-13:]))
    assert env.observation_space.contains(observation)
    assert info["proxygap_local_terrain_observation_enabled"] is True
    next_observation, _, _, _, step_info = env.step(
        np.zeros(8, dtype=np.float64)
    )
    assert next_observation.shape == (135,)
    assert np.all(np.isfinite(next_observation[-13:]))
    assert step_info["proxygap_local_terrain_observation_enabled"] is True
    env.close()


def test_hold_annulus_cannot_establish_success_before_true_arrival() -> None:
    wrapper = object.__new__(FixedGoalTerrainWrapper)
    wrapper.arrival_radius = 1.5
    wrapper.hold_radius = 2.0
    wrapper.required_hold_steps = 3
    wrapper._goal_entered = False
    wrapper._goal_hold_run_steps = 0
    wrapper._longest_goal_hold_run_steps = 0
    wrapper._task_success = False
    wrapper._success_step = None
    wrapper._task_steps = 10

    for _ in range(4):
        wrapper._update_goal_state(1.6)
    assert wrapper._goal_entered is False
    assert wrapper._goal_hold_run_steps == 0
    assert wrapper._task_success is False

    wrapper._update_goal_state(1.4)
    wrapper._update_goal_state(1.8)
    wrapper._task_steps = 13
    wrapper._update_goal_state(1.9)
    assert wrapper._goal_entered is True
    assert wrapper._longest_goal_hold_run_steps == 3
    assert wrapper._task_success is True
    assert wrapper._success_step == 13
