from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_goal_local_preview_pilot import validate_config
from run_fixed_goal_terrain_training import make_task_env, prepare_task_scenes


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_local_preview_pilot_v1_20260819.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_preview_pilot_freezes_an_observation_only_intervention() -> None:
    config = load_config()
    control, _ = validate_config(config)
    assert config["approved_map"] == control["approved_map"]
    assert config["evaluation"] == control["evaluation"]
    assert config["task_adapter"]["additional_task_reward"] == 0.0
    assert config["task_adapter"]["augment_local_terrain_observation"] is True
    assert config["observation_transfer"]["source_observation_dimension"] == 122
    assert config["observation_transfer"]["target_observation_dimension"] == 135
    assert config["observation_transfer"]["appended_columns"] == 13
    assert config["observation_transfer"]["optimizer_state"] == "fresh"


def test_preview_environment_appends_thirteen_values_after_the_122_value_prefix(
    tmp_path: Path,
) -> None:
    preview_config = load_config()
    control_config, v22_config = validate_config(preview_config)
    output = tmp_path / "preview_task"
    output.mkdir()
    scenes, _ = prepare_task_scenes(preview_config, output, [0.0])

    preview_env = make_task_env(
        preview_config,
        v22_config,
        xml_path=scenes[0],
        seed=811,
        spawn_fraction=0.0,
        max_episode_steps=20,
        cruise_speed=0.55,
        terminate_on_success=False,
    )
    control_env = make_task_env(
        control_config,
        v22_config,
        xml_path=scenes[0],
        seed=811,
        spawn_fraction=0.0,
        max_episode_steps=20,
        cruise_speed=0.55,
        terminate_on_success=False,
    )
    preview_observation, preview_info = preview_env.reset(seed=811)
    control_observation, _ = control_env.reset(seed=811)
    try:
        assert preview_observation.shape == (135,)
        assert control_observation.shape == (122,)
        np.testing.assert_array_equal(
            preview_observation[:122],
            control_observation,
        )
        terrain_preview = preview_observation[122:]
        assert np.all(np.isfinite(terrain_preview))
        np.testing.assert_allclose(
            np.linalg.norm(terrain_preview[9:12]),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        assert -math.pi / 2.0 <= float(terrain_preview[12]) <= math.pi / 2.0
        assert preview_info["proxygap_local_terrain_observation_enabled"] is True
    finally:
        preview_env.close()
        control_env.close()


def test_preview_pilot_rejects_a_friction_change() -> None:
    config = copy.deepcopy(load_config())
    config["approved_map"]["fixed_friction"][0] = 1.1
    with pytest.raises(ValueError, match="Approved map, friction or XML"):
        validate_config(config)
