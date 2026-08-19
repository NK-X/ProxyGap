from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_standard_support_curriculum import (  # noqa: E402
    build_standard_heights,
    contact_masks_from_data,
    install_substep_contact_audit,
    make_standard_env,
    prepare_standard_scenes,
    recursive_json_differences,
    reward_config_with_contact_gap_weight,
    summarise_substep_rows,
    validate_config,
)


CONFIG_PATH = ROOT / "configs" / "fixed_standard_support_curriculum_v1_20260819.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_reward(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_frozen_config_and_single_reward_path_validate() -> None:
    config = load_config()
    _, reward = validate_config(config)
    control = reward_config_with_contact_gap_weight(reward, 0.0)
    intervention = reward_config_with_contact_gap_weight(reward, 1.0)
    differences = recursive_json_differences(control, intervention)
    assert differences == [
        (
            "preserved_pre_pitch_reward.foot_contact_gap_shaping_weight",
            0.0,
            1.0,
        )
    ]


def test_standard_heightfields_have_declared_direction_and_bowl_exit() -> None:
    config = load_config()
    scene = config["standard_scenes"]
    heights = build_standard_heights(scene)
    extent = float(scene["map_half_extent_m"])
    spacing = 2.0 * extent / (int(scene["grid_cols"]) - 1)
    _, flat_dx = np.gradient(heights["flat"], spacing, spacing)
    _, up_dx = np.gradient(heights["uphill_8deg"], spacing, spacing)
    _, down_dx = np.gradient(heights["downhill_8deg"], spacing, spacing)
    expected = math.tan(math.radians(8.0))
    assert np.max(np.abs(flat_dx)) < 1e-6
    assert np.allclose(up_dx, expected, atol=1e-12, rtol=0.0)
    assert np.allclose(down_dx, -expected, atol=1e-12, rtol=0.0)
    bowl = heights["bowl_exit"]
    centre_column = int(round((float(scene["bowl_centre_xy_m"][0]) + extent) / (2 * extent) * (bowl.shape[1] - 1)))
    centre_row = bowl.shape[0] // 2
    goal_column = int(round((float(scene["goal_xy_m"][0]) + extent) / (2 * extent) * (bowl.shape[1] - 1)))
    assert bowl[centre_row, centre_column] < bowl[centre_row, goal_column]


def test_generated_scenes_preserve_robot_and_contact_contract(tmp_path: Path) -> None:
    config = load_config()
    records, manifest = prepare_standard_scenes(config, tmp_path)
    assert list(records) == config["standard_scenes"]["scene_order"]
    assert manifest["robot_signature"]["nu"] == 8
    for record in records.values():
        model = mujoco.MjModel.from_xml_path(str(Path(record["xml_path"])))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert np.array_equal(model.geom_friction[floor_id], np.asarray([1.0, 0.5, 0.5]))
        assert int(model.geom_condim[floor_id]) == 3
        assert int(model.nu) == 8
        assert record["robot_signature_matches_base"] is True


def test_endpoint_and_substep_summary_are_distinct() -> None:
    rows = [
        {
            "endpoint_zero_foot": True,
            "full_interval_zero_foot": False,
            "endpoint_zero_with_earlier_substep_contact": True,
            "zero_foot_substep_fraction": 0.8,
            "endpoint_nonfoot_robot_ground": False,
            "any_substep_nonfoot_robot_ground": True,
            "endpoint_torso_ground": False,
            "endpoint_last_substep_mask_mismatch": False,
        },
        {
            "endpoint_zero_foot": True,
            "full_interval_zero_foot": True,
            "endpoint_zero_with_earlier_substep_contact": False,
            "zero_foot_substep_fraction": 1.0,
            "endpoint_nonfoot_robot_ground": True,
            "any_substep_nonfoot_robot_ground": True,
            "endpoint_torso_ground": True,
            "endpoint_last_substep_mask_mismatch": False,
        },
    ]
    summary = summarise_substep_rows(rows)
    assert summary["endpoint_zero_foot_fraction"] == 1.0
    assert summary["full_interval_zero_foot_fraction"] == 0.5
    assert summary[
        "endpoint_zero_with_earlier_substep_contact_fraction_conditional"
    ] == 0.5
    assert summary[
        "endpoint_zero_with_any_substep_nonfoot_ground_fraction_conditional"
    ] == 1.0


def test_standard_env_supports_135d_and_v20_118d_interfaces(tmp_path: Path) -> None:
    config = load_config()
    records, _ = prepare_standard_scenes(config, tmp_path)
    source_reward = load_reward(config["frozen_sources"]["reward_configuration"])
    v20_reward = load_reward(
        config["frozen_sources"]["diagnostic_support_comparator_configuration"]
    )
    env_135 = make_standard_env(
        config,
        source_reward,
        records["flat"],
        condition_id="TEST_135D",
        seed=1,
        max_episode_steps=5,
        cruise_speed=0.55,
    )
    observation_135, _ = env_135.reset(seed=1)
    assert observation_135.shape == (135,)
    source = PPO.load(ROOT / config["frozen_sources"]["source_checkpoint"])
    assert env_135.observation_space == source.observation_space
    assert np.array_equal(env_135.observation_space.low[-13:-4], np.full(9, -6.0))
    assert np.array_equal(env_135.observation_space.high[-13:-4], np.full(9, 6.0))
    env_135.close()
    env_118 = make_standard_env(
        config,
        v20_reward,
        records["flat"],
        condition_id="TEST_118D",
        seed=1,
        max_episode_steps=5,
        cruise_speed=0.55,
        augment_local_terrain_observation=False,
    )
    observation_118, _ = env_118.reset(seed=1)
    assert observation_118.shape == (118,)
    env_118.close()


def test_substep_audit_endpoint_matches_wrapper_contact(tmp_path: Path) -> None:
    config = load_config()
    records, _ = prepare_standard_scenes(config, tmp_path)
    reward = load_reward(config["frozen_sources"]["reward_configuration"])
    env = make_standard_env(
        config,
        reward,
        records["flat"],
        condition_id="TEST_SUBSTEP",
        seed=2,
        max_episode_steps=5,
        cruise_speed=0.55,
    )
    env.reset(seed=2)
    state = install_substep_contact_audit(env)
    _, _, _, _, info = env.step(np.zeros(8, dtype=np.float32))
    last = state["last"]
    assert last is not None
    endpoint = np.asarray(info["proxygap_foot_contact_mask_step"], dtype=bool)
    assert np.array_equal(endpoint, np.asarray(last["foot_masks"])[-1])
    independent, _, _ = contact_masks_from_data(
        env.unwrapped.model,
        env.unwrapped.data,
        tuple(
            mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "left_ankle_geom",
                "right_ankle_geom",
                "third_ankle_geom",
                "fourth_ankle_geom",
            )
        ),
    )
    assert np.array_equal(endpoint, independent)
    env.close()
