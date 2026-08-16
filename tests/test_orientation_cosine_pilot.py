from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_orientation_pilot_changes_only_the_new_posture_term() -> None:
    config = load_config()
    reward = config["reward"]
    assert reward["forward_reward_weight"] == 1.0
    assert reward["healthy_reward"] == 1.0
    assert reward["ctrl_cost_weight"] == 0.5
    assert reward["contact_cost_weight"] == 5e-4
    assert reward["forward_progress_shaping_weight"] == 0.0
    assert reward["lateral_drift_shaping_weight"] == 0.0
    assert reward["effort_shaping_weight"] == 0.0
    assert config["orientation_shaping"]["function"] == "cosine"
    assert config["orientation_shaping"]["normalisation"] == "(1 - cos(theta)) / 2"


def test_orientation_pilot_preserves_formal_seed_holdout() -> None:
    config = load_config()
    development = set(config["training_seeds"])
    formal = set(config["reserved_formal_training_seeds"])
    assert development == {41201, 41204, 41205}
    assert development.isdisjoint(formal)


def test_orientation_pilot_matrix_and_video_duration_are_frozen() -> None:
    config = load_config()
    assert config["orientation_shaping"]["candidate_weights"] == [0.1, 0.25, 0.5]
    assert config["checkpoint_timesteps"] == [250000, 500000, 750000, 1000000]
    assert config["eval_episodes_per_checkpoint"] == 20
    assert config["video_plan"]["fps"] == 20
    assert config["video_plan"]["max_steps"] == 1000
    assert config["video_plan"]["playback_speed_ratio"] == 1.0
    assert config["formal_launch"] == "prohibited"
    assert config["second_intervention_launch"] == "prohibited"
