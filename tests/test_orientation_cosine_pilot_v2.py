from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "orientation_cosine_shaping_pilot_v2_20260815.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v2_changes_calibration_gate_only() -> None:
    v1 = json.loads(
        (ROOT / "configs" / "orientation_cosine_shaping_pilot_v1_20260815.json")
        .read_text(encoding="utf-8")
    )
    v2 = load_config()
    for key in [
        "ctrl_cost_weight",
        "orientation_shaping",
        "training_seeds",
        "reserved_formal_training_seeds",
        "evaluation_seeds",
        "timesteps_per_condition",
        "checkpoint_timesteps",
        "reward",
        "metrics",
        "pilot_gate",
        "video_plan",
        "ppo",
    ]:
        assert v2[key] == v1[key]


def test_v2_stratified_scale_gate_is_frozen() -> None:
    gate = load_config()["offline_calibration_gate"]
    assert gate["reference_mode_training_seed"] == 41201
    assert gate["adverse_mode_training_seed"] == 41204
    assert gate["largest_weight_adverse_mean_ratio_interval"] == [0.05, 0.25]
    assert gate["largest_weight_reference_mean_ratio_max"] == 0.10


def test_v2_remains_development_only() -> None:
    config = load_config()
    assert config["formal_launch"] == "prohibited"
    assert config["second_intervention_launch"] == "prohibited"
    assert set(config["training_seeds"]).isdisjoint(
        config["reserved_formal_training_seeds"]
    )
