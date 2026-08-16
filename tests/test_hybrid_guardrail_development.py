from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hybrid_guardrail_development_v1_20260816.json"
ROUND2_CONFIG = ROOT / "configs" / "hybrid_guardrail_round2_v1_20260816.json"
OBSERVABILITY_CORRECTION_CONFIG = (
    ROOT / "configs" / "hybrid_guardrail_observability_correction_v1_20260816.json"
)
RUNNER = ROOT / "scripts" / "run_hybrid_guardrail_development.py"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_runner_module():
    spec = importlib.util.spec_from_file_location("hybrid_guardrail_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_development_and_formal_seeds_are_disjoint() -> None:
    config = load_config()
    assert set(config["training_seeds"]).isdisjoint(
        config["reserved_formal_training_seeds"]
    )
    assert config["formal_launch"] == "prohibited"


def test_all_conditions_share_the_113_dimensional_observation() -> None:
    observation = load_config()["observation"]
    assert observation["default_dimensions"] + observation[
        "previous_applied_action_dimensions"
    ] == observation["dimensions"]
    assert observation["augment_previous_applied_action_for_all_conditions"] is True


def test_matrix_has_matched_reward_by_constraint_cells() -> None:
    config = load_config()
    cells = {
        (condition["reward_id"], condition["constraint_id"])
        for condition in config["conditions"]
    }
    assert cells == {
        ("R0_default", "K0_none"),
        ("Rtheta_0p1", "K0_none"),
        ("Rtheta_0p25", "K0_none"),
        ("R0_default", "Kslew_1p4"),
        ("Rtheta_0p1", "Kslew_1p4"),
        ("Rtheta_0p25", "Kslew_1p4"),
    }


def test_runner_rejects_unfrozen_or_overlapping_config() -> None:
    module = load_runner_module()
    config = load_config()
    module.validate_config(config)
    config["training_seeds"] = [config["reserved_formal_training_seeds"][0]]
    try:
        module.validate_config(config)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping seed partitions were accepted")


def test_development_step_log_path_stays_below_windows_max_path() -> None:
    config = load_config()
    longest_condition = max(
        config["conditions"], key=lambda row: len(row["condition_id"])
    )
    seed = max(config["training_seeds"])
    target = max(config["checkpoint_timesteps"])
    evaluation_seed = max(config["evaluation_seeds"])
    path = (
        ROOT
        / config["execution"]["output_root"]
        / "runs"
        / f"seed_{seed}"
        / longest_condition["condition_id"]
        / "logs"
        / "evaluation_steps"
        / f"tr{seed}_t{target}_ev{evaluation_seed}.csv.gz"
    )
    assert len(str(path.resolve())) < 260


def test_round2_matrix_is_frozen_bounded_and_formal_seeds_remain_protected() -> None:
    module = load_runner_module()
    config = json.loads(ROUND2_CONFIG.read_text(encoding="utf-8"))
    module.validate_config(config)
    assert config["revision_round"] == config["revision_limit"] == 2
    assert config["formal_launch"] == "prohibited"
    assert set(config["training_seeds"]).isdisjoint(
        config["reserved_formal_training_seeds"]
    )
    assert {row["lateral_drift_shaping_weight"] for row in config["conditions"]} == {
        0.0,
        0.05,
        0.1,
    }
    assert {row["action_slew_l2_limit"] for row in config["conditions"]} == {
        None,
        1.1,
    }
    assert 1.1**2 / 32 < 0.04


def test_observability_correction_uses_lateral_velocity_without_absolute_xy() -> None:
    config = json.loads(OBSERVABILITY_CORRECTION_CONFIG.read_text(encoding="utf-8"))
    assert config["formal_launch"] == "prohibited"
    assert config["observation"]["lateral_velocity_observed"] is True
    assert config["observation"]["absolute_xy_position_observed"] is False
    assert set(config["training_seeds"]).isdisjoint(
        config["reserved_formal_training_seeds"]
    )
    assert {
        condition["lateral_shaping_signal"]
        for condition in config["conditions"]
    } == {"velocity_tanh_squared"}
    assert {
        condition["lateral_velocity_target"]
        for condition in config["conditions"]
    } == {0.0}
