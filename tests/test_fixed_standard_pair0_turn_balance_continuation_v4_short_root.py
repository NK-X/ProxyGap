from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_standard_pair0_turn_balance_continuation_v4_short_root as runner  # noqa: E402

CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v4_short_root_20260819.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v4_preserves_old_failure_and_changes_only_output_root() -> None:
    config = load()
    evidence = runner.validate_old_failure(config)
    assert evidence["failure_record_sha256"] == runner.EXPECTED_OLD_FAILURE_SHA
    assert evidence["training_started"] is False
    assert evidence["zip_count"] == evidence["training_record_count"] == 0
    assert config["execution"]["only_change_from_V3_formal_execution"] == "canonical_output_root_shortening"
    assert config["execution"]["reuse_partial_weights"] is False
    assert config["execution"]["retry_after_V4_failure"] is False


def test_all_planned_paths_fit_and_the_previous_failure_suffix_is_covered() -> None:
    config = load()
    root = runner.formal_root(config)
    result = runner.validate_path_budget(root)
    assert result["all_within_limit"] is True
    assert result["maximum"]["characters"] <= 239
    paths = set(runner.planned_relative_paths())
    assert "standard_slope_assets/condition_assets/scenes/turn_balance_downhill_8deg/explicit_floor_distal_pair_margin0_candidate/ant_standard_scene.xml" in paths
    assert "standard_slope_assets/condition_assets/scenes/turn_balance_bowl_exit/explicit_floor_distal_pair_margin0_candidate/ant_standard_scene.xml" in paths
    assert all(len(str(root / path)) <= 239 for path in paths)


def test_path_preflight_fails_closed_for_long_root() -> None:
    with pytest.raises(ValueError, match="planned path exceeds"):
        runner.validate_path_budget(ROOT / ("x" * 180))


def test_config_runtime_smoke_and_science_boundaries_validate() -> None:
    config = load()
    v2, v1, protocol, reward, checkpoint, evidence = runner.validate_config(config)
    assert v2["formal"]["additional_timesteps_per_condition"] == 65536
    assert v1["training"]["master_seed"] == 63806
    assert checkpoint.name == "checkpoint_2727936.zip"
    assert evidence["V2_smoke_manifest_sha256"] == "a02b8dad18c94f75d50c8d41dbc9600884bb735726c9301c7967d1e8238d32be"
    assert evidence["V4_short_root_repair"]["old_V2_failure"]["failed_stage"] == "prepare_standard_slope_scenes"
    source = (ROOT / runner.RUNTIME_SELF).read_text(encoding="utf-8")
    assert ".learn(" not in source and "model.save(" not in source


def test_mutated_scientific_boundary_is_rejected() -> None:
    config = load()
    changed = copy.deepcopy(config)
    changed["frozen_scientific_protocol"]["seeds_changed"] = True
    with pytest.raises(ValueError, match="seeds_changed"):
        runner.validate_config(changed)


def test_unique_v4_root_is_sealed_as_a_pretraining_failure() -> None:
    root = runner.formal_root(load())
    failure = root / "FAILURE_RECORD.json"
    assert root.is_dir()
    assert runner.v3.sha256(failure) == (
        "9695bd3b5d628907053a2f785ec874efa18b2fc47a317c452a88566c0d624812"
    )
    assert not list(root.rglob("*.zip"))
    assert not list(root.rglob("training_record.json"))
