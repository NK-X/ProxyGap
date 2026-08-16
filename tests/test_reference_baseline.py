from __future__ import annotations

import copy
import ast
import json
from pathlib import Path

import pytest

from proxygap.reference_baseline import (
    classify_configuration,
    summarise_reference_endpoint,
    validate_reference_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def endpoint_rows(*, pass_count: int = 4) -> list[dict]:
    cfg = config()
    rows: list[dict] = []
    for policy_index, training_seed in enumerate(cfg["training_seeds"]):
        passing = policy_index < pass_count
        unhealthy_count = 4 if passing else 5
        velocity = 0.2 if passing else 0.05
        for episode_index, evaluation_seed in enumerate(cfg["evaluation_seeds"]):
            rows.append(
                {
                    "condition_id": "reference",
                    "ctrl_cost_weight": 0.5,
                    "training_seed": training_seed,
                    "target_timesteps": 1_000_000,
                    "seed": evaluation_seed,
                    "unhealthy_termination": episode_index < unhealthy_count,
                    "mean_forward_velocity": velocity,
                    "forward_progress_shaping_weight": 0.0,
                    "lateral_drift_shaping_weight": 0.0,
                    "reward_shaping_sum": 0.0,
                    "reward_forward_shaping_sum": 0.0,
                    "reward_lateral_shaping_sum": 0.0,
                    "reward_effort_shaping_sum": 0.0,
                    "reward_orientation_shaping_sum": 0.0,
                }
            )
    return rows


def test_v6_config_freezes_reference_only_scope_and_separates_seeds() -> None:
    cfg = config()
    validate_reference_config(cfg)
    assert cfg["ctrl_cost_weights"] == [0.5]
    assert set(cfg["training_seeds"]).isdisjoint(cfg["reserved_formal_training_seeds"])
    assert cfg["formal_launch"] == "prohibited"
    assert cfg["shaping_launch"] == "prohibited"


@pytest.mark.parametrize(
    ("passing", "expected"),
    [(5, "supported"), (4, "supported"), (3, "inconclusive"), (2, "inconclusive"), (1, "failed"), (0, "failed")],
)
def test_configuration_classification(passing: int, expected: str) -> None:
    assert classify_configuration(passing) == expected


def test_endpoint_aggregates_twenty_episodes_within_each_policy() -> None:
    result = summarise_reference_endpoint(endpoint_rows(pass_count=4), config())
    assert result["classification"] == "supported"
    assert result["passing_policies"] == 4
    assert result["total_policies"] == 5
    assert result["evaluation_episode_role"] == "nested repeated observation"
    assert all(row["evaluation_episodes"] == 20 for row in result["policy_results"])
    assert result["policy_results"][0]["unhealthy_termination_rate"] == 0.2


def test_endpoint_rejects_duplicate_evaluation_seed() -> None:
    rows = endpoint_rows()
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match="Duplicate"):
        summarise_reference_endpoint(rows, config())


def test_endpoint_rejects_nonzero_shaping() -> None:
    rows = endpoint_rows()
    rows[0]["reward_shaping_sum"] = 0.1
    with pytest.raises(ValueError, match="Non-zero shaping"):
        summarise_reference_endpoint(rows, config())


def test_config_rejects_candidate_condition_or_formal_seed_overlap() -> None:
    cfg = config()
    cfg["ctrl_cost_weights"] = [0.5, 0.21875]
    with pytest.raises(ValueError, match="only the 0.5 reference"):
        validate_reference_config(cfg)
    cfg = copy.deepcopy(config())
    cfg["reserved_formal_training_seeds"][0] = cfg["training_seeds"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_reference_config(cfg)


def test_reference_scripts_parse_without_unbound_json_booleans() -> None:
    for name in (
        "prepare_stage1_reference_fresh.py",
        "smoke_stage1_reference_fresh.py",
        "analyse_stage1_reference_fresh.py",
        "verify_stage1_reference_fresh.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        unbound_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}
        }
        assert not unbound_names
