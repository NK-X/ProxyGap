from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTCOME = ROOT / "configs" / "stage1_reference_fresh_1m_outcome_v7_20260814.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v7_outcome_matches_frozen_parent_and_evidence_hashes() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    parent = outcome["parent_frozen_design"]
    assert sha256(ROOT / parent["config_path"]) == parent["config_sha256"]
    assert sha256(ROOT / parent["protocol_path"]) == parent["protocol_sha256"]
    for evidence in outcome["evidence"].values():
        path = ROOT / evidence["path"]
        if not path.exists():
            pytest.skip(f"full local evidence package is not included: {path}")
        assert sha256(path) == evidence["sha256"]


def test_v7_outcome_preserves_stage_one_boundaries() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    result = outcome["reference_competence"]
    assert result["passing_policies"] == 2
    assert result["total_policies"] == 5
    assert result["configuration_classification"] == "inconclusive"
    assert outcome["decision"]["stage_one_hypothesis_tested"] is False
    assert outcome["decision"]["candidate_weight_launch"] == "prohibited"
    assert outcome["decision"]["formal_launch"] == "prohibited"
    assert outcome["decision"]["shaping_launch"] == "prohibited"


def test_v7_outcome_retains_every_fresh_training_seed() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    assert set(outcome["reference_competence"]["policy_results"]) == {
        "41201",
        "41202",
        "41203",
        "41204",
        "41205",
    }
