from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTCOME = ROOT / "configs" / "stage1_reference_construct_adjudication_v8_20260814.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v8_evidence_hashes_match_files() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    for evidence in outcome["evidence"].values():
        path = ROOT / evidence["path"]
        if not path.exists():
            pytest.skip(f"full local evidence package is not included: {path}")
        assert sha256(path) == evidence["sha256"]


def test_v8_keeps_simulator_health_separate_from_human_intent() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    assert outcome["environment_rule"]["healthy_z_range"] == [0.2, 1.0]
    assert outcome["decision"]["change_healthy_z_range_now"] is False
    assert outcome["decision"]["v6_reference_competence_gate"].startswith(
        "construct_insufficient"
    )


def test_v8_blocks_downstream_experiments_and_causal_overclaim() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    decision = outcome["decision"]
    assert decision["run_normalisation_pilot_now"] is False
    assert decision["candidate_weight_launch"] == "prohibited"
    assert decision["formal_launch"] == "prohibited"
    assert decision["shaping_launch"] == "prohibited"
    assert any(
        "causal contribution" in item
        for item in outcome["epistemic_adjudication"]["unresolved"]
    )
