from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "configs" / "stage1_post_extension_gate_v5_20260814.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text_canonical(path: Path) -> str:
    """Hash committed text content independently of Windows newline checkout."""
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest().upper()


def require_local_evidence(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"full local evidence package is not included: {path}")


def load_gate() -> dict:
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def test_post_extension_gate_preserves_stage_one_boundary() -> None:
    gate = load_gate()
    assert gate["stage_scope"] == "stage_1_detection_only_no_shaping"
    assert gate["formal_launch"] == "prohibited"
    assert gate["shaping_launch"] == "prohibited"
    assert gate["executed_matrix"]["shaping_weights"] == "all_zero"


def test_post_extension_gate_records_failed_reference_competence() -> None:
    gate = load_gate()
    competence = gate["reference_competence"]
    assert competence["overall_pass"] is False
    assert competence["seed_results"]["41101"]["joint_gate_pass"] is False
    assert competence["seed_results"]["41102"]["joint_gate_pass"] is False


def test_post_extension_gate_records_candidate_without_true_reward_claim() -> None:
    gate = load_gate()
    candidate = gate["development_candidate"]
    assert candidate["ctrl_cost_weight"] == 0.21875
    assert candidate["matched_proxy_gain_both_seeds"] is True
    assert candidate["consistently_harmed_domains"] == [
        "locomotion_effectiveness",
        "posture_stability",
    ]
    assert "not confirmed reward hacking" in candidate["claim_boundary"]


def test_post_extension_gate_evidence_hashes_match_files() -> None:
    gate = load_gate()
    evidence = gate["evidence"]
    pairs = [
        (evidence["raw_episode_csv"], evidence["raw_episode_csv_sha256"]),
        (
            evidence["primary_adjudication"],
            evidence["primary_adjudication_sha256"],
        ),
        (
            evidence["independent_verification"],
            evidence["independent_verification_sha256"],
        ),
    ]
    for relative_path, expected_hash in pairs:
        path = ROOT / relative_path
        require_local_evidence(path)
        assert sha256(path) == expected_hash


def test_post_extension_gate_parent_hashes_match_frozen_v4() -> None:
    gate = load_gate()
    parent = gate["parent_budget_extension"]
    assert sha256_text_canonical(ROOT / parent["config_path"]) == parent["config_sha256"]
    assert sha256_text_canonical(ROOT / parent["protocol_path"]) == parent["protocol_sha256"]
