from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair as runner  # noqa: E402


CONFIG = ROOT / "configs/fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair_20260819.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def exact_audit_fixture(v1: dict) -> tuple[dict, dict]:
    pair = runner.v2.l2._pair_contract(v1)
    contact = v1["contact_contract"]
    margin = float(contact["all_geom_margins_m"])
    compiled = [
        {
            "geom1": "floor",
            "geom2": foot,
            "margin": float(pair["margin_m"]),
            "gap": float(pair["gap_m"]),
            "condim": int(pair["condim"]),
            "friction": pair["friction"],
            "solref": pair["solref"],
            "solreffriction": pair["solreffriction"],
            "solimp": pair["solimp"],
            "adhesion": float(pair["adhesion"]),
        }
        for foot in runner.pair_tools.FOOT_NAMES
    ]
    audit = {
        "only_four_permitted_explicit_pairs_added": True,
        "explicit_pair_count": 4,
        "source_floor_margin_m": margin,
        "candidate_floor_geom_margin_m": margin,
        "source_distal_margins_m": {
            name: margin for name in runner.pair_tools.FOOT_NAMES
        },
        "candidate_distal_geom_margins_m": {
            name: margin for name in runner.pair_tools.FOOT_NAMES
        },
        "candidate_non_distal_margins_m": {
            name: margin for name in runner.pair_tools.NON_DISTAL_ROBOT_GEOMS
        },
        "default_geom_margin_m": margin,
        "root_joint_margin_m": margin,
        "compiled_explicit_pairs": compiled,
        "friction": contact["geom_friction"],
        "condim": int(contact["condim"]),
        "solref": contact["solref"],
        "solimp": contact["solimp"],
        "physics_timestep_seconds": 0.01,
    }
    return audit, pair


def test_v5_preserves_the_frozen_scientific_protocol() -> None:
    config = load()
    v2, v1, _, _, checkpoint, evidence = runner.validate_config(config)
    assert v2["formal"]["additional_timesteps_per_condition"] == 65_536
    assert v2["formal"]["absolute_final_checkpoint_timesteps"] == 2_793_472
    assert v2["formal"]["save_intermediate_checkpoints"] is False
    assert v2["formal"]["evaluate_intermediate_checkpoints"] is False
    assert v2["formal"]["select_intermediate_checkpoint"] is False
    assert v1["training"]["master_seed"] == 63_806
    assert v1["final_evaluation"]["heldout_seeds"] == [
        96131,
        96137,
        96149,
        96153,
        96177,
    ]
    assert checkpoint.name == "checkpoint_2727936.zip"
    assert runner.v3.sha256(checkpoint) == (
        "5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c"
    )
    assert evidence["scientific_protocol_changed"] is False


def test_exact_contract_succeeds_without_a_passed_field() -> None:
    _, v1, _, _, _, _ = runner.validate_config(load())
    audit, pair = exact_audit_fixture(v1)
    assert "passed" not in audit
    result = runner.validate_compiled_audit_exact(
        audit, pair, v1["contact_contract"]
    )
    assert result["compiled_contract_exact"] is True
    assert result["success_derived_from_all_exact_fields"] is True
    assert result["synthetic_passed_field_used"] is False


def test_synthetic_passed_field_and_contract_mutation_fail_closed() -> None:
    _, v1, _, _, _, _ = runner.validate_config(load())
    audit, pair = exact_audit_fixture(v1)
    with_passed = copy.deepcopy(audit)
    with_passed["passed"] = True
    with pytest.raises(ValueError, match="field set/order"):
        runner.validate_compiled_audit_exact(
            with_passed, pair, v1["contact_contract"]
        )
    mutated = copy.deepcopy(audit)
    mutated["compiled_explicit_pairs"][0]["margin"] = 0.001
    with pytest.raises(ValueError, match="pair 0 margin"):
        runner.validate_compiled_audit_exact(
            mutated, pair, v1["contact_contract"]
        )


def test_v2_and_v4_failure_roots_are_immutable_pretraining_evidence() -> None:
    evidence = runner.validate_frozen_failures(load())
    assert evidence["V2"]["failure_record_sha256"] == runner.EXPECTED_V2_FAILURE_SHA
    assert evidence["V4"]["failure_record_sha256"] == runner.EXPECTED_V4_FAILURE_SHA
    for row in evidence.values():
        assert row["training_started"] is False
        assert row["checkpoint_count"] == 0
        assert row["training_record_count"] == 0
        assert row["scientifically_evaluable"] is False
        assert row["same_root_retry_permitted"] is False


def test_v5_short_root_path_budget_and_once_only_boundaries() -> None:
    config = load()
    root = runner.formal_root(config)
    assert root == (ROOT / "artifacts/dev/tb_v5_20260819/a0").resolve()
    result = runner.validate_path_budget(root)
    assert result["all_within_limit"] is True
    assert result["maximum"]["characters"] <= 239
    assert config["execution"]["retry_in_same_root"] is False
    assert config["execution"]["reuse_partial_weights"] is False
    assert config["post_formal_boundary"]["further_optimisation_authorised"] is False


def test_v5_runtime_is_frozen_and_wrapper_contains_no_direct_learning() -> None:
    config = load()
    assert runner.validate_v5_runtime(config) == config["V5_runtime_contract"][
        "exact_relative_path_sha256"
    ]
    source = (ROOT / runner.RUNTIME_SELF).read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert "audit.get(\"passed\")" not in source
    assert "if not all(bool(audit.get('passed'))" not in source


def test_canonical_smoke_is_strictly_validated_if_present() -> None:
    config = load()
    smoke_root = runner.engineering_smoke_root(config)
    if not smoke_root.exists():
        return
    _, v1, _, _, checkpoint, _ = runner.validate_config(config)
    evidence = runner.validate_completed_engineering_smoke(config, v1, checkpoint)
    assert evidence["formal_pre_run_decision"] == "GO"
    assert evidence["training_performed"] is False
    assert evidence["checkpoint_written_or_selected"] is False
