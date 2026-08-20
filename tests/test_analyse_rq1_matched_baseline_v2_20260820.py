from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_rq1_matched_baseline_v2_20260820.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rq1_analysis_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boolean_parser_handles_csv_roundtrip() -> None:
    module = load_module()
    assert module.as_float("True") == 1.0
    assert module.as_float("False") == 0.0
    assert module.as_float(True) == 1.0
    assert module.as_float(False) == 0.0


def test_decision_is_based_on_three_training_pairs() -> None:
    module = load_module()
    pairs = []
    for seed in module.EXPECTED_TRAINING_SEEDS:
        pairs.append(
            {
                "training_seed": seed,
                "shaped_minus_default__target_speed_abs_error_m_per_s": -0.1,
                "shaped_minus_default__direction_error_degrees": -1.0,
                "shaped_minus_default__forward_path_efficiency": 0.1,
                "shaped_minus_default__normalised_action_roughness": -0.01,
                "shaped_minus_default__unhealthy_termination_rate": 0.0,
            }
        )
    result = module.build_decision(pairs)
    assert result["training_pairs_n"] == 3
    assert result["joint_descriptive_gate_pass"] is True


def test_failed_metric_is_not_hidden_by_other_metrics() -> None:
    module = load_module()
    pairs = []
    for seed in module.EXPECTED_TRAINING_SEEDS:
        pairs.append(
            {
                "training_seed": seed,
                "shaped_minus_default__target_speed_abs_error_m_per_s": -0.1,
                "shaped_minus_default__direction_error_degrees": 1.0,
                "shaped_minus_default__forward_path_efficiency": -0.1,
                "shaped_minus_default__normalised_action_roughness": 0.01,
                "shaped_minus_default__unhealthy_termination_rate": 0.1,
            }
        )
    result = module.build_decision(pairs)
    assert result["joint_descriptive_gate_pass"] is False
    assert result["quality_metrics_passing_2_of_3_rule_out_of_4"] == 1


def test_manifest_inventory_requires_exact_paths_sizes_and_hashes(tmp_path: Path) -> None:
    module = load_module()
    payload = tmp_path / "logs" / "raw.csv"
    payload.parent.mkdir(parents=True)
    payload.write_text("x\n1\n", encoding="utf-8")
    manifest = {
        "file_count": 1,
        "files": [
            {
                "path": "logs/raw.csv",
                "bytes": payload.stat().st_size,
                "sha256": module.sha256(payload),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = module.validate_manifest_inventory(tmp_path, manifest_path)
    assert result["inventory_exact"] is True
    (tmp_path / "unexpected.txt").write_text("late file", encoding="utf-8")
    try:
        module.validate_manifest_inventory(tmp_path, manifest_path)
    except ValueError as error:
        assert "inventory differs" in str(error)
    else:
        raise AssertionError("Late files must invalidate the sealed parent inventory")


def test_manifest_file_count_must_match_list(tmp_path: Path) -> None:
    module = load_module()
    payload = tmp_path / "raw.csv"
    payload.write_text("x\n1\n", encoding="utf-8")
    manifest = {
        "file_count": 2,
        "files": [
            {
                "path": "raw.csv",
                "bytes": payload.stat().st_size,
                "sha256": module.sha256(payload),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        module.validate_manifest_inventory(tmp_path, manifest_path)
    except ValueError as error:
        assert "file_count" in str(error)
    else:
        raise AssertionError("A false manifest file_count must fail closed")


def test_required_field_contract_includes_decision_and_vector_fields() -> None:
    module = load_module()
    assert "normalised_action_roughness" in module.REQUIRED_FINITE_FIELDS
    assert "unhealthy_termination" in module.REQUIRED_BOOLEAN_FIELDS
    assert module.REQUIRED_VECTOR_FIELDS["support_count_step_fractions_0_to_4"] == 5
