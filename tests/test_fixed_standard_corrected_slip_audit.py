from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_fixed_standard_corrected_slip.py"
CONFIG = ROOT / "configs" / "fixed_standard_explicit_pair_corrected_slip_audit_v1_20260819.json"


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_slip_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplementary_config_validates_without_reopening_parent_gate() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_config, protocol, _, manifest, parent_rows = module.validate_config(config)
    assert parent_config["standard_gate"]["minimum_best_progress_ratio"] == 0.9
    assert protocol["task_adapter"]["augment_local_terrain_observation"] is True
    assert set(manifest["scenes"]) == {"flat", "uphill_8deg", "downhill_8deg", "bowl_exit"}
    assert len(parent_rows) == 24
    assert config["exploratory_training_interpretation"]["formal_promotion_authorised"] is False


def test_aggregate_uses_supported_step_and_foot_sample_denominators() -> None:
    module = load_module()
    row = {
        "fall": False,
        "fixed_goal_success": False,
        "episode_steps": 10,
        "supported_step_count": 4,
        "distal_foot_contact_sample_count": 6,
        "mean_support_count": 0.6,
        "fixed_goal_best_progress_m": 1.0,
        "raw_step_count": 2,
        "qualified_step_count": 1,
        "duration_corrected_sustained_step_count": 1,
        "raw_foot_sample_count": 3,
        "qualified_foot_sample_count": 2,
        "duration_corrected_sustained_foot_sample_count": 1,
        "duration_corrected_slip_event_count": 1,
    }
    result = module.aggregate([row])
    assert result["pooled_raw_all_step_fraction"] == 0.2
    assert result["pooled_raw_per_supported_step_fraction"] == 0.5
    assert result["pooled_raw_per_distal_foot_contact_sample_fraction"] == 0.5
    assert result["events_per_100_supported_steps"] == 25.0


def test_zero_event_csv_retains_complete_header(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "events.csv"
    module.write_event_rows(path, [])
    assert path.stat().st_size > 0
    assert path.read_text(encoding="utf-8").strip().split(",") == list(
        module.EVENT_COLUMNS
    )
