from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_fixed_standard_distal_margin0_paired.py"
CONFIG = ROOT / "configs" / "fixed_standard_distal_margin0_paired_diagnostic_v1_20260819.json"


def load_module():
    spec = importlib.util.spec_from_file_location("distal_pair_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_config_validates_and_uses_explicit_pairs() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol, reward, scenes = module.validate_config(config)
    assert protocol["task_adapter"]["augment_local_terrain_observation"] is True
    assert reward["preserved_pre_pitch_reward"]["ctrl_cost_weight"] == 0.5
    assert set(scenes["scenes"]) == {"flat", "uphill_8deg", "downhill_8deg", "bowl_exit"}
    candidate = config["margin_conditions"][1]
    assert candidate["explicit_pair_count"] == 4
    assert candidate["floor_geom_margin_m"] == 0.01
    assert candidate["distal_ankle_geom_margin_m"] == 0.01


def test_explicit_pair_edit_is_exactly_reversible() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = (
        ROOT
        / "artifacts/dev/fixed_standard_support_curriculum_v1_20260819/paired_bound6_seed_62804/standard_scenes/flat/ant_standard_scene.xml"
    ).read_text(encoding="utf-8")
    contract = config["permitted_xml_change"]["explicit_pair_contract"]
    candidate = module.inject_explicit_pairs(source, contract)
    assert module.reverse_explicit_pairs(candidate, contract) == source
    assert candidate.count("<pair ") == 4
    assert candidate.count('margin="0"') == 4
    assert 'name="floor"' in candidate and '<geom conaffinity="1"' in candidate


def test_config_rejects_geom_level_candidate_or_non_distal_change() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["margin_conditions"][1]["floor_geom_margin_m"] = 0.0
    try:
        module.validate_config(changed)
    except ValueError as error:
        assert "margin contract" in str(error)
    else:
        raise AssertionError("geom-level floor margin change was not rejected")
    changed = copy.deepcopy(config)
    changed["margin_conditions"][1]["non_distal_robot_geom_margin_m"] = 0.0
    try:
        module.validate_config(changed)
    except ValueError as error:
        assert "margin contract" in str(error)
    else:
        raise AssertionError("non-distal margin change was not rejected")


def test_compiled_pair_audit_keeps_all_geom_margins(tmp_path: Path) -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol, _, manifest = module.validate_config(config)
    del protocol
    pair, audit = module.prepare_pair(
        manifest["scenes"]["flat"],
        tmp_path,
        "flat",
        config["permitted_xml_change"]["explicit_pair_contract"],
    )
    assert audit["only_four_permitted_explicit_pairs_added"] is True
    assert audit["explicit_pair_count"] == 4
    assert audit["candidate_floor_geom_margin_m"] == 0.01
    assert set(audit["candidate_distal_geom_margins_m"].values()) == {0.01}
    assert set(audit["candidate_non_distal_margins_m"].values()) == {0.01}
    assert pair[module.CONTROL_ID]["heights_sha256"] == pair[module.CANDIDATE_ID]["heights_sha256"]
