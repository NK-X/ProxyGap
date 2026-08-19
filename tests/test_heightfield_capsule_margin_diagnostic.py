from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_heightfield_capsule_margin.py"
SPEC = importlib.util.spec_from_file_location("heightfield_capsule_margin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_margin_addendum_config_validates() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    scenes = MODULE.validate_config(config)
    assert "native_plane" in scenes
    assert "hfield_plateau_129" in scenes
    assert "hfield_plateau_257" in scenes


def test_margin_matrix_is_single_factor() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    observed = {
        (float(item["floor_margin_m"]), float(item["foot_margin_m"]))
        for item in config["margin_conditions"]
    }
    assert observed == {(0.01, 0.01), (0.0, 0.01), (0.01, 0.0), (0.0, 0.0)}
    assert config["controlled_parameters"]["fixed_friction"] == [1.0, 0.5, 0.5]
