from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_heightfield_plane_contact.py"
SPEC = importlib.util.spec_from_file_location("heightfield_plane_contact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_config_validates() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    MODULE.validate_config(config)


def test_plateau_is_exactly_flat_away_from_remote_sentinel() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    scene = config["controlled_scene"]
    heights = MODULE.surface_heights("plateau", 257, 10.0, scene)
    assert np.count_nonzero(heights) == 1
    assert heights[-1, -1] == scene["remote_sentinel_height_m"]
    assert np.all(heights[:200, :200] == 0.0)


def test_microrelief_matches_frozen_amplitude_definition() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    scene = config["controlled_scene"]
    heights = MODULE.surface_heights("microrelief", 257, 10.0, scene)
    assert np.isclose(np.ptp(heights), 3.0e-6, atol=1e-15, rtol=0.0)


def test_surface_set_keeps_fixed_friction() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["controlled_scene"]["fixed_friction"] == [1.0, 0.5, 0.5]
    assert config["controlled_scene"]["condim"] == 3
