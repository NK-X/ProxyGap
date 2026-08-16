from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_hybrid_intent_sensitivity.py"
CONFIG = ROOT / "configs" / "hybrid_intent_sensitivity_v1_20260816.json"


def load_module():
    spec = importlib.util.spec_from_file_location("intent_sensitivity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_row() -> dict:
    return {
        "full_horizon_completed": True,
        "unhealthy_termination": False,
        "sustained_inversion": False,
        "fixed_horizon_mean_forward_velocity": 1.0,
        "torso_tilt_rms": 0.1,
        "net_displacement_direction_error_degrees": 2.0,
        "forward_path_efficiency": 0.95,
        "normalised_action_roughness": 0.02,
        "action_saturation_rate": 0.005,
    }


def test_sensitivity_grid_has_predeclared_729_cells() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(module.threshold_grid(config)) == 729


def test_unhealthy_episode_never_passes_relaxed_quality_thresholds() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = [passing_row(), {**passing_row(), "unhealthy_termination": True}]
    result = module.compliance(pd.DataFrame(rows), module.threshold_grid(config)[0])
    assert result.tolist() == [True, False]
