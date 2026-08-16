from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_body_smoothness_gsde_matrix import factorial_contrasts  # noqa: E402
from run_body_smoothness_gsde_matrix import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "body_smoothness_gsde_matrix_v1_20260816.json"


def test_body_smoothness_matrix_is_frozen_balanced_and_seed_separated() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    factors = {
        (item["body_dynamics_enabled"], item["use_sde"])
        for item in config["conditions"]
    }
    assert factors == {(False, False), (True, False), (False, True), (True, True)}
    assert not set(config["training_seeds"]) & set(config["reserved_formal_training_seeds"])
    assert config["evaluation_seeds"] == list(range(51501, 51511))


def test_body_penalty_scale_and_weight_are_calibration_pinned() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    body = config["body_dynamics"]
    assert np.isclose(body["vertical_velocity_shaping_scale"], 1.014092584749083)
    assert np.isclose(
        body["roll_pitch_angular_velocity_shaping_scale"],
        1.9893176307304792,
    )
    assert np.isclose(body["maximum_combined_penalty_per_step"], 0.1)
    assert body["estimated_mean_combined_penalty_per_step"] < 0.03


def test_factorial_contrast_uses_paired_training_seeds(tmp_path: Path, monkeypatch) -> None:
    import analyse_body_smoothness_gsde_matrix as module

    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    rows = []
    for seed in (1, 2, 3):
        for condition_id, value in {
            "B0__G0": 10.0,
            "B1__G0": 8.0,
            "B0__G8": 9.0,
            "B1__G8": 6.0,
        }.items():
            rows.append({"condition_id": condition_id, "training_seed": seed, "metric": value})
    contrasts, summary = factorial_contrasts(pd.DataFrame(rows), ["metric"], "synthetic")
    body = contrasts.loc[contrasts["effect"] == "body_main", "contrast"]
    gsde = contrasts.loc[contrasts["effect"] == "gsde_main", "contrast"]
    interaction = contrasts.loc[contrasts["effect"] == "interaction", "contrast"]
    assert np.allclose(body, -2.5)
    assert np.allclose(gsde, -1.5)
    assert np.allclose(interaction, -1.0)
    assert set(summary["effect"]) == {"body_main", "gsde_main", "interaction"}
