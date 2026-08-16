from __future__ import annotations

import json
from pathlib import Path

from proxygap.two_experiment_protocol import validate_two_experiment_protocol


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "two_experiment_revision_gate_20260813.json").read_text(
            encoding="utf-8"
        )
    )


def test_development_screen_is_ready_but_heldout_is_blocked() -> None:
    status = validate_two_experiment_protocol(load_config())
    assert status["development_status"] == "ready"
    assert status["heldout_status"] == "blocked"
    assert "No development-screened candidate" in status["heldout_blockers"][0]


def test_shaping_cannot_change_detected_control_weight() -> None:
    config = load_config()
    config["experiment_2_shaping"]["same_ctrl_cost_weight_as_detected_condition"] = False
    status = validate_two_experiment_protocol(config)
    assert any("retain the diagnosed" in item for item in status["development_blockers"])


def test_boundary_weight_cannot_silently_become_primary_candidate() -> None:
    config = load_config()
    config["heldout_confirmation"]["candidate_ctrl_cost_weight"] = 0.0625
    status = validate_two_experiment_protocol(config)
    assert any("outside the eligible core" in item for item in status["heldout_blockers"])


def test_effort_repenalisation_must_remain_prohibited() -> None:
    config = load_config()
    config["experiment_2_shaping"]["prohibited_signals"] = ["forward_reward"]
    status = validate_two_experiment_protocol(config)
    assert any("effort re-penalisation" in item for item in status["development_blockers"])
