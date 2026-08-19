from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_fixed_standard_pair0_turn_balance_v5_video_archive as subject  # noqa: E402


CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_v5_video_archive_v1_20260819.json"
)


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_archive_binds_the_frozen_failed_turn_gate_and_holds_stage_b() -> None:
    validated = subject.validate_config(load())
    assert validated["final_gate"]["decision"] == (
        "both_fail_turning_HOLD_retain_source_PAIR0"
    )
    assert validated["final_gate"]["C1_passed_both_turn_and_slope"] is False
    assert validated["final_gate"]["fixed_map_authorised"] is False
    assert validated["hard_stop"]["hard_stop"] is True
    assert validated["hard_stop"]["further_optimisation_authorised"] is False


def test_four_predeclared_rows_are_unique_and_exact_extent() -> None:
    validated = subject.validate_config(load())
    assert subject.EXPECTED_EPISODES == (
        ("C0_STRAIGHT_CONTINUE", "curve_left_020", 96131),
        ("C0_STRAIGHT_CONTINUE", "curve_right_020", 96131),
        ("C1_BALANCED_TURN", "curve_left_020", 96131),
        ("C1_BALANCED_TURN", "curve_right_020", 96131),
    )
    for branch_id, condition_name, seed in subject.EXPECTED_EPISODES:
        row = subject.formal_row_for(
            validated["branches"][branch_id]["rows"],
            branch_id,
            condition_name,
            seed,
        )
        assert row["control_steps"] == 600
        assert row["physics_substeps"] == 3000
        assert row["checkpoint_timesteps"] == 2_793_472
        assert row["branch_id"] == branch_id
        assert row["condition_name"] == condition_name
        assert row["evaluation_seed"] == 96131


def test_complete_row_comparator_is_order_and_value_exact() -> None:
    validated = subject.validate_config(load())
    row = subject.formal_row_for(
        validated["branches"]["C1_BALANCED_TURN"]["rows"],
        "C1_BALANCED_TURN",
        "curve_left_020",
        96131,
    )
    result = subject.compare_episode_rows(row, dict(row))
    assert result["field_order_exact"] is True
    assert result["all_fields_exact_match"] is True
    changed = dict(row)
    changed["actual_cumulative_yaw_change_rad"] = (
        float(changed["actual_cumulative_yaw_change_rad"]) + 1e-15
    )
    try:
        subject.compare_episode_rows(row, changed)
    except RuntimeError as error:
        assert "actual_cumulative_yaw_change_rad" in str(error)
    else:
        raise AssertionError("A modified formal field was not rejected")


def test_frame_contract_is_dual_view_and_uses_frozen_gate_label() -> None:
    config = load()
    left = np.zeros((subject.visual.VIEW_HEIGHT, subject.visual.VIEW_WIDTH, 3), dtype=np.uint8)
    right = np.full_like(left, 32)
    condition = next(
        row
        for row in subject.v2.expected_turn_conditions()
        if row["condition_name"] == "curve_left_020"
    )
    formal_row = {
        "actual_cumulative_yaw_change_rad": 0.5,
        "yaw_change_target_ratio": 0.15,
        "full_interval_zero_foot_count": 12,
    }
    frame = subject.compose_frame(
        left,
        right,
        branch_id="C1_BALANCED_TURN",
        condition=condition,
        seed=96131,
        step=600,
        formal_row=formal_row,
        actual_yaw_change=0.5,
        current_support=2,
        cumulative_zero_foot=12,
        squared_action=100.0,
        positive_work_j=2000.0,
    )
    assert frame.size == (1280, 720)
    pixels = np.asarray(frame)
    assert int(pixels[subject.visual.VIEW_HEIGHT + 10, 10, 0]) > int(
        pixels[subject.visual.VIEW_HEIGHT + 10, 10, 1]
    )
    assert config["render"]["mandatory_label"] == (
        "TURN GATE: FAIL | SLOPE CONTINUITY: PASS | FIXED-MAP: NOT AUTHORISED"
    )


def test_runner_is_read_only_fail_closed_and_once_only() -> None:
    source = (
        ROOT
        / "scripts"
        / "render_fixed_standard_pair0_turn_balance_v5_video_archive.py"
    ).read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert "policy.save(" not in source
    assert "PPO.load(" in source
    assert "except BaseException" in source
    config = load()
    assert config["replay"]["training"] is False
    assert config["replay"]["checkpoint_write"] is False
    assert config["output"]["fail_if_exists"] is True
    assert config["output"]["retry_same_root"] is False
