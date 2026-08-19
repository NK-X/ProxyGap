from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_post_seal_full_map_eval_video_v1 as subject  # noqa: E402


CONFIG = ROOT / "configs" / "post_seal_full_map_eval_video_v1_20260819.json"


def test_video_contract_binds_the_one_formal_failure_episode() -> None:
    config, execution, result = subject.validate_config(CONFIG)
    episode = config["episode_contract"]
    assert episode["evaluation_seed"] == 1763594348
    assert episode["checkpoint_sha256"] == (
        "5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c"
    )
    assert episode["horizon_control_steps"] == 12000
    assert episode["display_outcome"] == "FAILED TO REACH / HORIZON"
    assert result["goal_entered"] is False
    assert result["strict_stable_dwell_success"] is False
    assert result["safety_qualified_success"] is False
    assert execution["training_performed"] is False


def test_formal_trace_has_exact_frozen_extent_and_seed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    trace = ROOT / config["formal_evidence"]["control_trace"]
    rows = subject.load_trace(trace, 12000)
    assert int(rows[0]["evaluation_seed"]) == 1763594348
    assert int(rows[-1]["control_step"]) == 12000
    assert int(rows[-1]["environment_truncated"]) == 1


def test_exact_row_comparator_accepts_round_trip_and_rejects_change() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    trace = ROOT / config["formal_evidence"]["control_trace"]
    with trace.open("r", encoding="utf-8", newline="") as handle:
        formal_row = next(csv.DictReader(handle))
    replay: dict[str, object] = {"mode": formal_row["mode"], "action": formal_row["action"]}
    replay.update({key: float(formal_row[key]) for key in subject.FLOAT_FIELDS})
    replay.update({key: int(formal_row[key]) for key in subject.INT_FIELDS})
    examples: list[dict[str, object]] = []
    assert subject.compare_row(formal_row, replay, examples) == 0
    replay["x_m"] = float(replay["x_m"]) + 1e-12
    assert subject.compare_row(formal_row, replay, examples) == 1
    assert examples[0]["field"] == "x_m"


def test_composed_frame_has_dual_view_and_failure_panel() -> None:
    left = np.zeros((subject.dual.VIEW_HEIGHT, subject.dual.VIEW_WIDTH, 3), dtype=np.uint8)
    right = np.full_like(left, 32)
    map_base = Image.new("RGB", (subject.dual.MAP_SIZE, subject.dual.MAP_SIZE), (80, 100, 120))
    frame = subject.compose_frame(
        left,
        right,
        map_base=map_base,
        trail_xy=[np.asarray((-34.0, -34.0)), np.asarray((-33.0, -33.0))],
        start=np.asarray((-34.0, -34.0)),
        goal=np.asarray((34.0, 34.0)),
        position=np.asarray((-33.0, -33.0)),
        half_extent=40.0,
        physical_time=600.0,
        distance=83.58,
        best_progress=14.51,
        net_progress=12.53,
        terrain_tilt_degrees=4.0,
        current_support=4,
        cumulative_mean_support=1.8646,
        maximum_contact_speed=0.1,
        current_airborne=False,
        ever_airborne=True,
        cumulative_zero_foot_count=383,
        completed_steps=12000,
        floor_friction=np.asarray((1.0, 0.5, 0.5)),
        floor_condim=3,
        map_hash="59e60ddd91d799f44f84aa74a2ecff1",
        overview_profile_id="relief-v2",
        terrain_min_height_m=-4.0,
        terrain_max_height_m=4.0,
    )
    assert frame.size == (1280, 720)
    pixels = np.asarray(frame)
    assert int(pixels[690, 700, 0]) > int(pixels[690, 700, 1])


def test_renderer_contains_no_training_or_model_save_call() -> None:
    source = (ROOT / "scripts" / "render_post_seal_full_map_eval_video_v1.py").read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert "policy.save(" not in source
    assert "PPO.load(" in source
    assert "except BaseException" in source
