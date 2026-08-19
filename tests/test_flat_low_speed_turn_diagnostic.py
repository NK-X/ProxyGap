from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_flat_low_speed_turn import (  # noqa: E402
    build_env,
    commanded_yaw_rate,
    validate_config,
)


CONFIG_PATH = ROOT / "configs" / "flat_low_speed_turn_diagnostic_v1_20260819.json"


def load() -> tuple[dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = json.loads((ROOT / config["source_policy"]["config"]).read_text(encoding="utf-8"))
    return config, source


def test_diagnostic_is_read_only_and_out_of_distribution_is_explicit() -> None:
    config, source = load()
    validate_config(config, source)
    assert config["training_performed"] is False
    assert config["core_source_modified"] is False
    assert config["source_policy"]["training_speed_m_per_s"] == 0.8
    assert max(config["diagnostic_commands"]["positive_crawl_speeds_m_per_s"]) == 0.1


def test_yaw_rate_controller_saturates_and_stops_inside_tolerance() -> None:
    value = commanded_yaw_rate(
        0.0,
        math.pi,
        gain_per_second=1.0,
        maximum_abs_yaw_rate_rad_per_s=0.3,
        tolerance_rad=math.radians(10.0),
    )
    assert np.isclose(value, 0.3)
    stopped = commanded_yaw_rate(
        math.radians(85.0),
        math.radians(90.0),
        gain_per_second=1.0,
        maximum_abs_yaw_rate_rad_per_s=0.3,
        tolerance_rad=math.radians(10.0),
    )
    assert stopped == 0.0


def test_existing_adapter_rejects_true_zero_speed_external_yaw() -> None:
    config, source = load()
    env = build_env(source, config, seed=54801)
    try:
        observation, _ = env.reset(seed=54801)
        with pytest.raises(ValueError, match="external speed"):
            env.set_external_curve_command(
                observation,
                target_heading=math.pi / 2.0,
                yaw_rate=0.3,
                speed=0.0,
                lateral_speed=0.0,
            )
    finally:
        env.close()
