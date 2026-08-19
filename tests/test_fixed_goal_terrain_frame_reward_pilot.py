from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_goal_terrain_frame_reward_pilot import validate_config  # noqa: E402


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_terrain_frame_pilot_config_validates() -> None:
    reward = validate_config(_config())
    assert reward["preserved_pre_pitch_reward"]["ctrl_cost_weight"] == 0.5
    assert reward["preserved_pre_pitch_reward"]["airborne_shaping_weight"] == 4


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("approved_map", "fixed_friction"), [1.1, 0.5, 0.5]),
        (("training", "additional_target_timesteps"), 65536),
        (("task_adapter", "maximum_abs_curvature_per_m"), 0.2),
        (("energy_boundary", "relative_mission_energy_v2_status"), "reward"),
    ],
)
def test_frozen_fields_fail_closed(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_config(config)
