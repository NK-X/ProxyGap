from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_default_reward_construct.py"


def load_module():
    spec = importlib.util.spec_from_file_location("construct_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairwise_audit_detects_proxy_intent_and_pareto_inversion() -> None:
    module = load_module()
    common = {
        "target_velocity_absolute_error": 0.1,
        "unhealthy_termination": 0.0,
        "sustained_inversion": 0.0,
        "torso_tilt_rms_degrees": 10.0,
        "net_displacement_direction_error_degrees": 3.0,
        "path_inefficiency": 0.1,
        "normalised_action_roughness": 0.02,
        "action_saturation_rate": 0.0,
    }
    policy = pd.DataFrame(
        [
            {
                "training_seed": 1,
                "target_timesteps": 100,
                "base_proxy_return": 20.0,
                "intent_compliance_rate": 0.2,
                **{key: value + 0.1 for key, value in common.items()},
            },
            {
                "training_seed": 2,
                "target_timesteps": 100,
                "base_proxy_return": 10.0,
                "intent_compliance_rate": 0.8,
                **common,
            },
        ]
    )
    result = module.pairwise_audit(policy).iloc[0]
    assert result["proxy_intent_rank_inversion"]
    assert result["pareto_inversion"]
