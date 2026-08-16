from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_stage1_budget_extension.py"
)
SPEC = importlib.util.spec_from_file_location("budget_extension_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matched_proxy_uses_signed_contact_and_action_cost() -> None:
    rows = [
        {
            "reward_forward_sum": "10",
            "reward_survive_sum": "20",
            "reward_contact_sum": "-2",
            "cumulative_squared_action": "8",
        },
        {
            "reward_forward_sum": "14",
            "reward_survive_sum": "20",
            "reward_contact_sum": "-4",
            "cumulative_squared_action": "12",
        },
    ]
    assert MODULE.matched_proxy(rows, 0.25) == 26.5


def test_policy_rows_selects_weight_seed_and_checkpoint() -> None:
    rows = [
        {
            "ctrl_cost_weight": "0.5",
            "training_seed": "41101",
            "target_timesteps": "1000000",
        },
        {
            "ctrl_cost_weight": "0.5",
            "training_seed": "41102",
            "target_timesteps": "1000000",
        },
    ]
    selected = MODULE.policy_rows(
        rows, weight=0.5, training_seed=41101, checkpoint=1000000
    )
    assert selected == [rows[0]]


def test_numeric_parses_csv_booleans() -> None:
    assert MODULE.numeric("True") == 1.0
    assert MODULE.numeric("false") == 0.0
    assert MODULE.numeric(" 2.5 ") == 2.5
