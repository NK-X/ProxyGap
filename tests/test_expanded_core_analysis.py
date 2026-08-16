"""Regression tests for the post-hoc expanded-core endpoint analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyse_expanded_core_endpoint.py"
)
SPEC = importlib.util.spec_from_file_location("expanded_core_analysis", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_numeric_parses_csv_booleans_and_numbers() -> None:
    assert MODULE.numeric("True") == 1.0
    assert MODULE.numeric("false") == 0.0
    assert MODULE.numeric(" 2.5 ") == 2.5


def test_mean_accepts_boolean_metric_rows() -> None:
    rows = [{"unhealthy_termination": "True"}, {"unhealthy_termination": "False"}]
    assert MODULE.mean(rows, "unhealthy_termination") == 0.5
