from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_orientation_cosine_pilot.py"
SPEC = importlib.util.spec_from_file_location("orientation_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_numeric_frame_converts_native_and_string_booleans_to_numbers() -> None:
    frame = pd.DataFrame(
        {
            "native_bool": [True, False],
            "string_bool": ["True", "False"],
        }
    )
    converted = MODULE.numeric_frame(
        frame,
        ["native_bool", "string_bool"],
    )
    assert converted["native_bool"].tolist() == [1.0, 0.0]
    assert converted["string_bool"].tolist() == [1.0, 0.0]
    assert pd.api.types.is_numeric_dtype(converted["native_bool"])
    assert pd.api.types.is_numeric_dtype(converted["string_bool"])


def test_dataframe_to_markdown_has_no_optional_dependency() -> None:
    frame = pd.DataFrame({"weight": [0.1], "passed": [True]})
    rendered = MODULE.dataframe_to_markdown(frame)
    assert "| weight | passed |" in rendered
    assert "| 0.100 | True |" in rendered
