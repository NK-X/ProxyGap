from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terrain_generator import TerrainConfig, load_config  # noqa: E402


@pytest.fixture(scope="session")
def development_config() -> TerrainConfig:
    config = load_config(ROOT / "configs" / "terrain_development.json")
    return replace(config, nrow=257, ncol=257)
