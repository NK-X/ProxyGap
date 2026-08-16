from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_body_dynamics_replication import paired_contrasts  # noqa: E402


def test_replication_contrast_is_shaped_minus_paired_baseline(tmp_path: Path) -> None:
    rows = []
    for seed, baseline, shaped in ((1, 10.0, 8.0), (2, 12.0, 9.0), (3, 7.0, 8.0)):
        rows.extend(
            [
                {"condition_id": "B0__G0_REP", "training_seed": seed, "metric": baseline},
                {"condition_id": "B1__G0_REP", "training_seed": seed, "metric": shaped},
            ]
        )
    contrasts, summary = paired_contrasts(
        pd.DataFrame(rows), ["metric"], "synthetic", tmp_path
    )
    assert contrasts["contrast_shaped_minus_baseline"].tolist() == [-2.0, -3.0, 1.0]
    assert int(summary.loc[0, "seed_pairs_below_zero"]) == 2
