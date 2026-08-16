"""Screen a completed development coefficient run without formal claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.divergence import (  # noqa: E402
    choose_minimal_departure_candidate,
    screen_divergence_candidates,
    screen_pairwise_fixed_proxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "two_experiment_revision_gate_20260813.json"),
    )
    parser.add_argument("--early_checkpoints", nargs="+", type=int, default=None)
    parser.add_argument("--late_checkpoints", nargs="+", type=int, default=None)
    parser.add_argument("--exploration_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    detection = config["experiment_1_detection"]
    with (run_root / "logs" / "evaluation_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    early = args.early_checkpoints or detection["early_window"]
    late = args.late_checkpoints or detection["late_window"]
    temporal_screens = screen_divergence_candidates(
        rows,
        early_checkpoints=early,
        late_checkpoints=late,
        minimum_consistent_seeds=len(detection["development_training_seeds"]),
    )
    eligible = detection["screening_rule"]["eligible_reduced_weights"]
    endpoint = max(late)
    pairwise_screens = screen_pairwise_fixed_proxy(
        rows,
        checkpoint=endpoint,
        minimum_consistent_seeds=len(detection["development_training_seeds"]),
        minimum_consistent_harm_metrics=2,
    )
    selected = choose_minimal_departure_candidate(
        pairwise_screens,
        eligible_reduced_weights=eligible,
    )
    if args.exploration_only:
        status = "short_horizon_exploration_only_no_candidate_lock"
    else:
        status = (
            "candidate_requires_method_review_before_heldout"
            if selected is not None
            else "stop_before_shaping_no_core_candidate"
        )
    result = {
        "status": status,
        "selected_candidate_ctrl_cost_weight": selected,
        "candidate_is_locked": False,
        "early_checkpoints": early,
        "late_checkpoints": late,
        "selection_rule": detection["screening_rule"]["multiple_candidate_rule"],
        "claim_boundary": (
            "Development screening only. A candidate is not a confirmed finding "
            "of reward hacking or mitigation efficacy. Short-horizon exploration "
            "cannot lock the 300k condition."
        ),
        "pairwise_fixed_proxy_screens": [
            screen.to_dict() for screen in pairwise_screens
        ],
        "within_weight_temporal_screens": [
            screen.to_dict() for screen in temporal_screens
        ],
    }
    output = run_root / "development_divergence_screen.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
