"""Verify the paired W4/W12 support-priority videos and replay prefixes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_fixed_goal_training_video import sha256, validate_video  # noqa: E402


DEFAULT_RUN_ROOT = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819"
    / "seed_62803"
)
VIDEO_STEM = "fixed_map_final_policy_seed_73802_dual_view_v1"
CONDITIONS = {
    "W4_MATCHED_CONTINUATION_CONTROL": {
        "key": "w4",
        "weight": 4.0,
        "checkpoint_sha256": "c6073f2ea61edd7a50fee4c2d2623243ed3c9ce866b4f328cad1a4c6289ecbd2",
    },
    "W12_SUPPORT_PRIORITY_INTERVENTION": {
        "key": "w12",
        "weight": 12.0,
        "checkpoint_sha256": "e152df7462033144318549b0983ebbb651ec907d6b152f4644e42f245cdd752c",
    },
}
NUMERIC_PREFIX_FIELDS = (
    "x_m",
    "y_m",
    "terrain_height_m",
    "torso_z_m",
    "distance_to_goal_m",
    "support_count",
    "maximum_contact_tangential_speed_m_per_s",
    "reward",
)
BOOLEAN_PREFIX_FIELDS = ("airborne", "terminated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def verify_prefix(
    formal_trace: Path,
    video_trace: Path,
) -> dict[str, Any]:
    formal = _read_csv(formal_trace)
    video = _read_csv(video_trace)
    if len(video) > len(formal):
        raise ValueError("Video trace is longer than the formal evaluation trace")
    if [int(row["step"]) for row in video] != list(range(1, len(video) + 1)):
        raise ValueError("Video trace steps are not consecutive")
    maximum_errors: dict[str, float] = {}
    for field in NUMERIC_PREFIX_FIELDS:
        differences = [
            abs(float(formal[index][field]) - float(row[field]))
            for index, row in enumerate(video)
        ]
        maximum_errors[field] = max(differences, default=0.0)
        if maximum_errors[field] != 0.0:
            raise ValueError(f"Video replay differs from formal trace field {field}")
    boolean_mismatches: dict[str, int] = {}
    for field in BOOLEAN_PREFIX_FIELDS:
        mismatches = sum(
            _bool(formal[index][field]) != _bool(row[field])
            for index, row in enumerate(video)
        )
        boolean_mismatches[field] = mismatches
        if mismatches:
            raise ValueError(f"Video replay differs from formal trace field {field}")
    truncation_mismatches = sum(
        _bool(formal[index]["truncated"]) != _bool(row["truncated"])
        for index, row in enumerate(video)
    )
    if truncation_mismatches != 1 or not _bool(video[-1]["truncated"]):
        raise ValueError("Expected one final-step truncation due to the 45 s video horizon")
    return {
        "formal_trace": str(formal_trace),
        "formal_trace_sha256": sha256(formal_trace),
        "video_trace": str(video_trace),
        "video_trace_sha256": sha256(video_trace),
        "formal_rows": len(formal),
        "video_rows": len(video),
        "numeric_prefix_maximum_absolute_errors": maximum_errors,
        "boolean_prefix_mismatch_counts": boolean_mismatches,
        "expected_final_truncation_mismatch_count": truncation_mismatches,
        "exact_state_reward_prefix_match": True,
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = run_root / "videos" / "paired_support_video_qa.json"
    if output.exists() and not args.overwrite:
        raise RuntimeError(f"Refusing to overwrite {output}")
    execution = json.loads(
        (run_root / "execution_record.json").read_text(encoding="utf-8")
    )
    model_hashes = {
        str(item["condition_id"]): str(item["checkpoint_sha256"])
        for item in execution["model_records"]
    }
    records: list[dict[str, Any]] = []
    for condition_id, expected in CONDITIONS.items():
        key = str(expected["key"])
        video_dir = run_root / "videos" / f"{key}_seed_73802_dual_view_v3"
        video_path = video_dir / f"{VIDEO_STEM}.mp4"
        manifest_path = video_dir / f"{VIDEO_STEM}_video_manifest.json"
        video_trace = video_dir / f"{VIDEO_STEM}_trace.csv"
        formal_trace = (
            run_root
            / "traces"
            / (
                "w4_matched_continuation_control_seed_73802_trace.csv"
                if key == "w4"
                else "w12_support_priority_intervention_seed_73802_trace.csv"
            )
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rollout = manifest["rollout"]
        if rollout["condition_label"] != condition_id:
            raise ValueError("Video condition label mismatch")
        if float(rollout["airborne_shaping_weight"]) != float(expected["weight"]):
            raise ValueError("Video airborne weight mismatch")
        if int(rollout["evaluation_seed"]) != 73802:
            raise ValueError("Video seed mismatch")
        if rollout["paired_evaluation_group"] != [2, 3]:
            raise ValueError("Video must display seed 73802 as paired evaluation 2/3")
        expected_hash = str(expected["checkpoint_sha256"])
        if model_hashes.get(condition_id) != expected_hash:
            raise ValueError("Execution model hash differs from predeclared video hash")
        if rollout["model_sha256"] != expected_hash:
            raise ValueError("Video model hash mismatch")
        if sha256(video_path) != manifest["video"]["sha256"]:
            raise ValueError("Video file hash mismatch")
        decoded = validate_video(video_path, expected_width=1280, expected_height=720)
        if decoded["decoded_frames"] != int(manifest["video"]["frames"]):
            raise ValueError("Independent full decode frame count mismatch")
        if decoded["decoded_duration_seconds"] < 10.0:
            raise ValueError("Video is shorter than ten seconds")
        if not all(
            bool(manifest["qa"][key])
            for key in (
                "duration_at_least_10_seconds",
                "map_hash_matches_frozen_config",
                "checkpoint_hash_matches_execution_record",
                "friction_matches_frozen_config",
                "both_views_share_one_deterministic_rollout",
            )
        ):
            raise ValueError("Renderer QA flag failed")
        records.append(
            {
                "condition_id": condition_id,
                "airborne_shaping_weight": expected["weight"],
                "evaluation_seed": 73802,
                "model_sha256": expected_hash,
                "video": str(video_path),
                "video_sha256": sha256(video_path),
                "video_manifest": str(manifest_path),
                "video_manifest_sha256": sha256(manifest_path),
                "independent_full_decode": decoded,
                "formal_trace_prefix_verification": verify_prefix(
                    formal_trace,
                    video_trace,
                ),
            }
        )
    payload = {
        "schema_version": "proxygap-paired-support-video-qa-v1",
        "selection_rule": (
            "Both equal-budget conditions use predeclared validation seed 73802 "
            "(paired evaluation 2/3), the first 45 physical seconds, a five-times "
            "playback speed and identical cameras."
        ),
        "conditions": records,
        "pair_is_a_visualisation_of_formal_rollout_prefixes": True,
        "publication_boundary": {
            "publish_only": [
                "videos/w4_seed_73802_dual_view_v3",
                "videos/w12_seed_73802_dual_view_v3",
            ],
            "intermediate_do_not_publish": [
                "videos/w4_seed_73802_dual_view_v1",
                "videos/w4_seed_73802_dual_view_v2",
                "videos/w12_seed_73802_dual_view_v2",
            ],
        },
        "truncation_boundary": (
            "Formal evaluations run 180 s. Each video intentionally truncates at 45 s; "
            "therefore only the final truncated flag differs from the formal prefix."
        ),
        "claim_boundary": (
            "Videos are qualitative evidence and cannot establish superiority. "
            "Contact-speed exceedance includes brief landing transients and is not "
            "automatically sustained physical sliding."
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "output": str(output), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
