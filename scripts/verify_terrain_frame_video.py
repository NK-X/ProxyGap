"""Independently decode and match the terrain-frame representative video."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_fixed_goal_training_video import sha256, validate_video  # noqa: E402


RUN_ROOT = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819"
    / "seed_62803"
)
STEM = "fixed_map_final_policy_seed_73802_dual_view_v1"
NUMERIC_FIELDS = (
    "x_m",
    "y_m",
    "terrain_height_m",
    "torso_z_m",
    "distance_to_goal_m",
    "support_count",
    "maximum_contact_tangential_speed_m_per_s",
    "reward",
)
BOOLEAN_FIELDS = ("airborne", "terminated", "truncated")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def exact_trace_match(formal_path: Path, video_path: Path) -> dict[str, Any]:
    formal = _read(formal_path)
    video = _read(video_path)
    if len(formal) != 3600 or len(video) != len(formal):
        raise ValueError("formal and video traces must both contain 3600 steps")
    if [int(row["step"]) for row in video] != list(range(1, 3601)):
        raise ValueError("video trace step sequence is not consecutive")
    errors: dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        error = max(
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(formal, video)
        )
        errors[field] = error
        if error != 0.0:
            raise ValueError(f"video replay differs from formal trace: {field}")
    mismatches: dict[str, int] = {}
    for field in BOOLEAN_FIELDS:
        count = sum(
            _bool(left[field]) != _bool(right[field])
            for left, right in zip(formal, video)
        )
        mismatches[field] = count
        if count:
            raise ValueError(f"video replay differs from formal trace: {field}")
    return {
        "formal_trace": str(formal_path),
        "formal_trace_sha256": sha256(formal_path),
        "video_trace": str(video_path),
        "video_trace_sha256": sha256(video_path),
        "rows": len(formal),
        "numeric_maximum_absolute_errors": errors,
        "boolean_mismatch_counts": mismatches,
        "exact_state_reward_match": True,
    }


def main() -> None:
    video_root = RUN_ROOT / "videos" / "tf73802_v1"
    video = video_root / f"{STEM}.mp4"
    manifest_path = video_root / f"{STEM}_video_manifest.json"
    video_trace = video_root / f"{STEM}_trace.csv"
    formal_trace = (
        RUN_ROOT
        / "traces"
        / "terrain_frame_reward_intervention_seed_73802_trace.csv"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["rollout"]["evaluation_seed"] != 73802:
        raise ValueError("representative evaluation seed mismatch")
    if manifest["rollout"]["model_sha256"] != (
        "f43cb53fc1f11752c9a63895071aa00f6b9a67825d6df656454acc56884c0f31"
    ):
        raise ValueError("representative checkpoint mismatch")
    if sha256(video) != manifest["video"]["sha256"]:
        raise ValueError("video SHA-256 differs from renderer manifest")
    decoded = validate_video(video, expected_width=1280, expected_height=720)
    if decoded["decoded_frames"] != int(manifest["video"]["frames"]):
        raise ValueError("independent full decode frame count mismatch")
    if float(decoded["decoded_duration_seconds"]) < 10.0:
        raise ValueError("representative video is shorter than ten seconds")
    payload = {
        "schema_version": "proxygap-terrain-frame-video-qa-v1",
        "selection_rule": (
            "Predeclared validation seed 73802 (2/3), full 180 physical seconds, "
            "20x playback; retained even though the formal gate failed."
        ),
        "video": str(video),
        "video_sha256": sha256(video),
        "video_manifest": str(manifest_path),
        "video_manifest_sha256": sha256(manifest_path),
        "independent_full_decode": decoded,
        "formal_trace_verification": exact_trace_match(formal_trace, video_trace),
        "canonical_publish_path": "videos/tf73802_v1",
        "intermediate_do_not_publish": [
            "videos/terrain_frame_seed_73802_dual_view_v1"
        ],
        "claim_boundary": (
            "Qualitative negative-result evidence; it cannot establish generalisation "
            "or statistical superiority."
        ),
    }
    output = RUN_ROOT / "videos" / "terrain_frame_video_qa.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "output": str(output), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
