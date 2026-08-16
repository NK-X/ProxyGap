"""Independently verify the V8 high-z diagnostic from raw replay traces."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "exploration" / "stage1_reference_fresh_1m_v6_20260814"
ANALYSIS = ROOT / "artifacts" / "analysis" / "stage1_reference_high_z_diagnostic_v8_20260814"
TRACE_PATTERN = re.compile(r"train_(\d+)__target_1000000__eval_(\d+)\.csv\.gz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    original = [
        row
        for row in read_csv(RUN / "logs" / "evaluation_metrics.csv")
        if int(row["target_timesteps"]) == 1_000_000
    ]
    replay = read_csv(ANALYSIS / "replay_episode_metrics.csv")
    original_by_key = {
        (int(row["training_seed"]), int(row["seed"])): row for row in original
    }
    replay_by_key = {
        (int(row["training_seed"]), int(row["seed"])): row for row in replay
    }
    if len(original_by_key) != 100 or original_by_key.keys() != replay_by_key.keys():
        raise RuntimeError("The endpoint replay matrix does not match the original keys")
    for key in original_by_key:
        left = original_by_key[key]
        right = replay_by_key[key]
        if left["termination_category"] != right["termination_category"]:
            raise RuntimeError(f"Termination mismatch at {key}")
        if int(left["episode_length"]) != int(right["episode_length"]):
            raise RuntimeError(f"Episode-length mismatch at {key}")
        if not math.isclose(
            float(left["condition_objective_return"]),
            float(right["condition_objective_return"]),
            abs_tol=1e-10,
        ):
            raise RuntimeError(f"Return mismatch at {key}")

    per_seed = {
        seed: {
            "steps": 0,
            "inverted_steps": 0,
            "low_posture_steps": 0,
            "high_z_episodes": 0,
            "time_limit_episodes": 0,
            "episodes_majority_inverted": 0,
        }
        for seed in (41201, 41202, 41203, 41204, 41205)
    }
    high_z_terminal_velocity: list[float] = []
    high_z_last_second_gain: list[float] = []
    trace_keys: set[tuple[int, int]] = set()
    trace_hashes: list[dict[str, str | int]] = []
    for path in sorted((ANALYSIS / "step_traces").rglob("*.csv.gz")):
        match = TRACE_PATTERN.search(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected trace filename: {path}")
        training_seed, evaluation_seed = map(int, match.groups())
        key = (training_seed, evaluation_seed)
        trace_keys.add(key)
        rows = read_csv(path)
        if [int(row["step_index"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"Non-contiguous trace: {path}")
        final = rows[-1]
        expected = original_by_key[key]
        if final["termination_category"] != expected["termination_category"]:
            raise RuntimeError(f"Trace category mismatch at {key}")
        if truthy(final["truncated"]) != truthy(expected["truncated"]):
            raise RuntimeError(f"Trace truncation mismatch at {key}")

        heights = [float(row["torso_height"]) for row in rows]
        tilts = [float(row["torso_tilt_rad"]) for row in rows]
        inverted_count = sum(tilt >= math.pi / 2 for tilt in tilts)
        low_count = sum(height < 0.3 for height in heights)
        seed_record = per_seed[training_seed]
        seed_record["steps"] += len(rows)
        seed_record["inverted_steps"] += inverted_count
        seed_record["low_posture_steps"] += low_count
        seed_record["episodes_majority_inverted"] += inverted_count > len(rows) / 2
        if final["termination_category"] == "high_z_excursion":
            seed_record["high_z_episodes"] += 1
            high_z_terminal_velocity.append((heights[-1] - heights[-2]) / 0.05)
            start = max(0, len(heights) - 21)
            high_z_last_second_gain.append(heights[-1] - heights[start])
        elif truthy(final["truncated"]):
            seed_record["time_limit_episodes"] += 1
        else:
            raise RuntimeError(f"Unexpected endpoint category at {key}")
        trace_hashes.append(
            {
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "sha256": sha256(path),
            }
        )

    if trace_keys != original_by_key.keys():
        raise RuntimeError("Step-trace keys do not match the original endpoint")

    primary = json.loads((ANALYSIS / "high_z_diagnostic.json").read_text(encoding="utf-8"))
    for row in primary["policy_posture_diagnostic"]:
        seed = int(row["training_seed"])
        independent = per_seed[seed]
        inverted = independent["inverted_steps"] / independent["steps"]
        low = independent["low_posture_steps"] / independent["steps"]
        if not math.isclose(inverted, row["step_weighted_inverted_proportion"], abs_tol=1e-12):
            raise RuntimeError(f"Inversion proportion mismatch for {seed}")
        if not math.isclose(low, row["step_weighted_low_posture_proportion"], abs_tol=1e-12):
            raise RuntimeError(f"Low-posture proportion mismatch for {seed}")

    video_records = []
    for manifest_path in sorted((ANALYSIS / "videos").glob("*.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_path = Path(record["video_path"])
        if record["status"] != "complete_trajectory_video_rendered":
            raise RuntimeError(f"Incomplete video: {manifest_path}")
        if int(record["evaluation_seed"]) != 51216:
            raise RuntimeError(f"Unmatched video seed: {manifest_path}")
        if int(record["frames"]) != int(record["episode_summary"]["episode_length"]):
            raise RuntimeError(f"Video frame mismatch: {manifest_path}")
        if sha256(video_path) != record["video_sha256"]:
            raise RuntimeError(f"Video hash mismatch: {video_path}")
        video_records.append(record)
    if {int(record["training_seed"]) for record in video_records} != set(per_seed):
        raise RuntimeError("The matched five-policy video set is incomplete")

    result = {
        "status": "pass",
        "method": "Independent raw-CSV and gzip-trace recomputation; no primary diagnostic functions imported.",
        "original_endpoint_rows": len(original),
        "replay_rows": len(replay),
        "step_traces": len(trace_hashes),
        "high_z_episodes": sum(record["high_z_episodes"] for record in per_seed.values()),
        "time_limit_episodes": sum(record["time_limit_episodes"] for record in per_seed.values()),
        "per_seed": {
            str(seed): {
                **record,
                "inverted_step_proportion": record["inverted_steps"] / record["steps"],
                "low_posture_step_proportion": record["low_posture_steps"] / record["steps"],
            }
            for seed, record in per_seed.items()
        },
        "high_z_terminal_vertical_velocity_range": [
            min(high_z_terminal_velocity),
            max(high_z_terminal_velocity),
        ],
        "high_z_last_second_height_gain_range": [
            min(high_z_last_second_gain),
            max(high_z_last_second_gain),
        ],
        "matched_videos_verified": len(video_records),
        "selected_contact_sheet": str(ANALYSIS / "matched_video_contact_sheet.png"),
        "claim_boundary": (
            "The verification establishes the replayed kinematics and video integrity. "
            "Post-hoc posture descriptors remain diagnostic, not frozen formal endpoints."
        ),
    }
    output = ANALYSIS / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
