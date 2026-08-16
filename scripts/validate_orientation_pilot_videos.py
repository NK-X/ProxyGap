"""Validate the fixed complete-episode MP4 panel and save durable QA evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from _portable_runtime import (
        FFMPEG_TARGET_ENV,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )
except ModuleNotFoundError:  # Support module-style execution from the repository root.
    from scripts._portable_runtime import (
        FFMPEG_TARGET_ENV,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "orientation_cosine_shaping_pilot_v2_20260815.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--ffmpeg-target",
        type=Path,
        default=optional_ffmpeg_target(),
        help=(
            "Optional pip-target containing imageio-ffmpeg. Defaults to "
            f"{FFMPEG_TARGET_ENV}, then the active Python environment."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_root = PROJECT_ROOT / config["execution"]["output_root"]
    video_root = run_root / "videos"
    index_path = video_root / "VIDEO_INDEX.csv"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    prepend_optional_dependency_target(
        args.ffmpeg_target,
        package_marker="imageio_ffmpeg",
    )
    try:
        import imageio_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install imageio-ffmpeg in the active environment, or pass "
            f"--ffmpeg-target/set {FFMPEG_TARGET_ENV}."
        ) from exc

    plan = config["video_plan"]
    condition_count = 1 + len(config["orientation_shaping"]["candidate_weights"])
    expected_count = condition_count * (
        len(plan["progression_checkpoints"])
        + len(plan["additional_final_training_seeds"])
    )
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    checks: list[dict] = []
    for row in rows:
        video_path = Path(row["video_path"])
        manifest_path = video_path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decoded_frames, decoded_seconds = imageio_ffmpeg.count_frames_and_secs(
            str(video_path)
        )
        expected_frames = int(row["frames"])
        expected_seconds = float(row["duration_seconds"])
        checks.append(
            {
                "video_path": str(video_path),
                "exists": video_path.is_file(),
                "nonempty": video_path.stat().st_size > 0,
                "sha256_matches_index": sha256(video_path)
                == row["video_sha256"],
                "sha256_matches_manifest": sha256(video_path)
                == manifest["video_sha256"],
                "decoded_frames": decoded_frames,
                "expected_frames": expected_frames,
                "frame_count_matches": decoded_frames == expected_frames,
                "decoded_seconds": decoded_seconds,
                "expected_seconds": expected_seconds,
                "duration_matches": math.isclose(
                    decoded_seconds, expected_seconds, abs_tol=0.051
                ),
                "playback_speed_ratio": manifest["playback_speed_ratio"],
                "real_time_playback": math.isclose(
                    float(manifest["playback_speed_ratio"]), 1.0, abs_tol=1e-12
                ),
            }
        )

    required_flags = (
        "exists",
        "nonempty",
        "sha256_matches_index",
        "sha256_matches_manifest",
        "frame_count_matches",
        "duration_matches",
        "real_time_playback",
    )
    result = {
        "status": "pass"
        if len(rows) == expected_count
        and all(all(check[name] for name in required_flags) for check in checks)
        else "fail",
        "config_path": str(config_path),
        "expected_video_count": expected_count,
        "indexed_video_count": len(rows),
        "ffmpeg_executable": imageio_ffmpeg.get_ffmpeg_exe(),
        "checks": checks,
    }
    output_path = video_root / "VIDEO_QA.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    if result["status"] != "pass":
        raise RuntimeError(f"Video QA failed; inspect {output_path}")


if __name__ == "__main__":
    main()
