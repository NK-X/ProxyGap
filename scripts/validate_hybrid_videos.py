"""Validate rendered hybrid-development videos and their evidence manifests."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--expected-videos", type=int, default=8)
    parser.add_argument("--expected-duration", type=float, default=50.0)
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


def main() -> None:
    args = parse_args()
    video_root = (ROOT / args.video_root).resolve()
    prepend_optional_dependency_target(
        args.ffmpeg_target,
        package_marker="imageio_ffmpeg",
    )
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install imageio and imageio-ffmpeg in the active environment, or pass "
            f"--ffmpeg-target/set {FFMPEG_TARGET_ENV}."
        ) from exc

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    manifests = sorted(video_root.glob("*.json"))
    manifests = [path for path in manifests if path.name != "VIDEO_INDEX.json"]
    if len(manifests) != args.expected_videos:
        failures.append(
            f"manifest_count={len(manifests)} expected={args.expected_videos}"
        )

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_path = Path(manifest["video_path"])
        if not video_path.exists():
            failures.append(f"missing_video={video_path}")
            continue
        observed_hash = sha256(video_path)
        if observed_hash != manifest["video_sha256"]:
            failures.append(f"hash_mismatch={video_path.name}")
        if int(manifest["frames"]) != 1000:
            failures.append(f"frame_manifest_mismatch={video_path.name}")
        if not math.isclose(
            float(manifest["video_duration_seconds"]),
            args.expected_duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            failures.append(f"duration_manifest_mismatch={video_path.name}")

        reader = imageio.get_reader(video_path, format="FFMPEG")
        metadata = reader.get_meta_data()
        sampled_indices = [0, 499, 999]
        sample_shapes: list[list[int]] = []
        for index in sampled_indices:
            sample_shapes.append(list(reader.get_data(index).shape))
        reader.close()
        duration = float(metadata["duration"])
        if not math.isclose(
            duration, args.expected_duration, rel_tol=0.0, abs_tol=0.05
        ):
            failures.append(f"decoded_duration_mismatch={video_path.name}:{duration}")
        if float(metadata["fps"]) != 20.0:
            failures.append(f"decoded_fps_mismatch={video_path.name}")

        summary = manifest["episode_summary"]
        rows.append(
            {
                "video": video_path.name,
                "sha256": observed_hash,
                "bytes": video_path.stat().st_size,
                "fps": float(metadata["fps"]),
                "decoded_duration_seconds": duration,
                "manifest_frames": int(manifest["frames"]),
                "trajectory_frames": int(manifest["trajectory_frames"]),
                "padded_frames": int(manifest["padded_frames"]),
                "sample_shapes": sample_shapes,
                "termination_category": summary["termination_category"],
                "intent_compliant": bool(summary["intent_compliant"]),
            }
        )

    result = {
        "status": "passed" if not failures else "failed",
        "video_count": len(rows),
        "expected_video_count": args.expected_videos,
        "all_timeline_durations_seconds": sorted(
            {row["decoded_duration_seconds"] for row in rows}
        ),
        "all_manifest_frames": sorted({row["manifest_frames"] for row in rows}),
        "all_sampled_frames_decoded": all(
            len(row["sample_shapes"]) == 3 for row in rows
        ),
        "failures": failures,
        "videos": rows,
        "claim_boundary": "Qualitative development evidence for default flat-ground Ant-v5 only.",
    }
    output_path = video_root / "VIDEO_QA.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
