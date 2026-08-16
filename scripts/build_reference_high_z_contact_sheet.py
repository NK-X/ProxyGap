"""Extract matched pre-terminal and terminal frames from diagnostic videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from _portable_runtime import (
        FFMPEG_TARGET_ENV,
        LATIN_FONT_NAMES,
        iter_font_files,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )
except ModuleNotFoundError:  # Support module-style execution from the repository root.
    from scripts._portable_runtime import (
        FFMPEG_TARGET_ENV,
        LATIN_FONT_NAMES,
        iter_font_files,
        optional_ffmpeg_target,
        prepend_optional_dependency_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "stage1_reference_high_z_diagnostic_v8_20260814"
    / "videos"
)
DEFAULT_OUTPUT = DEFAULT_VIDEO_ROOT.parent / "matched_video_contact_sheet.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--imageio_target",
        type=Path,
        default=optional_ffmpeg_target(),
        help=(
            "Optional pip-target containing imageio-ffmpeg. Defaults to "
            f"{FFMPEG_TARGET_ENV}, then the active Python environment."
        ),
    )
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in iter_font_files(LATIN_FONT_NAMES):
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    for name in LATIN_FONT_NAMES:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    video_root = args.video_root.resolve()
    output = args.output.resolve()
    prepend_optional_dependency_target(
        args.imageio_target,
        package_marker="imageio_ffmpeg",
    )
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install imageio and imageio-ffmpeg in the active environment, or pass "
            f"--imageio_target/set {FFMPEG_TARGET_ENV}."
        ) from exc

    manifests = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in video_root.glob("*.json")),
        key=lambda record: int(record["training_seed"]),
    )
    if len(manifests) != 5:
        raise ValueError("Exactly five matched video manifests are required")

    panels: list[tuple[int, str, Image.Image]] = []
    frames_root = output.parent / "selected_frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        video_path = Path(manifest["video_path"])
        frame_count = int(manifest["frames"])
        fps = int(manifest["fps"])
        reader = imageio.get_reader(video_path)
        indices = {
            "two_seconds_before_end": max(0, frame_count - 1 - 2 * fps),
            "final_frame": frame_count - 1,
        }
        for label, index in indices.items():
            frame = Image.fromarray(reader.get_data(index)).convert("RGB")
            frame_path = frames_root / f"seed_{manifest['training_seed']}__{label}.png"
            frame.save(frame_path)
            panels.append((int(manifest["training_seed"]), label, frame))
        reader.close()

    panel_width = 490
    source_width, source_height = panels[0][2].size
    panel_height = round(source_height * panel_width / source_width)
    header_height = 48
    gap = 12
    canvas = Image.new(
        "RGB",
        (
            5 * panel_width + 6 * gap,
            2 * (panel_height + header_height) + 3 * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(19)
    label_font = font(15)
    for column, seed in enumerate(sorted({seed for seed, _, _ in panels})):
        for row, label in enumerate(("two_seconds_before_end", "final_frame")):
            image = next(
                frame for panel_seed, panel_label, frame in panels
                if panel_seed == seed and panel_label == label
            ).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = gap + column * (panel_width + gap)
            y = gap + row * (panel_height + header_height + gap)
            draw.text((x, y), f"Training seed {seed}", fill="#172B3A", font=title_font)
            subtitle = "2 s before episode end" if row == 0 else "Final recorded frame"
            draw.text((x, y + 25), subtitle, fill="#50606D", font=label_font)
            canvas.paste(image, (x, y + header_height))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)
    print(f"selected_frames={len(panels)}")


if __name__ == "__main__":
    main()
