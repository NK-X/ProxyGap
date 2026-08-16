"""Portable discovery helpers for standalone ProxyGap release scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import os
from pathlib import Path
import sys


FONT_DIR_ENV = "PROXYGAP_FONT_DIR"
FFMPEG_TARGET_ENV = "PROXYGAP_IMAGEIO_FFMPEG_TARGET"

CJK_FONT_PAIRS: tuple[tuple[str, str], ...] = (
    ("msyh.ttc", "msyhbd.ttc"),
    ("simsun.ttc", "simhei.ttf"),
    ("Deng.ttf", "Dengb.ttf"),
    ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"),
    ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"),
    ("SourceHanSansSC-Regular.otf", "SourceHanSansSC-Bold.otf"),
)
DENG_FIRST_CJK_FONT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Deng.ttf", "Dengb.ttf"),
    *(pair for pair in CJK_FONT_PAIRS if pair != ("Deng.ttf", "Dengb.ttf")),
)
LATIN_FONT_NAMES: tuple[str, ...] = (
    "segoeui.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
)


def optional_ffmpeg_target() -> Path | None:
    """Return an optional pip-target directory declared through the environment."""
    raw_value = os.environ.get(FFMPEG_TARGET_ENV)
    return Path(raw_value).expanduser() if raw_value else None


def prepend_optional_dependency_target(
    target: str | Path | None,
    *,
    package_marker: str,
) -> Path | None:
    """Prepend an optional pip-target directory after validating its package marker."""
    if target is None:
        return None
    resolved = Path(target).expanduser().resolve()
    if not (resolved / package_marker).exists():
        raise FileNotFoundError(
            f"{package_marker!r} was not found under dependency target {resolved}. "
            f"Install it in the active environment, set {FFMPEG_TARGET_ENV}, "
            "or pass the corresponding target option."
        )
    resolved_text = str(resolved)
    if resolved_text not in sys.path:
        sys.path.insert(0, resolved_text)
    return resolved


def font_directories() -> tuple[Path, ...]:
    """Return portable user, operating-system and common open-font locations."""
    directories: list[Path] = []
    explicit = os.environ.get(FONT_DIR_ENV)
    if explicit:
        directories.append(Path(explicit).expanduser())

    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        directories.append(Path(system_root) / "Fonts")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        directories.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")

    home = Path.home()
    directories.extend(
        (
            home / ".fonts",
            home / ".local" / "share" / "fonts",
            Path(os.sep) / "usr" / "share" / "fonts" / "opentype" / "noto",
            Path(os.sep) / "usr" / "share" / "fonts" / "truetype" / "noto",
            Path(os.sep) / "usr" / "share" / "fonts" / "truetype" / "dejavu",
            Path(os.sep) / "usr" / "share" / "fonts" / "truetype" / "liberation2",
            Path(os.sep) / "usr" / "local" / "share" / "fonts",
            Path(os.sep) / "Library" / "Fonts",
            Path(os.sep) / "System" / "Library" / "Fonts",
        )
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        key = str(directory).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return tuple(unique)


def iter_font_pairs(
    filename_pairs: Iterable[tuple[str, str]],
) -> Iterator[tuple[Path, Path]]:
    """Yield installed regular/bold font pairs in deterministic preference order."""
    directories = font_directories()
    for regular_name, bold_name in filename_pairs:
        for directory in directories:
            regular = directory / regular_name
            bold = directory / bold_name
            if regular.is_file() and bold.is_file():
                yield regular, bold


def iter_font_files(filenames: Iterable[str]) -> Iterator[Path]:
    """Yield installed font files in deterministic preference order."""
    directories = font_directories()
    for filename in filenames:
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                yield candidate
