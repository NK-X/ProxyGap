"""Build or verify the SHA-256 manifest for the staged public repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "handoff" / "FILE_MANIFEST_SHA256.csv"
MANIFEST_PATH = "handoff/FILE_MANIFEST_SHA256.csv"


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def build_rows() -> list[dict[str, str | int]]:
    paths = _git_bytes("ls-files", "--cached", "-z").decode("utf-8").split("\0")
    rows: list[dict[str, str | int]] = []
    for path in sorted(item for item in paths if item and item != MANIFEST_PATH):
        data = _git_bytes("show", f":{path}")
        rows.append(
            {
                "path": path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str | int]]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_manifest() -> list[dict[str, str | int]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "path": row["path"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the committed manifest with the current Git index.",
    )
    args = parser.parse_args()

    expected = build_rows()
    if args.check:
        observed = read_manifest()
        if observed != expected:
            raise SystemExit("Public manifest verification failed.")
        print(f"Public manifest verification passed: {len(expected)} files.")
        return

    write_manifest(expected)
    print(f"Wrote {MANIFEST.relative_to(ROOT)} for {len(expected)} files.")


if __name__ == "__main__":
    main()
