"""Build the final SHA-256 inventory for the fresh-reference V6/V7 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_files(root: Path, output: Path, summary: Path) -> list[tuple[str, Path]]:
    explicit = {
        "protocol_and_context": [
            root / "configs" / "stage1_reference_fresh_1m_v6_20260814.json",
            root / "configs" / "stage1_reference_fresh_1m_outcome_v7_20260814.json",
            root / "protocols" / "STAGE1_REFERENCE_FRESH_1M_PROTOCOL_V6_20260814.md",
            root / "protocols" / "STAGE1_REFERENCE_FRESH_1M_ADJUDICATION_V7_20260814.md",
            root / "docs" / "STAGE1_DEVIATION_REGISTER_20260814.md",
            root / "reports" / "STAGE1_REFERENCE_FRESH_1M_RESULT_20260814_CN.md",
            root.parent / "PROJECT_CONTEXT.md",
        ],
        "implementation": [
            root / "src" / "proxygap" / "reference_baseline.py",
            root / "scripts" / "prepare_stage1_reference_fresh.py",
            root / "scripts" / "smoke_stage1_reference_fresh.py",
            root / "scripts" / "analyse_stage1_reference_fresh.py",
            root / "scripts" / "verify_stage1_reference_fresh.py",
            root / "scripts" / "launch_stage1_reference_fresh_detached.py",
            root / "scripts" / "build_stage1_reference_manifest.py",
        ],
        "tests": [
            root / "tests" / "test_reference_baseline.py",
            root / "tests" / "test_stage1_reference_outcome.py",
        ],
    }
    files: list[tuple[str, Path]] = []
    for role, paths in explicit.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((role, path))

    recursive = {
        "smoke_evidence": root
        / "artifacts"
        / "smoke"
        / "stage1_reference_fresh_v6_20260814",
        "interrupted_attempt_1": root
        / "artifacts"
        / "exploration"
        / "stage1_reference_fresh_1m_v6_20260814_attempt1_interrupted",
        "interrupted_attempt_2": root
        / "artifacts"
        / "exploration"
        / "stage1_reference_fresh_1m_v6_20260814_attempt2_interrupted_host_timeout",
        "completed_run": root
        / "artifacts"
        / "exploration"
        / "stage1_reference_fresh_1m_v6_20260814",
        "analysis": root
        / "artifacts"
        / "analysis"
        / "stage1_reference_fresh_1m_v6_20260814",
    }
    excluded = {output.resolve(), summary.resolve()}
    for role, directory in recursive.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() not in excluded:
                files.append((role, path))

    unique: dict[Path, str] = {}
    for role, path in files:
        unique.setdefault(path.resolve(), role)
    return sorted(
        ((role, path) for path, role in unique.items()),
        key=lambda item: str(item[1]).casefold(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files(root, output, summary_path)

    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["relative_path", "role", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for role, path in files:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = f"../{path.name}"
            writer.writerow(
                {
                    "relative_path": relative,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    summary = {
        "status": "PASS_ENGINEERING_VALIDATED_REFERENCE_INCONCLUSIVE",
        "manifest_path": output.as_posix(),
        "manifest_entries": len(files),
        "manifest_sha256": sha256(output),
        "scientific_classification": "inconclusive",
        "passing_policies": 2,
        "total_policies": 5,
        "independent_verification": "pass",
        "candidate_weight_launch": "prohibited",
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
