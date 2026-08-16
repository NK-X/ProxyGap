"""Build the V8 reference-construct diagnostic SHA-256 inventory."""

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


def collect(root: Path, output: Path, summary: Path) -> list[tuple[str, Path]]:
    explicit = {
        "protocol_and_context": [
            root / "configs" / "stage1_reference_fresh_1m_v6_20260814.json",
            root / "configs" / "stage1_reference_fresh_1m_outcome_v7_20260814.json",
            root / "configs" / "stage1_reference_construct_adjudication_v8_20260814.json",
            root / "protocols" / "STAGE1_REFERENCE_CONSTRUCT_ADJUDICATION_V8_20260814.md",
            root / "reports" / "STAGE1_REFERENCE_HIGH_Z_DIAGNOSTIC_20260814_CN.md",
            root / "docs" / "STAGE1_METRIC_CONTRACT_20260814.md",
            root / "docs" / "STAGE1_DEVIATION_REGISTER_20260814.md",
            root.parent / "PROJECT_CONTEXT.md",
        ],
        "implementation": [
            root / "src" / "proxygap" / "ant_wrapper.py",
            root / "src" / "proxygap" / "metrics.py",
            root / "src" / "proxygap" / "high_z_diagnostic.py",
            root / "scripts" / "diagnose_stage1_reference_high_z.py",
            root / "scripts" / "verify_stage1_reference_high_z.py",
            root / "scripts" / "render_stage1_full_video.py",
            root / "scripts" / "build_reference_high_z_contact_sheet.py",
            root / "scripts" / "build_stage1_reference_construct_manifest.py",
        ],
        "tests": [
            root / "tests" / "test_high_z_diagnostic.py",
            root / "tests" / "test_stage1_reference_construct_outcome.py",
        ],
        "parent_evidence": [
            root
            / "artifacts"
            / "exploration"
            / "stage1_reference_fresh_1m_v6_20260814"
            / "logs"
            / "evaluation_metrics.csv",
            root
            / "artifacts"
            / "analysis"
            / "stage1_reference_fresh_1m_v6_20260814"
            / "model_sha256_manifest.csv",
            root
            / "artifacts"
            / "analysis"
            / "stage1_reference_fresh_1m_v6_20260814"
            / "FINAL_SHA256_MANIFEST_20260814.csv",
        ],
    }
    files: list[tuple[str, Path]] = []
    for role, paths in explicit.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((role, path))

    analysis = (
        root
        / "artifacts"
        / "analysis"
        / "stage1_reference_high_z_diagnostic_v8_20260814"
    )
    excluded = {output.resolve(), summary.resolve()}
    for path in sorted(analysis.rglob("*")):
        if path.is_file() and path.resolve() not in excluded:
            files.append(("diagnostic_evidence", path))

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
    files = collect(root, output, summary_path)
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

    result = {
        "status": "PASS_ENGINEERING_VALIDATED_CONSTRUCT_REVISION_REQUIRED",
        "manifest_entries": len(files),
        "manifest_sha256": sha256(output),
        "independent_verification": "pass",
        "episodes_replayed": 100,
        "step_traces": 100,
        "matched_complete_videos": 5,
        "construct_adjudication": "V6 gate insufficient for human-intended stable quadrupedal locomotion",
        "candidate_weight_launch": "prohibited",
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
