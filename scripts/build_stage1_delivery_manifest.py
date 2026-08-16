"""Create a SHA-256 manifest for the stage-one bidirectional delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, output: Path) -> list[tuple[str, Path]]:
    explicit = {
        "protocol": [
            root / "protocols" / "STAGE1_BIDIRECTIONAL_DEVELOPMENT_PROTOCOL_V2_20260814.md",
            root / "protocols" / "STAGE1_PREFORMAL_REVISION_GATE_V3_20260814.md",
            root / "docs" / "STAGE1_DEVIATION_REGISTER_20260814.md",
            root / "configs" / "stage1_bidirectional_development_v2_20260814.json",
            root / "configs" / "stage1_preformal_revision_gate_v3_20260814.json",
        ],
        "implementation": [
            root / "src" / "proxygap" / "stage1.py",
            root / "src" / "proxygap" / "experiment.py",
            root / "scripts" / "prepare_stage1_bidirectional_development.py",
            root / "scripts" / "resume_development_parallel.py",
            root / "scripts" / "analyse_stage1_dense_development.py",
            root / "scripts" / "audit_stage1_bidirectional_result.py",
            root / "scripts" / "render_stage1_full_video.py",
            root / "scripts" / "render_stage1_report_equations.py",
            root / "scripts" / "build_stage1_bidirectional_review_pdf.py",
            root / "scripts" / "build_stage1_delivery_manifest.py",
            root / "scripts" / "validate_stage1_delivery.py",
        ],
        "tests": [
            root / "tests" / "test_stage1.py",
            root / "tests" / "test_ant_wrapper.py",
        ],
        "report": [
            root / "reports" / "STAGE1_BIDIRECTIONAL_DEVELOPMENT_REVIEW_20260814_CN.md",
            root / "output" / "pdf" / "ProxyGap_Stage1_Bidirectional_Review_20260814_CN.pdf",
        ],
        "input_data": [
            root
            / "artifacts"
            / "exploration"
            / "stage1_harmonised_existing_models_v2_20260814"
            / "logs"
            / "evaluation_metrics.csv",
            root
            / "artifacts"
            / "exploration"
            / "stage1_dense_development_300k_20260814"
            / "logs"
            / "evaluation_metrics.csv",
        ],
    }

    files: list[tuple[str, Path]] = []
    for role, paths in explicit.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((role, path))

    recursive_sources = {
        "upper_development_run": root
        / "artifacts"
        / "exploration"
        / "stage1_upper_development_300k_20260814",
        "analysis_output": root
        / "artifacts"
        / "analysis"
        / "stage1_bidirectional_development_v2_20260814",
    }
    for role, directory in recursive_sources.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() != output.resolve():
                files.append((role, path))

    unique: dict[Path, str] = {}
    for role, path in files:
        unique.setdefault(path.resolve(), role)
    return sorted(((role, path) for path, role in unique.items()), key=lambda item: str(item[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files(root, output)

    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["relative_path", "role", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for role, path in files:
            writer.writerow(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    print(output)
    print(f"entries={len(files)}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
