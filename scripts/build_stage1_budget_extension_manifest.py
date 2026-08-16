"""Build a SHA-256 manifest for the stage-one 1M extension evidence package."""

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
            root / "protocols" / "STAGE1_DEVELOPMENT_BUDGET_EXTENSION_V4_20260814.md",
            root
            / "protocols"
            / "STAGE1_DEVELOPMENT_BUDGET_EXTENSION_ADJUDICATION_V5_20260814.md",
            root / "docs" / "STAGE1_DEVIATION_REGISTER_20260814.md",
            root / "configs" / "stage1_development_budget_extension_v4_20260814.json",
            root / "configs" / "stage1_post_extension_gate_v5_20260814.json",
            root.parent / "PROJECT_CONTEXT.md",
        ],
        "implementation": [
            root / "src" / "proxygap" / "budget_extension.py",
            root / "scripts" / "run_stage1_budget_extension.py",
            root / "scripts" / "smoke_stage1_budget_extension.py",
            root / "scripts" / "analyse_stage1_budget_extension.py",
            root / "scripts" / "verify_stage1_budget_extension.py",
            root / "scripts" / "build_stage1_1m_adjudication_pdf.py",
            root / "scripts" / "validate_stage1_1m_adjudication_pdf.py",
            root / "scripts" / "build_stage1_budget_extension_manifest.py",
        ],
        "report": [
            root / "reports" / "STAGE1_1M_EXTENSION_ADJUDICATION_20260814_CN.md",
            root
            / "output"
            / "pdf"
            / "ProxyGap_Stage1_1M_Extension_Adjudication_20260814_CN.pdf",
        ],
        "tests": [
            root / "tests" / "test_budget_extension.py",
            root / "tests" / "test_budget_extension_analysis.py",
            root / "tests" / "test_budget_extension_verifier.py",
            root / "tests" / "test_stage1_post_extension_gate.py",
        ],
    }

    files: list[tuple[str, Path]] = []
    for role, paths in explicit.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((role, path))

    recursive_sources = {
        "failed_smoke_evidence": root
        / "artifacts"
        / "smoke"
        / "stage1_budget_extension_resume_smoke_v4_20260814",
        "successful_smoke_evidence": root
        / "artifacts"
        / "smoke"
        / "stage1_budget_extension_resume_smoke_v4_20260814_attempt2",
        "extension_run": root
        / "artifacts"
        / "exploration"
        / "stage1_budget_extension_1m_v4_20260814",
        "analysis_output": root
        / "artifacts"
        / "analysis"
        / "stage1_budget_extension_1m_v4_20260814_attempt2",
        "report_qa": root
        / "artifacts"
        / "reports"
        / "stage1_budget_extension_1m_20260814",
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
    return sorted(
        ((role, path) for path, role in unique.items()),
        key=lambda item: str(item[1]).casefold(),
    )


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
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError:
                relative_path = f"../{path.name}"
            writer.writerow(
                {
                    "relative_path": relative_path,
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
