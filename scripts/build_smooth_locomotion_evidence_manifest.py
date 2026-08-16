"""Create a SHA-256 evidence manifest for the smooth-locomotion exploration."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "SMOOTH_LOCOMOTION_EVIDENCE_MANIFEST_20260816.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "configs":
        return "frozen_config"
    if relative.parts[0] == "protocols":
        return "protocol"
    if relative.parts[0] in {"src", "scripts", "tests"}:
        return "code_or_test"
    if path.suffix.lower() == ".mp4":
        return "full_endpoint_video"
    if relative.parts[0] == "reports":
        return "report_source"
    if relative.parts[0] == "output":
        return "deliverable"
    return "generated_evidence"


def main() -> None:
    explicit = [
        ROOT / "configs" / "body_smoothness_gsde_matrix_v1_20260816.json",
        ROOT / "protocols" / "BODY_SMOOTHNESS_GSDE_MATRIX_PROTOCOL_20260816.md",
        ROOT / "docs" / "INTENDED_BEHAVIOUR_CONSTRUCT_AUDIT_V2_20260816.md",
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
        ROOT / "src" / "proxygap" / "experiment.py",
        ROOT / "src" / "proxygap" / "metrics.py",
        ROOT / "scripts" / "run_body_smoothness_gsde_matrix.py",
        ROOT / "scripts" / "analyse_body_smoothness_gsde_matrix.py",
        ROOT / "scripts" / "render_body_smoothness_gsde_videos.py",
        ROOT / "scripts" / "build_smooth_locomotion_report_pdf.py",
        ROOT / "tests" / "test_body_smoothness_gsde.py",
        ROOT / "reports" / "SMOOTH_LOCOMOTION_MECHANISM_EXPLORATION_20260816_CN.md",
        ROOT / "output" / "pdf" / "SMOOTH_LOCOMOTION_MECHANISM_EXPLORATION_20260816_CN.pdf",
        ROOT / "output" / "pdf" / "SMOOTH_LOCOMOTION_MECHANISM_EXPLORATION_20260816_CN_QA.json",
    ]
    generated_roots = [
        ROOT / "artifacts" / "dev" / "body_smoothness_gsde_matrix_v1" / "analysis",
        ROOT / "artifacts" / "dev" / "body_smoothness_gsde_matrix_v1" / "8_16_trials_4",
    ]
    files = {path.resolve() for path in explicit if path.is_file()}
    for directory in generated_roots:
        if directory.exists():
            files.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    rows = []
    for path in sorted(files):
        rows.append({
            "category": category(path),
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evidence records to {OUTPUT}")


if __name__ == "__main__":
    main()
