from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "evidence_manifest_20260816.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [
        ROOT / "configs" / "hybrid_guardrail_observability_correction_v1_20260816.json",
        ROOT / "src" / "proxygap" / "ant_wrapper.py",
        ROOT / "src" / "proxygap" / "metrics.py",
        ROOT / "src" / "proxygap" / "experiment.py",
        ROOT / "scripts" / "run_hybrid_guardrail_development.py",
        ROOT / "scripts" / "analyse_lateral_velocity_correction.py",
        ROOT / "scripts" / "render_hybrid_guardrail_videos.py",
        ROOT / "scripts" / "validate_hybrid_videos.py",
        ROOT / "scripts" / "build_ten_hour_report_figures.py",
        ROOT / "scripts" / "build_proxygap_ten_hour_pdf.py",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "data_quality_qa.json",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "development_gate_adjudication.json",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "endpoint_condition_summary.csv",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "lateral_velocity" / "lateral_velocity_qa.json",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "lateral_velocity" / "endpoint_condition_lateral_velocity_summary.csv",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "analysis" / "intent_sensitivity" / "sensitivity_manifest.json",
        ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "videos" / "VIDEO_QA.json",
        ROOT / "reports" / "PROXYGAP_TEN_HOUR_DEVELOPMENT_AUDIT_20260816_CN.md",
        ROOT / "output" / "pdf" / "PROXYGAP_TEN_HOUR_DEVELOPMENT_AUDIT_20260816_CN.pdf",
        ROOT / "output" / "pdf_qa_20260816" / "PDF_QA.json",
    ]
    paths.extend(sorted((ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1" / "videos").glob("*.mp4")))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evidence files:\n" + "\n".join(missing))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    print(f"{len(paths)} evidence files -> {OUTPUT}")


if __name__ == "__main__":
    main()
