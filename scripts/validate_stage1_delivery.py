"""Validate the final stage-one report and key evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    pdf_path = root / "output" / "pdf" / "ProxyGap_Stage1_Bidirectional_Review_20260814_CN.pdf"
    markdown_path = root / "reports" / "STAGE1_BIDIRECTIONAL_DEVELOPMENT_REVIEW_20260814_CN.md"
    analysis = root / "artifacts" / "analysis" / "stage1_bidirectional_development_v2_20260814"
    result_path = analysis / "stage1_development_result.json"
    audit_path = analysis / "stage1_bidirectional_audit.json"
    video_paths = [
        analysis / "videos" / "reference_seed41101_eval51103_300k.mp4",
        analysis / "videos" / "ctrl_0p21875_seed41101_eval51103_300k.mp4",
    ]

    required = [pdf_path, markdown_path, result_path, audit_path, *video_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing delivery files: {missing}")

    reader = PdfReader(str(pdf_path))
    if len(reader.pages) != 14:
        raise AssertionError(f"Expected 14 PDF pages, found {len(reader.pages)}")
    if reader.is_encrypted:
        raise AssertionError("Final PDF must not be encrypted")

    with pdfplumber.open(pdf_path) as pdf:
        page_text_lengths = [len((page.extract_text() or "").strip()) for page in pdf.pages]
        if any(length < 80 for length in page_text_lengths):
            raise AssertionError(f"Potentially blank PDF page: {page_text_lengths}")
        extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    required_phrases = [
        "双向权重开发审查报告",
        "scientifically unresolved",
        "w=0.21875",
        "w=0.125",
        "正式 held-out training",
        "参考文献",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in extracted_text]
    if missing_phrases:
        raise AssertionError(f"Missing required PDF phrases: {missing_phrases}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if result["selected_candidates_by_side"] != {"lower": 0.21875, "upper": None}:
        raise AssertionError("Unexpected selected candidates")
    if audit["status"] != "development_candidate_nominated_formal_protocol_blocked":
        raise AssertionError("Unexpected audit status")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    validation = {
        "status": "PASS",
        "pdf": {
            "path": str(pdf_path),
            "sha256": sha256(pdf_path),
            "pages": len(reader.pages),
            "encrypted": reader.is_encrypted,
            "page_text_lengths": page_text_lengths,
            "visual_review": "completed_manually_from_120_dpi_page_renders",
        },
        "markdown_sha256": sha256(markdown_path),
        "result_status": result["status"],
        "audit_status": audit["status"],
        "selected_candidates_by_side": result["selected_candidates_by_side"],
        "video_sha256": {path.name: sha256(path) for path in video_paths},
    }
    output.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
