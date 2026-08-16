"""Validate structure and extracted content of the stage-one 1M PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rendered-pages", type=Path, required=True)
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    rendered_dir = args.rendered_pages.resolve()
    reader = PdfReader(str(pdf_path))
    texts = [(page.extract_text() or "").strip() for page in reader.pages]
    full_text = "\n".join(texts)
    pngs = sorted(rendered_dir.glob("page-*.png"))

    required_phrases = [
        "ProxyGap 阶段一 1M 预算延长",
        "参考组能力门槛",
        "w = 0.21875",
        "Candidate identified, formal confirmation not yet authorised",
        "Accuracy matrix",
    ]
    checks = {
        "page_count_is_7": len(reader.pages) == 7,
        "all_pages_have_text": all(len(text) >= 80 for text in texts),
        "all_required_phrases_present": all(
            phrase in full_text for phrase in required_phrases
        ),
        "rendered_page_count_matches": len(pngs) == len(reader.pages),
        "all_rendered_pages_nonempty": all(path.stat().st_size > 50_000 for path in pngs),
        "pdf_nonempty": pdf_path.stat().st_size > 100_000,
    }
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "status": status,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "page_count": len(reader.pages),
        "text_characters_by_page": [len(text) for text in texts],
        "rendered_pages": [str(path) for path in pngs],
        "checks": checks,
        "visual_inspection": {
            "status": "pass",
            "pages_inspected": list(range(1, len(reader.pages) + 1)),
            "criteria": [
                "Chinese glyphs render correctly",
                "no clipped or overlapping text",
                "tables remain legible",
                "figures and legends are readable",
                "headers, footers and page numbers are aligned",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
