"""Validate the final stage-one PDF and record machine-checkable QA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--rendered_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visual_review_pass", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    rendered_dir = Path(args.rendered_dir).resolve()
    output_path = Path(args.output).resolve()

    reader = PdfReader(str(pdf_path))
    texts = [(page.extract_text() or "") for page in reader.pages]
    rendered = sorted(rendered_dir.glob("page-*.png"))
    dimensions = []
    for path in rendered:
        with Image.open(path) as image:
            dimensions.append([image.width, image.height])

    joined_text = "\n".join(texts)
    required_phrases = [
        "阶段一",
        "proxy–diagnostic divergence",
        "Scientifically unresolved",
        "正式实验仍被阻断",
        "附录 A：关键文件",
    ]
    required_phrase_presence = {
        phrase: phrase in joined_text for phrase in required_phrases
    }
    checks = {
        "pdf_exists": pdf_path.is_file(),
        "page_count_is_13": len(reader.pages) == 13,
        "all_pages_have_extractable_text": all(text.strip() for text in texts),
        "no_unicode_replacement_character": "\ufffd" not in joined_text,
        "rendered_page_count_matches_pdf": len(rendered) == len(reader.pages),
        "rendered_pages_have_consistent_dimensions": len({tuple(x) for x in dimensions}) == 1,
        "required_phrases_present": all(required_phrase_presence.values()),
        "visual_review_recorded": bool(args.visual_review_pass),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pdf": str(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size,
        "page_count": len(reader.pages),
        "page_text_characters": [len(text) for text in texts],
        "required_phrase_presence": required_phrase_presence,
        "rendered_page_count": len(rendered),
        "rendered_page_dimensions": dimensions,
        "checks": checks,
        "visual_review": {
            "status": "PASS" if args.visual_review_pass else "NOT_RECORDED",
            "scope": "All rendered pages inspected at original detail in the Codex app.",
            "criteria": [
                "no clipping or overlap",
                "formula-caption pairing",
                "legible charts, tables and captions",
                "clean appendix wrapping",
            ],
        },
        "validator_corrections": [
            {
                "issue": "The first phrase check expected 正式训练仍被阻断, while the verified report source says 正式实验仍被阻断.",
                "resolution": "The expectation was aligned with the source wording; the PDF was not changed.",
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
