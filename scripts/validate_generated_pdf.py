from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "PROXYGAP_TEN_HOUR_DEVELOPMENT_AUDIT_20260816_CN.pdf"
OUTPUT = ROOT / "output" / "pdf_qa_20260816"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(PDF))
    rendered: list[Path] = []
    for page_index in range(len(document)):
        bitmap = document[page_index].render(scale=1.45)
        image = bitmap.to_pil().convert("RGB")
        path = OUTPUT / f"page_{page_index + 1:02d}.png"
        image.save(path, optimize=True)
        rendered.append(path)

    thumbs = []
    for index, path in enumerate(rendered, start=1):
        page = Image.open(path).convert("RGB")
        page.thumbnail((310, 440))
        tile = Image.new("RGB", (330, 475), "white")
        tile.paste(page, ((330 - page.width) // 2, 25))
        ImageDraw.Draw(tile).text((12, 6), f"Page {index}", fill="#173753")
        thumbs.append(tile)
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 330, rows * 475), "#DDE5EC")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % columns) * 330, (index // columns) * 475))
    contact_sheet = OUTPUT / "contact_sheet.png"
    sheet.save(contact_sheet, optimize=True)

    reader = PdfReader(str(PDF))
    with pdfplumber.open(PDF) as pdf:
        extracted = [page.extract_text() or "" for page in pdf.pages]
    required_phrases = [
        "默认平地 Ant-v5",
        "Development gate",
        "不能宣称",
        "参考文献",
    ]
    combined = "\n".join(extracted)
    qa = {
        "status": "passed",
        "pdf": str(PDF),
        "sha256": sha256(PDF),
        "bytes": PDF.stat().st_size,
        "page_count_pdfium": len(document),
        "page_count_pypdf": len(reader.pages),
        "rendered_pages": len(rendered),
        "blank_text_pages": [index + 1 for index, text in enumerate(extracted) if not text.strip()],
        "required_phrase_presence": {phrase: phrase in combined for phrase in required_phrases},
        "contact_sheet": str(contact_sheet),
    }
    if (
        qa["page_count_pdfium"] != qa["page_count_pypdf"]
        or qa["rendered_pages"] != qa["page_count_pypdf"]
        or qa["blank_text_pages"]
        or not all(qa["required_phrase_presence"].values())
    ):
        qa["status"] = "failed"
    (OUTPUT / "PDF_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
