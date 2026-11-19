"""Real scanned documents from RVL-CDIP, mapped to our page types.

    python -m classify.rvl      # OCR all downloaded RVL pages once (cached in data/rvl_ocr.jsonl)

RVL-CDIP has 16 document types; we use the 5 that resemble pages in a claim packet.
The mapping is approximate (an RVL "form" is a generic office form, not an insurance
loss notice), so this adds realistic scan variety with some label noise.
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from .data import ROOT, PageRecord

RVL_DIR = ROOT / "data" / "raw" / "rvl_cdip"
OCR_CACHE = ROOT / "data" / "rvl_ocr.jsonl"

MAPPING = {
    "letter": "correspondence",
    "email": "correspondence",
    "memo": "correspondence",
    "form": "claim_form",
    "invoice": "other",
}


def rvl_files(split: str) -> list[tuple[Path, str]]:
    """(file, rvl_class) for one split: 'train' or 'validation'."""
    return [(f, cls) for cls in MAPPING for f in sorted((RVL_DIR / split / cls).glob("*.tif"))]


def _ocr_one(path: Path) -> dict:
    import pytesseract

    img = Image.open(path).convert("L")
    img = img.resize((img.width * 2, img.height * 2))  # RVL scans are ~100 dpi; upscale helps Tesseract
    data = pytesseract.image_to_data(img, config="--psm 3", output_type=pytesseract.Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confs = [float(c) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
    return {"path": str(path.relative_to(RVL_DIR)), "text": " ".join(words),
            "confidence": round(sum(confs) / len(confs), 1) if confs else 0.0}


def ocr_all(workers: int = 6) -> None:
    done = set()
    if OCR_CACHE.exists():
        done = {json.loads(l)["path"] for l in OCR_CACHE.read_text().splitlines()}
    todo = [f for split in ("train", "validation") for f, _ in rvl_files(split)
            if str(f.relative_to(RVL_DIR)) not in done]
    print(f"OCR: {len(todo)} pages to do ({len(done)} cached), {workers} workers")
    with Pool(workers) as pool, open(OCR_CACHE, "a") as out:
        for i, row in enumerate(pool.imap_unordered(_ocr_one, todo, chunksize=8), 1):
            out.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")


def load_rvl(split: str) -> list[PageRecord]:
    """RVL pages as PageRecords. Each page counts as its own 'claim' (they are unrelated documents)."""
    texts = {}
    if OCR_CACHE.exists():
        texts = {r["path"]: r["text"] for r in map(json.loads, OCR_CACHE.read_text().splitlines())}
    records = []
    for f, cls in rvl_files(split):
        key = str(f.relative_to(RVL_DIR))
        records.append(PageRecord(
            claim_id=f"RVL-{split}-{cls}-{f.stem}", page=1, label=MAPPING[cls], scanned=True,
            method="ocr", text=texts.get(key, ""), pdf=None, image=f,
        ))
    return records


if __name__ == "__main__":
    ocr_all()
