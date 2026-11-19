"""Read all packets and save page text.

    python -m ingest.run                 # Tesseract OCR on scanned pages
    python -m ingest.run --ocr none      # skip scans (fails if a packet has one)

Writes data/ingested/<claim_id>.jsonl, one line per page.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .reader import read_packet

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ingested")
    ap.add_argument("--ocr", choices=["tesseract", "none"], default="tesseract")
    ap.add_argument("--limit", type=int, default=None, help="only the first N packets")
    args = ap.parse_args()

    ocr = None
    if args.ocr == "tesseract":
        from .ocr import TesseractOCR
        ocr = TesseractOCR()

    args.out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.packets.glob("*/packet.pdf"))[: args.limit]
    methods, ocr_secs, n_ocr = Counter(), 0.0, 0
    t0 = time.time()
    for pdf in pdfs:
        claim_id = pdf.parent.name
        start = time.time()
        pages = read_packet(pdf, ocr)
        with open(args.out / f"{claim_id}.jsonl", "w") as f:
            for p in pages:
                f.write(json.dumps({"claim_id": claim_id, **p.to_dict()}) + "\n")
                methods[p.method] += 1
        k = sum(p.method == "ocr" for p in pages)
        if k:
            ocr_secs += time.time() - start
            n_ocr += k
        print(f"{claim_id}: {len(pages)} pages  " + "  ".join(f"{m}={c}" for m, c in Counter(p.method for p in pages).items()))
    print(f"\n{len(pdfs)} packets in {time.time() - t0:.1f}s | pages by method: {dict(methods)}")
    if n_ocr:
        print(f"OCR: {n_ocr} pages, ~{ocr_secs / n_ocr:.2f}s per scanned page")


if __name__ == "__main__":
    main()
