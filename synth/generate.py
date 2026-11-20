"""Generate synthetic claim packets with ground-truth labels.

    python -m synth.generate --n 50 --seed 42

Writes, per claim, data/synthetic/<claim_id>/packet.pdf and labels.json, plus
data/synthetic/page_labels.csv (classifier labels) and eval/gold_qa.jsonl
(gold questions with answers and evidence pages). Same seed -> same output.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import pymupdf
from faker import Faker
from PIL import Image, ImageFilter

from .claim import build_claim, inject_anomalies
from .deep_qa import build_deep_qa
from .qa import build_qa
from .render import RenderedPage, render_packet

ROOT = Path(__file__).resolve().parent.parent
REAL_PHOTOS = ROOT / "data" / "raw" / "car_damage" / "photos"  # CC BY 4.0, see SOURCE.md there


def load_photo_pool(photo_dir: Path | None) -> list[Path] | None:
    """Sorted list of real damage photos, or None to fall back to drawn photos."""
    if photo_dir is None or not photo_dir.is_dir():
        return None
    pool = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return pool or None


def _scan(page_pdf: bytes, rng: random.Random) -> bytes:
    """Make a page look scanned: rasterize, tilt, blur, add noise, JPEG. Text is no longer extractable."""
    src = pymupdf.open("pdf", page_pdf)
    pix = src[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    img = img.rotate(rng.uniform(-1.8, 1.8), expand=False, fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    img = Image.blend(img, Image.effect_noise(img.size, rng.uniform(20, 40)), 0.08)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(45, 75))
    return buf.getvalue()


def _page_text(page_pdf: bytes) -> str:
    return pymupdf.open("pdf", page_pdf)[0].get_text()


def build_packet(index: int, seed: int, out_dir: Path, scan_rate: float = 0.3,
                 photo_pool: list[Path] | None = None) -> dict:
    claim_seed = seed * 100_003 + index
    rng = random.Random(claim_seed)
    fake = Faker("en_US")
    fake.seed_instance(claim_seed)

    claim = build_claim(index, rng, fake)
    # Separate RNGs: planting inconsistencies and phrasing deep questions must not shift the main
    # random stream, so claims without an anomaly stay byte-for-byte the same.
    inject_anomalies(claim, random.Random(claim_seed ^ 0xA5A5))
    pages: list[RenderedPage] = render_packet(claim, rng, photo_pool)
    order = list(range(len(pages)))
    rng.shuffle(order)  # real packets arrive in no particular order

    out = pymupdf.open()
    page_meta, fact_pages = [], defaultdict(list)
    for pno, idx in enumerate(order, start=1):
        rp = pages[idx]
        scanned = rp.type != "photo" and rng.random() < scan_rate
        if scanned:
            new = out.new_page(width=612, height=792)
            new.insert_image(new.rect, stream=_scan(rp.pdf, rng))
        else:
            out.insert_pdf(pymupdf.open("pdf", rp.pdf))
        page_meta.append({
            "page": pno,
            "type": rp.type,
            "doc_id": rp.doc_id,
            "scanned": scanned,
            "text": _page_text(rp.pdf),  # ground-truth text (what perfect OCR would return)
        })
        if rp.source:
            page_meta[-1]["photo_source"] = rp.source
        for key in rp.facts:
            fact_pages[key].append(pno)

    claim_dir = out_dir / claim.claim_id
    claim_dir.mkdir(parents=True, exist_ok=True)
    out.save(claim_dir / "packet.pdf", garbage=3, deflate=True)

    qa = build_qa(claim, fact_pages, rng)
    # evidence_pages = the official source page(s) of a fact (strict).
    # answer_pages   = every page where the answer string is printed (lenient): e.g. the vehicle
    # also appears on the police report, estimate and letter. Short Yes/No answers and the
    # cross-page comparison question keep their strict pages only.
    for q in qa:
        q["answer_pages"] = list(q["evidence_pages"])
        if q["answerable"] and q["type"] not in ("lowest_estimate", "injury_reported"):
            q["answer_pages"] = sorted(set(q["evidence_pages"])
                                       | {p["page"] for p in page_meta if q["answer"] in p["text"]})

    labels = {
        "claim_id": claim.claim_id,
        "seed": claim_seed,
        "num_pages": len(page_meta),
        "pages": page_meta,
        "qa": qa,
        "anomalies": claim.anomalies,
        "deep_qa": build_deep_qa(claim, fact_pages, random.Random(claim_seed + 7)),
    }
    (claim_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of claim packets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan-rate", type=float, default=0.3, help="share of non-photo pages rendered as scans")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold_qa.jsonl")
    ap.add_argument("--deep-gold", type=Path, default=ROOT / "eval" / "gold_deep_qa.jsonl")
    ap.add_argument("--photos", type=Path, default=REAL_PHOTOS,
                    help="folder of real damage photos (falls back to drawn photos if missing)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    pool = load_photo_pool(args.photos)
    print(f"photos: {len(pool)} real from {args.photos}" if pool else "photos: drawn (no real photo folder found)")
    all_labels = [build_packet(i, args.seed, args.out, args.scan_rate, pool) for i in range(1, args.n + 1)]

    with open(args.out / "page_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "page", "type", "scanned"])
        for lab in all_labels:
            for p in lab["pages"]:
                w.writerow([lab["claim_id"], p["page"], p["type"], p["scanned"]])

    with open(args.gold, "w") as f:
        for lab in all_labels:
            for q in lab["qa"]:
                f.write(json.dumps(q) + "\n")

    with open(args.deep_gold, "w") as f:
        for lab in all_labels:
            for q in lab["deep_qa"]:
                f.write(json.dumps(q) + "\n")
    n_deep = sum(len(lab["deep_qa"]) for lab in all_labels)
    n_anom = sum(bool(lab["anomalies"]) for lab in all_labels)
    print(f"deep questions: {n_deep} | claims with a planted inconsistency: {n_anom}")

    n_pages = sum(lab["num_pages"] for lab in all_labels)
    n_q = sum(len(lab["qa"]) for lab in all_labels)
    by_type: dict[str, int] = defaultdict(int)
    for lab in all_labels:
        for p in lab["pages"]:
            by_type[p["type"]] += 1
    print(f"{args.n} packets, {n_pages} pages, {n_q} gold questions -> {args.out}")
    print("pages by type:", dict(sorted(by_type.items())))


if __name__ == "__main__":
    main()
