"""Page records for the sorter: true label (from labels.json) + extracted text (from data/ingested).

Train/test split is by CLAIM, not by page: all pages of one claim land on the same side.
Splitting by page would leak names, amounts and claim IDs between train and test and
inflate the score.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKETS = ROOT / "data" / "synthetic"
INGESTED = ROOT / "data" / "ingested"

LABELS = ["claim_form", "police_report", "repair_estimate", "medical_bill",
          "correspondence", "photo", "other"]


@dataclass
class PageRecord:
    claim_id: str
    page: int
    label: str
    scanned: bool
    method: str   # how the text was read: text / ocr / image
    text: str
    pdf: Path | None
    image: Path | None = None   # set for real scans (RVL-CDIP) that come as images, not PDFs


def load_pages(packets: Path = PACKETS, ingested: Path = INGESTED) -> list[PageRecord]:
    records = []
    for lab_file in sorted(packets.glob("*/labels.json")):
        lab = json.loads(lab_file.read_text())
        ing_file = ingested / f"{lab['claim_id']}.jsonl"
        if not ing_file.exists():
            raise FileNotFoundError(f"{ing_file} missing: run `python -m ingest.run` first")
        got = {r["page"]: r for r in map(json.loads, ing_file.read_text().splitlines())}
        for lp in lab["pages"]:
            g = got[lp["page"]]
            records.append(PageRecord(lab["claim_id"], lp["page"], lp["type"], lp["scanned"],
                                      g["method"], g["text"], lab_file.parent / "packet.pdf"))
    return records


def split_by_claim(records: list[PageRecord], test_share: float = 0.3, seed: int = 0):
    claims = sorted({r.claim_id for r in records})
    random.Random(seed).shuffle(claims)
    n_test = max(1, round(len(claims) * test_share))
    test_claims = set(claims[:n_test])
    train = [r for r in records if r.claim_id not in test_claims]
    test = [r for r in records if r.claim_id in test_claims]
    return train, test
