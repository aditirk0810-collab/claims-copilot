"""Score page reading against the ground truth in each packet's labels.json.

    python -m eval.ocr_eval

Metrics:
  - method accuracy: did we pick text / ocr / image correctly for each page?
  - CER (character error rate): edits needed to turn OCR text into the true text, / true length
  - fact survival: share of gold answers (amounts, IDs, dates, names) found intact in the page text.
    This matters most: one wrong digit in "$2,346.78" means a wrong answer later.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(pred: str, truth: str) -> float:
    pred, truth = norm(pred), norm(truth)
    return Levenshtein.distance(pred, truth) / max(1, len(truth))


def expected_method(label_page: dict) -> str:
    if label_page["type"] == "photo":
        return "image"
    return "ocr" if label_page["scanned"] else "text"


def evaluate(packets: Path, ingested: Path) -> dict:
    method_ok = method_n = 0
    cers = defaultdict(list)                      # method -> CER per page
    facts = defaultdict(lambda: [0, 0])           # method -> [found, total]
    misses = []
    for lab_file in sorted(packets.glob("*/labels.json")):
        lab = json.loads(lab_file.read_text())
        out = ingested / f"{lab['claim_id']}.jsonl"
        if not out.exists():
            continue
        got = {r["page"]: r for r in map(json.loads, out.read_text().splitlines())}
        for lp in lab["pages"]:
            g = got[lp["page"]]
            method_n += 1
            method_ok += g["method"] == expected_method(lp)
            if lp["type"] != "photo":
                cers[g["method"]].append(cer(g["text"], lp["text"]))
        for q in lab["qa"]:
            if not q["answerable"]:
                continue
            if q["type"] == "lowest_estimate":
                continue  # comparison across pages: the shop name is printed on only one of them
            for p in q["evidence_pages"]:
                g = got[p]
                f = facts[g["method"]]
                f[1] += 1
                if norm(q["answer"]) in norm(g["text"]):
                    f[0] += 1
                elif g["method"] == "ocr":
                    misses.append((lab["claim_id"], p, q["answer"]))
    return {"method_acc": method_ok / max(1, method_n), "pages": method_n,
            "cers": cers, "facts": facts, "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--ingested", type=Path, default=ROOT / "data" / "ingested")
    args = ap.parse_args()
    r = evaluate(args.packets, args.ingested)

    print(f"Method choice accuracy: {r['method_acc']:.1%} of {r['pages']} pages")
    print("\n| Method | Pages | Mean CER | Fact survival |")
    print("|---|---|---|---|")
    for m in sorted(r["cers"]):
        found, total = r["facts"][m]
        print(f"| {m} | {len(r['cers'][m])} | {mean(r['cers'][m]):.2%} | "
              f"{found}/{total} ({found / max(1, total):.1%}) |")
    if r["misses"]:
        print(f"\nFacts lost in OCR ({len(r['misses'])}), first 10:")
        for claim, page, ans in r["misses"][:10]:
            print(f"  {claim} p{page}: {ans!r}")


if __name__ == "__main__":
    main()
