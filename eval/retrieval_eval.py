"""Grade search on the gold questions: is the right page in the results?

    python -m eval.retrieval_eval

For each answerable gold question, search inside its claim and check the pages of the
top-k chunks against the question's evidence pages.

  hit@1     the very first result is on a correct page
  recall@5  a correct page is somewhere in the top 5 (what the answer step will read)
  MRR       mean of 1/rank of the first correct result (rewards ranking it higher)
  latency   milliseconds per search
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from index.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [("keyword", "chroma", "BM25 keyword"), ("vector", "faiss", "vector (FAISS)"),
           ("vector", "chroma", "vector (Chroma)"), ("hybrid", "chroma", "hybrid (Chroma + BM25, RRF)")]


def load_gold() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "gold_qa.jsonl").read_text().splitlines()]
    return [q for q in rows if q["answerable"]]


def page_methods() -> dict[tuple[str, int], str]:
    out = {}
    for f in (ROOT / "data" / "ingested").glob("CLM-*.jsonl"):
        for r in map(json.loads, f.read_text().splitlines()):
            out[(r["claim_id"], r["page"])] = r["method"]
    return out


def evaluate(searcher: Searcher, gold: list[dict], mode: str, store: str, k: int = 5) -> dict:
    """Two answer keys:
    strict  = evidence_pages: the official source page (claim form for the policy number)
    any     = answer_pages: any page where the answer is printed (also the letter, estimate...)
    The answer step only needs SOME page with the answer, so `any` is the main metric."""
    methods = page_methods()
    hits1, recall, recall_strict, rr, times = [], [], [], [], []
    by_type, by_method = defaultdict(list), defaultdict(list)
    misses = []
    for q in gold:
        t0 = time.perf_counter()
        results = searcher.search(q["question"], q["claim_id"], mode=mode, k=k, store=store)
        times.append((time.perf_counter() - t0) * 1000)
        pages = [c.page for c, _ in results]
        good = set(q.get("answer_pages", q["evidence_pages"]))
        first = next((i for i, p in enumerate(pages) if p in good), None)
        hits1.append(first == 0)
        recall.append(first is not None)
        recall_strict.append(any(p in set(q["evidence_pages"]) for p in pages))
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        by_type[q["type"]].append(first is not None)
        ev_method = "ocr" if any(methods[(q["claim_id"], p)] == "ocr" for p in q["evidence_pages"]) else "text"
        by_method[ev_method].append(first is not None)
        if first is None:
            misses.append((q["claim_id"], q["question"], sorted(good), pages))
    return {"hit@1": mean(hits1), f"recall@{k}": mean(recall), f"recall@{k}_strict": mean(recall_strict),
            "mrr": mean(rr), "ms": mean(times), "by_type": {t: mean(v) for t, v in by_type.items()},
            "by_method": {m: (mean(v), len(v)) for m, v in by_method.items()}, "misses": misses}


def main() -> None:
    gold = load_gold()
    searcher = Searcher()
    print(f"{len(searcher.chunks)} chunks, {len(gold)} answerable questions, "
          f"embedding: {searcher.embedder.name} ({searcher.embed_secs:.1f}s incl. cache)")
    results = {}
    print("\n| Search | hit@1 | recall@5 | recall@5 (strict: source page) | MRR | ms/query |")
    print("|---|---|---|---|---|---|")
    for mode, store, label in CONFIGS:
        r = evaluate(searcher, gold, mode, store)
        results[label] = r
        print(f"| {label} | {r['hit@1']:.1%} | {r['recall@5']:.1%} | {r['recall@5_strict']:.1%} | "
              f"{r['mrr']:.3f} | {r['ms']:.1f} |")

    best_label = max(results, key=lambda k: (results[k]["recall@5"], results[k]["mrr"]))
    best = results[best_label]
    print(f"\nBest: {best_label}. Recall@5 by question type:")
    for t, v in sorted(best["by_type"].items(), key=lambda x: x[1]):
        print(f"  {t:<22} {v:.1%}")
    print("Recall@5 by how the source page was read:",
          {m: f"{v:.1%} of {n}" for m, (v, n) in best["by_method"].items()})
    if best["misses"]:
        print(f"\nMisses ({len(best['misses'])}), first 8:")
        for claim, q, ev, got in best["misses"][:8]:
            print(f"  {claim} | {q} | evidence {ev} | got pages {got}")

    out = ROOT / "eval" / "results" / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "misses"} | {"n_misses": len(v["misses"])}
                               for k, v in results.items()}, indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
