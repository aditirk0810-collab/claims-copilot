"""Grade the full copilot (search + Gemini + verification) on the gold questions.

    python -m eval.qa_eval                 # the 15 held-out test claims (~200 questions)
    python -m eval.qa_eval --all           # all 50 claims (679 questions; slow on the free tier)
    python -m eval.qa_eval --limit 40      # quick check

Metrics
  answerable questions:
    accuracy          shown answer matches the gold answer
    citation accuracy shown answer cites a page that really holds the answer
    needs_review      model answered but verification failed -> sent to a human
    false refusal     said "not in packet" although the answer was there
  unanswerable (NOT_IN_PACKET) questions:
    refusal rate      correctly said "not in packet"
    hallucination     invented an answer (the dangerous failure)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from index.search import Searcher
from qa.answer import answer_question, norm

ROOT = Path(__file__).resolve().parent.parent


def correct(pred: str, gold: str) -> bool:
    p, g = norm(pred), norm(gold)
    if not p:
        return False
    if g in ("yes", "no"):
        return p.split()[0].strip(".,") == g
    return g == p or g in p          # model may add a little context ("$2,346.78 total")


def test_claims() -> set[str]:
    claims = set()
    for f in (ROOT / "data" / "ingested").glob("CLM-*.jsonl"):
        first = json.loads(f.read_text().splitlines()[0])
        if first.get("split") == "test":
            claims.add(first["claim_id"])
    return claims


def main() -> None:
    from qa.llm import GeminiLLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    gold = [json.loads(l) for l in (ROOT / "eval" / "gold_qa.jsonl").read_text().splitlines()]
    if not args.all:
        keep = test_claims()
        gold = [q for q in gold if q["claim_id"] in keep]
    gold = gold[: args.limit]
    llm = GeminiLLM(model=args.model)
    searcher = Searcher()
    print(f"{len(gold)} questions | model {llm.model} | ~{llm.min_gap:.1f}s between new calls")

    rows, t0 = [], time.time()
    for i, q in enumerate(gold, 1):
        r = answer_question(q["question"], q["claim_id"], searcher, llm)
        rows.append((q, r))
        if i % 20 == 0 or i == len(gold):
            print(f"  {i}/{len(gold)}  ({time.time() - t0:.0f}s, new API calls: {llm.calls})")

    ans = [(q, r) for q, r in rows if q["answerable"]]
    una = [(q, r) for q, r in rows if not q["answerable"]]
    shown = [(q, r) for q, r in ans if r.status == "answered"]
    acc = mean(r.status == "answered" and correct(r.answer, q["answer"]) for q, r in ans)
    cite = mean(bool(set(r.source_pages) & set(q["answer_pages"])) for q, r in shown) if shown else 0
    shown_acc = mean(correct(r.answer, q["answer"]) for q, r in shown) if shown else 0
    status = Counter(r.status for _, r in ans)
    refuse = mean(r.status == "not_in_packet" for _, r in una) if una else None
    halluc = mean(r.status == "answered" for _, r in una) if una else None
    lat = [r.info["latency_ms"] for _, r in rows if not r.info.get("cached")]
    toks_in = [r.info.get("input_tokens") or 0 for _, r in rows]
    toks_out = [r.info.get("output_tokens") or 0 for _, r in rows]

    print(f"\nAnswerable questions: {len(ans)}")
    print(f"  accuracy (correct answer shown)        {acc:.1%}")
    print(f"  accuracy of answers shown (verified)   {shown_acc:.1%}   [{len(shown)} shown]")
    print(f"  citation accuracy (of shown)           {cite:.1%}")
    print(f"  needs human review (failed checks)     {status['needs_review'] / len(ans):.1%}")
    print(f"  false refusal ('not in packet')        {status['not_in_packet'] / len(ans):.1%}")
    if una:
        print(f"Unanswerable questions: {len(una)}")
        print(f"  correct refusal                        {refuse:.1%}")
        print(f"  hallucinated an answer                 {halluc:.1%}")
    if lat:
        print(f"Latency (uncached): median {median(lat):.0f} ms | tokens/question: "
              f"{mean(toks_in):.0f} in, {mean(toks_out):.0f} out")

    by_type = defaultdict(list)
    for q, r in ans:
        by_type[q["type"]].append(r.status == "answered" and correct(r.answer, q["answer"]))
    print("\nAccuracy by question type:")
    for t, v in sorted(by_type.items(), key=lambda x: mean(x[1])):
        print(f"  {t:<22} {mean(v):.0%} ({len(v)})")

    wrong = [(q, r) for q, r in rows if q["answerable"] and not (r.status == "answered" and correct(r.answer, q["answer"]))]
    wrong += [(q, r) for q, r in una if r.status != "not_in_packet"]
    if wrong:
        print(f"\nMisses ({len(wrong)}), first 10:")
        for q, r in wrong[:10]:
            print(f"  {q['claim_id']} | {q['question']} | gold {q['answer']!r} | got {r.status}: {r.answer!r} p{r.source_pages}")

    out = ROOT / "eval" / "results" / "qa.json"
    out.write_text(json.dumps({
        "model": llm.model, "questions": len(rows), "answerable": len(ans), "unanswerable": len(una),
        "accuracy": acc, "shown_accuracy": shown_acc, "citation_accuracy": cite,
        "needs_review_rate": status["needs_review"] / len(ans), "false_refusal_rate": status["not_in_packet"] / len(ans),
        "correct_refusal_rate": refuse, "hallucination_rate": halluc,
        "median_latency_ms": median(lat) if lat else None,
        "by_type": {t: mean(v) for t, v in by_type.items()},
    }, indent=2))
    (ROOT / "eval" / "results" / "qa_answers.jsonl").write_text(
        "".join(json.dumps({"gold": q, "pred": r.to_dict()}) + "\n" for q, r in rows))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
