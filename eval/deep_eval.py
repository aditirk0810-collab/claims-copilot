"""Deep questions: agent vs. the simple one-shot copilot, on the same questions.

    python -m eval.deep_eval                  # up to 4 questions per type (balanced Yes/No), ~40 questions
    python -m eval.deep_eval --per-type 10

Questions come from eval/gold_deep_qa.jsonl (total cost, estimate gap, days to treatment,
at-fault check, date / injury / vehicle consistency, treatment before accident, red flags).
Sampling is balanced across answers, so planted inconsistencies ("No" / "Yes" flags) are included.

Grading
  money: within $0.01      days: same whole number      yes/no: same first word
  flags: "none" must say none; a planted inconsistency must be named (date / injury / vehicle / treatment)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from index.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
FLAG_WORDS = {"police_date_mismatch": ["date"], "injury_mismatch": ["injur"],
              "vehicle_mismatch": ["vehicle", "car"], "treatment_before_loss": ["before", "treatment"]}


def grade(pred: str, gold: str, kind: str) -> bool:
    p = (pred or "").strip().lower()
    if not p:
        return False
    if kind == "money":
        nums = [float(n.replace(",", "")) for n in re.findall(r"\$\s?([\d,]+\.\d{2})", pred)]
        return any(abs(n - float(gold.replace("$", "").replace(",", ""))) < 0.011 for n in nums)
    if kind == "days":
        m = re.search(r"-?\d+", p)
        return bool(m) and int(m.group()) == int(gold)
    if kind == "yes_no":
        return p.split()[0].strip(".,:;") == gold.lower()
    if kind == "flags":
        says_none = p.startswith("none") or "no inconsisten" in p or "none found" in p
        if gold == "none":
            return says_none
        return not says_none and all(any(w in p for w in FLAG_WORDS[f]) for f in gold.split(","))
    return False


def sample(gold: list[dict], per_type: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in gold:
        by_type[q["type"]][q["answer"] if q["answer_kind"] in ("yes_no", "flags") else "x"].append(q)
    out = []
    for t, groups in by_type.items():
        buckets = [rng.sample(v, len(v)) for _, v in sorted(groups.items())]
        picked = []
        while len(picked) < per_type and any(buckets):   # round-robin across answers
            for b in buckets:
                if b and len(picked) < per_type:
                    picked.append(b.pop())
        out += picked
    return out


def main() -> None:
    from qa.agent import ClaimAgent
    from qa.answer import answer_question
    from qa.llm import GeminiLLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    gold = [json.loads(l) for l in (ROOT / "eval" / "gold_deep_qa.jsonl").read_text().splitlines()]
    qs = sample(gold, args.per_type)
    llm, searcher = GeminiLLM(), Searcher()
    agent = ClaimAgent(searcher, llm)
    print(f"{len(qs)} deep questions ({len({q['type'] for q in qs})} types) | agent model {agent.model}")

    rows, t0 = [], time.time()
    for i, q in enumerate(qs, 1):
        a = agent.answer(q["question"], q["claim_id"])
        b = None if args.no_baseline else answer_question(q["question"], q["claim_id"], searcher, llm)
        rows.append((q, a, b))
        if i % 5 == 0 or i == len(qs):
            print(f"  {i}/{len(qs)}  ({time.time() - t0:.0f}s, API calls {llm.calls})")

    def score(pick):
        ok = [r.status == "answered" and grade(r.answer, q["answer"], q["answer_kind"]) for q, r in pick]
        raw = [grade(r.answer, q["answer"], q["answer_kind"]) for q, r in pick]
        return mean(ok), mean(raw)

    agent_rows = [(q, a) for q, a, _ in rows]
    a_ver, a_raw = score(agent_rows)
    print("\n| System | Correct + verified | Correct (any status) | Needs review | Avg steps | Avg docs cited | Avg latency |")
    print("|---|---|---|---|---|---|---|")
    print(f"| Agent (multi-step) | {a_ver:.1%} | {a_raw:.1%} | "
          f"{mean(a.status == 'needs_review' for _, a in agent_rows):.1%} | "
          f"{mean(a.info['steps'] for _, a in agent_rows):.1f} | "
          f"{mean(a.checks.get('documents_cited', 0) for _, a in agent_rows):.1f} | "
          f"{mean(a.info['latency_ms'] for _, a in agent_rows) / 1000:.1f}s |")
    if not args.no_baseline:
        base_rows = [(q, b) for q, _, b in rows]
        b_ver, b_raw = score(base_rows)
        print(f"| Simple (1 search + 1 answer) | {b_ver:.1%} | {b_raw:.1%} | "
              f"{mean(b.status == 'needs_review' for _, b in base_rows):.1%} | 1.0 | "
              f"{mean(len(b.source_pages) for _, b in base_rows):.1f} | - |")

    print("\nBy question type (agent correct+verified / simple correct+verified):")
    by_t = defaultdict(list)
    for q, a, b in rows:
        by_t[q["type"]].append((a.status == "answered" and grade(a.answer, q["answer"], q["answer_kind"]),
                                b is not None and b.status == "answered" and grade(b.answer, q["answer"], q["answer_kind"])))
    for t, v in sorted(by_t.items()):
        print(f"  {t:<26} agent {mean(x for x, _ in v):.0%}   simple {mean(y for _, y in v):.0%}   ({len(v)})")

    misses = [(q, a) for q, a, _ in rows if not (a.status == "answered" and grade(a.answer, q["answer"], q["answer_kind"]))]
    if misses:
        print(f"\nAgent misses ({len(misses)}):")
        for q, a in misses[:12]:
            print(f"  {q['claim_id']} {q['type']}: gold {q['answer']!r} | got [{a.status}] {a.answer[:80]!r} {a.checks.get('unbacked_numbers') or ''}")

    out = ROOT / "eval" / "results"
    (out / "deep.json").write_text(json.dumps({
        "questions": len(rows), "agent_model": agent.model,
        "agent_correct_verified": a_ver, "agent_correct_any": a_raw,
        "simple_correct_verified": None if args.no_baseline else b_ver,
        "simple_correct_any": None if args.no_baseline else b_raw,
        "by_type": {t: {"agent": mean(x for x, _ in v), "simple": mean(y for _, y in v), "n": len(v)} for t, v in by_t.items()},
    }, indent=2))
    (out / "deep_answers.jsonl").write_text("".join(
        json.dumps({"gold": q, "agent": a.to_dict(), "simple": b.to_dict() if b else None}) + "\n" for q, a, b in rows))
    print("\nsaved", out / "deep.json")


if __name__ == "__main__":
    main()
