"""Grade the router on 30 hand-written questions (worded differently from the training templates).

    python -m eval.router_eval

Compares rules only, classifier only, and the combined router. The costly mistake is sending
a deep question to Quick (wrong or missing answer); sending a simple one to Deep only costs time.
"""

from __future__ import annotations

import json
from pathlib import Path

from qa.router import Router

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    router = Router()
    results = {}
    for name in ("router_questions", "router_holdout"):
        print(f"\n===== {name} =====")
        results[name] = grade(router, [json.loads(l) for l in (ROOT / "eval" / f"{name}.jsonl").read_text().splitlines()])
    (ROOT / "eval" / "results" / "router.json").write_text(json.dumps(results, indent=2))


def grade(router, qs) -> dict:
    print(f"{len(qs)} hand-written questions | classifier trained on {router.n_train} template questions\n")
    configs = {"rules only": dict(use_rules=True, use_classifier=False),
               "classifier only": dict(use_rules=False, use_classifier=True),
               "combined router": dict(use_rules=True, use_classifier=True)}
    print("| Router | Accuracy | Deep sent to Quick (costly) | Quick sent to Deep (slower) |")
    print("|---|---|---|---|")
    results = {}
    for name, kw in configs.items():
        preds = [router.route(q["question"], **kw) for q in qs]
        acc = sum(p.route == q["route"] for p, q in zip(preds, qs)) / len(qs)
        miss_deep = sum(q["route"] == "deep" and p.route == "quick" for p, q in zip(preds, qs))
        miss_quick = sum(q["route"] == "quick" and p.route == "deep" for p, q in zip(preds, qs))
        results[name] = {"accuracy": acc, "deep_to_quick": miss_deep, "quick_to_deep": miss_quick}
        print(f"| {name} | {acc:.0%} | {miss_deep} | {miss_quick} |")

    print("\nCombined router, question by question:")
    for q in qs:
        r = router.route(q["question"])
        mark = "ok " if r.route == q["route"] else "XX "
        print(f"  {mark}{r.route:5} [{r.method:10}] {q['question']}  -- {r.reason}")
    return results


if __name__ == "__main__":
    main()
