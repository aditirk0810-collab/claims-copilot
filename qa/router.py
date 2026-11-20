"""Router: decide whether a question needs a Quick answer or Deep analysis.

    python -m qa.router "What is our total exposure on this claim?"

Three layers, cheapest first, all local (no API calls):

  1. rules       keyword patterns for math, comparisons, consistency, time gaps, fault/recovery
                 -> instant and explainable ("needs a calculation: 'add'")
  2. classifier  question embedding (bge-small) + logistic regression, trained on the gold
                 questions: simple lookups (gold_qa) vs. multi-document questions (gold_deep_qa)
  3. default     no rule and the classifier is not confident -> Quick.

Safety net (in route_and_answer): if a Quick answer refuses ("not in packet") or fails its
verification, the question is escalated to Deep automatically, so a routing miss costs time,
not a wrong answer.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# (reason, pattern). Patterns are deliberately specific: "total" alone is NOT here, because
# "What is the total on the medical bill?" is a single-document lookup.
RULES = [
    ("needs a calculation", r"\b(add|adding|added|sum|combined|altogether|overall|exposure|all (the )?costs|owe in total|grand total|total (known )?cost)\b"),
    ("compares documents", r"\b(compare|comparison|cheaper|more expensive|by how much|difference|versus|vs\.?|higher .* than|lower .* than)\b"),
    ("checks consistency", r"\b(consistent|inconsisten\w*|match(es)?|agree|line up|lines up|discrepanc\w*|mismatch\w*|contradict\w*|same (one|vehicle|car|date|day|driver|person)\b)"),
    ("looks for red flags", r"\b(suspicious|red flags?|fraud\w*|anything (odd|wrong|unusual))\b"),
    ("needs date math", r"\b(how many days|how long|days (between|after|before)|before the (accident|crash|loss)|after the (accident|crash|loss))\b"),
    ("needs fault / recovery analysis", r"\b(subrogat\w*|recover\w*|our (policyholder|insured|driver)\b.*\bfault|at fault\b.*\b(our|policyholder|insured))\b"),
    ("needs the whole packet", r"\b(whole packet|all (the )?documents|every (document|estimate)|summari[sz]e)\b"),
]
DEEP_IF_P = 0.80   # classifier must be confident to send a no-rule question to Deep (it is weak: few examples)


@dataclass
class Route:
    route: str          # "quick" | "deep"
    method: str         # "rule" | "classifier" | "fallback"
    reason: str
    p_deep: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def rule_route(question: str) -> Route | None:
    q = question.lower()
    for reason, pattern in RULES:
        m = re.search(pattern, q)
        if m:
            return Route("deep", "rule", f"{reason} ('{m.group(0)}')")
    return None


def training_questions() -> tuple[list[str], list[int]]:
    """Unique gold questions labelled 0 = quick (single fact) / 1 = deep (multi-document)."""
    simple = {json.loads(l)["question"] for l in (ROOT / "eval" / "gold_qa.jsonl").read_text().splitlines()}
    deep = {json.loads(l)["question"] for l in (ROOT / "eval" / "gold_deep_qa.jsonl").read_text().splitlines()}
    qs = sorted(simple) + sorted(deep)
    return qs, [0] * len(simple) + [1] * len(deep)


class Router:
    def __init__(self, embedder=None):
        from sklearn.linear_model import LogisticRegression

        if embedder is None:
            from index.stores import Embedder
            embedder = Embedder()
        self.embedder = embedder
        qs, y = training_questions()
        X = self.embedder.docs(qs)
        self.clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(X, y)
        self.n_train = len(qs)

    def p_deep(self, question: str) -> float:
        return float(self.clf.predict_proba(self.embedder.docs([question]))[0, 1])

    def route(self, question: str, use_rules: bool = True, use_classifier: bool = True) -> Route:
        if use_rules:
            r = rule_route(question)
            if r:
                return r
        if not use_classifier:
            return Route("quick", "rule", "no multi-document signal found")
        p = self.p_deep(question)
        if p >= DEEP_IF_P:
            return Route("deep", "classifier", f"similar to multi-document questions ({p:.0%})", p)
        return Route("quick", "classifier", f"no multi-document signal (deep probability {p:.0%})", p)


def route_and_answer(question: str, claim_id: str, router: Router, quick_fn, deep_fn) -> dict:
    """Route, answer, and escalate Quick -> Deep when the quick answer can't be trusted.

    quick_fn / deep_fn: (question, claim_id) -> result object with .status and .to_dict()."""
    r = router.route(question)
    if r.route == "deep":
        return {"route": r.to_dict(), "mode": "deep", "escalated": False, "result": deep_fn(question, claim_id).to_dict()}
    quick = quick_fn(question, claim_id)
    if quick.status == "answered":
        return {"route": r.to_dict(), "mode": "quick", "escalated": False, "result": quick.to_dict()}
    return {"route": r.to_dict(), "mode": "deep", "escalated": True,
            "escalation_reason": f"quick answer was '{quick.status}'", "result": deep_fn(question, claim_id).to_dict()}


if __name__ == "__main__":
    router = Router()
    for q in sys.argv[1:] or ["What is the policy number?", "What is our total exposure on this claim?"]:
        r = router.route(q)
        print(f"{r.route.upper():5} [{r.method}] {r.reason}  <- {q}")
