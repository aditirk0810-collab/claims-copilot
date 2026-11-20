"""Agent: runs tools, does math in code, and code-checks every citation and number."""

import pytest

from eval.deep_eval import grade, sample
from qa.agent import ClaimAgent, calculate, days_between, numbers_in


class FakeSearcher:
    PAGES = {
        10: ("repair_estimate", "Repair Estimate Estimate total $2,346.78"),
        5: ("medical_bill", "Patient Statement Total charges $1,037.01 Date of service: 08/22/2026"),
        1: ("other", "Towing Invoice Amount paid $271.85"),
    }

    class C:
        def __init__(self, page, page_type, text):
            self.page, self.page_type, self.text = page, page_type, f"[{page_type}] {text}"
            self.claim_id, self.method, self.chunk_id = "CLM-1", "text", f"CLM-1/p{page:02d}/c0"

    def search(self, query, claim_id, mode="hybrid", k=5, store="chroma", page_types=None):
        return [(self.C(p, t, x), 1.0) for p, (t, x) in self.PAGES.items() if not page_types or t in page_types][:k]

    def page_text(self, claim_id, page):
        return self.PAGES.get(page)

    def page_types(self, claim_id):
        return [(p, t) for p, (t, _) in self.PAGES.items()]


class ScriptedLLM:
    """Plays back a fixed list of tool calls, one per agent turn."""

    def __init__(self, calls):
        self.calls = list(calls)

    def agent_turn(self, model, system, history, tools, force=None):
        name, args = self.calls.pop(0)
        return {"role": "model", "parts": [{"function_call": {"name": name, "args": args}}]}, {"latency_ms": 1}


def run(final_args, before=(("search", {"query": "totals"}),)):
    calls = list(before) + [("final_answer", final_args)]
    return ClaimAgent(FakeSearcher(), ScriptedLLM(calls), model="fake").answer("total cost?", "CLM-1")


CITES = [{"page": 10, "quote": "Estimate total $2,346.78"}, {"page": 5, "quote": "Total charges $1,037.01"},
         {"page": 1, "quote": "Amount paid $271.85"}]


def test_multi_document_answer_with_computed_total_is_verified():
    r = run({"answer": "$3,655.64", "explanation": "sum", "citations": CITES, "not_in_packet": False},
            before=[("search", {"query": "totals"}), ("calculate", {"expression": "2346.78 + 1037.01 + 271.85"})])
    assert r.status == "answered"
    assert r.checks["documents_cited"] == 3 and r.checks["numbers_grounded"]
    assert [s["tool"] for s in r.steps] == ["search", "calculate", "final_answer"]


def test_total_the_model_did_in_its_head_goes_to_review():
    # no calculate call: $3,655.64 is not in any quote and was never computed by code
    r = run({"answer": "$3,655.64", "explanation": "sum", "citations": CITES, "not_in_packet": False})
    assert r.status == "needs_review" and r.checks["unbacked_numbers"] == ["3655.64"]


def test_fake_quote_goes_to_review():
    bad = [{"page": 10, "quote": "Estimate total $9,999.99"}]
    r = run({"answer": "$9,999.99", "explanation": "x", "citations": bad, "not_in_packet": False})
    assert r.status == "needs_review" and not r.checks["all_citations_verified"]


def test_quote_with_ellipsis_is_checked_part_by_part():
    good = [{"page": 5, "quote": "Patient Statement ... Total charges $1,037.01"}]
    r = run({"answer": "$1,037.01", "explanation": "x", "citations": good, "not_in_packet": False})
    assert r.status == "answered"
    bad = [{"page": 5, "quote": "Patient Statement ... Total charges $5,000.00"}]
    r = run({"answer": "$5,000.00", "explanation": "x", "citations": bad, "not_in_packet": False})
    assert r.status == "needs_review"


def test_page_numbers_in_the_answer_are_not_flagged():
    r = run({"answer": "Yes, page 10 shows it", "explanation": "x",
             "citations": [{"page": 10, "quote": "Estimate total $2,346.78"}], "not_in_packet": False})
    assert r.checks["numbers_grounded"]


def test_citing_a_page_never_opened_goes_to_review():
    calls = [("read_page", {"page": 10}), ("final_answer", {"answer": "$271.85", "explanation": "x",
             "citations": [{"page": 1, "quote": "Amount paid $271.85"}], "not_in_packet": False})]
    r = ClaimAgent(FakeSearcher(), ScriptedLLM(calls), model="fake").answer("tow?", "CLM-1")
    assert r.status == "needs_review"


def test_agent_stops_after_max_steps():
    calls = [("search", {"query": "x"})] * 9 + [("final_answer", {"answer": "None found", "explanation": "x",
             "citations": [{"page": 1, "quote": "Amount paid $271.85"}], "not_in_packet": False})]
    r = ClaimAgent(FakeSearcher(), ScriptedLLM(calls), model="fake").answer("q", "CLM-1")
    assert r.info["steps"] == 10 and r.status == "answered"


def test_tools():
    assert calculate("$2,346.78 + 1,037.01") == 3383.79
    assert days_between("08/21/2026", "08/19/2026") == -2
    assert numbers_in("on 08/21/2026 total $1,037.01") == {"08/21/2026", "1037.01"}
    with pytest.raises(ValueError):
        calculate("__import__('os').system('ls')")


def test_grading():
    assert grade("$3,655.64", "$3,655.64", "money") and not grade("$3,655.00", "$3,655.64", "money")
    assert grade("2 days", "2", "days") and not grade("3", "2", "days")
    assert grade("No, the dates differ.", "No", "yes_no")
    assert grade("None found.", "none", "flags") and not grade("Vehicle differs", "none", "flags")
    assert grade("Estimate 2 lists a different vehicle", "vehicle_mismatch", "flags")
    assert not grade("None found", "vehicle_mismatch", "flags")


def test_sampling_balances_answers():
    gold = [{"type": "t", "answer": "Yes", "answer_kind": "yes_no"}] * 9 + [{"type": "t", "answer": "No", "answer_kind": "yes_no"}]
    picked = sample(gold, per_type=4)
    assert len(picked) == 4 and any(q["answer"] == "No" for q in picked)
