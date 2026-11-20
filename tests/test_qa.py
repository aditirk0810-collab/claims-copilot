"""Answer step: grounded answers pass, made-up or mis-cited answers are flagged, refusals pass through."""

from eval.qa_eval import correct
from index.chunking import chunk_page
from qa.answer import answer_question, build_prompt
from qa.llm import FakeLLM

ESTIMATE = "Lewis Collision Repair\nRepair Estimate\nEstimate total $2,346.78"
LETTER = "Dear Brooke,\nThank you for reporting your loss of 08/21/2026."


class FakeSearcher:
    def __init__(self):
        self.chunks = (chunk_page("CLM-1", 10, "repair_estimate", "ocr", ESTIMATE)
                       + chunk_page("CLM-1", 6, "correspondence", "text", LETTER))

    def search(self, query, claim_id, mode="hybrid", k=5, store="chroma"):
        return [(c, 1.0) for c in self.chunks if c.claim_id == claim_id][:k]


def run(model_output: dict):
    return answer_question("What is the repair estimate total?", "CLM-1", FakeSearcher(), FakeLLM(lambda p: model_output))


def test_grounded_answer_is_shown():
    r = run({"answer": "$2,346.78", "quote": "Estimate total $2,346.78", "source_pages": [10], "not_in_packet": False})
    assert r.status == "answered" and r.source_pages == [10]
    assert all(r.checks.values())


def test_made_up_amount_goes_to_review():
    r = run({"answer": "$2,999.00", "quote": "Estimate total $2,999.00", "source_pages": [10], "not_in_packet": False})
    assert r.status == "needs_review"
    assert not r.checks["answer_on_cited_page"]


def test_wrong_page_citation_goes_to_review():
    r = run({"answer": "$2,346.78", "quote": "Estimate total $2,346.78", "source_pages": [6], "not_in_packet": False})
    assert r.status == "needs_review"


def test_citing_a_page_that_was_not_retrieved_goes_to_review():
    r = run({"answer": "$2,346.78", "quote": "Estimate total $2,346.78", "source_pages": [99], "not_in_packet": False})
    assert r.status == "needs_review"
    assert not r.checks["cited_pages_retrieved"]


def test_refusal_passes_through():
    r = run({"answer": "", "quote": "", "source_pages": [], "not_in_packet": True})
    assert r.status == "not_in_packet" and r.answer == ""


def test_prompt_numbers_sources_with_pages():
    p = build_prompt("q?", [(c, 1.0) for c in FakeSearcher().chunks])
    assert "[S1] page 10 (repair_estimate)" in p and "Question: q?" in p


def test_correct_matching():
    assert correct("$2,346.78", "$2,346.78")
    assert correct("The total is $2,346.78", "$2,346.78")
    assert not correct("$2,346.70", "$2,346.78")
    assert correct("Yes.", "Yes") and not correct("No", "Yes")
    assert not correct("", "$1.00")
