"""Router: rules, classifier threshold, and Quick -> Deep escalation."""

import numpy as np
import pytest

from qa.router import Route, Router, route_and_answer, rule_route


@pytest.mark.parametrize("q", [
    "Add the repair estimate and the tow bill.", "Which shop is cheaper and by how much?",
    "Do the dates match?", "Is there anything suspicious here?", "How many days until treatment?",
    "Can we pursue subrogation?", "Is the car on the estimate the same one as on the claim form?",
    "What's our total exposure?",
])
def test_rules_send_multi_document_questions_to_deep(q):
    r = rule_route(q)
    assert r is not None and r.route == "deep" and r.method == "rule"


@pytest.mark.parametrize("q", [
    "What is the policy number?", "What did the medical bill charge in total?", "Who is the adjuster?",
])
def test_rules_ignore_single_fact_questions(q):
    assert rule_route(q) is None


class FakeEmbedder:
    """'deep'-ish questions point one way, others the other way (enough to test the threshold logic)."""
    def docs(self, texts):
        return np.array([[1.0, 0.0] if any(w in t.lower() for w in ("consistent", "agree", "total cost", "fault"))
                         else [0.0, 1.0] for t in texts], dtype="float32")


def test_router_uses_classifier_when_no_rule_fires():
    r = Router(embedder=FakeEmbedder())
    assert r.route("What is the VIN?").route == "quick"
    assert r.route("What is the VIN?").method == "classifier"


class Res:
    def __init__(self, status):
        self.status = status

    def to_dict(self):
        return {"status": self.status}


class FixedRouter:
    def __init__(self, route):
        self.r = route

    def route(self, q):
        return self.r


def test_quick_answer_kept_when_verified():
    out = route_and_answer("q", "C", FixedRouter(Route("quick", "rule", "x")),
                           lambda q, c: Res("answered"), lambda q, c: Res("deep"))
    assert out["mode"] == "quick" and not out["escalated"]


@pytest.mark.parametrize("status", ["not_in_packet", "needs_review"])
def test_unverified_quick_answer_escalates_to_deep(status):
    calls = []
    out = route_and_answer("q", "C", FixedRouter(Route("quick", "classifier", "x")),
                           lambda q, c: Res(status), lambda q, c: calls.append(1) or Res("answered"))
    assert out["mode"] == "deep" and out["escalated"] and calls == [1]


def test_deep_route_skips_quick():
    out = route_and_answer("q", "C", FixedRouter(Route("deep", "rule", "x")),
                           lambda q, c: pytest.fail("quick should not run"), lambda q, c: Res("answered"))
    assert out["mode"] == "deep" and not out["escalated"]
