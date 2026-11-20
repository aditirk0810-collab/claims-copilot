"""LangChain / LangGraph versions: retriever, LCEL quick chain, and create_agent deep agent (fake models)."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from qa.lc import Answer, ClaimRetriever, deep_answer, quick_answer
from tests.test_agent import FakeSearcher


def test_retriever_returns_documents_with_page_metadata():
    docs = ClaimRetriever(searcher=FakeSearcher(), claim_id="CLM-1", page_types=["medical_bill"]).invoke("charges")
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 5 and docs[0].metadata["page_type"] == "medical_bill"
    assert "Total charges $1,037.01" in docs[0].page_content


def test_quick_chain_verified_and_made_up_answers():
    good = RunnableLambda(lambda _: Answer(answer="$271.85", quote="Amount paid $271.85", source_pages=[1], not_in_packet=False))
    r = quick_answer("towing?", "CLM-1", FakeSearcher(), good)
    assert r.status == "answered" and r.source_pages == [1]

    bad = RunnableLambda(lambda _: Answer(answer="$999.00", quote="Amount paid $999.00", source_pages=[1], not_in_packet=False))
    assert quick_answer("towing?", "CLM-1", FakeSearcher(), bad).status == "needs_review"

    refuse = RunnableLambda(lambda _: Answer(answer="", quote="", source_pages=[], not_in_packet=True))
    assert quick_answer("police report?", "CLM-1", FakeSearcher(), refuse).status == "not_in_packet"


def test_quick_chain_prompt_contains_numbered_sources():
    seen = {}

    def capture(prompt_value):
        seen["text"] = prompt_value.to_string()
        return Answer(answer="", quote="", source_pages=[], not_in_packet=True)

    quick_answer("q?", "CLM-1", FakeSearcher(), RunnableLambda(capture))
    assert "[S1] page" in seen["text"] and "Question: q?" in seen["text"]


class ScriptedChat(BaseChatModel):
    """Fake chat model that returns pre-written AIMessages (with tool calls), one per turn."""

    script: list[AIMessage]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[self.i]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}])


CITES = [{"page": 10, "quote": "Estimate total $2,346.78"}, {"page": 5, "quote": "Total charges $1,037.01"},
         {"page": 1, "quote": "Amount paid $271.85"}]


def test_langgraph_agent_runs_tools_and_verifies():
    model = ScriptedChat(script=[
        call("search", {"query": "totals"}, "c1"),
        call("calculate", {"expression": "2346.78 + 1037.01 + 271.85"}, "c2"),
        call("FinalAnswer", {"answer": "$3,655.64", "explanation": "sum of three documents",
                             "citations": CITES, "not_in_packet": False}, "c3"),
    ])
    r = deep_answer("total cost?", "CLM-1", FakeSearcher(), model)
    assert r.status == "answered", r.checks
    assert [s["tool"] for s in r.steps] == ["search", "calculate", "final_answer"]
    assert r.checks["documents_cited"] == 3 and r.info["impl"] == "langgraph"


def test_langgraph_agent_flags_mental_math():
    model = ScriptedChat(script=[
        call("search", {"query": "totals"}, "c1"),
        call("FinalAnswer", {"answer": "$3,655.64", "explanation": "x", "citations": CITES, "not_in_packet": False}, "c2"),
    ])
    r = deep_answer("total cost?", "CLM-1", FakeSearcher(), model)
    assert r.status == "needs_review" and r.checks["unbacked_numbers"] == ["3655.64"]
