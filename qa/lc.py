"""LangChain / LangGraph version of the copilot.

    python -m qa.lc --claim CLM-2026-00008 --q "What is the repair estimate total?"
    python -m qa.lc --claim CLM-2026-00005 --q "Do the loss notice and police report agree on the date?" --deep

Same pipeline, built from LangChain parts:

  ClaimRetriever      a LangChain BaseRetriever over our hybrid search (Chroma + BM25), filtered
                      to one claim and optionally to document types; returns Documents with page metadata
  quick_chain         LCEL chain: retriever -> prompt -> Gemini with structured output (Answer schema)
  deep agent          langchain.agents.create_agent (runs on LangGraph): LangChain @tools for search,
                      read_page, calculate, days_between; structured final answer (FinalAnswer schema)

Verification is the same code as the native version: quotes and numbers are checked against
the real page text, so LangChain changes the plumbing, not the safety rules.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from .agent import MAX_STEPS, SYSTEM as AGENT_SYSTEM, AgentResult, ClaimAgent, _short, calculate as calc, days_between as days
from .answer import SYSTEM as QUICK_SYSTEM, AnswerResult, verify

ROOT = Path(__file__).resolve().parent.parent
PageType = Literal["claim_form", "police_report", "repair_estimate", "medical_bill", "correspondence", "photo", "other"]


# ---------- retriever ----------

class ClaimRetriever(BaseRetriever):
    """LangChain retriever over our hybrid search, restricted to one claim."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    searcher: Any
    claim_id: str
    k: int = 5
    page_types: list[str] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hits = self.searcher.search(query, self.claim_id, mode="hybrid", k=self.k, page_types=self.page_types)
        return [Document(page_content=c.text, metadata={"claim_id": c.claim_id, "page": c.page,
                                                        "page_type": c.page_type, "method": c.method,
                                                        "chunk_id": c.chunk_id, "score": s})
                for c, s in hits]


# ---------- quick answer: LCEL chain ----------

class Answer(BaseModel):
    """Answer one question from the numbered sources."""
    answer: str = Field(description="only the value, copied exactly as printed")
    quote: str = Field(description="short exact excerpt from one source containing the answer")
    source_pages: list[int] = Field(description="page numbers of the sources used")
    not_in_packet: bool = Field(description="true if the sources do not contain the answer")


QUICK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUICK_SYSTEM),
    ("human", "Sources:\n{sources}\n\nQuestion: {question}"),
])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[S{i}] page {d.metadata['page']} ({d.metadata['page_type']}):\n{d.page_content}"
                       for i, d in enumerate(docs, 1))


def build_quick_chain(retriever: BaseRetriever, structured_llm: Runnable) -> Runnable:
    """question -> {docs, question, answer}. LCEL: retrieve in parallel with passing the question through,
    then prompt -> LLM with structured output. Docs are kept so code can verify the answer."""
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(
            answer=(lambda x: {"sources": format_docs(x["docs"]), "question": x["question"]})
            | QUICK_PROMPT | structured_llm)
    )


class _Chunk:
    """Adapter so LangChain Documents can go through the same verify() as the native version."""
    def __init__(self, d: Document):
        self.page, self.text = d.metadata["page"], d.page_content


def quick_answer(question: str, claim_id: str, searcher, structured_llm: Runnable) -> AnswerResult:
    t0 = time.time()
    out = build_quick_chain(ClaimRetriever(searcher=searcher, claim_id=claim_id), structured_llm).invoke(question)
    ans: Answer = out["answer"]
    docs = out["docs"]
    retrieved = sorted({d.metadata["page"] for d in docs})
    info = {"model": "langchain + gemini", "impl": "langchain", "latency_ms": round((time.time() - t0) * 1000),
            "cached": False}
    if ans.not_in_packet:
        return AnswerResult(claim_id, question, "not_in_packet", "", "", [], retrieved, {}, info)
    checks = verify(ans.model_dump(), [(_Chunk(d), 0) for d in docs])
    ok = checks["cited_pages_retrieved"] and checks["answer_on_cited_page"]
    return AnswerResult(claim_id, question, "answered" if ok else "needs_review", ans.answer, ans.quote,
                        sorted(set(ans.source_pages)), retrieved, checks, info)


# ---------- deep agent: LangGraph via create_agent ----------

class Citation(BaseModel):
    page: int
    quote: str = Field(description="exact contiguous line copied from a tool result")


class FinalAnswer(BaseModel):
    """Final answer with one citation per fact used. Ends the task."""
    answer: str
    explanation: str = Field(description="2-4 sentences: which documents, values and calculation")
    citations: list[Citation]
    not_in_packet: bool = False


def make_tools(searcher, claim_id: str, seen: set[int], computed: set[str]):
    """LangChain tools bound to one claim. They record seen pages / computed numbers for verification."""

    @tool
    def search(query: str, page_type: PageType | None = None) -> dict:
        """Search this claim's pages; optionally only one document type. Returns chunks with page numbers."""
        hits = searcher.search(query, claim_id, mode="hybrid", k=4, page_types=[page_type] if page_type else None)
        seen.update(c.page for c, _ in hits)
        return {"results": [{"page": c.page, "page_type": c.page_type, "text": c.text.split("] ", 1)[-1]}
                            for c, _ in hits]}

    @tool
    def read_page(page: int) -> dict:
        """Read the full text of one page of this claim."""
        got = searcher.page_text(claim_id, page)
        if not got:
            return {"error": "no such page"}
        seen.add(page)
        return {"page": page, "page_type": got[0], "text": got[1][:3000]}

    @tool
    def calculate(expression: str) -> dict:
        """Evaluate arithmetic like '2346.78 + 1037.01'. Numbers and + - * / ( ) only."""
        try:
            v = calc(expression)
        except Exception as e:
            return {"error": str(e)}
        computed.update({f"{v:.2f}", str(int(v)) if float(v).is_integer() else f"{v:.2f}"})
        return {"result": v}

    @tool
    def days_between(start_date: str, end_date: str) -> dict:
        """Whole days from start_date to end_date (MM/DD/YYYY); negative if end is before start."""
        try:
            d = days(start_date, end_date)
        except Exception as e:
            return {"error": f"use MM/DD/YYYY ({e})"}
        computed.update({str(d), str(abs(d))})
        return {"days": d}

    return [search, read_page, calculate, days_between]


def deep_answer(question: str, claim_id: str, searcher, chat_model) -> AgentResult:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_core.messages import AIMessage, ToolMessage

    seen: set[int] = set()
    computed: set[str] = set()
    agent = create_agent(
        model=chat_model,
        tools=make_tools(searcher, claim_id, seen, computed),
        system_prompt=AGENT_SYSTEM,
        response_format=ToolStrategy(FinalAnswer),   # the agent ends by "calling" FinalAnswer
    )
    toc = ", ".join(f"p{p}={t}" for p, t in searcher.page_types(claim_id))
    t0 = time.time()
    state = agent.invoke(
        {"messages": [{"role": "user", "content": f"Claim {claim_id}. Pages: {toc}.\n\nQuestion: {question}"}]},
        config={"recursion_limit": 2 * MAX_STEPS + 2},
    )

    # Rebuild the step trace (same shape as the native agent, so the webpage shows it the same way).
    steps, pending = [], {}
    for m in state["messages"]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                pending[tc["id"]] = tc
        elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
            tc = pending.pop(m.tool_call_id)
            if tc["name"] == "FinalAnswer":
                continue
            try:
                result = json.loads(m.content) if isinstance(m.content, str) else m.content
            except json.JSONDecodeError:
                result = {"text": str(m.content)}
            steps.append({"tool": tc["name"], "args": tc["args"],
                          "result": _short(result) if isinstance(result, dict) else result})
    steps.append({"tool": "final_answer", "args": {}})

    info = {"model": getattr(chat_model, "model", "langchain"), "impl": "langgraph", "steps": len(steps),
            "latency_ms": round((time.time() - t0) * 1000), "llm_calls": sum(isinstance(m, AIMessage) for m in state["messages"])}
    final: FinalAnswer | None = state.get("structured_response")
    if final is None:
        return AgentResult(claim_id, question, "needs_review", steps=steps, info=info,
                           explanation="The agent did not produce a final answer.")
    fd = final.model_dump()
    if final.not_in_packet:
        return AgentResult(claim_id, question, "not_in_packet", final.answer, final.explanation, steps=steps, info=info)
    checks = ClaimAgent(searcher, llm=None, model="-").verify(claim_id, fd, seen, computed)
    ok = checks["all_citations_verified"] and checks["numbers_grounded"]
    return AgentResult(claim_id, question, "answered" if ok else "needs_review", final.answer,
                       final.explanation, fd["citations"], checks, steps, info)


# ---------- Gemini via LangChain, with a disk cache ----------

class JsonlCache(BaseCache):
    """Persistent LangChain LLM cache (one JSON line per entry): re-runs cost no API quota."""

    def __init__(self, path: Path = ROOT / "data" / "lc_cache.jsonl"):
        self.path = path
        self.mem: dict[tuple[str, str], str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.mem[(row["p"], row["l"])] = row["v"]

    def lookup(self, prompt: str, llm_string: str):
        v = self.mem.get((prompt, llm_string))
        if not v:
            return None
        import warnings
        with warnings.catch_warnings():          # langchain_core.load is marked beta; our own cache file is trusted
            warnings.simplefilter("ignore")
            return loads(v)

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        v = dumps(return_val)
        self.mem[(prompt, llm_string)] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"p": prompt, "l": llm_string, "v": v}) + "\n")

    def clear(self, **kwargs) -> None:
        self.mem.clear()


def gemini_chat(model: str | None = None):
    from dotenv import load_dotenv
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv(ROOT / ".env")
    return ChatGoogleGenerativeAI(
        model=model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=3,
        rate_limiter=InMemoryRateLimiter(requests_per_second=float(os.environ.get("GEMINI_RPM", "12")) / 60),
        cache=JsonlCache(),
    )


def main() -> None:
    from index.search import Searcher

    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    searcher, chat = Searcher(), gemini_chat(args.model)
    if args.deep:
        r = deep_answer(args.q, args.claim, searcher, chat)
        for i, s in enumerate(r.steps, 1):
            print(f"  step {i}: {s['tool']}({json.dumps(s['args'])[:100]})")
        print(f"\n[{r.status}] {r.answer}\n  {r.explanation}")
        for c in r.citations:
            print(f"  - page {c['page']}: \"{c['quote']}\"")
        print(f"  checks: {r.checks}")
    else:
        r = quick_answer(args.q, args.claim, searcher, chat.with_structured_output(Answer))
        print(f"[{r.status}] {r.answer}  (page {r.source_pages})  quote: \"{r.quote}\"  checks: {r.checks}")


if __name__ == "__main__":
    main()
