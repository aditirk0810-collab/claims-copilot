"""Deep questions: a tool-using agent that works across several documents.

    python -m qa.agent --claim CLM-2026-00008 --q "What is the total known cost of this claim?"

Why an agent: "total cost = cheaper estimate + medical + towing" or "was treatment before
the accident?" need several documents, a comparison or arithmetic. One search + one answer
cannot do that reliably. The agent loops (max 10 steps):

    think -> call a tool -> read the result -> ... -> final_answer

Tools
  search(query, page_type?)      hybrid search inside this claim, optionally one document type
  read_page(page)                full text of a page (tables, every line of an estimate)
  calculate(expression)          arithmetic done by code, never by the model's head
  days_between(start, end)       date math done by code
  final_answer(...)              answer + explanation + one citation per fact used

Then code verifies (not the model):
  - every cited page was actually opened by the agent (search result or read_page)
  - every quote really appears on its cited page
  - every number in the answer appears in a verified quote or came out of calculate/days_between
Anything that fails is shown as "needs review".
"""

from __future__ import annotations

import ast
import argparse
import json
import operator
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from index.search import Searcher

MAX_STEPS = 10
PAGE_TYPES = ["claim_form", "police_report", "repair_estimate", "medical_bill", "correspondence", "photo", "other"]

SYSTEM = """You are a senior auto-claims analyst assisting an insurance adjuster with ONE claim packet.
You can only see the packet through your tools. Work step by step:
1. Plan which documents the question needs (loss notice = claim_form, police_report, repair_estimate,
   medical_bill, correspondence = letters/emails, other = towing invoice). The table of contents is given.
2. Gather each fact with search (use page_type to target a document) or read_page for full pages.
   For questions about "every" estimate or consistency, open EVERY relevant page.
3. Never do arithmetic or date math yourself: use calculate and days_between.
4. Compare names, dates and vehicles exactly as printed.
5. Call final_answer. Cite every page you relied on, each with a short quote copied exactly from a tool result.

For an inconsistency review, check ALL of these, opening every relevant page:
  a. date of loss on the loss notice vs. the date on the police report (if there is one)
  b. "Injuries reported" on the loss notice vs. whether a medical bill is in the packet
  c. vehicle on the loss notice vs. the vehicle on EVERY repair estimate
  d. medical date of service vs. date of loss (treatment before the accident is a red flag)
  Only factual conflicts count; different wording or extra detail in letters is not an inconsistency.

Answer formats (the "answer" field is short; details go in "explanation"):
- money: "$1,234.56"   - day counts: a whole number like "3"   - yes/no questions: start with "Yes" or "No"
- inconsistency reviews: list each inconsistency in a few words, or "None found".
Quotes: copy one exact contiguous line per citation; to cite two facts on the same page, add two citations.
If the packet does not contain what is needed, set not_in_packet true. Never guess. Ignore OCR noise."""

TOOLS = [
    {"name": "search", "description": "Search this claim's pages. Returns the best matching text chunks with page numbers.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "what to look for"},
         "page_type": {"type": "string", "enum": PAGE_TYPES, "description": "optional: only this document type"}},
         "required": ["query"]}},
    {"name": "read_page", "description": "Read the full text of one page of this claim.",
     "parameters": {"type": "object", "properties": {"page": {"type": "integer"}}, "required": ["page"]}},
    {"name": "calculate", "description": "Evaluate arithmetic, e.g. '2346.78 + 1037.01 + 271.85'. Numbers only, + - * / ( ).",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "days_between", "description": "Whole days from start_date to end_date (negative if end is before start). Dates MM/DD/YYYY.",
     "parameters": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}},
                    "required": ["start_date", "end_date"]}},
    {"name": "final_answer", "description": "Give the final answer with citations. Ends the task.",
     "parameters": {"type": "object", "properties": {
         "answer": {"type": "string"},
         "explanation": {"type": "string", "description": "2-4 sentences: which documents, which values, which calculation"},
         "citations": {"type": "array", "items": {"type": "object", "properties": {
             "page": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["page", "quote"]}},
         "not_in_packet": {"type": "boolean"}},
         "required": ["answer", "explanation", "citations", "not_in_packet"]}},
]


# ---------- tools ----------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> float:
    """Safe arithmetic: parses the expression and allows only numbers and + - * / ( )."""
    expr = expression.replace("$", "").replace(",", "")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ( ) are allowed")

    return round(ev(ast.parse(expr, mode="eval")), 2)


def days_between(start_date: str, end_date: str) -> int:
    f = "%m/%d/%Y"
    return (datetime.strptime(end_date.strip(), f) - datetime.strptime(start_date.strip(), f)).days


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


NUM = re.compile(r"\d{2}/\d{2}/\d{4}|\$?\d[\d,]*(?:\.\d+)?")   # dates first, so 08/21/2026 stays whole


def numbers_in(text: str) -> set[str]:
    return {n.replace("$", "").replace(",", "") for n in NUM.findall(text)}


# ---------- result ----------

@dataclass
class AgentResult:
    claim_id: str
    question: str
    status: str                      # answered | not_in_packet | needs_review
    answer: str = ""
    explanation: str = ""
    citations: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)   # trace shown in the UI
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimAgent:
    def __init__(self, searcher: Searcher, llm, model: str | None = None):
        import os

        self.searcher = searcher
        self.llm = llm
        self.model = model or os.environ.get("GEMINI_AGENT_MODEL", "gemini-3.5-flash-lite")

    def _run_tool(self, claim_id: str, name: str, args: dict, seen: set[int], computed: set[str]):
        if name == "search":
            pt = args.get("page_type")
            hits = self.searcher.search(args.get("query", ""), claim_id, mode="hybrid", k=4,
                                        page_types=[pt] if pt else None)
            seen.update(c.page for c, _ in hits)
            return {"results": [{"page": c.page, "page_type": c.page_type,
                                 "text": c.text.split("] ", 1)[-1]} for c, _ in hits]}
        if name == "read_page":
            got = self.searcher.page_text(claim_id, int(args.get("page", 0)))
            if not got:
                return {"error": "no such page"}
            seen.add(int(args["page"]))
            return {"page": int(args["page"]), "page_type": got[0], "text": got[1][:3000]}
        if name == "calculate":
            try:
                v = calculate(str(args.get("expression", "")))
            except Exception as e:
                return {"error": str(e)}
            computed.add(f"{v:.2f}")
            computed.add(str(int(v)) if float(v).is_integer() else f"{v:.2f}")
            return {"result": v}
        if name == "days_between":
            try:
                d = days_between(str(args.get("start_date")), str(args.get("end_date")))
            except Exception as e:
                return {"error": f"use MM/DD/YYYY ({e})"}
            computed.add(str(d))
            computed.add(str(abs(d)))
            return {"days": d}
        return {"error": f"unknown tool {name}"}

    def verify(self, claim_id: str, final: dict, seen: set[int], computed: set[str]) -> dict:
        cites = [c for c in final.get("citations", []) if isinstance(c, dict)]
        cite_ok, backed = [], set()
        for c in cites:
            page, quote = c.get("page"), str(c.get("quote", ""))
            got = self.searcher.page_text(claim_id, page) if isinstance(page, int) else None
            # A quote may join excerpts with "..." -> every excerpt must be on the page, word for word.
            parts = [norm(p) for p in re.split(r"\s*(?:\.\.\.|…)\s*", quote) if len(p.strip()) >= 3]
            ok = bool(got) and page in seen and bool(parts) and all(p in norm(got[1]) for p in parts)
            cite_ok.append(ok)
            if ok:
                # Numbers printed on a verified cited page (and its page number) back the answer.
                backed |= numbers_in(got[1]) | {str(page)}
        answer_nums = numbers_in(final.get("answer", ""))
        unbacked = sorted(n for n in answer_nums if n not in backed and n not in computed)
        return {
            "has_citations": bool(cites),
            "all_citations_verified": bool(cites) and all(cite_ok),
            "numbers_grounded": not unbacked,
            "documents_cited": len({c.get("page") for c in cites}),
            "unbacked_numbers": unbacked,
        }

    def answer(self, question: str, claim_id: str) -> AgentResult:
        toc = ", ".join(f"p{p}={t}" for p, t in self.searcher.page_types(claim_id))
        history = [{"role": "user", "parts": [{"text": f"Claim {claim_id}. Pages: {toc}.\n\nQuestion: {question}"}]}]
        seen: set[int] = set()
        computed: set[str] = set()
        steps, latency, calls_cached = [], 0, 0
        final = None
        for step in range(MAX_STEPS):
            force = ["final_answer"] if step == MAX_STEPS - 1 else None
            content, info = self.llm.agent_turn(self.model, SYSTEM, history, TOOLS, force)
            latency += info.get("latency_ms") or 0
            calls_cached += bool(info.get("cached"))
            history.append(content)
            calls = [p["function_call"] for p in content.get("parts", []) if "function_call" in p]
            if not calls:
                history.append({"role": "user", "parts": [{"text": "Use a tool; finish with final_answer."}]})
                continue
            responses = []
            for call in calls:
                name, args = call.get("name"), call.get("args") or {}
                if name == "final_answer":
                    final = args
                    steps.append({"tool": name, "args": {k: v for k, v in args.items() if k != "citations"}})
                    break
                result = self._run_tool(claim_id, name, args, seen, computed)
                steps.append({"tool": name, "args": args, "result": _short(result)})
                responses.append({"function_response": {"name": name, "response": result,
                                                        **({"id": call["id"]} if call.get("id") else {})}})
            if final is not None:
                break
            history.append({"role": "user", "parts": responses})

        info = {"model": self.model, "steps": len(steps), "latency_ms": latency,
                "llm_calls": len([h for h in history if h.get("role") == "model"]), "cached_calls": calls_cached}
        if final is None:
            return AgentResult(claim_id, question, "needs_review", steps=steps, info=info,
                               explanation="The agent ran out of steps.")
        if final.get("not_in_packet"):
            return AgentResult(claim_id, question, "not_in_packet", final.get("answer", ""),
                               final.get("explanation", ""), steps=steps, info=info)
        checks = self.verify(claim_id, final, seen, computed)
        ok = checks["all_citations_verified"] and checks["numbers_grounded"]
        return AgentResult(claim_id, question, "answered" if ok else "needs_review", final.get("answer", ""),
                           final.get("explanation", ""), final.get("citations", []), checks, steps, info)


def _short(result: dict) -> dict:
    """Trim tool results for the trace shown to people (the model sees the full result)."""
    if "results" in result:
        return {"results": [{"page": r["page"], "page_type": r["page_type"], "text": r["text"][:140]}
                            for r in result["results"]]}
    if "text" in result:
        return {**result, "text": result["text"][:200]}
    return result


def main() -> None:
    from .llm import GeminiLLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    r = ClaimAgent(Searcher(), GeminiLLM(), args.model).answer(args.q, args.claim)
    for i, s in enumerate(r.steps, 1):
        print(f"  step {i}: {s['tool']}({json.dumps(s['args'])[:110]})")
    print(f"\n[{r.status}] {r.answer}\n  {r.explanation}")
    for c in r.citations:
        print(f"  - page {c.get('page')}: \"{c.get('quote')}\"")
    print(f"  checks: {r.checks}\n  {r.info}")


if __name__ == "__main__":
    main()
