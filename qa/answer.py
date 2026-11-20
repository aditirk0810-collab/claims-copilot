"""Answer an adjuster's question about ONE claim, with a page citation, or refuse.

    python -m qa.answer --claim CLM-2026-00008 --q "What is the repair estimate total?"

Flow:
  1. search: top-5 chunks from this claim only (hybrid search, Step 3)
  2. Gemini: answer ONLY from those chunks, as JSON: answer, exact quote, source pages,
     or not_in_packet=true when the chunks don't contain the answer
  3. verify (code, not the model): the cited pages were actually retrieved, and the quote
     and the answer really appear in the cited page text. If not -> status "needs_review",
     so a made-up answer is never shown as fact.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass, field

from index.search import Searcher

SYSTEM = """You are a claims intake assistant helping an insurance adjuster.
Answer the question using ONLY the numbered sources from this one claim packet.
Rules:
- "answer" is ONLY the value itself, copied exactly as printed (e.g. "$271.85", "03/14/2026", "HMI-AU-5120854",
  "Rush Collision Center", "Yes"). No sentence, no explanation, no reformatting, no calculation.
- If several amounts appear, give the final total the question asks for (e.g. "Amount paid", "Estimate total",
  "Total charges"), not a line item.
- "quote" must be an exact, short excerpt from one source that contains the answer.
- "source_pages" lists the page numbers of the sources you used.
- If the sources do not contain the answer, set not_in_packet to true, answer "", quote "", source_pages [].
  Never guess and never use outside knowledge.
- Sources come from OCR and may contain noise; ignore garbled text."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "quote": {"type": "string"},
        "source_pages": {"type": "array", "items": {"type": "integer"}},
        "not_in_packet": {"type": "boolean"},
    },
    "required": ["answer", "quote", "source_pages", "not_in_packet"],
}


@dataclass
class AnswerResult:
    claim_id: str
    question: str
    status: str                  # "answered" | "not_in_packet" | "needs_review"
    answer: str
    quote: str
    source_pages: list[int]
    retrieved_pages: list[int]
    checks: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def build_prompt(question: str, chunks) -> str:
    sources = "\n\n".join(f"[S{i}] page {c.page} ({c.page_type}):\n{c.text}" for i, (c, _) in enumerate(chunks, 1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def verify(out: dict, chunks) -> dict:
    """Code-side checks. The model's claim is trusted only if the text backs it up."""
    page_text: dict[int, str] = {}
    for c, _ in chunks:
        page_text[c.page] = page_text.get(c.page, "") + " " + c.text
    cited = [p for p in out.get("source_pages", []) if isinstance(p, int)]
    cited_text = norm(" ".join(page_text.get(p, "") for p in cited))
    answer, quote = norm(out.get("answer", "")), norm(out.get("quote", ""))
    quote_ok = bool(quote) and quote in cited_text
    return {
        "cited_pages_retrieved": bool(cited) and all(p in page_text for p in cited),
        "quote_on_cited_page": quote_ok,
        # The answer must be on the cited page itself, or inside a quote that is on it.
        # (Trusting the quote alone would let a made-up quote "prove" a made-up answer.)
        "answer_on_cited_page": bool(answer) and (answer in cited_text or (quote_ok and answer in quote)),
    }


def answer_question(question: str, claim_id: str, searcher: Searcher, llm, k: int = 5) -> AnswerResult:
    chunks = searcher.search(question, claim_id, mode="hybrid", k=k)
    retrieved = sorted({c.page for c, _ in chunks})
    out, info = llm.generate_json(SYSTEM, build_prompt(question, chunks), SCHEMA)

    if out.get("not_in_packet"):
        return AnswerResult(claim_id, question, "not_in_packet", "", "", [], retrieved, {}, info)

    checks = verify(out, chunks)
    # Answer must be grounded: cited pages were retrieved AND the answer text is on them.
    ok = checks["cited_pages_retrieved"] and checks["answer_on_cited_page"]
    return AnswerResult(claim_id, question, "answered" if ok else "needs_review",
                        out.get("answer", ""), out.get("quote", ""),
                        sorted(set(out.get("source_pages", []))), retrieved, checks, info)


def main() -> None:
    from .llm import GeminiLLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--q", required=True)
    args = ap.parse_args()
    r = answer_question(args.q, args.claim, Searcher(), GeminiLLM())
    if r.status == "answered":
        print(f"{r.answer}   (page {', '.join(map(str, r.source_pages))})\n  quote: \"{r.quote}\"")
    elif r.status == "not_in_packet":
        print("Not in this claim packet.")
    else:
        print(f"Needs human review. Model said: {r.answer!r} citing pages {r.source_pages}; checks: {r.checks}")
    print(f"  [{r.info.get('model')}, {r.info.get('latency_ms')} ms, cached={r.info.get('cached')}]")


if __name__ == "__main__":
    main()
