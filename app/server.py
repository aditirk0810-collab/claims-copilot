"""Claims Intake Copilot demo: a small web app over the whole pipeline.

    .venv/bin/python -m uvicorn app.server:app --port 8765

  GET  /                                  the page (app/static/index.html)
  GET  /api/claims                        claims with page counts
  GET  /api/claims/{claim_id}             pages: type, confidence, how the text was read
  GET  /api/claims/{claim_id}/page/{n}    page image (PNG); ?hl=<quote> highlights the quote
  POST /api/ask {claim_id, question}      answer + citation + verification checks + sources
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
PACKETS = ROOT / "data" / "synthetic"
INGESTED = ROOT / "data" / "ingested"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Claims Intake Copilot")
# "langchain" (default): LCEL chain + LangGraph agent from qa/lc.py. "native": plain Gemini SDK version.
IMPL = __import__("os").environ.get("COPILOT_IMPL", "langchain")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def engine():
    """Load search + LLM once (embedding model, indexes, Gemini client)."""
    from index.search import Searcher
    from qa.llm import GeminiLLM

    return Searcher(), GeminiLLM()


@lru_cache(maxsize=1)
def lc_chat():
    from qa.lc import gemini_chat
    return gemini_chat()


def _pages(claim_id: str) -> list[dict]:
    f = INGESTED / f"{claim_id}.jsonl"
    if not f.exists():
        raise HTTPException(404, f"unknown claim {claim_id}")
    return [json.loads(l) for l in f.read_text().splitlines()]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/claims")
def claims() -> list[dict]:
    out = []
    for f in sorted(INGESTED.glob("CLM-*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        out.append({"claim_id": rows[0]["claim_id"], "pages": len(rows), "split": rows[0].get("split"),
                    "scanned": sum(r["method"] == "ocr" for r in rows)})
    return out


@app.get("/api/claims/{claim_id}")
def claim(claim_id: str) -> dict:
    pages = [{k: r.get(k) for k in ("page", "method", "page_type", "page_type_conf", "needs_review", "confidence")}
             for r in _pages(claim_id)]
    return {"claim_id": claim_id, "pages": pages}


@app.on_event("startup")
def warm_up() -> None:
    """Load the embedding model and indexes in the background so the first question is fast."""
    import threading

    threading.Thread(target=engine, daemon=True).start()


def _tok(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in "$.,/-")


def _highlight_scan(img, quote: str):
    """Scanned pages have no text layer, so find the quote with OCR word boxes and draw over it."""
    import pytesseract
    from PIL import Image, ImageDraw

    want = [t for t in (_tok(w) for w in quote.split()) if t]
    if not want:
        return img
    data = pytesseract.image_to_data(img.convert("L"), config="--psm 6", output_type=pytesseract.Output.DICT)
    words = [(i, _tok(w)) for i, w in enumerate(data["text"]) if _tok(w)]
    toks = [t for _, t in words]
    for start in range(len(toks) - len(want) + 1):
        if toks[start:start + len(want)] == want:
            idx = [words[start + j][0] for j in range(len(want))]
            overlay = img.convert("RGBA")
            layer = overlay.copy()
            draw = ImageDraw.Draw(layer)
            for i in idx:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill=(255, 210, 60, 140))
            return Image.alpha_composite(overlay, layer).convert("RGB")
    return img


@app.get("/api/claims/{claim_id}/page/{n}")
def page_image(claim_id: str, n: int, hl: str = "", dpi: int = 80) -> Response:
    import io

    from PIL import Image

    pdf = PACKETS / claim_id / "packet.pdf"
    if not pdf.exists():
        raise HTTPException(404, "packet not found")
    dpi = min(max(dpi, 30), 150)
    with pymupdf.open(pdf) as doc:
        if not 1 <= n <= doc.page_count:
            raise HTTPException(404, "no such page")
        page = doc[n - 1]
        scanned = not page.get_text().strip()
        if hl and not scanned:
            for rect in page.search_for(hl.strip()):   # text layer: exact positions
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.82, 0.25))
                annot.update()
        if hl and scanned:
            big = Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png")))
            img = _highlight_scan(big, hl)
            if dpi != 150:
                img = img.resize((int(img.width * dpi / 150), int(img.height * dpi / 150)))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            png = buf.getvalue()
        else:
            png = page.get_pixmap(dpi=dpi).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


class Ask(BaseModel):
    claim_id: str
    question: str


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    from qa.answer import answer_question

    q = body.question.strip()
    if not q:
        raise HTTPException(400, "question is empty")
    _pages(body.claim_id)
    searcher, llm = engine()
    if IMPL == "langchain":
        from qa.lc import Answer, quick_answer
        r = quick_answer(q, body.claim_id, searcher, lc_chat().with_structured_output(Answer))
    else:
        r = answer_question(q, body.claim_id, searcher, llm)
    sources = [{"page": c.page, "page_type": c.page_type, "method": c.method, "text": c.text}
               for c, _ in searcher.search(q, body.claim_id, mode="hybrid", k=5)]
    return {**r.to_dict(), "sources": sources}


@app.post("/api/deep")
def deep(body: Ask) -> dict:
    """Multi-document agent: plans, searches several document types, calculates, cites each fact."""
    from qa.agent import ClaimAgent

    q = body.question.strip()
    if not q:
        raise HTTPException(400, "question is empty")
    types = {r["page"]: r.get("page_type") for r in _pages(body.claim_id)}
    searcher, llm = engine()
    if IMPL == "langchain":
        from qa.lc import deep_answer
        r = deep_answer(q, body.claim_id, searcher, lc_chat())
    else:
        r = ClaimAgent(searcher, llm).answer(q, body.claim_id)
    out = r.to_dict()
    for c in out["citations"]:
        c["page_type"] = types.get(c.get("page"))
    return out
