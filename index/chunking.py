"""Cut page text into small chunks, each carrying where it came from.

Chunks follow LINES, not fixed character counts, so a label and its value
("Estimate total  $2,893.30") stay together. Each chunk keeps claim_id, page and
page_type so search can filter by claim and the answer can cite the page.
A one-line overlap between neighbouring chunks keeps facts that sit on a boundary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 400
OVERLAP_LINES = 1


@dataclass
class Chunk:
    chunk_id: str      # "<claim_id>/p<page>/c<n>"
    claim_id: str
    page: int
    page_type: str
    method: str        # text / ocr / image (how the page text was read)
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_page(claim_id: str, page: int, page_type: str, method: str, text: str,
               max_chars: int = MAX_CHARS, overlap: int = OVERLAP_LINES) -> list[Chunk]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks, current = [], []
    for line in lines:
        if current and sum(len(l) + 1 for l in current) + len(line) > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
        current.append(line)
    if current:
        chunks.append(current)
    # Every chunk starts with the page type so the meaning survives on its own
    # ("repair_estimate: Estimate total $2,893.30").
    return [Chunk(f"{claim_id}/p{page:02d}/c{i}", claim_id, page, page_type, method,
                  f"[{page_type}] " + " ".join(c)) for i, c in enumerate(chunks)]


def chunk_claim(ingested_file: Path) -> list[Chunk]:
    out = []
    for row in map(json.loads, ingested_file.read_text().splitlines()):
        if not row["text"].strip():
            continue
        out += chunk_page(row["claim_id"], row["page"], row.get("page_type", "unknown"),
                          row["method"], row["text"])
    return out
