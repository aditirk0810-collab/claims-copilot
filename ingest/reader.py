"""Read every page of a claim packet into text.

Cheapest method that works, per page:
  - "text":  the PDF has a text layer -> copy it (instant, exact)
  - "ocr":   the page is one big image with no text (a scan) -> OCR it
  - "image": a photo with a short caption -> keep the caption, don't OCR the photo
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pymupdf

MIN_TEXT_WORDS = 20      # fewer words than this = no real text layer
SCAN_COVERAGE = 0.80     # an image covering 80%+ of the page = a scan


class OCREngine(Protocol):
    name: str

    def read(self, page: pymupdf.Page) -> tuple[str, float]:
        """Return (text, mean confidence 0-100)."""


@dataclass
class PageText:
    page: int
    method: str              # "text" | "ocr" | "image"
    text: str
    confidence: float | None  # OCR confidence 0-100, None when text was copied
    word_count: int
    image_coverage: float    # share of the page covered by images

    def to_dict(self) -> dict:
        return asdict(self)


def image_coverage(page: pymupdf.Page) -> float:
    """Share of the page area covered by images (0.0 to 1.0)."""
    page_area = page.rect.width * page.rect.height
    covered = 0.0
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            covered += (rect & page.rect).get_area()
    return min(1.0, covered / page_area)


def classify_method(word_count: int, coverage: float) -> str:
    if word_count >= MIN_TEXT_WORDS:
        return "text"
    if coverage >= SCAN_COVERAGE:
        return "ocr"
    return "image"


def read_page(page: pymupdf.Page, ocr: OCREngine | None) -> PageText:
    words = page.get_text("words")
    coverage = image_coverage(page)
    method = classify_method(len(words), coverage)

    if method == "ocr":
        if ocr is None:
            raise RuntimeError(f"page {page.number + 1} is a scan but no OCR engine was given")
        text, conf = ocr.read(page)
        return PageText(page.number + 1, "ocr", text, round(conf, 1), len(text.split()), round(coverage, 2))

    return PageText(page.number + 1, method, page.get_text(), None, len(words), round(coverage, 2))


def read_packet(pdf_path: str | Path, ocr: OCREngine | None = None) -> list[PageText]:
    with pymupdf.open(pdf_path) as doc:
        return [read_page(page, ocr) for page in doc]
