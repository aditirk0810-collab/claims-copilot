"""OCR engines. Tesseract is the free baseline; cloud engines can be added with the same `read()` shape."""

from __future__ import annotations

import io

import pymupdf
from PIL import Image


class TesseractOCR:
    """Tesseract 4+ (CNN + bidirectional LSTM + CTC), run locally.

    Needs the tesseract program installed (macOS: `brew install tesseract`) and `pip install pytesseract`.
    """

    name = "tesseract"

    def __init__(self, dpi: int = 300, lang: str = "eng", psm: int = 6):
        import pytesseract  # imported here so the rest of the project works without it

        self.pt = pytesseract
        self.dpi = dpi          # 300 dpi is the usual sweet spot for OCR accuracy
        self.lang = lang
        self.config = f"--psm {psm}"  # psm 6 = treat the page as one block of text lines

    def _image(self, page: pymupdf.Page) -> Image.Image:
        pix = page.get_pixmap(dpi=self.dpi, colorspace=pymupdf.csGRAY)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def read(self, page: pymupdf.Page) -> tuple[str, float]:
        img = self._image(page)
        data = self.pt.image_to_data(img, lang=self.lang, config=self.config,
                                     output_type=self.pt.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            conf = float(data["conf"][i])
            if conf >= 0:
                confs.append(conf)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        return text, (sum(confs) / len(confs) if confs else 0.0)


class GroundTruthOCR:
    """Test double: returns the known correct text for each page (perfect OCR)."""

    name = "ground_truth"

    def __init__(self, page_texts: dict[int, str]):
        self.page_texts = page_texts

    def read(self, page: pymupdf.Page) -> tuple[str, float]:
        return self.page_texts[page.number + 1], 100.0
