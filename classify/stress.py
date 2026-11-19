"""Stress test: damage the TEST pages and see which sorter breaks first.

    python -m classify.stress        (run text_baseline and cnn first; uses their saved models)

Both sorters score 100% on clean synthetic pages because the templates are uniform.
That shows the code works, not that the models are robust. So we degrade the inputs:
  - text:  keep only a share of the words (very bad OCR, torn or cropped pages)
  - image: blur and shrink the page (low-quality phone photo or fax)
"""

from __future__ import annotations

import random

import joblib
import numpy as np
import torch
from PIL import Image, ImageFilter

from .cnn import TEST_TF, build_model, device, page_image
from .data import LABELS, ROOT, load_pages, split_by_claim


def drop_words(text: str, keep: float, rng: random.Random) -> str:
    words = text.split()
    return " ".join(w for w in words if rng.random() < keep)


def degrade(img: Image.Image, blur: float, shrink: int) -> Image.Image:
    w, h = img.size
    small = img.resize((max(8, w // shrink), max(8, h // shrink)))
    return small.resize((w, h)).filter(ImageFilter.GaussianBlur(blur))


def main() -> None:
    _, test = split_by_claim(load_pages(), seed=0)
    y = np.array([r.label for r in test])

    text_model = joblib.load(ROOT / "models" / "text_tfidf_logreg.joblib")
    print("TEXT sorter: keep only some words")
    print(f"{'words kept':>12}{'accuracy':>10}{'to human (<80%)':>17}")
    for keep in [1.0, 0.5, 0.2, 0.1, 0.05]:
        rng = random.Random(0)
        proba = text_model.predict_proba([drop_words(r.text, keep, rng) for r in test])
        pred = text_model.classes_[proba.argmax(1)]
        print(f"{keep:>12.0%}{(pred == y).mean():>10.1%}{(proba.max(1) < 0.8).mean():>17.1%}")

    dev = device()
    cnn = build_model(pretrained=False)
    cnn.load_state_dict(torch.load(ROOT / "models" / "image_resnet18.pt", map_location="cpu"))
    cnn.to(dev).eval()
    imgs = [Image.open(page_image(r)) for r in test]
    print("\nIMAGE sorter: blur + shrink the page")
    print(f"{'damage':>12}{'accuracy':>10}{'to human (<80%)':>17}")
    for blur, shrink, name in [(0, 1, "none"), (1, 2, "light"), (2, 4, "medium"), (3, 6, "heavy"), (4, 10, "extreme")]:
        x = torch.stack([TEST_TF(degrade(im, blur, shrink)) for im in imgs]).to(dev)
        with torch.no_grad():
            proba = torch.softmax(cnn(x), 1).cpu().numpy()
        pred = np.array([LABELS[i] for i in proba.argmax(1)])
        print(f"{name:>12}{(pred == y).mean():>10.1%}{(proba.max(1) < 0.8).mean():>17.1%}")


if __name__ == "__main__":
    main()
