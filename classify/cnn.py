"""Sorter 2: ResNet-18 (CNN) fine-tuned on page IMAGES.

    python -m classify.cnn --epochs 4

Looks at the page like a person glancing at the layout, without reading:
letterhead, tables, photo, form boxes. Works even when OCR text is garbage.

Transfer learning: start from ResNet-18 pretrained on ImageNet (1.2M photos),
replace the last layer (1000 ImageNet classes -> our 7 page types), fine-tune.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pymupdf
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import LABELS, ROOT, PageRecord, load_pages, split_by_claim
from .metrics import print_report, save, score

IMG_DIR = ROOT / "data" / "page_images"
SIZE = 224  # ResNet's native input size


def page_image(r: PageRecord, cache: Path = IMG_DIR) -> Path:
    """Render one page to a grayscale PNG once, then reuse it.

    PDF pages are rendered at 40 dpi (340x440 px for US Letter); real scans that arrive
    as images (RVL-CDIP) are resized to the same size so both look alike to the network.
    """
    path = cache / f"{r.claim_id}_p{r.page:02d}.png"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        if r.image is not None:
            Image.open(r.image).convert("L").resize((340, 440)).save(path)
        else:
            with pymupdf.open(r.pdf) as doc:
                doc[r.page - 1].get_pixmap(dpi=40, colorspace=pymupdf.csGRAY).save(path)
    return path


# ImageNet statistics: the pretrained network expects inputs normalized this way.
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomRotation(2, fill=255),                 # pages arrive slightly tilted
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # different scanners
    transforms.ToTensor(),
    NORMALIZE,
])
TEST_TF = transforms.Compose([transforms.Grayscale(3), transforms.Resize((SIZE, SIZE)),
                              transforms.ToTensor(), NORMALIZE])


class PageImages(Dataset):
    def __init__(self, records: list[PageRecord], tf):
        self.paths = [page_image(r) for r in records]
        self.y = [LABELS.index(r.label) for r in records]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.y[i]


def build_model(pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))  # 512 features -> 7 page types
    return model


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit(train: list[PageRecord], seed: int = 0, epochs: int = 4, pretrained: bool = True) -> nn.Module:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dev = device()
    train_dl = DataLoader(PageImages(train, TRAIN_TF), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    model = build_model(pretrained).to(dev)
    # Class weights: rare types (towing, medical) count more, like class_weight="balanced".
    counts = np.bincount([LABELS.index(r.label) for r in train], minlength=len(LABELS))
    weights = torch.tensor(len(train) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)      # forward + how wrong
            optimizer.zero_grad()
            loss.backward()                  # autograd: gradient for all ~11M weights
            optimizer.step()                 # nudge the weights
            total, n = total + loss.item() * len(y), n + len(y)
        print(f"epoch {epoch}/{epochs}  train loss {total / n:.4f}  ({time.time() - t0:.1f}s)")
    return model


def predict_proba(model: nn.Module, records: list[PageRecord]) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in DataLoader(PageImages(records, TEST_TF), batch_size=64):
            probs.append(torch.softmax(model(x.to(dev)), dim=1).cpu())
    return torch.cat(probs).numpy()


def train_and_eval(records, seed: int = 0, epochs: int = 4, threshold: float = 0.8, pretrained: bool = True):
    train, test = split_by_claim(records, seed=seed)
    dev = device()
    model = fit(train, seed=seed, epochs=epochs, pretrained=pretrained)
    proba = predict_proba(model, test)
    pred = [LABELS[i] for i in proba.argmax(axis=1)]
    result = score("image_resnet18", [r.label for r in test], pred, proba.max(axis=1),
                   [r.scanned for r in test], threshold)
    result.update(train_pages=len(train), epochs=epochs, device=str(dev))
    return model, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    model, result = train_and_eval(load_pages(), seed=args.seed, epochs=args.epochs)
    print_report(result)
    args.models.mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.models / "image_resnet18.pt")
    print("\nsaved", save(result, ROOT / "eval" / "results"))


if __name__ == "__main__":
    main()
