"""One scoring function for every sorter, so models are compared the same way.

Accuracy alone hides weak small classes (photos are 37% of pages, medical bills 5%),
so we also report macro-F1 (every class counts equally), a confusion matrix, and how
many pages would go to a human at a confidence threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS


def score(name: str, y_true: list[str], y_pred: list[str], confidence: np.ndarray,
          scanned: list[bool], threshold: float = 0.8) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scanned = np.array(scanned)
    confident = confidence >= threshold
    result = {
        "model": name,
        "test_pages": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # macro-F1 over the classes present in this test set (the RVL test set has only 3 of the 7)
        "macro_f1": float(f1_score(y_true, y_pred, labels=[c for c in LABELS if c in set(y_true)],
                                   average="macro", zero_division=0)),
        "accuracy_scanned": float(accuracy_score(y_true[scanned], y_pred[scanned])) if scanned.any() else None,
        "threshold": threshold,
        "to_human_share": float(1 - confident.mean()),
        "accuracy_when_confident": float(accuracy_score(y_true[confident], y_pred[confident])) if confident.any() else None,
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "errors": [(str(t), str(p)) for t, p in zip(y_true, y_pred) if t != p],
    }
    return result


def print_report(r: dict) -> None:
    print(f"\n=== {r['model']} ({r['test_pages']} test pages) ===")
    scan = f"{r['accuracy_scanned']:.1%}" if r["accuracy_scanned"] is not None else "n/a"
    print(f"accuracy {r['accuracy']:.1%} | macro-F1 {r['macro_f1']:.3f} | scanned pages {scan}")
    rest = f"{r['accuracy_when_confident']:.1%}" if r["accuracy_when_confident"] is not None else "n/a (none confident)"
    print(f"below {r['threshold']:.0%} confidence -> human: {r['to_human_share']:.1%} of pages; "
          f"accuracy on the rest: {rest}")
    print(f"{'class':<16}{'precision':>10}{'recall':>8}{'f1':>7}{'pages':>7}")
    for c in LABELS:
        m = r["per_class"][c]
        print(f"{c:<16}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1-score']:>7.2f}{int(m['support']):>7}")
    short = [c[:6] for c in LABELS]
    print("confusion (rows = true, cols = predicted):")
    print(" " * 16 + "".join(f"{s:>7}" for s in short))
    for c, row in zip(LABELS, r["confusion"]):
        print(f"{c:<16}" + "".join(f"{v:>7}" for v in row))
    if r["errors"]:
        print("mistakes (true -> predicted):", r["errors"][:10])


def save(r: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{r['model']}.json"
    path.write_text(json.dumps(r, indent=2))
    return path
