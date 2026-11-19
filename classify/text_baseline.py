"""Sorter 1: TF-IDF + logistic regression on the page text (from data/ingested).

    python -m classify.text_baseline

Reads like a person reading the words. Fast, explainable (top words per class),
and gives probabilities we can use to route unsure pages to a human.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import ROOT, load_pages, split_by_claim
from .metrics import print_report, save, score


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),      # single words and 2-word phrases ("estimate total")
            min_df=2,                # ignore words seen on only 1 page
            sublinear_tf=True,       # 10 repeats is not 10x as important
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]+\b",  # words only: ignore numbers, names of amounts/IDs
        )),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
    ])


def top_words(model: Pipeline, n: int = 6) -> dict[str, list[str]]:
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    clf = model.named_steps["clf"]
    return {c: vocab[np.argsort(w)[::-1][:n]].tolist() for c, w in zip(clf.classes_, clf.coef_)}


def train_and_eval(records, seed: int = 0, threshold: float = 0.8):
    train, test = split_by_claim(records, seed=seed)
    model = build_model()
    model.fit([r.text for r in train], [r.label for r in train])
    proba = model.predict_proba([r.text for r in test])
    pred = model.classes_[proba.argmax(axis=1)]
    result = score("text_tfidf_logreg", [r.label for r in test], list(pred), proba.max(axis=1),
                   [r.scanned for r in test], threshold)
    result["train_pages"] = len(train)
    return model, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    model, result = train_and_eval(load_pages(), seed=args.seed)
    print_report(result)
    print("\nTop words per class (why the model decides):")
    for c, ws in top_words(model).items():
        print(f"  {c:<16} {', '.join(ws)}")
    args.models.mkdir(exist_ok=True)
    joblib.dump(model, args.models / "text_tfidf_logreg.joblib")
    print("\nsaved", save(result, ROOT / "eval" / "results"))


if __name__ == "__main__":
    main()
