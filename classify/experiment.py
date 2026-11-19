"""Does adding REAL scans (RVL-CDIP) help the sorters?

    python -m classify.experiment

Train each sorter two ways, test each on two test sets:

  train A: synthetic only            (402 pages, 35 claims)
  train B: synthetic + real RVL      (402 + 800 pages)

  test 1:  synthetic held-out claims (161 pages, 15 claims never seen)
  test 2:  real RVL scans held out   (200 pages: letter/email/memo, form, invoice)

The RVL test pages come from RVL's own validation split, so none were seen in training.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from . import cnn
from .data import LABELS, ROOT, load_pages, split_by_claim
from .metrics import print_report, score
from .rvl import load_rvl
from .text_baseline import build_model as build_text_model


def evaluate(name, proba, classes, test, threshold=0.8):
    pred = [classes[i] for i in proba.argmax(axis=1)]
    return score(name, [r.label for r in test], pred, proba.max(axis=1), [r.scanned for r in test], threshold)


def main() -> None:
    syn_train, syn_test = split_by_claim(load_pages(), seed=0)
    rvl_train, rvl_test = load_rvl("train"), load_rvl("validation")
    if not rvl_train[0].text:
        raise SystemExit("run `python -m classify.rvl` first to OCR the RVL pages")
    trains = {"A_synthetic": syn_train, "B_synthetic+real": syn_train + rvl_train}
    tests = {"synthetic_test": syn_test, "real_rvl_test": rvl_test}
    print("train sets:", {k: len(v) for k, v in trains.items()},
          "| real test labels:", dict(Counter(r.label for r in rvl_test)))

    results = []
    for tname, train in trains.items():
        text_model = build_text_model().fit([r.text for r in train], [r.label for r in train])
        print(f"\n--- image model, train {tname} ---")
        img_model = cnn.fit(train, seed=0, epochs=4)
        for sname, test in tests.items():
            p_text = text_model.predict_proba([r.text for r in test])
            results.append(evaluate(f"text|{tname}|{sname}", p_text, list(text_model.classes_), test))
            p_img = cnn.predict_proba(img_model, test)
            results.append(evaluate(f"image|{tname}|{sname}", p_img, LABELS, test))
            # Combined: average the two sorters' probabilities (columns aligned to LABELS).
            # If they disagree, the average is low-confidence and the page goes to a human.
            p_text_aligned = p_text[:, [list(text_model.classes_).index(c) for c in LABELS]]
            results.append(evaluate(f"combined|{tname}|{sname}", (p_text_aligned + p_img) / 2, LABELS, test))

    print("\n| Sorter | Trained on | Tested on | Accuracy | Macro-F1 | To human (<80%) | Acc. when confident |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        sorter, tname, sname = r["model"].split("|")
        print(f"| {sorter} | {tname} | {sname} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | "
              f"{r['to_human_share']:.1%} | {r['accuracy_when_confident']:.1%} |"
              if r["accuracy_when_confident"] is not None else
              f"| {sorter} | {tname} | {sname} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | "
              f"{r['to_human_share']:.1%} | n/a |")

    for r in results:
        if r["model"].endswith("real_rvl_test"):
            print_report(r)

    out = ROOT / "eval" / "results" / "rvl_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "errors"} for r in results], indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
