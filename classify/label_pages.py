"""Tag every page with its predicted type, for search filters and answer citations.

    python -m classify.label_pages

Uses the best setup from the experiment: text + image sorters trained on synthetic
training claims + real RVL scans, probabilities averaged. Adds to each line of
data/ingested/<claim_id>.jsonl:

    page_type        predicted type, e.g. "repair_estimate"
    page_type_conf   confidence 0-1
    needs_review     True when confidence < 0.8 (a human should check the type)
    split            "train" or "test" (claims the sorter was trained on vs never saw)
"""

from __future__ import annotations

import json
from collections import Counter

import joblib
import torch

from . import cnn
from .data import INGESTED, LABELS, ROOT, load_pages, split_by_claim
from .rvl import load_rvl
from .text_baseline import build_model as build_text_model

THRESHOLD = 0.8


def main() -> None:
    pages = load_pages()
    syn_train, syn_test = split_by_claim(pages, seed=0)
    test_claims = {r.claim_id for r in syn_test}
    train = syn_train + load_rvl("train")

    text_model = build_text_model().fit([r.text for r in train], [r.label for r in train])
    img_model = cnn.fit(train, seed=0, epochs=4)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(text_model, models_dir / "final_text.joblib")
    torch.save(img_model.state_dict(), models_dir / "final_image_resnet18.pt")

    p_text = text_model.predict_proba([r.text for r in pages])
    p_text = p_text[:, [list(text_model.classes_).index(c) for c in LABELS]]
    proba = (p_text + cnn.predict_proba(img_model, pages)) / 2

    by_claim: dict[str, dict[int, tuple[str, float]]] = {}
    for r, p in zip(pages, proba):
        by_claim.setdefault(r.claim_id, {})[r.page] = (LABELS[int(p.argmax())], float(p.max()))

    stats = Counter()
    for claim_id, preds in by_claim.items():
        path = INGESTED / f"{claim_id}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        for row in rows:
            t, conf = preds[row["page"]]
            row.update(page_type=t, page_type_conf=round(conf, 3), needs_review=conf < THRESHOLD,
                       split="test" if claim_id in test_claims else "train")
            stats["review" if conf < THRESHOLD else "auto"] += 1
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    truth = {(r.claim_id, r.page): r.label for r in pages}
    test_rows = [(c, p) for c in test_claims for p in by_claim[c]]
    acc = sum(by_claim[c][p][0] == truth[(c, p)] for c, p in test_rows) / len(test_rows)
    print(f"labeled {len(pages)} pages in {len(by_claim)} claims | auto: {stats['auto']}, needs review: {stats['review']}")
    print(f"accuracy on the {len(test_claims)} test claims (never trained on): {acc:.1%}")


if __name__ == "__main__":
    main()
