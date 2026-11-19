"""Sorter: no train/test leakage, scoring is right, both models train and predict."""

import numpy as np
import pytest
import torch

from classify.cnn import build_model
from classify.data import LABELS, PageRecord, split_by_claim
from classify.metrics import score
from classify.stress import drop_words
from classify.text_baseline import build_model as build_text_model


def fake_records(n_claims=10, pages=6):
    texts = {
        "claim_form": "automobile loss notice policy number date of loss signature",
        "police_report": "motor vehicle accident report officer narrative driver plate",
        "repair_estimate": "repair estimate parts operations labor estimate total replace",
        "medical_bill": "patient statement itemized bill cpt charges balance due",
        "correspondence": "dear claim received regards adjuster next steps",
        "photo": "img jpg damage photo uploaded",
        "other": "towing invoice hook-up fee mileage amount paid",
    }
    recs = []
    for c in range(n_claims):
        for p in range(pages):
            label = LABELS[(c + p) % len(LABELS)]
            recs.append(PageRecord(f"CLM-{c:03d}", p + 1, label, p % 3 == 0, "text", texts[label], None))
    return recs


def test_split_keeps_each_claim_on_one_side():
    train, test = split_by_claim(fake_records(), test_share=0.3, seed=1)
    assert {r.claim_id for r in train}.isdisjoint({r.claim_id for r in test})
    assert len({r.claim_id for r in test}) == 3


def test_split_is_repeatable():
    a = split_by_claim(fake_records(), seed=5)[1]
    b = split_by_claim(fake_records(), seed=5)[1]
    assert [r.claim_id for r in a] == [r.claim_id for r in b]


def test_score_counts_human_routing_and_errors():
    r = score("t", ["photo", "photo", "other", "other"], ["photo", "other", "other", "other"],
              np.array([0.95, 0.6, 0.9, 0.99]), [False, True, False, False], threshold=0.8)
    assert r["accuracy"] == 0.75
    assert r["to_human_share"] == 0.25
    assert r["accuracy_when_confident"] == 1.0   # the one mistake was the unsure page
    assert r["accuracy_scanned"] == 0.0
    assert r["errors"] == [("photo", "other")]


def test_text_model_learns_and_gives_probabilities():
    recs = fake_records()
    model = build_text_model().fit([r.text for r in recs], [r.label for r in recs])
    proba = model.predict_proba(["patient itemized bill cpt charges"])
    assert model.classes_[proba.argmax()] == "medical_bill"
    assert proba.sum() == pytest.approx(1.0)


def test_cnn_outputs_one_score_per_page_type():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, len(LABELS))


def test_rvl_mapping_targets_our_labels():
    from classify.rvl import MAPPING
    assert set(MAPPING.values()) <= set(LABELS)
    assert MAPPING["letter"] == MAPPING["email"] == MAPPING["memo"] == "correspondence"


def test_real_scan_images_are_resized_like_pdf_pages(tmp_path):
    from PIL import Image
    from classify.cnn import page_image

    src = tmp_path / "scan.tif"
    Image.new("L", (754, 1000), 200).save(src)
    rec = PageRecord("RVL-train-form-1", 1, "claim_form", True, "ocr", "x", None, image=src)
    out = page_image(rec, cache=tmp_path / "cache")
    assert Image.open(out).size == (340, 440)


def test_drop_words_keeps_roughly_the_share():
    import random
    text = " ".join(["w"] * 1000)
    kept = len(drop_words(text, 0.2, random.Random(0)).split())
    assert 150 < kept < 250
