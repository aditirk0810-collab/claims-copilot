"""The synthetic data is the ground truth for every metric, so check it hard."""

import json

import pymupdf
import pytest

from synth.generate import build_packet
from synth.qa import NOT_IN_PACKET
from synth.render import PAGE_TYPES

N = 6


@pytest.fixture(scope="module")
def packets(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    return out, [build_packet(i, seed=7, out_dir=out) for i in range(1, N + 1)]


def test_pdf_matches_labels(packets):
    out, labels = packets
    for lab in labels:
        doc = pymupdf.open(out / lab["claim_id"] / "packet.pdf")
        assert doc.page_count == lab["num_pages"] == len(lab["pages"])
        assert [p["page"] for p in lab["pages"]] == list(range(1, doc.page_count + 1))


def test_page_types_valid_and_core_docs_present(packets):
    _, labels = packets
    for lab in labels:
        types = [p["type"] for p in lab["pages"]]
        assert set(types) <= set(PAGE_TYPES)
        assert types.count("claim_form") == 2
        assert "repair_estimate" in types and "photo" in types


def test_every_answer_is_printed_on_a_cited_page(packets):
    _, labels = packets
    for lab in labels:
        for q in lab["qa"]:
            if not q["answerable"]:
                continue
            texts = [lab["pages"][p - 1]["text"] for p in q["evidence_pages"]]
            assert any(q["answer"] in t for t in texts), q


def test_single_fact_answers_appear_on_every_cited_page(packets):
    _, labels = packets
    for lab in labels:
        for q in lab["qa"]:
            if q["answerable"] and q["type"] != "lowest_estimate":
                for p in q["evidence_pages"]:
                    assert q["answer"] in lab["pages"][p - 1]["text"], (q, p)


def test_unanswerable_questions_have_no_source_document(packets):
    _, labels = packets
    missing_doc = {"police_report_number": "police_report", "medical_total": "medical_bill", "tow_amount": "other"}
    for lab in labels:
        types = {p["type"] for p in lab["pages"]}
        for q in lab["qa"]:
            if q["answer"] == NOT_IN_PACKET:
                assert q["evidence_pages"] == []
                assert missing_doc[q["type"]] not in types


def test_scanned_pages_need_ocr_and_clean_pages_do_not(packets):
    out, labels = packets
    for lab in labels:
        doc = pymupdf.open(out / lab["claim_id"] / "packet.pdf")
        for p in lab["pages"]:
            extracted = doc[p["page"] - 1].get_text().strip()
            if p["scanned"]:
                assert extracted == ""
            elif p["type"] != "photo":
                assert extracted and extracted.split()[0] in p["text"]


def test_same_seed_same_output(tmp_path):
    a = build_packet(3, seed=11, out_dir=tmp_path / "a")
    b = build_packet(3, seed=11, out_dir=tmp_path / "b")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_real_photos_used_when_pool_given(tmp_path):
    from PIL import Image

    pool = []
    for i in range(8):
        f = tmp_path / f"car_{i:03d}.jpg"
        Image.new("RGB", (64, 64), (i * 30, 80, 120)).save(f)
        pool.append(f)
    lab = build_packet(1, seed=7, out_dir=tmp_path / "out", photo_pool=pool)
    photo_pages = [p for p in lab["pages"] if p["type"] == "photo"]
    assert photo_pages
    names = [p["photo_source"] for p in photo_pages]
    assert set(names) <= {f.name for f in pool}
    assert len(names) == len(set(names))  # no repeated photo within one packet


def test_mix_of_scanned_and_missing_docs_across_dataset(packets):
    _, labels = packets
    pages = [p for lab in labels for p in lab["pages"]]
    assert any(p["scanned"] for p in pages) and not all(p["scanned"] for p in pages)
