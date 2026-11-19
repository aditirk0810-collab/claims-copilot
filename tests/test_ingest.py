"""Page reading: right method per page, and text that matches the ground truth."""

import json
import shutil

import pytest

from eval.ocr_eval import cer, evaluate, expected_method
from ingest.ocr import GroundTruthOCR
from ingest.reader import classify_method, read_packet
from synth.generate import build_packet


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("pk")
    lab = build_packet(4, seed=7, out_dir=out, scan_rate=0.5)
    return out / lab["claim_id"] / "packet.pdf", lab


def test_method_rule():
    assert classify_method(120, 0.0) == "text"
    assert classify_method(0, 1.0) == "ocr"
    assert classify_method(9, 0.4) == "image"     # photo with a caption


def test_picks_right_method_for_every_page(packet):
    pdf, lab = packet
    truth = {p["page"]: p["text"] for p in lab["pages"]}
    pages = read_packet(pdf, GroundTruthOCR(truth))
    assert [p.method for p in pages] == [expected_method(lp) for lp in lab["pages"]]


def test_text_pages_are_copied_exactly(packet):
    pdf, lab = packet
    truth = {p["page"]: p["text"] for p in lab["pages"]}
    for p in read_packet(pdf, GroundTruthOCR(truth)):
        if p.method == "text":
            assert cer(p.text, truth[p.page]) == 0.0
            assert p.confidence is None


def test_scan_without_ocr_engine_fails_loudly(packet):
    pdf, lab = packet
    assert any(p["scanned"] for p in lab["pages"])
    with pytest.raises(RuntimeError, match="scan"):
        read_packet(pdf)


def test_eval_scores_perfect_ocr_as_perfect(packet, tmp_path):
    pdf, lab = packet
    pk_dir = tmp_path / "packets" / lab["claim_id"]
    pk_dir.mkdir(parents=True)
    shutil.copy(pdf.parent / "labels.json", pk_dir / "labels.json")
    truth = {p["page"]: p["text"] for p in lab["pages"]}
    ing = tmp_path / "ingested"
    ing.mkdir()
    with open(ing / f"{lab['claim_id']}.jsonl", "w") as f:
        for p in read_packet(pdf, GroundTruthOCR(truth)):
            f.write(json.dumps(p.to_dict()) + "\n")
    r = evaluate(tmp_path / "packets", ing)
    assert r["method_acc"] == 1.0
    assert all(found == total for found, total in r["facts"].values())


def test_cer():
    assert cer("$2,346.78", "$2,346.78") == 0.0
    assert cer("$2,B46.78", "$2,346.78") == pytest.approx(1 / 9)
