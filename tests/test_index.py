"""Search: chunks keep their source, stores never leak another claim's pages, fusion works."""

import numpy as np
import pytest

from index.chunking import chunk_page
from index.stores import BM25Store, ChromaStore, FaissStore, rrf

PAGE = "\n".join([
    "Lewis Collision Repair",
    "Repair Estimate",
    "Estimate number: EST-90123",
    "1 Headlamp assembly LH - replace $763.33",
    "2 Hood panel - repair $221.49",
    "Subtotal $2,206.66",
    "Estimate total $2,346.78",
] * 3)


def test_chunks_keep_source_and_label_value_lines_together():
    chunks = chunk_page("CLM-1", 10, "repair_estimate", "ocr", PAGE)
    assert len(chunks) > 1
    assert all(c.claim_id == "CLM-1" and c.page == 10 for c in chunks)
    assert all(c.text.startswith("[repair_estimate]") for c in chunks)
    assert any("Estimate total $2,346.78" in c.text for c in chunks)   # a line is never split
    assert all(len(c.text) <= 400 + 200 for c in chunks)


def test_overlap_repeats_the_boundary_line():
    chunks = chunk_page("CLM-1", 1, "x", "text", PAGE, max_chars=120, overlap=1)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-12:]                       # end of a's last line
        assert b.text[:60].find(tail) != -1       # ...is repeated at the start of b


def _two_claims():
    a = chunk_page("CLM-A", 1, "repair_estimate", "text", "Estimate total $100.00\nShop Alpha")
    b = chunk_page("CLM-B", 1, "repair_estimate", "text", "Estimate total $999.00\nShop Beta")
    chunks = a + b
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(chunks), 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs


@pytest.mark.parametrize("store_cls", [FaissStore, ChromaStore])
def test_vector_stores_only_search_inside_the_claim(store_cls):
    chunks, vecs = _two_claims()
    store = store_cls()
    store.add(chunks, vecs)
    # query with claim B's own vector, but restricted to claim A: must return only A
    results = store.search(vecs[-1], "CLM-A", k=5)
    assert results and all(c.claim_id == "CLM-A" for c, _ in results)


def test_bm25_finds_exact_tokens_inside_the_claim():
    chunks, _ = _two_claims()
    bm = BM25Store()
    bm.add(chunks)
    top, _ = bm.search_text("Shop Beta total", "CLM-B", k=1)[0]
    assert top.claim_id == "CLM-B" and "Beta" in top.text
    assert all(c.claim_id == "CLM-A" for c, _ in bm.search_text("Shop Beta", "CLM-A", k=5))


def test_rrf_rewards_chunks_ranked_high_in_both_lists():
    chunks, _ = _two_claims()
    x, y = chunks[0], chunks[1]
    fused = rrf([[(x, 0.9), (y, 0.1)], [(x, 5.0), (y, 1.0)]], top=2)
    assert fused[0][0].chunk_id == x.chunk_id
    assert fused[0][1] > fused[1][1]
