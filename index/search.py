"""Build the search index for all claims and answer search queries.

    python -m index.search --claim CLM-2026-00008 --q "What is the repair estimate total?"

Search modes:
  vector  - embeddings in Chroma (meaning: "repair cost" ~ "estimate total")
  keyword - BM25 (exact tokens: policy numbers, shop names)
  hybrid  - both, fused with Reciprocal Rank Fusion
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_claim
from .stores import BM25Store, ChromaStore, Embedder, FaissStore, rrf

ROOT = Path(__file__).resolve().parent.parent
INGESTED = ROOT / "data" / "ingested"
INDEX_DIR = ROOT / "data" / "index"


def load_chunks(ingested: Path = INGESTED) -> list[Chunk]:
    chunks = []
    for f in sorted(ingested.glob("CLM-*.jsonl")):
        chunks += chunk_claim(f)
    return chunks


def embed_chunks(chunks: list[Chunk], embedder: Embedder, cache: Path = INDEX_DIR) -> np.ndarray:
    """Embed once and cache, keyed by chunk ids + model, so reruns are instant."""
    cache.mkdir(parents=True, exist_ok=True)
    vec_file, ids_file = cache / "vectors.npy", cache / "vector_ids.json"
    ids = [c.chunk_id for c in chunks]
    meta = {"model": embedder.name, "ids": ids, "texts_hash": hash(tuple(c.text for c in chunks))}
    if vec_file.exists() and ids_file.exists():
        saved = json.loads(ids_file.read_text())
        if saved["model"] == meta["model"] and saved["ids"] == ids and saved["texts_hash"] == meta["texts_hash"]:
            return np.load(vec_file)
    vectors = embedder.docs([c.text for c in chunks])
    np.save(vec_file, vectors)
    ids_file.write_text(json.dumps(meta))
    return vectors


class Searcher:
    def __init__(self, chunks: list[Chunk] | None = None, persist: bool = True):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.embedder = Embedder()
        t0 = time.time()
        vectors = embed_chunks(self.chunks, self.embedder) if persist else self.embedder.docs([c.text for c in self.chunks])
        self.embed_secs = time.time() - t0
        self.chroma = ChromaStore(INDEX_DIR / "chroma" if persist else None)
        self.chroma.add(self.chunks, vectors)
        self.faiss = FaissStore()
        self.faiss.add(self.chunks, vectors)
        self.bm25 = BM25Store()
        self.bm25.add(self.chunks)

    def search(self, query: str, claim_id: str, mode: str = "hybrid", k: int = 5,
               store: str = "chroma") -> list[tuple[Chunk, float]]:
        if mode == "keyword":
            return self.bm25.search_text(query, claim_id, k)
        qvec = self.embedder.query(query)
        vstore = self.chroma if store == "chroma" else self.faiss
        if mode == "vector":
            return vstore.search(qvec, claim_id, k)
        # hybrid: take a deeper list from each, fuse, keep top k
        return rrf([vstore.search(qvec, claim_id, 20), self.bm25.search_text(query, claim_id, 20)], top=k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    s = Searcher()
    for chunk, sc in s.search(args.q, args.claim, args.mode, args.k):
        print(f"p{chunk.page:<3} {chunk.page_type:<16} {sc:.3f}  {chunk.text[:110]}")


if __name__ == "__main__":
    main()
