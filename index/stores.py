"""Embeddings + three ways to search chunks, all restricted to ONE claim.

  - ChromaStore: vector database with metadata filters (where claim_id = ...)
  - FaissStore:  fast vector library; it has no metadata filter, so we keep one small index per claim
  - BM25Store:   classic keyword search (exact words: policy numbers, shop names, amounts)

All three return (chunk, score) lists so they can be compared and combined.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .chunking import Chunk

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on QUERIES (not on documents) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False).astype("float32")

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0].astype("float32")


class ChromaStore:
    name = "chroma"

    def __init__(self, path: Path | None = None, collection: str | None = None):
        import uuid

        import chromadb

        # In-memory by default, with a unique collection name per instance: two processes
        # (e.g. the web app and the evaluation) must never reset each other's collection.
        # Rebuilding takes ~2 s from the cached vectors, so persistence isn't needed.
        self.client = chromadb.PersistentClient(str(path)) if path else chromadb.EphemeralClient()
        name = collection or f"claims_{uuid.uuid4().hex[:8]}"
        self.col = self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
        self.chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for i in range(0, len(chunks), 1000):
            part = chunks[i:i + 1000]
            self.col.add(ids=[c.chunk_id for c in part], embeddings=vectors[i:i + 1000].tolist(),
                         documents=[c.text for c in part],
                         metadatas=[{"claim_id": c.claim_id, "page": c.page, "page_type": c.page_type}
                                    for c in part])
        self.chunks.update({c.chunk_id: c for c in chunks})

    def search(self, qvec: np.ndarray, claim_id: str, k: int = 5) -> list[tuple[Chunk, float]]:
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=k, where={"claim_id": claim_id})
        return [(self.chunks[i], 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


class FaissStore:
    name = "faiss"

    def __init__(self):
        self.indexes: dict[str, tuple[object, list[Chunk]]] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        import faiss

        by_claim: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            by_claim.setdefault(c.claim_id, []).append(i)
        for claim_id, idx in by_claim.items():
            index = faiss.IndexFlatIP(vectors.shape[1])   # inner product = cosine on normalized vectors
            index.add(vectors[idx])
            self.indexes[claim_id] = (index, [chunks[i] for i in idx])

    def search(self, qvec: np.ndarray, claim_id: str, k: int = 5) -> list[tuple[Chunk, float]]:
        index, chunks = self.indexes[claim_id]
        scores, ids = index.search(qvec[None, :], min(k, len(chunks)))
        return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$][a-z0-9$.,\-]*[a-z0-9]|[a-z0-9]", text.lower())


class BM25Store:
    name = "bm25"

    def __init__(self):
        self.indexes: dict[str, tuple[object, list[Chunk]]] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        from rank_bm25 import BM25Okapi

        by_claim: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_claim.setdefault(c.claim_id, []).append(c)
        for claim_id, cs in by_claim.items():
            self.indexes[claim_id] = (BM25Okapi([_tokens(c.text) for c in cs]), cs)

    def search_text(self, query: str, claim_id: str, k: int = 5) -> list[tuple[Chunk, float]]:
        bm25, chunks = self.indexes[claim_id]
        scores = bm25.get_scores(_tokens(query))
        top = np.argsort(scores)[::-1][:k]
        return [(chunks[i], float(scores[i])) for i in top]


def rrf(result_lists: list[list[tuple[Chunk, float]]], k: int = 60, top: int = 5) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: combine rankings by position, not raw score
    (vector and keyword scores are on different scales)."""
    fused: dict[str, float] = {}
    keep: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            keep[chunk.chunk_id] = chunk
    best = sorted(fused, key=fused.get, reverse=True)[:top]
    return [(keep[i], fused[i]) for i in best]
