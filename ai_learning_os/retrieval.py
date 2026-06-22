"""Hybrid retrieval: BM25 + dense embeddings + Reciprocal Rank Fusion."""

import re
from typing import Any

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi


# ── Chunking ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter — split on '. ', '! ', '? ', '\n\n'."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size_words: int = 400, overlap_words: int = 50) -> list[str]:
    """
    Split text into overlapping word-count chunks, breaking at sentence boundaries.
    For markdown, splits on ## headers first so section structure is preserved.
    """
    # If markdown-like, split on top-level headers first
    if re.search(r"^#{1,3} ", text, re.MULTILINE):
        sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    else:
        sections = [text]

    chunks = []
    for section in sections:
        sentences = _split_sentences(section)
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            wc = len(sentence.split())
            if current_words + wc > chunk_size_words and current:
                chunks.append(" ".join(current))
                # keep overlap
                overlap: list[str] = []
                ow = 0
                for s in reversed(current):
                    sw = len(s.split())
                    if ow + sw > overlap_words:
                        break
                    overlap.insert(0, s)
                    ow += sw
                current = overlap
                current_words = ow
            current.append(sentence)
            current_words += wc

        if current:
            chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    if not texts:
        return []
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def embed_query(query: str, client: OpenAI, model: str) -> list[float]:
    return embed_texts([query], client, model)[0]


# ── Index ──────────────────────────────────────────────────────────────────────

class HybridIndex:
    """
    In-memory hybrid search index over KB chunks.
    Combines BM25 (keyword) + cosine similarity (semantic) with RRF.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: list[dict] = []         # full chunk dicts with metadata
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None  # shape (N, D)

    def build(self, chunks: list[dict]) -> None:
        """Build index from a list of chunk dicts (must have 'text' key)."""
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            self.embeddings = None
            return
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        if chunks[0].get("embedding"):
            self.embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        else:
            self.embeddings = None

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        num_results: int = 5,
    ) -> list[dict]:
        if not self.chunks:
            return []

        n = len(self.chunks)
        results: list[dict] = []

        # ── BM25 ranking ──
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(-bm25_scores)  # descending

        if query_embedding is not None and self.embeddings is not None:
            # ── Dense ranking ──
            qv = np.array(query_embedding, dtype=np.float32)
            # cosine similarity (embeddings already normalised by OpenAI)
            dense_scores = self.embeddings @ qv
            dense_ranks = np.argsort(-dense_scores)

            # ── RRF fusion ──
            bm25_rank_of = {idx: rank for rank, idx in enumerate(bm25_ranks)}
            dense_rank_of = {idx: rank for rank, idx in enumerate(dense_ranks)}
            k = self.rrf_k
            fused = {
                i: 1 / (k + bm25_rank_of.get(i, n)) + 1 / (k + dense_rank_of.get(i, n))
                for i in range(n)
            }
            ranked = sorted(fused, key=lambda i: -fused[i])
            scores = fused
        else:
            # BM25-only fallback
            ranked = list(bm25_ranks)
            scores = {i: float(bm25_scores[i]) for i in range(n)}

        for idx in ranked[:num_results]:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(scores[idx]), 4)
            chunk.pop("embedding", None)  # don't return raw vector
            results.append(chunk)

        return results


# ── Singleton index (rebuilt from kb.json on each load) ───────────────────────

_index: HybridIndex | None = None


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex()
    return _index


def rebuild_index(kb_entries: list[dict], rrf_k: int = 60) -> HybridIndex:
    """Flatten all chunks from KB entries and (re)build the index."""
    global _index
    all_chunks = []
    for entry in kb_entries:
        for chunk in entry.get("chunks", []):
            all_chunks.append(chunk)
    _index = HybridIndex(rrf_k=rrf_k)
    _index.build(all_chunks)
    return _index
