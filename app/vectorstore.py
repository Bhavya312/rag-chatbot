"""Tiny in-memory vector store plus text chunking.

Usage:
    store = VectorStore()
    store.add("doc1#c0", "chunk text...", vector)
    results = store.search(query_vector, top_k=2)  # -> [(id, text, score)]

In production this would be pgvector / Qdrant / Pinecone. The API shape
(add a vector, search by similarity) is the same — only the scale changes.
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into word chunks with overlap.

    Why chunk: embedding models have token limits, and one giant vector
    for a whole manual blurs every topic together. Why overlap: so a
    sentence cut at a boundary still appears whole in the next chunk.
    """
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """In-memory store of (id, text, vector) triples, searched by cosine."""

    def __init__(self) -> None:
        self._items: list[tuple[str, str, list[float]]] = []

    def add(self, chunk_id: str, text: str, vector: list[float]) -> None:
        self._items.append((chunk_id, text, vector))

    def search(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[tuple[str, str, float]]:
        scored = [
            (chunk_id, text, cosine_similarity(query_vector, vec))
            for chunk_id, text, vec in self._items
        ]
        scored.sort(key=lambda r: r[2], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)
