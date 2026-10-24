"""Embedding adapters: real OpenAI embeddings or a deterministic mock.

Both expose the same interface:
    embed(text) -> list[float]
    embed_batch(texts) -> list[list[float]]

Mock mode is the default: free, offline, and deterministic, so the whole
project (including the test suite) runs with zero API keys and zero cost.
Set OPENAI_API_KEY to switch to real embeddings with no code changes.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

DIM = 2048  # mock vector size (real model uses 1536; bigger = fewer hash collisions)

# Skip filler words so content words dominate the mock vector.
STOPWORDS = frozenset(
    """
    a an the and or of to in on for with is are was were be been you your we our
    they their it its this that these those how what when where do does did i me
    my get got can will would should could there here from by as at not no any
    all if then than so such into out up down over under again once per each
    """.split()
)


def _stem(word: str) -> str:
    """Crude suffix stripper so 'takes' and 'take' hash to the same bucket.

    Deliberately naive — a real stemmer (NLTK/Snowball) is overkill for a
    mock whose only job is to make the RAG plumbing testable offline.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class MockEmbedder:
    """Deterministic word-hash vectors. Free, offline.

    Honest limitation: similarity here mostly reflects WORD OVERLAP,
    not meaning. "refund" and "money back" look unrelated to this mock,
    while a real model knows they're close. Good enough to learn the RAG
    plumbing; the real model adds the semantics.
    """

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in text.lower().split():
            word = word.strip(".,!?;:()\"'")
            if not word or word in STOPWORDS:
                continue
            word = _stem(word)
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder:
    """Real embeddings. Requires an OpenAI API key."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # imported here so mock mode needs no install
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder(settings: Settings) -> MockEmbedder | OpenAIEmbedder:
    """Pick real or mock embeddings based on the configured mode."""
    if settings.mode == "openai":
        return OpenAIEmbedder(settings.openai_api_key or "", settings.embedding_model)
    return MockEmbedder()
