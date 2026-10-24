"""The RAG pipeline: chunk docs -> embed -> retrieve -> answer with citations.

Flow for one question:
    1. embed the question
    2. cosine-search the chunk index for the top_k nearest chunks
    3. if the best match is below the similarity threshold, refuse honestly
       ("I don't know") instead of hallucinating
    4. otherwise hand the retrieved chunks to an Answerer:
       - mock mode    -> ExtractiveAnswerer quotes the best chunk (free, offline)
       - openai mode  -> OpenAIAnswerer asks the LLM, grounded in the context
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings, load_settings
from .embeddings import get_embedder
from .llm_client import OpenAIClient
from .vectorstore import VectorStore, chunk_text

REFUSAL = (
    "I don't know — nothing in the docs covers that. "
    "Try asking about refunds, support hours, pricing, or project timelines."
)


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)
    mode: str = "mock"


class Answerer(Protocol):
    """Anything that turns retrieved chunks into a final answer string."""

    def answer(self, question: str, hits: list[Hit]) -> str: ...


class ExtractiveAnswerer:
    """Mock-mode answerer: quote the best chunk, never invent.

    Deterministic and free. The "(mock)" prefix makes it obvious in the
    API response that no LLM was involved.
    """

    def answer(self, question: str, hits: list[Hit]) -> str:
        top = hits[0]
        snippet = top.text[:300] + ("..." if len(top.text) > 300 else "")
        return f"(mock) From '{top.doc_id}': {snippet}"


class OpenAIAnswerer:
    """Real LLM answering, grounded strictly in the retrieved context."""

    SYSTEM_PROMPT = (
        "You answer questions about a small business's services. "
        "Answer ONLY from the context below. If the answer is not in the "
        "context, say you don't know. Cite the doc id like [refunds]. "
        "Keep answers under 120 words."
    )

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def answer(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return self.client.chat(messages, temperature=0.0)["content"]


class RagPipeline:
    """Owns the chunk index and answers questions against it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = get_embedder(settings)
        if settings.mode == "openai":
            self.answerer: Answerer = OpenAIAnswerer(
                OpenAIClient(settings.openai_api_key or "", settings.llm_model)
            )
        else:
            self.answerer = ExtractiveAnswerer()
        self.store = VectorStore()
        self._meta: dict[str, tuple[str, str]] = {}  # chunk_id -> (doc_id, title)
        self.doc_count = 0

    def index(self, docs: list[dict]) -> int:
        """Chunk and embed docs into the store. Returns the chunk count."""
        n = 0
        for doc in docs:
            chunks = chunk_text(
                doc["text"], self.settings.chunk_size, self.settings.chunk_overlap
            )
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}#c{i}"
                self.store.add(chunk_id, chunk, self.embedder.embed(chunk))
                self._meta[chunk_id] = (doc["id"], doc.get("title", ""))
                n += 1
        self.doc_count = len(docs)
        return n

    @property
    def chunk_count(self) -> int:
        return len(self.store)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Retrieve the most relevant chunks and answer. Refuses when unsure."""
        k = top_k or self.settings.default_top_k
        qvec = self.embedder.embed(question)
        scored = self.store.search(qvec, top_k=k)

        if not scored or scored[0][2] < self.settings.similarity_threshold:
            return Answer(text=REFUSAL, hits=[], mode=self.settings.mode)

        hits = [
            Hit(
                doc_id=self._meta[chunk_id][0],
                title=self._meta[chunk_id][1],
                text=text,
                score=score,
            )
            for chunk_id, text, score in scored
        ]
        return Answer(
            text=self.answerer.answer(question, hits),
            hits=hits,
            mode=self.settings.mode,
        )


def load_docs(path: str) -> list[dict]:
    """Load the knowledge base; every doc needs id, title, and text."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        missing = {"id", "title", "text"} - set(doc)
        if missing:
            raise ValueError(f"doc {doc.get('id')!r} is missing keys: {missing}")
    return docs


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    """Factory: settings -> embedder + answerer -> fully indexed pipeline."""
    settings = settings or load_settings()
    pipeline = RagPipeline(settings)
    pipeline.index(load_docs(settings.docs_path))
    return pipeline
