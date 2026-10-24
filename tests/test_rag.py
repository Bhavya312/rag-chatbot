"""Tests for the RAG chatbot. All run offline in mock mode — no API keys."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import MockEmbedder
from app.main import create_app
from app.rag import build_pipeline
from app.vectorstore import chunk_text, cosine_similarity

DOCS_PATH = str(Path(__file__).resolve().parent.parent / "data" / "docs.json")


def make_settings(**overrides) -> Settings:
    return Settings(docs_path=DOCS_PATH, **overrides)


# --- chunking ---


def test_chunk_text_splits_with_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
    assert len(chunks) == 3
    # overlap: the tail of chunk 0 is the head of chunk 1
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("hello world", chunk_size=100) == ["hello world"]


# --- vectors ---


def test_cosine_similarity_identical_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_mock_embedder_is_deterministic_and_normalized():
    e = MockEmbedder()
    v1, v2 = e.embed("refund policy"), e.embed("refund policy")
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9


# --- pipeline ---


def test_retrieval_finds_the_right_doc():
    pipe = build_pipeline(make_settings())
    assert pipe.doc_count == 8
    ans = pipe.ask("How do I get a refund?")
    assert ans.hits, "expected at least one hit"
    assert ans.hits[0].doc_id == "refunds"
    assert "14-day" in ans.text  # quoted from the chunk, not invented


def test_refuses_when_nothing_is_relevant():
    pipe = build_pipeline(make_settings())
    ans = pipe.ask("zxqw blorpt purple elephant quantum")
    assert ans.hits == []
    assert "don't know" in ans.text


def test_top_k_override_is_respected():
    pipe = build_pipeline(make_settings())
    ans = pipe.ask("How do I get a refund?", top_k=1)
    assert len(ans.hits) == 1


# --- HTTP API ---


def test_chat_endpoint_answers_with_sources():
    pipe = build_pipeline(make_settings())
    with TestClient(create_app(pipe)) as client:
        r = client.post("/chat", json={"question": "What are your support hours?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "mock"
    assert body["sources"], "expected cited sources"
    assert body["sources"][0]["doc_id"] == "support-hours"


def test_chat_endpoint_rejects_empty_question():
    pipe = build_pipeline(make_settings())
    with TestClient(create_app(pipe)) as client:
        r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_health_and_stats():
    pipe = build_pipeline(make_settings())
    with TestClient(create_app(pipe)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        stats = client.get("/stats").json()
    assert stats == {"mode": "mock", "docs": 8, "chunks": pipe.chunk_count}
    assert stats["chunks"] >= 8
