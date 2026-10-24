# RAG Chatbot

A support chatbot that answers **only from your own documents** — with citations,
and an honest "I don't know" when nothing is relevant. Built as a portfolio
project for Python + LLM roles: the whole thing runs **offline in mock mode**
(no API keys, no cost), and switches to real OpenAI embeddings + chat with a
single environment variable.

```
question ──▶ embed ──▶ cosine search over chunk index ──▶ top_k chunks
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                    best score < threshold? ──▶ "I don't know" (no hallucinating)
                              │
                              ▼
                    mock mode: quote the best chunk with its doc id
                    openai mode: LLM answers, grounded in the retrieved context
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI demo (no server needed)
python scripts/demo.py

# API server
uvicorn app.main:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

## API

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I get a refund?"}' | python -m json.tool
```

```json
{
  "answer": "(mock) From 'refunds': We offer a 14-day money-back guarantee...",
  "sources": [
    {"doc_id": "refunds", "title": "Refund policy", "score": 0.42}
  ],
  "mode": "mock"
}
```

| Endpoint      | What it does                                    |
|---------------|-------------------------------------------------|
| `GET /`       | service info                                    |
| `GET /health` | liveness probe                                  |
| `GET /stats`  | mode, indexed doc count, chunk count            |
| `POST /chat`  | `{"question": "...", "top_k": 2}` → answer + sources |

## Use your own docs

`data/docs.json` is a list of `{"id", "title", "text"}` objects. Replace it
with your own content (FAQs, help-center articles, product docs) and restart —
or point `DOCS_PATH` at another file. Chunking (word windows with overlap),
embedding, and indexing all happen at startup.

## Real OpenAI mode

```bash
pip install openai          # only needed for real mode
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

Mock embeddings are deterministic word-hash vectors: similarity reflects word
overlap, not meaning. Real embeddings (`text-embedding-3-small`) add the
semantics; the pipeline code is identical either way.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers chunking/overlap, cosine sanity, mock-embedder determinism, retrieval
accuracy, the honest-refusal path, and the HTTP endpoints via TestClient.

## Project structure

```
app/
  main.py         FastAPI app (factory pattern — easy to test)
  rag.py          RagPipeline: index, retrieve, refuse-or-answer
  embeddings.py   OpenAIEmbedder / MockEmbedder (same interface)
  vectorstore.py  in-memory cosine store + chunk_text
  llm_client.py   OpenAI chat client (lazy import — mock needs no install)
  config.py       env-driven Settings dataclass
  schemas.py      Pydantic v2 request/response models
data/docs.json    sample knowledge base
scripts/demo.py   CLI demo
tests/            pytest suite (offline, mock mode)
```

## Design notes (interview talking points)

- **Why chunk with overlap?** Embedding models have token limits, and one
  vector per whole document blurs topics together. Overlap keeps sentences cut
  at a boundary intact in the next chunk.
- **Why a similarity threshold?** Retrieval always returns *something*; the
  threshold is what turns "nearest neighbor" into "I don't know" instead of a
  hallucinated answer.
- **Why the Answerer protocol?** Mock and OpenAI answering are swappable
  strategies behind one interface — the pipeline never branches on mode.
- **Production upgrades:** swap `VectorStore` for pgvector/Qdrant, persist the
  index, add re-ranking, stream tokens over SSE.

## Roadmap

- [x] Project 1 — RAG chatbot (this repo)
- [ ] Project 2 — ticket triage API (classify + prioritize support tickets)
- [ ] Project 3 — LLM agent with tools (function calling)
