"""FastAPI service exposing the RAG chatbot over HTTP.

Endpoints:
    GET  /        service info
    GET  /health  liveness probe
    GET  /stats   index stats (mode, doc count, chunk count)
    POST /chat    ask a question -> answer + cited sources

Run locally:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .rag import RagPipeline, build_pipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceRef,
    StatsResponse,
)


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    """App factory. Pass a pipeline in tests to skip disk/env reads."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline if pipeline is not None else build_pipeline()
        yield

    app = FastAPI(title="RAG Chatbot", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def root() -> dict:
        return {"name": "rag-chatbot", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        p: RagPipeline = app.state.pipeline
        return StatsResponse(mode=p.settings.mode, docs=p.doc_count, chunks=p.chunk_count)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        p: RagPipeline = app.state.pipeline
        ans = p.ask(req.question, top_k=req.top_k)
        return ChatResponse(
            answer=ans.text,
            sources=[
                SourceRef(doc_id=h.doc_id, title=h.title, score=round(h.score, 4))
                for h in ans.hits
            ],
            mode=ans.mode,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
