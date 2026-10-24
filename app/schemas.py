"""Pydantic v2 request/response models for the HTTP API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceRef(BaseModel):
    doc_id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    mode: str  # "mock" or "openai"


class StatsResponse(BaseModel):
    mode: str
    docs: int
    chunks: int


class HealthResponse(BaseModel):
    status: str = "ok"
