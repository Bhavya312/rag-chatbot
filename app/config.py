"""Runtime configuration.

Every knob comes from the environment so the same code runs locally,
in tests, and in production without edits. Copy `.env.example` to `.env`
to set values for local development. Nothing here is required: with no
variables set at all, the bot runs fully offline in mock mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """All tunables for the pipeline. Frozen so startup config can't drift."""

    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 120          # words per chunk
    chunk_overlap: int = 20        # words shared between adjacent chunks
    default_top_k: int = 2         # chunks retrieved per question
    similarity_threshold: float = 0.08  # below this, the bot says "I don't know"
    docs_path: str = str(PROJECT_ROOT / "data" / "docs.json")

    @property
    def mode(self) -> str:
        """'openai' when a key is configured, otherwise free offline 'mock'."""
        return "openai" if self.openai_api_key else "mock"


def load_settings() -> Settings:
    """Build Settings from the environment (and an optional .env file)."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        chunk_size=_get_int("CHUNK_SIZE", 120),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 20),
        default_top_k=_get_int("TOP_K", 2),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.08),
        docs_path=os.getenv("DOCS_PATH", str(PROJECT_ROOT / "data" / "docs.json")),
    )
