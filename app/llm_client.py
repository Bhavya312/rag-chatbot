"""Chat-completion client for the real (OpenAI) answering path.

The `openai` package is imported lazily inside the constructor so that
mock mode — the default — never needs it installed.
"""

from __future__ import annotations


class OpenAIClient:
    """Thin wrapper over OpenAI chat completions."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # local import: mock mode needs no install
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Send chat messages; return {"content": str, "usage": {...}}."""
        resp = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        u = resp.usage
        return {
            "content": resp.choices[0].message.content or "",
            "usage": {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
            },
        }
