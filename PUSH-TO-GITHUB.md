# Push to GitHub — RAG Chatbot (Project 1 of 3)

Prepared 2026-10-12. The scaffold is complete: FastAPI app, mock + OpenAI modes,
tests (passed at build, 2026-10-11), README with quickstart and API table.
`.gitignore` already excludes `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`.
It is NOT a git repo yet — nothing has been pushed anywhere.

## Suggested repo settings
- **Name:** `rag-chatbot`
- **Description:** Support chatbot that answers only from your own documents — FastAPI + embeddings + cosine retrieval, runs offline in mock mode, switches to OpenAI with one env var.
- **Topics:** `fastapi`, `rag`, `llm`, `embeddings`, `python`, `openai`, `portfolio`
- **Visibility:** Public (portfolio piece)

## Commands (run inside this folder)
```bash
git init -b main
git add .
git commit -m "RAG chatbot: FastAPI service with mock/OpenAI modes, retrieval with citations and honest fallback"
git remote add origin https://github.com/Bhavya312/rag-chatbot.git
git push -u origin main
```
(Create the empty repo on github.com first — no README/license, those are already here or can be added after.)

GitHub profile: https://github.com/Bhavya312

## After the push
- Pin the repo on your GitHub profile (Customize pins → rag-chatbot).
- Projects 2 and 3 from the refresher plan come next: ticket-triage API, then an LLM agent with tools.
- Once 1–2 projects are live, the paused Python/AI role applications can restart (paused 2026-09-26 until GitHub projects exist).
