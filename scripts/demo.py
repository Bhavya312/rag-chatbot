"""CLI demo: ask the bot questions without starting the server.

Usage:
    python scripts/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import build_pipeline

QUESTIONS = [
    "How do I get a refund?",
    "My WordPress site was hacked, how fast can you fix it?",
    "What are your support hours?",
    "How long does a Laravel project take?",
    "Do you sell frozen fish?",  # not in the docs -> the bot must refuse
]


def main() -> None:
    pipe = build_pipeline()
    print(f"mode={pipe.settings.mode}  docs={pipe.doc_count}  chunks={pipe.chunk_count}\n")
    for q in QUESTIONS:
        ans = pipe.ask(q)
        print("Q:", q)
        print("A:", ans.text)
        if ans.hits:
            print(
                "sources:",
                ", ".join(f"{h.doc_id} ({h.score:.2f})" for h in ans.hits),
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
