"""Data loading node: loads and cleans documents from SQLite into pipeline state."""

from __future__ import annotations

from pathlib import Path

from expert_digest.pipeline.state import DigestState
from expert_digest.processing.cleaner import clean_text
from expert_digest.storage.sqlite_store import (
    get_documents_by_author,
    list_documents,
)

_MAX_DOCS_FOR_ANALYSIS = 30


def run_load_data(state: DigestState) -> dict:
    """Load documents from SQLite database and sample for LLM analysis.

    Reads the configured db_path and author from state, loads documents,
    and writes a sampled subset to ``documents`` for downstream analysis.
    """
    db_path = state.get("db_path", "")
    author = state.get("author", "")

    if not db_path or not Path(db_path).exists():
        return {"documents": [], "errors": state.get("errors", [])}

    if author:
        docs = get_documents_by_author(db_path, author)
    else:
        docs = list_documents(db_path)

    sampled = docs[:_MAX_DOCS_FOR_ANALYSIS]
    serialized = [
        {
            "id": d.id,
            "title": d.title,
            "content": clean_text(d.content or ""),
            "author": d.author,
            "url": d.url,
            "created_at": d.created_at,
        }
        for d in sampled
    ]
    return {"documents": serialized}
