"""Data loading node: loads documents and wiki pages into pipeline state."""

from __future__ import annotations

from pathlib import Path

from expert_digest.pipeline.state import DigestState
from expert_digest.processing.cleaner import clean_text
from expert_digest.storage.sqlite_store import (
    get_documents_by_author,
    list_documents,
)
from expert_digest.wiki.vault import WikiVault


def run_load_data(state: DigestState) -> dict:
    """Load documents from SQLite and wiki pages from vault into pipeline state."""
    db_path = state.get("db_path", "")
    wiki_root = state.get("wiki_root", "")
    author = state.get("author", "")
    errors: list = state.get("errors", [])

    documents: list[dict] = []
    if db_path and Path(db_path).exists():
        if author:
            docs = get_documents_by_author(db_path, author)
        else:
            docs = list_documents(db_path)
        documents = [
            {
                "id": d.id,
                "title": d.title,
                "content": clean_text(d.content or ""),
                "author": d.author,
                "url": d.url,
                "created_at": d.created_at,
            }
            for d in docs
        ]

    wiki_pages: list[dict] = []
    if wiki_root and Path(wiki_root).exists():
        try:
            vault = WikiVault(root=Path(wiki_root))
            pages = vault.list_pages()
            wiki_pages = [
                {
                    "path": p.path,
                    "page_type": p.page_type,
                    "title": p.title,
                    "body": p.body,
                    "sources": [
                        {"source_id": s.source_id, "title": s.title, "url": s.url}
                        for s in p.sources
                    ],
                    "confidence": p.confidence,
                }
                for p in pages
            ]
        except Exception as exc:
            errors.append(
                {"node": "load_data", "message": f"failed to load wiki: {exc}"}
            )

    return {
        "documents": documents,
        "wiki_pages": wiki_pages,
        "errors": errors,
    }
