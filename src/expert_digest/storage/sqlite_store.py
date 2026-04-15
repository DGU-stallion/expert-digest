"""SQLite persistence for imported source documents."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from expert_digest.domain.models import Document

DEFAULT_DATABASE_PATH = Path("data/processed/expert_digest.sqlite3")


def save_documents(db_path: str | Path, documents: Iterable[Document]) -> int:
    """Save documents to SQLite and return the number of submitted documents."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(documents)

    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO documents (
                id,
                author,
                title,
                content,
                source,
                url,
                created_at,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    document.id,
                    document.author,
                    document.title,
                    document.content,
                    document.source,
                    document.url,
                    document.created_at,
                )
                for document in submitted
            ],
        )

    return len(submitted)


def list_documents(db_path: str | Path) -> list[Document]:
    """Return every stored document in stable title order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, author, title, content, source, url, created_at
            FROM documents
            ORDER BY title, id
            """
        ).fetchall()
    return [_document_from_row(row) for row in rows]


def get_documents_by_author(db_path: str | Path, author: str) -> list[Document]:
    """Return stored documents for one author in stable title order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, author, title, content, source, url, created_at
            FROM documents
            WHERE author = ?
            ORDER BY title, id
            """,
            (author,),
        ).fetchall()
    return [_document_from_row(row) for row in rows]


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            url TEXT,
            created_at TEXT,
            imported_at TEXT NOT NULL
        )
        """
    )


def _document_from_row(row: sqlite3.Row | tuple[str, ...]) -> Document:
    return Document(
        id=row[0],
        author=row[1],
        title=row[2],
        content=row[3],
        source=row[4],
        url=row[5],
        created_at=row[6],
    )
