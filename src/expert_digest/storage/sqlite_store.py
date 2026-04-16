"""SQLite persistence for imported source documents."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from expert_digest.domain.models import Chunk, Document

DEFAULT_DATABASE_PATH = Path("data/processed/expert_digest.sqlite3")


def save_documents(db_path: str | Path, documents: Iterable[Document]) -> int:
    """Save documents to SQLite and return the number of submitted documents."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(documents)

    with _connect(database_path) as connection:
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


def save_chunks(db_path: str | Path, chunks: Iterable[Chunk]) -> int:
    """Save chunks to SQLite and return the number of submitted chunks."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(chunks)

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id,
                document_id,
                text,
                chunk_index,
                start_char,
                end_char,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.text,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                )
                for chunk in submitted
            ],
        )

    return len(submitted)


def list_documents(db_path: str | Path) -> list[Document]:
    """Return every stored document in stable title order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
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

    with _connect(database_path) as connection:
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


def list_chunks(db_path: str | Path) -> list[Chunk]:
    """Return every stored chunk in stable document/chunk order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, text, chunk_index, start_char, end_char
            FROM chunks
            ORDER BY document_id, chunk_index, id
            """
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def list_chunks_for_document(db_path: str | Path, document_id: str) -> list[Chunk]:
    """Return chunks for one document in chunk index order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, text, chunk_index, start_char, end_char
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index, id
            """,
            (document_id,),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def clear_chunks(db_path: str | Path) -> int:
    """Delete all stored chunks and return number of removed rows."""
    database_path = Path(db_path)
    if not database_path.exists():
        return 0

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        connection.execute("DELETE FROM chunks")
    return count


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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            imported_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        """
    )


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    # Keep SQLite temp/journal data in memory to avoid sidecar-file failures
    # in restricted environments while preserving local single-process usage.
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


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


def _chunk_from_row(row: sqlite3.Row | tuple[str, ...]) -> Chunk:
    return Chunk(
        id=row[0],
        document_id=row[1],
        text=row[2],
        chunk_index=row[3],
        start_char=row[4],
        end_char=row[5],
    )
