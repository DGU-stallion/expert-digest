"""SQLite persistence for imported source documents."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from expert_digest.domain.models import (
    Chunk,
    ChunkEmbedding,
    Document,
    EvidenceSpan,
    ParentSection,
)

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
                parent_section_id,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.text,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.parent_section_id,
                )
                for chunk in submitted
            ],
        )

    return len(submitted)


def save_chunk_embeddings(
    db_path: str | Path,
    embeddings: Iterable[ChunkEmbedding],
) -> int:
    """Save chunk embeddings and return number of submitted embeddings."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(embeddings)

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunk_embeddings (
                id,
                chunk_id,
                model,
                dimensions,
                vector_json,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    embedding.id,
                    embedding.chunk_id,
                    embedding.model,
                    embedding.dimensions,
                    json.dumps(embedding.vector),
                )
                for embedding in submitted
            ],
        )
    return len(submitted)


def save_parent_sections(
    db_path: str | Path,
    sections: Iterable[ParentSection],
) -> int:
    """Save parent sections and return the number of submitted sections."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(sections)

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO parent_sections (
                id,
                document_id,
                title,
                text,
                section_index,
                start_char,
                end_char,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    section.id,
                    section.document_id,
                    section.title,
                    section.text,
                    section.section_index,
                    section.start_char,
                    section.end_char,
                )
                for section in submitted
            ],
        )

    return len(submitted)


def save_evidence_spans(
    db_path: str | Path,
    spans: Iterable[EvidenceSpan],
) -> int:
    """Save evidence spans and return the number of submitted spans."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(spans)

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO evidence_spans (
                id,
                document_id,
                parent_section_id,
                chunk_id,
                text,
                span_index,
                start_char,
                end_char,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    span.id,
                    span.document_id,
                    span.parent_section_id,
                    span.chunk_id,
                    span.text,
                    span.span_index,
                    span.start_char,
                    span.end_char,
                )
                for span in submitted
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
            SELECT id, document_id, text, chunk_index, start_char, end_char,
                   parent_section_id
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
            SELECT id, document_id, text, chunk_index, start_char, end_char,
                   parent_section_id
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index, id
            """,
            (document_id,),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def list_parent_sections(db_path: str | Path) -> list[ParentSection]:
    """Return every stored parent section in stable document/section order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, title, text, section_index, start_char, end_char
            FROM parent_sections
            ORDER BY document_id, section_index, id
            """
        ).fetchall()
    return [_parent_section_from_row(row) for row in rows]


def list_evidence_spans(db_path: str | Path) -> list[EvidenceSpan]:
    """Return every stored evidence span in stable source order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, parent_section_id, chunk_id, text,
                   span_index, start_char, end_char
            FROM evidence_spans
            ORDER BY document_id, parent_section_id, span_index, id
            """
        ).fetchall()
    return [_evidence_span_from_row(row) for row in rows]


def list_chunk_embeddings(
    db_path: str | Path,
    *,
    model: str | None = None,
) -> list[ChunkEmbedding]:
    """Return chunk embeddings in stable chunk order."""
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        if model:
            rows = connection.execute(
                """
                SELECT id, chunk_id, model, dimensions, vector_json
                FROM chunk_embeddings
                WHERE model = ?
                ORDER BY chunk_id, id
                """,
                (model,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, chunk_id, model, dimensions, vector_json
                FROM chunk_embeddings
                ORDER BY model, chunk_id, id
                """
            ).fetchall()
    return [_chunk_embedding_from_row(row) for row in rows]


def clear_chunks(db_path: str | Path) -> int:
    """Delete all stored chunks and return number of removed rows."""
    database_path = Path(db_path)
    if not database_path.exists():
        return 0

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM chunk_embeddings")
    return count


def clear_evidence(db_path: str | Path) -> dict[str, int]:
    """Delete hierarchical evidence rows and return removed row counts."""
    database_path = Path(db_path)
    if not database_path.exists():
        return {"parent_sections": 0, "chunks": 0, "evidence_spans": 0}

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        section_count = connection.execute(
            "SELECT COUNT(*) FROM parent_sections"
        ).fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        span_count = connection.execute(
            "SELECT COUNT(*) FROM evidence_spans"
        ).fetchone()[0]
        connection.execute("DELETE FROM evidence_spans")
        connection.execute("DELETE FROM chunk_embeddings")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM parent_sections")
    return {
        "parent_sections": section_count,
        "chunks": chunk_count,
        "evidence_spans": span_count,
    }


def clear_chunk_embeddings(
    db_path: str | Path,
    *,
    model: str | None = None,
) -> int:
    """Delete chunk embeddings and return number of removed rows."""
    database_path = Path(db_path)
    if not database_path.exists():
        return 0

    with _connect(database_path) as connection:
        _ensure_schema(connection)
        if model:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM chunk_embeddings
                WHERE model = ?
                """,
                (model,),
            ).fetchone()[0]
            connection.execute(
                "DELETE FROM chunk_embeddings WHERE model = ?",
                (model,),
            )
            return count

        count = connection.execute(
            "SELECT COUNT(*) FROM chunk_embeddings"
        ).fetchone()[0]
        connection.execute("DELETE FROM chunk_embeddings")
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
        CREATE TABLE IF NOT EXISTS parent_sections (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            section_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            imported_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
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
            parent_section_id TEXT,
            imported_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (parent_section_id) REFERENCES parent_sections (id)
        )
        """
    )
    _ensure_column(connection, "chunks", "parent_section_id", "TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            UNIQUE (chunk_id, model),
            FOREIGN KEY (chunk_id) REFERENCES chunks (id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_spans (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            parent_section_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            text TEXT NOT NULL,
            span_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            imported_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (parent_section_id) REFERENCES parent_sections (id),
            FOREIGN KEY (chunk_id) REFERENCES chunks (id)
        )
        """
    )


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in rows}
    if column_name not in existing:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
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
        parent_section_id=row[6],
    )


def _parent_section_from_row(row: sqlite3.Row | tuple[str, ...]) -> ParentSection:
    return ParentSection(
        id=row[0],
        document_id=row[1],
        title=row[2],
        text=row[3],
        section_index=row[4],
        start_char=row[5],
        end_char=row[6],
    )


def _evidence_span_from_row(row: sqlite3.Row | tuple[str, ...]) -> EvidenceSpan:
    return EvidenceSpan(
        id=row[0],
        document_id=row[1],
        parent_section_id=row[2],
        chunk_id=row[3],
        text=row[4],
        span_index=row[5],
        start_char=row[6],
        end_char=row[7],
    )


def _chunk_embedding_from_row(
    row: sqlite3.Row | tuple[str, str, str, int, str],
) -> ChunkEmbedding:
    return ChunkEmbedding(
        id=row[0],
        chunk_id=row[1],
        model=row[2],
        dimensions=row[3],
        vector=list(json.loads(row[4])),
    )
