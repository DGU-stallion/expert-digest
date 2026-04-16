"""Core domain models used across the local data pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Document:
    """A source article imported into the local knowledge base."""

    id: str
    author: str
    title: str
    content: str
    source: str
    url: str | None = None
    created_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        author: str,
        title: str,
        content: str,
        source: str,
        url: str | None = None,
        created_at: str | None = None,
    ) -> Document:
        payload = {
            "author": author,
            "title": title,
            "content": content,
            "source": source,
            "url": url,
            "created_at": created_at,
        }
        return cls(id=_stable_hash(payload), **payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    """A text chunk derived from a source document."""

    id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        text: str,
        chunk_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> Chunk:
        payload = {
            "document_id": document_id,
            "text": text,
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
        }
        return cls(id=_stable_hash(payload), **payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkEmbedding:
    """A deterministic embedding vector for one chunk."""

    id: str
    chunk_id: str
    model: str
    vector: list[float]
    dimensions: int

    @classmethod
    def create(
        cls,
        *,
        chunk_id: str,
        model: str,
        vector: list[float],
    ) -> ChunkEmbedding:
        rounded = [round(value, 8) for value in vector]
        payload = {
            "chunk_id": chunk_id,
            "model": model,
            "vector": rounded,
        }
        return cls(
            id=_stable_hash(payload),
            chunk_id=chunk_id,
            model=model,
            vector=rounded,
            dimensions=len(rounded),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
