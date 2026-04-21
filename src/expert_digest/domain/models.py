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
class ParentSection:
    """A larger context section derived from a source document."""

    id: str
    document_id: str
    title: str
    text: str
    section_index: int
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        title: str,
        text: str,
        section_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> ParentSection:
        payload = {
            "document_id": document_id,
            "title": title,
            "text": text,
            "section_index": section_index,
            "start_char": start_char,
            "end_char": end_char,
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
    parent_section_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        text: str,
        chunk_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
        parent_section_id: str | None = None,
    ) -> Chunk:
        payload = {
            "document_id": document_id,
            "text": text,
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "parent_section_id": parent_section_id,
        }
        return cls(id=_stable_hash(payload), **payload)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.parent_section_id is None:
            data.pop("parent_section_id")
        return data


@dataclass(frozen=True)
class EvidenceSpan:
    """A minimum source-backed citation span used by wiki claims."""

    id: str
    document_id: str
    parent_section_id: str
    chunk_id: str
    text: str
    span_index: int
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        parent_section_id: str,
        chunk_id: str,
        text: str,
        span_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> EvidenceSpan:
        payload = {
            "document_id": document_id,
            "parent_section_id": parent_section_id,
            "chunk_id": chunk_id,
            "text": text,
            "span_index": span_index,
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


@dataclass(frozen=True)
class Handbook:
    """A generated learning handbook for one author or author set."""

    author: str
    title: str
    markdown: str
    source_document_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
