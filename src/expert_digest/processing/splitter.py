"""Split source documents into deterministic chunks."""

from __future__ import annotations

from collections.abc import Iterable

from expert_digest.domain.models import Chunk, Document


def split_document(
    document: Document,
    max_chars: int = 1000,
    min_chars: int = 1,
) -> list[Chunk]:
    """Split one document into chunks with a character limit."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if min_chars <= 0:
        raise ValueError("min_chars must be > 0")

    chunk_texts = _split_text(document.content.strip(), max_chars=max_chars)
    chunk_texts = _merge_small_chunks(chunk_texts, min_chars=min_chars)
    return [
        Chunk.create(
            document_id=document.id,
            text=text,
            chunk_index=index,
        )
        for index, text in enumerate(chunk_texts)
    ]


def split_documents(
    documents: Iterable[Document],
    max_chars: int = 1000,
    min_chars: int = 1,
) -> list[Chunk]:
    """Split all documents and return one flat chunk list."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            split_document(
                document,
                max_chars=max_chars,
                min_chars=min_chars,
            )
        )
    return chunks


def _split_text(text: str, *, max_chars: int) -> list[str]:
    if not text:
        return []

    paragraphs = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        chunks.extend(_hard_split(paragraph, max_chars=max_chars))

    if current:
        chunks.append(current)

    return chunks


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    return [
        text[start : start + max_chars]
        for start in range(0, len(text), max_chars)
        if text[start : start + max_chars]
    ]


def _merge_small_chunks(chunks: list[str], *, min_chars: int) -> list[str]:
    if len(chunks) <= 1 or min_chars <= 1:
        return chunks

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
            continue
        merged.append(chunk)

    if len(merged) > 1 and len(merged[0]) < min_chars:
        merged[1] = f"{merged[0]}\n\n{merged[1]}".strip()
        return merged[1:]

    return merged
