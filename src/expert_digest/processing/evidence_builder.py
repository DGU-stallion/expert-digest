"""Build hierarchical source evidence for ExpertDigest 2.0 wiki ingest."""

from __future__ import annotations

from dataclasses import dataclass

from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection


@dataclass(frozen=True)
class DocumentEvidence:
    document: Document
    parent_sections: list[ParentSection]
    chunks: list[Chunk]
    evidence_spans: list[EvidenceSpan]


def build_document_evidence(
    document: Document,
    *,
    parent_max_chars: int = 2400,
    child_max_chars: int = 700,
    child_min_chars: int = 80,
    span_max_chars: int = 220,
) -> DocumentEvidence:
    if parent_max_chars <= 0:
        raise ValueError("parent_max_chars must be > 0")
    if child_max_chars <= 0:
        raise ValueError("child_max_chars must be > 0")
    if child_min_chars <= 0:
        raise ValueError("child_min_chars must be > 0")
    if span_max_chars <= 0:
        raise ValueError("span_max_chars must be > 0")

    parent_sections = _build_parent_sections(
        document,
        parent_max_chars=parent_max_chars,
    )
    chunks: list[Chunk] = []
    spans: list[EvidenceSpan] = []
    chunk_index = 0

    for section in parent_sections:
        child_texts = _split_by_paragraphs(
            section.text,
            max_chars=child_max_chars,
            min_chars=child_min_chars,
            merge_small=False,
        )
        for child_text in child_texts:
            start = document.content.find(child_text)
            end = start + len(child_text) if start >= 0 else None
            chunk = Chunk.create(
                document_id=document.id,
                parent_section_id=section.id,
                text=child_text,
                chunk_index=chunk_index,
                start_char=start if start >= 0 else None,
                end_char=end,
            )
            chunks.append(chunk)
            for span_index, span_text in enumerate(
                _split_sentences(child_text, max_chars=span_max_chars)
            ):
                span_start = document.content.find(span_text)
                span_end = span_start + len(span_text) if span_start >= 0 else None
                spans.append(
                    EvidenceSpan.create(
                        document_id=document.id,
                        parent_section_id=section.id,
                        chunk_id=chunk.id,
                        text=span_text,
                        span_index=span_index,
                        start_char=span_start if span_start >= 0 else None,
                        end_char=span_end,
                    )
                )
            chunk_index += 1

    return DocumentEvidence(
        document=document,
        parent_sections=parent_sections,
        chunks=chunks,
        evidence_spans=spans,
    )


def _build_parent_sections(
    document: Document,
    *,
    parent_max_chars: int,
) -> list[ParentSection]:
    blocks = _split_markdown_heading_blocks(document.content)
    if not blocks:
        blocks = [(document.title, document.content.strip())]

    sections: list[ParentSection] = []
    section_index = 0
    for title, text in blocks:
        for section_text in _split_by_paragraphs(
            text,
            max_chars=parent_max_chars,
            min_chars=1,
        ):
            start = document.content.find(section_text)
            end = start + len(section_text) if start >= 0 else None
            sections.append(
                ParentSection.create(
                    document_id=document.id,
                    title=title or document.title,
                    text=section_text,
                    section_index=section_index,
                    start_char=start if start >= 0 else None,
                    end_char=end,
                )
            )
            section_index += 1
    return sections


def _split_markdown_heading_blocks(text: str) -> list[tuple[str, str]]:
    current_title = ""
    current_lines: list[str] = []
    blocks: list[tuple[str, str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    blocks.append((current_title, body))
                current_lines = []
            current_title = stripped.lstrip("#").strip()
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append((current_title, body))
    return blocks


def _split_by_paragraphs(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    merge_small: bool = True,
) -> list[str]:
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    if not merge_small:
        return [
            piece
            for paragraph in paragraphs
            for piece in _hard_split(paragraph, max_chars=max_chars)
        ]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    return [
        text[start : start + max_chars]
        for start in range(0, len(text), max_chars)
        if text[start : start + max_chars]
    ]


def _split_sentences(text: str, *, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in text.replace("\n", " "):
        current += char
        if char in "。！？.!?" or len(current) >= max_chars:
            stripped = current.strip()
            if stripped:
                pieces.append(stripped)
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return pieces
