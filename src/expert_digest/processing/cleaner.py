"""Deterministic text cleanup helpers for chunk building."""

from __future__ import annotations

import html
import re

from expert_digest.domain.models import Document

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Normalize raw article text into chunk-friendly plain text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = html.unescape(normalized)
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"(?i)<br\s*/?>", "\n", normalized)
    normalized = re.sub(r"(?i)</p\s*>", "\n\n", normalized)
    normalized = re.sub(r"(?i)<p[^>]*>", "", normalized)
    normalized = _HTML_TAG_RE.sub("", normalized)
    normalized = _MARKDOWN_LINK_RE.sub(r"\1", normalized)

    lines = [
        _MULTI_SPACE_RE.sub(" ", line).strip()
        for line in normalized.split("\n")
    ]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def clean_document(document: Document) -> Document:
    """Return a copy of the document with cleaned content, preserving id."""
    return Document(
        id=document.id,
        author=document.author,
        title=document.title,
        content=clean_text(document.content),
        source=document.source,
        url=document.url,
        created_at=document.created_at,
    )
