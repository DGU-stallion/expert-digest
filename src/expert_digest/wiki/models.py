"""Wiki domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    title: str
    url: str | None = None
    evidence_span_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WikiPage:
    path: str
    page_type: str
    title: str
    body: str
    sources: list[SourceRef] = field(default_factory=list)
    confidence: str = "medium"
    updated_at: str | None = None


@dataclass(frozen=True)
class SourceAnalysis:
    source_id: str
    source_title: str
    author: str
    url: str | None
    summary: str
    key_claims: list[str]
    concepts: list[str]
    topics: list[str]
    evidence_span_ids: list[str]
    confidence: str
