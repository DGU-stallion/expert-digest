"""Small YAML-like frontmatter renderer/parser for wiki pages."""

from __future__ import annotations

from expert_digest.wiki.models import SourceRef, WikiPage


def render_frontmatter(page: WikiPage) -> str:
    lines = [
        "---",
        f"type: {page.page_type}",
        f"title: {page.title}",
        f"confidence: {page.confidence}",
    ]
    if page.updated_at:
        lines.append(f"updated_at: {page.updated_at}")
    lines.append("sources:")
    if not page.sources:
        lines.append("  []")
    else:
        for source in page.sources:
            lines.append(f"  - source_id: {source.source_id}")
            lines.append(f"    title: {source.title}")
            if source.url:
                lines.append(f"    url: {source.url}")
            if source.evidence_span_ids:
                joined = ", ".join(source.evidence_span_ids)
                lines.append(f"    evidence_span_ids: [{joined}]")
    lines.append("---")
    lines.append("")
    lines.append(page.body.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def parse_frontmatter(text: str, *, path: str = "") -> WikiPage:
    if not text.startswith("---\n"):
        return _page_without_frontmatter(text=text, path=path)

    marker = text.find("\n---", 4)
    if marker < 0:
        return _page_without_frontmatter(text=text, path=path)

    raw_meta = text[4:marker].strip().splitlines()
    body = text[marker + 4 :].lstrip("\n")
    page_type = "unknown"
    title = ""
    confidence = "medium"
    updated_at: str | None = None
    sources: list[SourceRef] = []
    current: dict[str, object] | None = None

    for raw_line in raw_meta:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped == "sources:" or stripped == "[]":
            continue
        if stripped.startswith("- source_id:"):
            if current:
                sources.append(_source_from_mapping(current))
            current = {"source_id": stripped.split(":", 1)[1].strip()}
            continue
        if current is not None and line.startswith("    ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip()
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key == "type":
            page_type = value.strip()
        elif key == "title":
            title = value.strip()
        elif key == "confidence":
            confidence = value.strip()
        elif key == "updated_at":
            updated_at = value.strip()

    if current:
        sources.append(_source_from_mapping(current))

    return WikiPage(
        path=path,
        page_type=page_type,
        title=title or _extract_h1(body) or path,
        body=body.rstrip(),
        sources=sources,
        confidence=confidence,
        updated_at=updated_at,
    )


def _source_from_mapping(mapping: dict[str, object]) -> SourceRef:
    raw_span_ids = str(mapping.get("evidence_span_ids", "")).strip()
    evidence_span_ids: list[str] = []
    if raw_span_ids.startswith("[") and raw_span_ids.endswith("]"):
        inner = raw_span_ids[1:-1].strip()
        evidence_span_ids = [item.strip() for item in inner.split(",") if item.strip()]
    return SourceRef(
        source_id=str(mapping.get("source_id", "")).strip(),
        title=str(mapping.get("title", "")).strip(),
        url=str(mapping.get("url")).strip() if mapping.get("url") else None,
        evidence_span_ids=evidence_span_ids,
    )


def _page_without_frontmatter(*, text: str, path: str) -> WikiPage:
    return WikiPage(
        path=path,
        page_type="unknown",
        title=_extract_h1(text) or path,
        body=text.rstrip(),
        sources=[],
        confidence="low",
    )


def _extract_h1(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return None
