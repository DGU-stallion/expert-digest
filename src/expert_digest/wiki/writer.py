"""Write source analysis results into a Markdown wiki vault."""

from __future__ import annotations

import re
from dataclasses import dataclass

from expert_digest.domain.models import EvidenceSpan
from expert_digest.wiki.models import SourceAnalysis, SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault

_MAX_CONCEPT_CLAIMS = 12
_MAX_TOPIC_SUMMARIES = 8
_MAX_TOPIC_CONCEPTS = 12
_MAX_PAGE_SOURCES = 24


@dataclass(frozen=True)
class WikiWriteResult:
    sources: list[str]
    concepts: list[str]
    topics: list[str]


def write_analysis_to_vault(
    *,
    vault: WikiVault,
    analysis: SourceAnalysis,
    evidence_spans: list[EvidenceSpan],
) -> dict[str, list[str]]:
    source_path = f"sources/{analysis.source_id}.md"
    source_ref = SourceRef(
        source_id=analysis.source_id,
        title=analysis.source_title,
        url=analysis.url,
        evidence_span_ids=analysis.evidence_span_ids,
    )
    evidence_lookup = {span.id: span for span in evidence_spans}
    source_body = _render_source_body(
        analysis=analysis,
        evidence_lookup=evidence_lookup,
    )
    vault.write_page(
        WikiPage(
            path=source_path,
            page_type="source",
            title=analysis.source_title,
            body=source_body,
            sources=[source_ref],
            confidence=analysis.confidence,
        )
    )

    concept_paths = []
    for concept in analysis.concepts[:8]:
        path = f"concepts/{_slug(concept)}.md"
        concept_paths.append(path)
        existing = _read_existing_page(vault=vault, path=path)
        vault.write_page(
            WikiPage(
                path=path,
                page_type="concept",
                title=concept,
                body=_render_concept_body(
                    concept=concept,
                    analysis=analysis,
                    existing_body=existing.body if existing else None,
                ),
                sources=_merge_sources(
                    existing_sources=existing.sources if existing else [],
                    incoming_source=source_ref,
                ),
                confidence=analysis.confidence,
            )
        )

    topic_paths = []
    for topic in analysis.topics[:4]:
        path = f"topics/{_slug(topic)}.md"
        topic_paths.append(path)
        existing = _read_existing_page(vault=vault, path=path)
        vault.write_page(
            WikiPage(
                path=path,
                page_type="topic",
                title=topic,
                body=_render_topic_body(
                    topic=topic,
                    analysis=analysis,
                    existing_body=existing.body if existing else None,
                ),
                sources=_merge_sources(
                    existing_sources=existing.sources if existing else [],
                    incoming_source=source_ref,
                ),
                confidence=analysis.confidence,
            )
        )

    _append_index(
        vault=vault,
        analysis=analysis,
        source_path=source_path,
        concept_paths=concept_paths,
        topic_paths=topic_paths,
    )
    vault.append_log(
        f"- ingested source `{analysis.source_id}`: {analysis.source_title}"
    )
    return {
        "sources": [source_path],
        "concepts": concept_paths,
        "topics": topic_paths,
    }


def _render_source_body(
    *,
    analysis: SourceAnalysis,
    evidence_lookup: dict[str, EvidenceSpan],
) -> str:
    lines = [f"# {analysis.source_title}", "", "## 摘要", "", analysis.summary, ""]
    lines.extend(["## 核心判断", ""])
    for claim in analysis.key_claims:
        lines.append(f"- {claim}")
    lines.extend(["", "## 证据片段", ""])
    for span_id in analysis.evidence_span_ids:
        span = evidence_lookup.get(span_id)
        if span is not None:
            lines.append(f"- `{span.id}` {span.text}")
    return "\n".join(lines).rstrip()


def _render_concept_body(
    *,
    concept: str,
    analysis: SourceAnalysis,
    existing_body: str | None = None,
) -> str:
    existing_sources = (
        _section_bullets(existing_body, "## 来源") if existing_body else []
    )
    existing_claims = (
        _section_bullets(existing_body, "## 相关判断") if existing_body else []
    )
    sources = _dedupe_keep_order(existing_sources + [f"[[{analysis.source_title}]]"])[
        -_MAX_PAGE_SOURCES:
    ]
    claims = _dedupe_keep_order(existing_claims + analysis.key_claims[:3])[
        -_MAX_CONCEPT_CLAIMS:
    ]
    source_lines = "\n".join(f"- {item}" for item in sources) or "- 暂无来源"
    claim_lines = "\n".join(f"- {claim}" for claim in claims) or "- 暂无核心判断"
    return (
        f"# {concept}\n\n"
        f"## 来源\n\n{source_lines}\n\n"
        f"## 相关判断\n\n{claim_lines}\n"
    ).rstrip()


def _render_topic_body(
    *,
    topic: str,
    analysis: SourceAnalysis,
    existing_body: str | None = None,
) -> str:
    existing_sources = (
        _section_bullets(existing_body, "## 来源") if existing_body else []
    )
    existing_summaries = (
        _section_bullets(existing_body, "## 主题摘要") if existing_body else []
    )
    existing_concepts = (
        _section_bullets(existing_body, "## 相关概念") if existing_body else []
    )

    sources = _dedupe_keep_order(existing_sources + [f"[[{analysis.source_title}]]"])[
        -_MAX_PAGE_SOURCES:
    ]
    summaries = _dedupe_keep_order(existing_summaries + [analysis.summary])[
        -_MAX_TOPIC_SUMMARIES:
    ]
    concepts = _dedupe_keep_order(
        existing_concepts + [f"[[{concept}]]" for concept in analysis.concepts[:6]]
    )[-_MAX_TOPIC_CONCEPTS:]

    source_lines = "\n".join(f"- {item}" for item in sources) or "- 暂无来源"
    summary_lines = "\n".join(f"- {item}" for item in summaries) or "- 暂无摘要"
    concept_lines = "\n".join(f"- {item}" for item in concepts) or "- 暂无"
    return (
        f"# {topic}\n\n"
        f"## 来源\n\n{source_lines}\n\n"
        f"## 主题摘要\n\n{summary_lines}\n\n"
        f"## 相关概念\n\n{concept_lines}\n"
    ).rstrip()


def _append_index(
    *,
    vault: WikiVault,
    analysis: SourceAnalysis,
    source_path: str,
    concept_paths: list[str],
    topic_paths: list[str],
) -> None:
    index_path = vault.root / "index.md"
    text = (
        index_path.read_text(encoding="utf-8") if index_path.exists() else "# Index\n"
    )
    additions = [
        "",
        f"## {analysis.source_title}",
        "",
        f"- Source: [{analysis.source_title}]({source_path})",
    ]
    for path in topic_paths:
        title = path.removeprefix("topics/").removesuffix(".md")
        additions.append(f"- Topic: [{title}]({path})")
    for path in concept_paths[:5]:
        title = path.removeprefix("concepts/").removesuffix(".md")
        additions.append(f"- Concept: [{title}]({path})")
    index_path.write_text(
        text.rstrip() + "\n" + "\n".join(additions) + "\n",
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-")
    return compact.lower() or "untitled"


def _read_existing_page(*, vault: WikiVault, path: str) -> WikiPage | None:
    page_path = vault.root / path
    if not page_path.exists():
        return None
    return vault.read_page(path)


def _merge_sources(
    *,
    existing_sources: list[SourceRef],
    incoming_source: SourceRef,
) -> list[SourceRef]:
    merged = list(existing_sources)
    if any(item.source_id == incoming_source.source_id for item in merged):
        return merged[-_MAX_PAGE_SOURCES:]
    merged.append(incoming_source)
    return merged[-_MAX_PAGE_SOURCES:]


def _section_bullets(body: str, header: str) -> list[str]:
    lines = body.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == header:
            start = index + 1
            break
    if start is None:
        return []

    items: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("- "):
            item = stripped.removeprefix("- ").strip()
            if item:
                items.append(item)
    return items


def _dedupe_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
