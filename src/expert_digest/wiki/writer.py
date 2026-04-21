"""Write source analysis results into a Markdown wiki vault."""

from __future__ import annotations

import re
from dataclasses import dataclass

from expert_digest.domain.models import EvidenceSpan
from expert_digest.wiki.models import SourceAnalysis, SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault


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
        vault.write_page(
            WikiPage(
                path=path,
                page_type="concept",
                title=concept,
                body=_render_concept_body(concept=concept, analysis=analysis),
                sources=[source_ref],
                confidence=analysis.confidence,
            )
        )

    topic_paths = []
    for topic in analysis.topics[:4]:
        path = f"topics/{_slug(topic)}.md"
        topic_paths.append(path)
        vault.write_page(
            WikiPage(
                path=path,
                page_type="topic",
                title=topic,
                body=_render_topic_body(topic=topic, analysis=analysis),
                sources=[source_ref],
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
    vault.append_log(f"- ingested source `{analysis.source_id}`: {analysis.source_title}")
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


def _render_concept_body(*, concept: str, analysis: SourceAnalysis) -> str:
    claims = "\n".join(f"- {claim}" for claim in analysis.key_claims[:3]) or "- 暂无核心判断"
    return (
        f"# {concept}\n\n"
        f"## 来源\n\n- [[{analysis.source_title}]]\n\n"
        f"## 相关判断\n\n{claims}\n"
    ).rstrip()


def _render_topic_body(*, topic: str, analysis: SourceAnalysis) -> str:
    concepts = "、".join(f"[[{concept}]]" for concept in analysis.concepts[:6])
    return (
        f"# {topic}\n\n"
        f"## 主题摘要\n\n{analysis.summary}\n\n"
        f"## 相关概念\n\n{concepts if concepts else '暂无'}\n"
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
    text = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Index\n"
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
