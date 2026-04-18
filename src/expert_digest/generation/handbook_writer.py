"""Generate Markdown learning handbooks from local ExpertDigest data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from expert_digest.domain.models import Chunk, Document, Handbook
from expert_digest.generation.prompts import build_theme_summary_prompts
from expert_digest.knowledge.author_profile import (
    AuthorProfile,
    extract_author_profile_from_documents,
)
from expert_digest.knowledge.topic_clusterer import (
    TopicCluster,
    TopicRepresentativeDocument,
    cluster_chunks_by_embeddings,
)
from expert_digest.processing.embedder import DEFAULT_EMBEDDING_MODEL, embed_text
from expert_digest.retrieval.retriever import (
    RetrievedChunk,
    hydrate_scored_chunks,
    rank_chunk_embeddings,
)
from expert_digest.storage.sqlite_store import (
    get_documents_by_author,
    list_chunk_embeddings,
    list_chunks,
    list_documents,
)


@dataclass(frozen=True)
class ThemeDefinition:
    name: str
    question: str


@dataclass(frozen=True)
class ThemeSection:
    definition: ThemeDefinition
    summary: str
    evidence: list[RetrievedChunk]
    representative_documents: list[TopicRepresentativeDocument]


@dataclass(frozen=True)
class TopicTaxonomyRule:
    name: str
    keywords: list[str]


DEFAULT_THEME_DEFINITIONS: list[ThemeDefinition] = [
    ThemeDefinition("核心能力与方法", "作者最核心的能力和方法论是什么？"),
    ThemeDefinition("决策与复盘机制", "作者如何做决策、复盘以及纠偏？"),
    ThemeDefinition("学习与行动路径", "如果要学习这位作者内容，建议路径是什么？"),
]

DEFAULT_TOPIC_TAXONOMY: list[TopicTaxonomyRule] = [
    TopicTaxonomyRule(
        name="股票与交易",
        keywords=[
            "A股",
            "股市",
            "ETF",
            "交易",
            "仓位",
            "回撤",
            "择时",
            "估值",
            "量化",
            "市场行情",
            "牛市",
            "右侧",
            "左侧",
        ],
    ),
    TopicTaxonomyRule(
        name="宏观经济与政策",
        keywords=[
            "宏观",
            "通胀",
            "通缩",
            "汇率",
            "美元",
            "美联储",
            "财政",
            "货币",
            "利率",
            "债券",
            "关税",
            "进出口",
            "社融",
            "M2",
        ],
    ),
    TopicTaxonomyRule(
        name="房地产与资产配置",
        keywords=[
            "楼市",
            "房价",
            "地产",
            "房地产",
            "公寓",
            "住宅",
            "租售",
            "买房",
            "土拍",
        ],
    ),
    TopicTaxonomyRule(
        name="行业与公司研究",
        keywords=[
            "泡泡玛特",
            "价值投资",
            "护城河",
            "商业模式",
            "公司",
            "行业",
            "业绩",
            "预期",
            "消费",
            "科技",
            "AI",
            "能源",
        ],
    ),
    TopicTaxonomyRule(
        name="方法论与认知",
        keywords=[
            "复盘",
            "方法论",
            "认知",
            "框架",
            "证据",
            "推理",
            "风险",
            "纪律",
            "策略",
            "决策",
            "不确定性",
        ],
    ),
]


class HandbookLLMClient(Protocol):
    """Future LLM API adapter protocol for handbook generation."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


class ThemeSynthesizer(Protocol):
    """Protocol for composing one theme summary from evidence."""

    def summarize_theme(
        self,
        *,
        theme_name: str,
        question: str,
        evidence_chunks: list[RetrievedChunk],
    ) -> str: ...


class DeterministicThemeSynthesizer:
    """Rule-based fallback synthesizer with deterministic output."""

    def summarize_theme(
        self,
        *,
        theme_name: str,
        question: str,
        evidence_chunks: list[RetrievedChunk],
    ) -> str:
        if not evidence_chunks:
            return (
                f"围绕“{theme_name}”暂无足够证据，当前无法稳定回答“{question}”，"
                "建议补充相关原文后再生成手册。"
            )

        primary = _clip(evidence_chunks[0].text)
        if len(evidence_chunks) == 1:
            return f"围绕“{theme_name}”，当前最强证据显示：{primary}。"

        secondary = _clip(evidence_chunks[1].text)
        return (
            f"围绕“{theme_name}”，当前最强证据显示：{primary}。"
            f"补充证据强调：{secondary}。"
            "结论仅基于当前检索片段，后续可接入 LLM 做风格化润色。"
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "fallback_used": True,
            "error_reason": "deterministic_mode",
        }


class HybridThemeSynthesizer:
    """LLM-first synthesizer with deterministic fallback."""

    def __init__(
        self,
        *,
        llm_client: HandbookLLMClient | None = None,
        fallback: ThemeSynthesizer | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback or DeterministicThemeSynthesizer()
        self._llm_attempts = 0
        self._llm_failures = 0
        self._last_error_reason: str | None = None

    def summarize_theme(
        self,
        *,
        theme_name: str,
        question: str,
        evidence_chunks: list[RetrievedChunk],
    ) -> str:
        if self._llm_client is not None:
            self._llm_attempts += 1
            system_prompt, user_prompt = build_theme_summary_prompts(
                theme_name=theme_name,
                question=question,
                evidence_chunks=evidence_chunks,
            )
            try:
                llm_output = self._llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ).strip()
                if llm_output and not _is_low_quality_summary(llm_output):
                    return llm_output
                if llm_output:
                    self._llm_failures += 1
                    self._last_error_reason = "llm_low_quality_output"
            except Exception:
                # Fail-closed: if llm invocation fails, keep deterministic behavior.
                self._llm_failures += 1
                self._last_error_reason = "llm_generation_error"
        return self._fallback.summarize_theme(
            theme_name=theme_name,
            question=question,
            evidence_chunks=evidence_chunks,
        )

    def runtime_metadata(self) -> dict[str, object]:
        if self._llm_client is None:
            return {
                "fallback_used": True,
                "error_reason": "llm_client_unavailable",
            }
        if self._llm_failures > 0:
            return {
                "fallback_used": True,
                "error_reason": self._last_error_reason or "llm_generation_error",
            }
        return {
            "fallback_used": False,
            "error_reason": None,
        }


def build_handbook(
    *,
    db_path: str | Path,
    author: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3,
    max_themes: int = 3,
    theme_source: str = "preset",
    num_topics: int = 3,
    topic_taxonomy_path: str | Path | None = None,
    synthesizer: ThemeSynthesizer | None = None,
) -> Handbook:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if max_themes <= 0:
        raise ValueError("max_themes must be > 0")
    if num_topics <= 0:
        raise ValueError("num_topics must be > 0")
    if theme_source not in {"preset", "cluster"}:
        raise ValueError(f"unsupported theme_source: {theme_source}")

    documents = (
        get_documents_by_author(db_path, author)
        if author
        else list_documents(db_path)
    )
    if not documents:
        raise ValueError("no documents available for handbook generation")

    document_ids = {document.id for document in documents}
    chunks = [
        chunk for chunk in list_chunks(db_path) if chunk.document_id in document_ids
    ]
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    documents_by_id = {document.id: document for document in documents}

    chunk_embeddings = [
        embedding
        for embedding in list_chunk_embeddings(db_path, model=model)
        if embedding.chunk_id in chunks_by_id
    ]
    summarizer: ThemeSynthesizer = synthesizer or HybridThemeSynthesizer()
    topic_clusters_for_render: list[TopicCluster] = []

    if theme_source == "cluster":
        raw_clusters = cluster_chunks_by_embeddings(
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
            chunk_embeddings=chunk_embeddings,
            num_topics=num_topics,
            top_docs_per_topic=top_k,
        )
        topic_clusters_for_render = _merge_topic_clusters_by_taxonomy(
            topic_clusters=raw_clusters,
            taxonomy_rules=_load_topic_taxonomy(topic_taxonomy_path),
            max_topics=max_themes,
        )
        theme_sections = _build_theme_sections_from_topics(
            topic_clusters=topic_clusters_for_render,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
            max_themes=max_themes,
            top_k=top_k,
            synthesizer=summarizer,
        )
        if not theme_sections:
            theme_sections = _build_theme_sections_from_definitions(
                max_themes=max_themes,
                top_k=top_k,
                chunks=chunks,
                chunks_by_id=chunks_by_id,
                documents_by_id=documents_by_id,
                chunk_embeddings=chunk_embeddings,
                synthesizer=summarizer,
            )
    else:
        theme_sections = _build_theme_sections_from_definitions(
            max_themes=max_themes,
            top_k=top_k,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
            chunk_embeddings=chunk_embeddings,
            synthesizer=summarizer,
        )
        topic_clusters_for_render = _build_topic_clusters_from_theme_sections(
            theme_sections
        )

    author_label = author or _resolve_author_label(documents)
    profile = extract_author_profile_from_documents(documents)
    synthesis_mode = _infer_synthesis_mode(summarizer)
    markdown = _render_handbook_markdown(
        author_label=author_label,
        documents=documents,
        chunks=chunks,
        profile=profile,
        theme_sections=theme_sections,
        topic_clusters=topic_clusters_for_render,
        theme_source=theme_source,
        synthesis_mode=synthesis_mode,
    )
    return Handbook(
        author=author_label,
        title=f"{author_label}学习手册",
        markdown=markdown,
        source_document_ids=[document.id for document in documents],
    )


def _build_theme_sections_from_definitions(
    *,
    max_themes: int,
    top_k: int,
    chunks: list[Chunk],
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
    chunk_embeddings: list,
    synthesizer: ThemeSynthesizer,
) -> list[ThemeSection]:
    theme_sections: list[ThemeSection] = []
    for definition in DEFAULT_THEME_DEFINITIONS[:max_themes]:
        evidence = _collect_theme_evidence(
            question=definition.question,
            top_k=top_k,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
            chunk_embeddings=chunk_embeddings,
        )
        summary = synthesizer.summarize_theme(
            theme_name=definition.name,
            question=definition.question,
            evidence_chunks=evidence,
        )
        theme_sections.append(
            ThemeSection(
                definition=definition,
                summary=summary,
                evidence=evidence,
                representative_documents=_representative_documents_from_evidence(
                    evidence
                ),
            )
        )
    return theme_sections


def _build_theme_sections_from_topics(
    *,
    topic_clusters: list[TopicCluster],
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
    max_themes: int,
    top_k: int,
    synthesizer: ThemeSynthesizer,
) -> list[ThemeSection]:
    sections: list[ThemeSection] = []
    for topic in topic_clusters[:max_themes]:
        score_by_chunk = {
            item.supporting_chunk_id: item.score
            for item in topic.representative_documents
        }
        evidence: list[RetrievedChunk] = []
        seen_documents: set[str] = set()
        ranked_chunk_ids = sorted(
            topic.representative_chunk_ids,
            key=lambda chunk_id: score_by_chunk.get(chunk_id, 0.0),
            reverse=True,
        )
        for chunk_id in ranked_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                continue
            score = score_by_chunk.get(chunk.id, 0.0)
            if score <= 0.0 and evidence:
                # Prefer chunks that are explicitly linked to representative docs.
                continue
            if document.id in seen_documents:
                continue
            evidence.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    score=score,
                    document_id=document.id,
                    title=document.title,
                    author=document.author,
                    text=chunk.text,
                    url=document.url,
                )
            )
            seen_documents.add(document.id)
            if len(evidence) >= top_k:
                break

        if not evidence:
            continue

        question = "该主题最核心的观点与证据是什么？"
        summary = synthesizer.summarize_theme(
            theme_name=topic.label,
            question=question,
            evidence_chunks=evidence,
        )
        sections.append(
            ThemeSection(
                definition=ThemeDefinition(name=topic.label, question=question),
                summary=summary,
                evidence=evidence,
                representative_documents=topic.representative_documents,
            )
        )
    return sections


def write_handbook(handbook: Handbook, *, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(handbook.markdown, encoding="utf-8")
    return path


def _collect_theme_evidence(
    *,
    question: str,
    top_k: int,
    chunks: list[Chunk],
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
    chunk_embeddings: list,
) -> list[RetrievedChunk]:
    if chunk_embeddings:
        query_vector = embed_text(question, dim=chunk_embeddings[0].dimensions)
        ranked = rank_chunk_embeddings(
            query_vector=query_vector,
            chunk_embeddings=chunk_embeddings,
            top_k=max(top_k * 3, top_k),
        )
        hydrated = hydrate_scored_chunks(
            ranked,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
        )
    else:
        hydrated = []

    if not hydrated:
        hydrated = [
            RetrievedChunk(
                chunk_id=chunk.id,
                score=0.0,
                document_id=chunk.document_id,
                title=documents_by_id[chunk.document_id].title,
                author=documents_by_id[chunk.document_id].author,
                text=chunk.text,
                url=documents_by_id[chunk.document_id].url,
            )
            for chunk in chunks[:top_k]
        ]
    return _dedupe_evidence(hydrated)[:top_k]


def _dedupe_evidence(evidence: list[RetrievedChunk]) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item.document_id, _normalize_text(item.text))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def _render_handbook_markdown(
    *,
    author_label: str,
    documents: list[Document],
    chunks: list[Chunk],
    profile: AuthorProfile,
    theme_sections: list[ThemeSection],
    topic_clusters: list[TopicCluster],
    theme_source: str,
    synthesis_mode: str,
) -> str:
    reading_path = _build_reading_path(documents, theme_sections)
    document_ids_with_chunks = {chunk.document_id for chunk in chunks}
    docs_with_chunks = sum(
        1 for document in documents if document.id in document_ids_with_chunks
    )
    avg_chunks_per_doc = (
        len(chunks) / len(documents) if documents else 0.0
    )
    mode_text = {
        "hybrid": "本版手册由混合模式生成：优先 LLM，失败时回退确定性模板。",
        "deterministic": "本版手册由确定性模式生成：不依赖 LLM。",
    }.get(synthesis_mode, f"本版手册由 {synthesis_mode} 模式生成。")

    lines: list[str] = []
    lines.append(f"# {author_label}学习手册")
    lines.append("")
    lines.append("## 简介")
    lines.append(
        "这是一份围绕作者公开文章构建的结构化学习手册，"
        "目标是把分散原文整理为可执行的认知框架与阅读路径。"
    )
    lines.append("")
    lines.append("## 目录")
    lines.append("- [简介](#简介)")
    lines.append("- [总览](#总览)")
    lines.append("- [作者画像](#作者画像)")
    lines.append("- [主题地图](#主题地图)")
    lines.append("- [主题章节](#主题章节)")
    lines.append("- [推荐阅读路径](#推荐阅读路径)")
    lines.append("")
    lines.append("## 总览")
    lines.append(
        f"当前样本包含 {len(documents)} 篇原文，切分为 {len(chunks)} 个 chunk。"
        f"其中 {docs_with_chunks} 篇进入了主题建模，平均每篇约 {avg_chunks_per_doc:.2f} 个 chunk。"
        f"主题组织方式：{theme_source}（机器聚类后按人工规则命名与合并）。"
        f"{mode_text}"
    )
    lines.append("")
    lines.append("## 作者画像")
    lines.append(f"- 作者：{profile.author}")
    lines.append(f"- 文档数：{profile.document_count}")
    lines.append(
        "- 关注主题："
        + ("、".join(profile.focus_topics) if profile.focus_topics else "（无）")
    )
    lines.append(
        "- 高频关键词："
        + (
            "、".join(item.keyword for item in profile.keywords[:10])
            if profile.keywords
            else "（无）"
        )
    )
    lines.append(
        "- 推理模式："
        + (
            "、".join(item.pattern for item in profile.reasoning_patterns)
            if profile.reasoning_patterns
            else "（无）"
        )
    )
    lines.append("")
    lines.append("## 主题地图")
    if not topic_clusters:
        lines.append("- （未生成主题聚类）")
    for index, topic in enumerate(topic_clusters, start=1):
        lines.append(
            f"{index}. **{topic.label}**："
            f"{topic.chunk_count} 个证据片段，"
            f"{len(topic.representative_documents)} 篇代表文章"
        )
        if topic.representative_documents:
            references = "；".join(
                _format_title_link(item.title, item.url)
                for item in topic.representative_documents[:3]
            )
            lines.append(f"   代表文章：{references}")
    lines.append("")
    lines.append("## 主题章节")
    for index, section in enumerate(theme_sections, start=1):
        lines.append("")
        lines.append(f"### 主题 {index}：{section.definition.name}")
        lines.append("")
        lines.append(f"研究问题：{section.definition.question}")
        lines.append("")
        lines.append("#### 主题综述")
        lines.append(section.summary if section.summary else "（无）")
        lines.append("")
        lines.append("#### 文章池（Top）")
        if not section.representative_documents:
            lines.append("- （无）")
        else:
            for doc_index, item in enumerate(
                section.representative_documents,
                start=1,
            ):
                lines.append(
                    f"- {doc_index}. score={item.score:.4f} "
                    f"{_format_title_link(item.title, item.url)}"
                )
        lines.append("")
        lines.append("#### 观点蒸馏")
        if not section.evidence:
            lines.append("- （无）")
        else:
            for evidence_index, evidence in enumerate(section.evidence, start=1):
                lines.append(
                    f"- {evidence_index}. score={evidence.score:.4f} "
                    f"{_format_title_link(evidence.title, evidence.url)} | "
                    f"{_clip(evidence.text, limit=120)}"
                )
    lines.append("")
    lines.append("## 推荐阅读路径")
    for index, document in enumerate(reading_path, start=1):
        lines.append(
            f"{index}. {_format_title_link(document.title, document.url)} "
            f"（作者：{document.author}）"
        )
    lines.append("")
    return "\n".join(lines)


def _build_reading_path(
    documents: list[Document],
    theme_sections: list[ThemeSection],
    *,
    limit: int = 12,
) -> list[Document]:
    scores: Counter[str] = Counter()
    for section in theme_sections:
        for rank, evidence in enumerate(section.evidence, start=1):
            scores[evidence.document_id] += max(1, len(section.evidence) - rank + 1)
    ranked_docs = sorted(
        documents,
        key=lambda item: (
            -scores[item.id],
            item.created_at or "",
            item.title,
            item.id,
        ),
    )
    return ranked_docs[:limit]


def _resolve_author_label(documents: list[Document]) -> str:
    authors = sorted({document.author for document in documents})
    if len(authors) == 1:
        return authors[0]
    return "多作者"


def _infer_synthesis_mode(synthesizer: ThemeSynthesizer) -> str:
    if isinstance(synthesizer, DeterministicThemeSynthesizer):
        return "deterministic"
    if isinstance(synthesizer, HybridThemeSynthesizer):
        return "hybrid"
    return "custom"


def _format_title_link(title: str, url: str | None) -> str:
    if not url:
        return title
    return f"[{title}]({url})"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, *, limit: int = 100) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _is_low_quality_summary(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if len(normalized) < 60:
        return True
    if normalized.endswith(("：", ":", "，", ",", "、", "（", "(", "-", "—")):
        return True
    if normalized[-1] in {"的", "了", "和", "与", "及", "并", "在", "对", "是"}:
        return True
    if not any(mark in normalized for mark in ("。", "！", "？", ".", "!", "?")):
        return True
    return False


def _load_topic_taxonomy(
    topic_taxonomy_path: str | Path | None,
) -> list[TopicTaxonomyRule]:
    if topic_taxonomy_path is None:
        return DEFAULT_TOPIC_TAXONOMY
    path = Path(topic_taxonomy_path)
    if not path.exists():
        return DEFAULT_TOPIC_TAXONOMY
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_TOPIC_TAXONOMY
    rules: list[TopicTaxonomyRule] = []
    raw_rules = payload.get("rules", []) if isinstance(payload, dict) else []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        keywords_raw = item.get("keywords", [])
        if not name or not isinstance(keywords_raw, list):
            continue
        keywords = [
            str(keyword).strip()
            for keyword in keywords_raw
            if str(keyword).strip()
        ]
        if not keywords:
            continue
        rules.append(TopicTaxonomyRule(name=name, keywords=keywords))
    return rules or DEFAULT_TOPIC_TAXONOMY


def _merge_topic_clusters_by_taxonomy(
    *,
    topic_clusters: list[TopicCluster],
    taxonomy_rules: list[TopicTaxonomyRule],
    max_topics: int,
) -> list[TopicCluster]:
    if not topic_clusters:
        return []
    merged: dict[str, TopicCluster] = {}
    for index, topic in enumerate(topic_clusters, start=1):
        target_name = _match_topic_taxonomy_name(
            topic=topic,
            taxonomy_rules=taxonomy_rules,
        )
        merged_topic = merged.get(target_name)
        if merged_topic is None:
            merged[target_name] = TopicCluster(
                topic_id=f"topic-merged-{index}",
                label=target_name,
                chunk_count=topic.chunk_count,
                representative_chunk_ids=list(topic.representative_chunk_ids),
                representative_documents=list(topic.representative_documents),
            )
            continue
        merged[merged_topic.label] = _merge_two_topics(merged_topic, topic)

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -item.chunk_count,
            item.label,
        ),
    )
    normalized: list[TopicCluster] = []
    for index, topic in enumerate(ranked[:max_topics], start=1):
        normalized.append(
            TopicCluster(
                topic_id=f"topic-{index}",
                label=topic.label,
                chunk_count=topic.chunk_count,
                representative_chunk_ids=topic.representative_chunk_ids[:8],
                representative_documents=topic.representative_documents[:6],
            )
        )
    return normalized


def _match_topic_taxonomy_name(
    *,
    topic: TopicCluster,
    taxonomy_rules: list[TopicTaxonomyRule],
) -> str:
    haystack_parts = [topic.label]
    haystack_parts.extend(item.title for item in topic.representative_documents)
    haystack = " ".join(haystack_parts).lower()
    best_name = topic.label
    best_score = 0
    for rule in taxonomy_rules:
        score = 0
        for keyword in rule.keywords:
            if keyword.lower() in haystack:
                score += 1
        if score > best_score:
            best_name = rule.name
            best_score = score
    return best_name


def _merge_two_topics(left: TopicCluster, right: TopicCluster) -> TopicCluster:
    chunk_ids = list(
        dict.fromkeys(
            left.representative_chunk_ids + right.representative_chunk_ids
        )
    )
    doc_map: dict[str, TopicRepresentativeDocument] = {}
    for item in left.representative_documents + right.representative_documents:
        current = doc_map.get(item.document_id)
        if current is None or item.score > current.score:
            doc_map[item.document_id] = item
    docs = sorted(doc_map.values(), key=lambda item: item.score, reverse=True)
    return TopicCluster(
        topic_id=left.topic_id,
        label=left.label,
        chunk_count=left.chunk_count + right.chunk_count,
        representative_chunk_ids=chunk_ids,
        representative_documents=docs,
    )


def _representative_documents_from_evidence(
    evidence: list[RetrievedChunk],
) -> list[TopicRepresentativeDocument]:
    by_doc: dict[str, TopicRepresentativeDocument] = {}
    for item in evidence:
        current = by_doc.get(item.document_id)
        candidate = TopicRepresentativeDocument(
            document_id=item.document_id,
            title=item.title,
            author=item.author,
            url=item.url,
            score=item.score,
            supporting_chunk_id=item.chunk_id,
        )
        if current is None or candidate.score > current.score:
            by_doc[item.document_id] = candidate
    return sorted(by_doc.values(), key=lambda item: item.score, reverse=True)


def _build_topic_clusters_from_theme_sections(
    theme_sections: list[ThemeSection],
) -> list[TopicCluster]:
    topics: list[TopicCluster] = []
    for index, section in enumerate(theme_sections, start=1):
        topics.append(
            TopicCluster(
                topic_id=f"topic-{index}",
                label=section.definition.name,
                chunk_count=len(section.evidence),
                representative_chunk_ids=[item.chunk_id for item in section.evidence],
                representative_documents=section.representative_documents,
            )
        )
    return topics
