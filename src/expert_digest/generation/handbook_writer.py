"""Generate Markdown learning handbooks from local ExpertDigest data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from expert_digest.domain.models import Chunk, Document, Handbook
from expert_digest.generation.prompts import build_theme_summary_prompts
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


DEFAULT_THEME_DEFINITIONS: list[ThemeDefinition] = [
    ThemeDefinition("核心能力与方法", "作者最核心的能力和方法论是什么？"),
    ThemeDefinition("决策与复盘机制", "作者如何做决策、复盘以及纠偏？"),
    ThemeDefinition("学习与行动路径", "如果要学习这位作者内容，建议路径是什么？"),
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
                if llm_output:
                    return llm_output
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
    synthesizer: ThemeSynthesizer | None = None,
) -> Handbook:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if max_themes <= 0:
        raise ValueError("max_themes must be > 0")

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
        summary = summarizer.summarize_theme(
            theme_name=definition.name,
            question=definition.question,
            evidence_chunks=evidence,
        )
        theme_sections.append(
            ThemeSection(
                definition=definition,
                summary=summary,
                evidence=evidence,
            )
        )

    author_label = author or _resolve_author_label(documents)
    markdown = _render_handbook_markdown(
        author_label=author_label,
        documents=documents,
        chunks=chunks,
        theme_sections=theme_sections,
    )
    return Handbook(
        author=author_label,
        title=f"{author_label}学习手册",
        markdown=markdown,
        source_document_ids=[document.id for document in documents],
    )


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
    theme_sections: list[ThemeSection],
) -> str:
    reading_path = _build_reading_path(documents, theme_sections)

    lines: list[str] = []
    lines.append(f"# {author_label}学习手册")
    lines.append("")
    lines.append("## 目录")
    lines.append("- [专家内容总览](#专家内容总览)")
    lines.append("- [核心主题初稿](#核心主题初稿)")
    lines.append("- [每个主题的核心观点](#每个主题的核心观点)")
    lines.append("- [推荐阅读路径](#推荐阅读路径)")
    lines.append("- [原文索引](#原文索引)")
    lines.append("")
    lines.append("## 专家内容总览")
    lines.append(
        f"当前样本包含 {len(documents)} 篇原文，切分为 {len(chunks)} 个 chunk。"
        "本版手册由混合模式生成：优先 LLM（可选），失败时回退确定性模板。"
    )
    lines.append("")
    lines.append("## 核心主题初稿")
    for index, section in enumerate(theme_sections, start=1):
        lines.append(
            f"{index}. **{section.definition.name}**：{section.definition.question}"
        )
    lines.append("")
    lines.append("## 每个主题的核心观点")
    for index, section in enumerate(theme_sections, start=1):
        lines.append("")
        lines.append(f"### 主题 {index}：{section.definition.name}")
        lines.append(f"问题：{section.definition.question}")
        lines.append("")
        lines.append(section.summary)
        lines.append("")
        lines.append("依据：")
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
    lines.append("## 原文索引")
    for document in documents:
        lines.append(
            f"- {_format_title_link(document.title, document.url)} | "
            f"author={document.author} | id={document.id}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_reading_path(
    documents: list[Document],
    theme_sections: list[ThemeSection],
    *,
    limit: int = 8,
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
