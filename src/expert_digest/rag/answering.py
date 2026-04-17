"""Build deterministic structured answers from retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass

from expert_digest.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class AnswerEvidence:
    chunk_id: str
    score: float
    title: str
    author: str
    snippet: str
    url: str | None = None


@dataclass(frozen=True)
class StructuredAnswer:
    answer: str
    evidence: list[AnswerEvidence]
    recommended_original: list[str]
    uncertainty: str
    refused: bool


def build_structured_answer(
    *,
    question: str,
    evidence_chunks: list[RetrievedChunk],
    max_evidence: int = 3,
    min_top_score: float = 0.05,
    min_avg_score: float = 0.03,
) -> StructuredAnswer:
    selected_chunks = _select_evidence_chunks(
        evidence_chunks,
        max_evidence=max_evidence,
    )
    if not selected_chunks:
        return StructuredAnswer(
            answer="抱歉，我无法基于当前知识库回答这个问题。",
            evidence=[],
            recommended_original=[],
            uncertainty="未检索到相关证据，结论风险过高。",
            refused=True,
        )

    evidence = [
        AnswerEvidence(
            chunk_id=item.chunk_id,
            score=item.score,
            title=item.title,
            author=item.author,
            snippet=_clip_text(item.text),
            url=item.url,
        )
        for item in selected_chunks
    ]
    top = evidence[0]
    avg_score = sum(item.score for item in evidence) / len(evidence)

    if top.score < min_top_score or avg_score < min_avg_score:
        return StructuredAnswer(
            answer="抱歉，我无法基于当前知识库回答这个问题。",
            evidence=evidence,
            recommended_original=[_reference_label(item) for item in evidence],
            uncertainty=(
                "证据置信度不足："
                f"top={top.score:.4f} (阈值 {min_top_score:.4f})，"
                f"avg={avg_score:.4f} (阈值 {min_avg_score:.4f})。"
            ),
            refused=True,
        )

    answer = (
        f"针对问题“{question}”，当前最相关证据指出：{top.snippet}"
        " 这个回答仅基于已检索到的本地资料。"
    )
    recommended_original = [_reference_label(item) for item in evidence]
    uncertainty = (
        f"仅检索到 {len(evidence)} 条证据，最高相似度为 {top.score:.4f}；"
        "如需更高置信度，请补充数据或扩大检索范围。"
    )
    return StructuredAnswer(
        answer=answer,
        evidence=evidence,
        recommended_original=recommended_original,
        uncertainty=uncertainty,
        refused=False,
    )


def _select_evidence_chunks(
    evidence_chunks: list[RetrievedChunk],
    *,
    max_evidence: int,
) -> list[RetrievedChunk]:
    if max_evidence <= 0:
        raise ValueError("max_evidence must be > 0")

    ranked = sorted(evidence_chunks, key=lambda item: item.score, reverse=True)
    deduplicated: list[RetrievedChunk] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in ranked:
        key = (item.document_id, _normalize_text(item.text))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(item)
        if len(deduplicated) >= max_evidence:
            break
    return deduplicated


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _clip_text(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _reference_label(item: AnswerEvidence) -> str:
    if item.url:
        return f"{item.title} - {item.url}"
    return f"{item.title} - (无链接)"
