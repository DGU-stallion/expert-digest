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
) -> StructuredAnswer:
    if not evidence_chunks:
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
        for item in evidence_chunks
    ]
    top = evidence[0]
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


def _clip_text(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _reference_label(item: AnswerEvidence) -> str:
    if item.url:
        return f"{item.title} - {item.url}"
    return f"{item.title} - (无链接)"
