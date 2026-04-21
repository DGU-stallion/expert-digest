"""Deterministic source analysis baseline for wiki ingest."""

from __future__ import annotations

import re
from collections import Counter

from expert_digest.processing.evidence_builder import DocumentEvidence
from expert_digest.wiki.models import SourceAnalysis

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}|[\u4e00-\u9fff]{2,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "这个",
    "那个",
    "因为",
    "所以",
    "但是",
    "如果",
    "那么",
    "可以",
    "不是",
    "而是",
    "他们",
    "我们",
}
_TITLE_SUFFIXES = ("复盘", "分析", "研究", "观察", "案例", "笔记", "总结", "报告")
_QUESTION_PATTERNS = (
    "如何看待",
    "发生了什么",
    "是什么意思",
    "为什么",
    "请问",
    "怎么走",
)
_SHORT_KEEP = {"AI", "IP", "A股", "美股", "港股"}


def analyze_document_evidence(evidence: DocumentEvidence) -> SourceAnalysis:
    spans = evidence.evidence_spans
    claims = [_normalize(span.text) for span in spans[:5] if _normalize(span.text)]
    concepts = _extract_concepts(
        evidence.document.title + "\n" + evidence.document.content
    )
    topics = _extract_topics(
        title=evidence.document.title,
        concepts=concepts,
    )
    summary = _build_summary(
        title=evidence.document.title,
        claims=claims,
    )
    return SourceAnalysis(
        source_id=evidence.document.id,
        source_title=evidence.document.title,
        author=evidence.document.author,
        url=evidence.document.url,
        summary=summary,
        key_claims=claims,
        concepts=concepts,
        topics=topics,
        evidence_span_ids=[span.id for span in spans[:8]],
        confidence="medium" if claims else "low",
    )


def _extract_concepts(text: str, *, limit: int = 12) -> list[str]:
    counts: Counter[str] = Counter()
    title, _, body = text.partition("\n")

    for token in _title_candidates(title):
        counts[token] += 3

    for token in _possessive_terms(body):
        counts[token] += 2

    for token in _TOKEN_RE.findall(text):
        normalized = token.strip()
        if _is_candidate(normalized):
            counts[normalized] += 1

    return [token for token, _ in counts.most_common(limit)]


def _extract_topics(*, title: str, concepts: list[str], limit: int = 4) -> list[str]:
    candidates = []
    for token in re.split(r"[\s:：,，。;；、\-_/|]+", title):
        stripped = token.strip()
        if _is_candidate(stripped):
            candidates.append(stripped)
    candidates.extend(concepts[:limit])
    return list(dict.fromkeys(candidates))[:limit] or ["未分类主题"]


def _build_summary(*, title: str, claims: list[str]) -> str:
    if not claims:
        return f"《{title}》暂无可稳定抽取的核心判断。"
    return f"《{title}》的核心线索：{claims[0]}"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _title_candidates(title: str) -> list[str]:
    candidates: list[str] = []
    for token in _TOKEN_RE.findall(title):
        if _is_candidate(token):
            candidates.append(token)
        for suffix in _TITLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                stem = token.removesuffix(suffix)
                if _is_candidate(stem):
                    candidates.append(stem)
    return candidates


def _possessive_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(
        r"([A-Za-z][A-Za-z0-9_+-]{1,}|[\u4e00-\u9fff]{2,})的", text
    ):
        token = match.group(1).strip()
        if _is_candidate(token):
            terms.append(token)
    return terms


def _is_candidate(token: str) -> bool:
    normalized = token.strip()
    if not normalized:
        return False
    if normalized in _STOPWORDS:
        return False
    if any(pattern in normalized for pattern in _QUESTION_PATTERNS):
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if normalized.endswith(("吗", "呢", "么")):
        return False
    if normalized.startswith(("日", "月")) and len(normalized) <= 3:
        return False
    if len(normalized) > 20:
        return False
    if len(normalized) <= 2:
        if normalized in _SHORT_KEEP:
            return True
        if re.fullmatch(r"[A-Z]{2,3}", normalized):
            return True
        return False
    if len(normalized) < 2:
        return False
    return bool(_TOKEN_RE.fullmatch(normalized) or _CHINESE_RE.fullmatch(normalized))
