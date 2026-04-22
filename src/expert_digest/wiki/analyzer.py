"""Deterministic source analysis baseline for wiki ingest."""

from __future__ import annotations

import re
from collections import Counter

from expert_digest.processing.evidence_builder import DocumentEvidence
from expert_digest.wiki.models import SourceAnalysis

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}|[\u4e00-\u9fff]{2,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_CONCEPT_LIMIT = 8
_TOPIC_LIMIT = 3
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
    "今日",
    "后续",
    "目前",
    "市场",
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
_NOISE_SUBSTRINGS = (
    "全市场逾",
    "只个股涨停",
    "个股涨停",
    "日午间",
    "午间盘中",
    "发生了什么",
    "请问后续",
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


def _extract_concepts(text: str, *, limit: int = _CONCEPT_LIMIT) -> list[str]:
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

    selected: list[str] = []
    for token, score in counts.most_common():
        if _passes_concept_score(token=token, score=score):
            selected.append(token)
        if len(selected) >= limit:
            break
    return selected


def _extract_topics(
    *,
    title: str,
    concepts: list[str],
    limit: int = _TOPIC_LIMIT,
) -> list[str]:
    candidates = []
    for token in re.split(r"[\s:：,，。;；、\-_/|]+", title):
        stripped = token.strip()
        if _is_topic_candidate(stripped):
            candidates.append(stripped)
    candidates.extend([token for token in concepts if _is_topic_candidate(token)])
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
        token = token.strip()
        stem_added = False
        for suffix in _TITLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                stem = token.removesuffix(suffix)
                if _is_candidate(stem):
                    candidates.append(stem)
                    stem_added = True
        if stem_added:
            continue
        if _is_candidate(token):
            candidates.append(token)
    return _dedupe_keep_order(candidates)


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
    if any(pattern in normalized for pattern in _NOISE_SUBSTRINGS):
        return False
    if any(pattern in normalized for pattern in _QUESTION_PATTERNS):
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if normalized.endswith(("吗", "呢", "么", "是", "在", "了")):
        return False
    if "个股" in normalized or "涨停" in normalized:
        return False
    if normalized.startswith(("日", "月")) and len(normalized) <= 3:
        return False
    if len(normalized) > 16:
        return False
    if len(normalized) <= 2:
        if normalized in _SHORT_KEEP:
            return True
        if re.fullmatch(r"[A-Z]{2,3}", normalized):
            return True
    if _is_short_chinese_token(normalized) and normalized not in _SHORT_KEEP:
        return True
    if len(normalized) < 2:
        return False
    return bool(_TOKEN_RE.fullmatch(normalized) or _CHINESE_RE.fullmatch(normalized))


def _passes_concept_score(*, token: str, score: int) -> bool:
    if token in _SHORT_KEEP:
        return score >= 1
    if _is_short_chinese_token(token) and token not in _SHORT_KEEP:
        return score >= 3
    return score >= 2


def _is_topic_candidate(token: str) -> bool:
    if not _is_candidate(token):
        return False
    if _is_short_chinese_token(token) and token not in _SHORT_KEEP:
        return False
    return True


def _is_short_chinese_token(token: str) -> bool:
    return len(token) <= 3 and bool(re.fullmatch(r"[\u4e00-\u9fff]{2,3}", token))


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
