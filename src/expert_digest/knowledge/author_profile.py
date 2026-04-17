"""Deterministic author profile extraction from imported documents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from expert_digest.domain.models import Document
from expert_digest.storage.sqlite_store import get_documents_by_author, list_documents

_TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}")
_TITLE_SPLIT_PATTERN = re.compile(r"[\s:：,，。;；、\-_/|]+")
_STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一个",
    "一些",
    "以及",
    "进行",
    "通过",
    "自己",
    "没有",
    "如何",
    "什么",
    "就是",
    "可以",
    "但是",
    "然后",
    "如果",
    "那么",
    "因为",
    "所以",
}


@dataclass(frozen=True)
class KeywordStat:
    keyword: str
    count: int


@dataclass(frozen=True)
class ReasoningPatternStat:
    pattern: str
    count: int


@dataclass(frozen=True)
class AuthorProfile:
    author: str
    document_count: int
    source_document_ids: list[str]
    focus_topics: list[str]
    keywords: list[KeywordStat]
    reasoning_patterns: list[ReasoningPatternStat]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_author_profile(
    *,
    db_path: str | Path,
    author: str | None = None,
    max_topics: int = 6,
    max_keywords: int = 12,
    max_patterns: int = 5,
) -> AuthorProfile:
    documents = (
        get_documents_by_author(db_path, author) if author else list_documents(db_path)
    )
    if not documents:
        raise ValueError("no documents available for author profile")
    return extract_author_profile_from_documents(
        documents,
        max_topics=max_topics,
        max_keywords=max_keywords,
        max_patterns=max_patterns,
    )


def extract_author_profile_from_documents(
    documents: list[Document],
    *,
    max_topics: int = 6,
    max_keywords: int = 12,
    max_patterns: int = 5,
) -> AuthorProfile:
    if not documents:
        raise ValueError("documents must not be empty")

    primary_author = _resolve_primary_author(documents)
    source_document_ids = [document.id for document in documents]
    focus_topics = _extract_focus_topics(documents, limit=max_topics)
    keywords = _extract_keywords(documents, limit=max_keywords)
    reasoning_patterns = _extract_reasoning_patterns(documents, limit=max_patterns)
    return AuthorProfile(
        author=primary_author,
        document_count=len(documents),
        source_document_ids=source_document_ids,
        focus_topics=focus_topics,
        keywords=keywords,
        reasoning_patterns=reasoning_patterns,
    )


def _resolve_primary_author(documents: list[Document]) -> str:
    counts = Counter(document.author for document in documents)
    return counts.most_common(1)[0][0]


def _extract_focus_topics(documents: list[Document], *, limit: int) -> list[str]:
    ranked = Counter()
    for document in documents:
        for token in _TITLE_SPLIT_PATTERN.split(document.title):
            normalized = token.strip().lower()
            if len(normalized) < 2:
                continue
            if normalized in _STOPWORDS:
                continue
            ranked[normalized] += 1
    return [topic for topic, _ in ranked.most_common(limit)]


def _extract_keywords(documents: list[Document], *, limit: int) -> list[KeywordStat]:
    ranked = Counter()
    for document in documents:
        for token in _TOKEN_PATTERN.findall(document.content):
            normalized = token.strip().lower()
            if len(normalized) < 2:
                continue
            if normalized in _STOPWORDS:
                continue
            ranked[normalized] += 1
    return [
        KeywordStat(keyword=keyword, count=count)
        for keyword, count in ranked.most_common(limit)
    ]


def _extract_reasoning_patterns(
    documents: list[Document], *, limit: int
) -> list[ReasoningPatternStat]:
    detectors = {
        "因为...所以...": re.compile(r"因为.{0,40}?所以"),
        "如果...那么...": re.compile(r"如果.{0,40}?那么"),
        "首先...然后...": re.compile(r"首先.{0,40}?然后"),
        "一方面...另一方面...": re.compile(r"一方面.{0,40}?另一方面"),
        "不是...而是...": re.compile(r"不是.{0,40}?而是"),
    }
    ranked = Counter()
    for document in documents:
        for label, pattern in detectors.items():
            ranked[label] += len(pattern.findall(document.content))
    return [
        ReasoningPatternStat(pattern=label, count=count)
        for label, count in ranked.most_common(limit)
        if count > 0
    ]
