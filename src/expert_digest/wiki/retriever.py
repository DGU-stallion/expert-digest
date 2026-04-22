"""Wiki-native retrieval over Markdown pages."""

from __future__ import annotations

import re

from expert_digest.wiki.models import WikiSearchHit
from expert_digest.wiki.vault import WikiVault


def search_wiki(
    *,
    vault: WikiVault,
    query: str,
    top_k: int = 5,
) -> list[WikiSearchHit]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    terms = _tokenize(query)
    if not terms:
        return []

    hits: list[WikiSearchHit] = []
    for page in vault.list_pages():
        title = page.title.lower()
        body = page.body.lower()
        source_text = _source_text(page)
        matched: list[str] = []
        score = 0.0
        for term in terms:
            lowered = term.lower()
            term_matched = False
            if lowered in title:
                score += 3.0
                term_matched = True
            if lowered in source_text:
                score += 2.0
                term_matched = True
            if lowered in body:
                score += 1.0
                term_matched = True
            if term_matched and term not in matched:
                matched.append(term)
        if score <= 0.0:
            continue
        hits.append(
            WikiSearchHit(
                page=page,
                score=score,
                matched_terms=matched,
                source_ids=[source.source_id for source in page.sources],
            )
        )
    return sorted(
        hits,
        key=lambda item: (
            -item.score,
            item.page.page_type,
            item.page.title,
            item.page.path,
        ),
    )[:top_k]


def _source_text(page) -> str:
    return " ".join(
        " ".join([source.source_id, source.title, *(source.evidence_span_ids or [])])
        for source in page.sources
    ).lower()


def _tokenize(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9_+-]{2,}", normalized):
        terms.append(term)
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.append(phrase)
        terms.extend(_cjk_ngrams(phrase))
    if not terms:
        terms.append(normalized)
    return _unique(terms)


def _cjk_ngrams(text: str) -> list[str]:
    grams: list[str] = []
    max_size = min(6, len(text))
    for size in range(max_size, 1, -1):
        for start in range(0, len(text) - size + 1):
            grams.append(text[start : start + size])
    return grams


def _unique(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in terms:
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_terms.append(term)
    return unique_terms
