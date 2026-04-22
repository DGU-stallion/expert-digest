"""Wiki lint checks for structural/content quality gates."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

from expert_digest.wiki.models import WikiLintReport, WikiPage
from expert_digest.wiki.vault import WikiVault

_META_PAGES = {"purpose.md", "schema.md", "index.md", "log.md"}
_SHORT_KEEP = {"AI", "IP", "A股", "美股", "港股"}
_LOW_INFO_TITLES = {"未分类主题", "发生了什么", "什么意思", "请问"}
_LOW_INFO_SUBSTRINGS = ("发生了什么", "请问", "是什么意思", "如何看待", "怎么走")
_TITLE_SUFFIXES = ("复盘", "分析", "研究", "观察", "案例", "笔记", "总结", "报告")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_QUALITY_PAGE_TYPES = {"concept", "topic", "thesis", "review"}


def lint_wiki(*, vault: WikiVault) -> WikiLintReport:
    all_pages = vault.list_pages()
    typed_pages = [
        page
        for page in all_pages
        if page.page_type != "unknown" and page.path not in _META_PAGES
    ]
    page_paths = {page.path for page in typed_pages}
    title_to_paths: dict[str, list[str]] = defaultdict(list)
    for page in typed_pages:
        title_to_paths[page.title].append(page.path)

    inbound_count = {page.path: 0 for page in typed_pages}
    for page in all_pages:
        for target in _extract_markdown_targets(page.body):
            resolved = _resolve_link_path(base_path=page.path, target=target)
            if (
                resolved is not None
                and resolved in page_paths
                and resolved != page.path
            ):
                inbound_count[resolved] += 1
        for title in _extract_wiki_targets(page.body):
            for resolved in title_to_paths.get(title, []):
                if resolved != page.path:
                    inbound_count[resolved] += 1

    pages_missing_sources = sorted(
        page.path for page in typed_pages if not page.sources
    )
    low_info_title_pages = sorted(
        page.path
        for page in typed_pages
        if page.page_type in _QUALITY_PAGE_TYPES and _is_low_info_title(page.title)
    )
    isolated_pages = sorted(
        page.path
        for page in typed_pages
        if page.page_type in _QUALITY_PAGE_TYPES
        and inbound_count.get(page.path, 0) == 0
    )
    near_duplicate_title_groups = _find_near_duplicate_title_groups(typed_pages)

    issue_count = (
        len(low_info_title_pages)
        + len(pages_missing_sources)
        + len(isolated_pages)
        + sum(max(0, len(group) - 1) for group in near_duplicate_title_groups)
    )
    return WikiLintReport(
        page_count=len(typed_pages),
        issue_count=issue_count,
        low_info_title_pages=low_info_title_pages,
        pages_missing_sources=pages_missing_sources,
        isolated_pages=isolated_pages,
        near_duplicate_title_groups=near_duplicate_title_groups,
    )


def _is_low_info_title(title: str) -> bool:
    normalized = _normalize_title(title)
    if not normalized:
        return True
    if title in _LOW_INFO_TITLES:
        return True
    if any(marker in title for marker in _LOW_INFO_SUBSTRINGS):
        return True
    if len(normalized) <= 2:
        if title in _SHORT_KEEP:
            return False
        if re.fullmatch(r"[A-Za-z]{1,2}", title):
            return True
        if re.fullmatch(r"[0-9]{1,2}", normalized):
            return True
    if re.fullmatch(r"[0-9年月日季度qQ]+", normalized):
        return True
    return False


def _find_near_duplicate_title_groups(pages: list[WikiPage]) -> list[list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for page in pages:
        if page.page_type not in {"concept", "topic"}:
            continue
        key = _near_duplicate_key(page.title)
        if len(key) < 2:
            continue
        groups[(page.page_type, key)].append(page.path)
    return sorted(
        [sorted(paths) for paths in groups.values() if len(paths) >= 2],
        key=lambda group: group[0],
    )


def _near_duplicate_key(title: str) -> str:
    normalized = _normalize_title(title)
    for suffix in _TITLE_SUFFIXES:
        suffix_norm = _normalize_title(suffix)
        if normalized.endswith(suffix_norm) and len(normalized) > len(suffix_norm) + 1:
            return normalized[: -len(suffix_norm)]
    return normalized


def _normalize_title(title: str) -> str:
    compact = re.sub(r"[\s\W_]+", "", title, flags=re.UNICODE)
    return compact.lower()


def _extract_markdown_targets(body: str) -> list[str]:
    return [match.group(1).strip() for match in _MARKDOWN_LINK_RE.finditer(body)]


def _extract_wiki_targets(body: str) -> list[str]:
    return [match.group(1).strip() for match in _WIKI_LINK_RE.finditer(body)]


def _resolve_link_path(*, base_path: str, target: str) -> str | None:
    candidate = target.split("#", 1)[0].strip()
    if not candidate:
        return None
    if re.match(r"^[a-zA-Z]+://", candidate) or candidate.startswith("mailto:"):
        return None
    if candidate.startswith("/"):
        return _normalize_path(candidate.lstrip("/"))
    base_parent = PurePosixPath(base_path).parent.as_posix()
    joined = PurePosixPath(base_parent) / candidate
    return _normalize_path(joined.as_posix())


def _normalize_path(path: str) -> str:
    parts: list[str] = []
    for part in PurePosixPath(path).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)
