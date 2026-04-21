"""Wiki quality evaluation helpers."""

from __future__ import annotations

from expert_digest.wiki.models import WikiQualityReport
from expert_digest.wiki.vault import WikiVault


def evaluate_wiki(
    *,
    vault: WikiVault,
    expected_source_count: int,
) -> WikiQualityReport:
    if expected_source_count < 0:
        raise ValueError("expected_source_count must be >= 0")
    pages = [
        page
        for page in vault.list_pages()
        if page.page_type not in {"unknown"}
        and page.path not in {"purpose.md", "schema.md", "index.md", "log.md"}
    ]
    source_pages = [page for page in pages if page.page_type == "source"]
    pages_with_sources = [page for page in pages if page.sources]
    missing = [page.path for page in pages if not page.sources]
    traceability_ratio = (
        round(len(pages_with_sources) / len(pages), 4) if pages else 1.0
    )
    coverage_ratio = (
        round(len(source_pages) / expected_source_count, 4)
        if expected_source_count > 0
        else 1.0
    )
    if coverage_ratio > 1.0:
        coverage_ratio = 1.0
    return WikiQualityReport(
        page_count=len(pages),
        source_page_count=len(source_pages),
        pages_with_sources=len(pages_with_sources),
        pages_missing_sources=missing,
        traceability_ratio=traceability_ratio,
        coverage_ratio=coverage_ratio,
    )
