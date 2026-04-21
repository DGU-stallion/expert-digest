from expert_digest.wiki.evaluator import evaluate_wiki
from expert_digest.wiki.models import SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault


def test_evaluate_wiki_reports_traceability_and_coverage(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")
    vault.write_page(
        WikiPage(
            path="sources/doc-1.md",
            page_type="source",
            title="泡泡玛特复盘",
            body="## 摘要\n\n有来源。",
            sources=[
                SourceRef(
                    source_id="doc-1",
                    title="泡泡玛特复盘",
                    evidence_span_ids=["span-1"],
                )
            ],
        )
    )
    vault.write_page(
        WikiPage(
            path="topics/ip.md",
            page_type="topic",
            title="IP 运营",
            body="## 判断\n\n有来源。",
            sources=[
                SourceRef(
                    source_id="doc-1",
                    title="泡泡玛特复盘",
                    evidence_span_ids=["span-1"],
                )
            ],
        )
    )

    report = evaluate_wiki(vault=vault, expected_source_count=1)

    assert report.page_count >= 2
    assert report.source_page_count == 1
    assert report.pages_with_sources >= 2
    assert report.traceability_ratio == 1.0
    assert report.coverage_ratio == 1.0


def test_evaluate_wiki_detects_pages_without_sources(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")
    vault.write_page(
        WikiPage(
            path="topics/no-source.md",
            page_type="topic",
            title="无来源",
            body="没有来源。",
        )
    )

    report = evaluate_wiki(vault=vault, expected_source_count=1)

    assert "topics/no-source.md" in report.pages_missing_sources
    assert report.traceability_ratio < 1.0
