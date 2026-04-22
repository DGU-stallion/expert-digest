import json

from expert_digest.wiki.linter import lint_wiki
from expert_digest.wiki.models import SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault


def test_lint_wiki_reports_standard_checks(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(
        expert_id="huang",
        expert_name="黄彦臻",
        purpose="沉淀公开文章。",
    )
    vault.write_page(
        WikiPage(
            path="sources/doc-1.md",
            page_type="source",
            title="泡泡玛特复盘",
            body="# 泡泡玛特复盘\n\n## 核心判断\n\n- 支撑 [[IP运营]]。",
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
            path="concepts/ip-a.md",
            page_type="concept",
            title="IP运营复盘",
            body="# IP运营复盘\n\n## 相关判断\n\n- a",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘")],
        )
    )
    vault.write_page(
        WikiPage(
            path="concepts/ip-b.md",
            page_type="concept",
            title="IP运营分析",
            body="# IP运营分析\n\n## 相关判断\n\n- b",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘")],
        )
    )
    vault.write_page(
        WikiPage(
            path="topics/no-source.md",
            page_type="topic",
            title="发生了什么",
            body="# 发生了什么\n\n没有来源。",
            sources=[],
        )
    )
    vault.write_page(
        WikiPage(
            path="topics/isolated.md",
            page_type="topic",
            title="孤立主题",
            body="# 孤立主题\n\n没有其他页面链接到这里。",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘")],
        )
    )
    (vault.root / "notes.md").write_text(
        json.dumps({"x": 1}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = lint_wiki(vault=vault)

    assert "topics/no-source.md" in report.pages_missing_sources
    assert "topics/no-source.md" in report.low_info_title_pages
    assert "topics/isolated.md" in report.isolated_pages
    duplicate_sets = [set(group) for group in report.near_duplicate_title_groups]
    assert {"concepts/ip-a.md", "concepts/ip-b.md"} in duplicate_sets
    assert report.issue_count >= 4
