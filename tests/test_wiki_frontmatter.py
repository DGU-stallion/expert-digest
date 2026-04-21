from expert_digest.wiki.frontmatter import parse_frontmatter, render_frontmatter
from expert_digest.wiki.models import SourceRef, WikiPage


def test_render_and_parse_frontmatter_roundtrip():
    page = WikiPage(
        path="topics/popmart-core.md",
        page_type="topic",
        title="泡泡玛特的核心能力",
        body="## 核心判断\n\n这是正文。",
        sources=[
            SourceRef(
                source_id="doc-1",
                title="原文标题",
                url="https://example.com/a",
                evidence_span_ids=["span-1", "span-2"],
            )
        ],
        confidence="medium",
        updated_at="2026-04-21",
    )

    text = render_frontmatter(page)
    parsed = parse_frontmatter(text)

    assert parsed.page_type == "topic"
    assert parsed.title == "泡泡玛特的核心能力"
    assert parsed.sources[0].source_id == "doc-1"
    assert parsed.sources[0].evidence_span_ids == ["span-1", "span-2"]
    assert parsed.body == "## 核心判断\n\n这是正文。"


def test_parse_page_without_frontmatter_uses_unknown_type():
    parsed = parse_frontmatter("# 普通页面\n\n正文")

    assert parsed.page_type == "unknown"
    assert parsed.title == "普通页面"
    assert parsed.sources == []
