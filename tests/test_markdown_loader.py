from expert_digest.ingest.markdown_loader import load_markdown_documents


def test_load_markdown_documents_reads_front_matter(tmp_path):
    article = tmp_path / "article.md"
    article.write_text(
        """---
author: 陈老师
title: 系统思考
url: https://example.com/system
created_at: 2026-01-03
---

# 文件内标题

系统思考先看关系，再看局部。
""",
        encoding="utf-8",
    )

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].author == "陈老师"
    assert documents[0].title == "系统思考"
    assert documents[0].content == "# 文件内标题\n\n系统思考先看关系，再看局部。"
    assert documents[0].source == str(article)
    assert documents[0].url == "https://example.com/system"
    assert documents[0].created_at == "2026-01-03"


def test_load_markdown_documents_uses_h1_then_filename_fallback(tmp_path):
    with_h1 = tmp_path / "with-h1.md"
    with_h1.write_text("# H1 标题\n\n正文", encoding="utf-8")
    without_h1 = tmp_path / "without-h1.md"
    without_h1.write_text("只有正文", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert [document.title for document in documents] == ["H1 标题", "without-h1"]
    assert [document.author for document in documents] == ["unknown", "unknown"]


def test_load_markdown_documents_reads_nested_files_in_stable_order(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.md"
    first = tmp_path / "a.md"
    second.write_text("# 第二篇\n\n内容二", encoding="utf-8")
    first.write_text("# 第一篇\n\n内容一", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert [document.title for document in documents] == ["第一篇", "第二篇"]
