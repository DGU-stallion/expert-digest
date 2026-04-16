from expert_digest.processing.cleaner import clean_text


def test_clean_text_removes_html_and_decodes_entities():
    raw = "<p>第一段&nbsp;内容</p><p>第二段</p>"

    cleaned = clean_text(raw)

    assert cleaned == "第一段 内容\n\n第二段"


def test_clean_text_rewrites_markdown_links_and_collapses_noise():
    raw = "  [原文链接](https://example.com/a)\n\n\n  这是正文   "

    cleaned = clean_text(raw)

    assert cleaned == "原文链接\n\n这是正文"
