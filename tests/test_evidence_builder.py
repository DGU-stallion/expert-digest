from expert_digest.domain.models import Document
from expert_digest.processing.cleaner import clean_document
from expert_digest.processing.evidence_builder import build_document_evidence


def test_build_document_evidence_creates_sections_chunks_and_spans():
    document = Document.create(
        author="黄彦臻",
        title="关于泡泡玛特的极简复盘",
        content="# 核心判断\n\n泡泡玛特的核心能力不是单纯卖潮玩，而是 IP 运营。\n\n第二段继续解释。",
        source="sample",
        url="https://example.com/popmart",
    )

    result = build_document_evidence(
        clean_document(document),
        parent_max_chars=80,
        child_max_chars=40,
        child_min_chars=10,
        span_max_chars=30,
    )

    assert len(result.parent_sections) == 1
    assert result.parent_sections[0].title == "核心判断"
    assert len(result.chunks) >= 2
    assert all(
        chunk.parent_section_id == result.parent_sections[0].id for chunk in result.chunks
    )
    assert len(result.evidence_spans) >= 2
    assert all(
        span.parent_section_id == result.parent_sections[0].id
        for span in result.evidence_spans
    )
    assert all(span.chunk_id for span in result.evidence_spans)


def test_build_document_evidence_falls_back_to_document_title_without_heading():
    document = Document.create(
        author="黄彦臻",
        title="没有标题层级的文章",
        content="第一段内容。\n\n第二段内容。",
        source="sample",
    )

    result = build_document_evidence(document, parent_max_chars=100)

    assert result.parent_sections[0].title == "没有标题层级的文章"
    assert result.parent_sections[0].document_id == document.id
