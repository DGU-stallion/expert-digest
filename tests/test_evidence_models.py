from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection


def test_parent_section_create_has_stable_id():
    document = Document.create(
        author="黄彦臻",
        title="关于泡泡玛特的极简复盘",
        content="第一段\n\n第二段",
        source="sample",
        url="https://example.com/a",
    )

    first = ParentSection.create(
        document_id=document.id,
        title="关于泡泡玛特的极简复盘",
        text="第一段\n\n第二段",
        section_index=0,
        start_char=0,
        end_char=7,
    )
    second = ParentSection.create(
        document_id=document.id,
        title="关于泡泡玛特的极简复盘",
        text="第一段\n\n第二段",
        section_index=0,
        start_char=0,
        end_char=7,
    )

    assert first.id == second.id
    assert first.document_id == document.id
    assert first.section_index == 0
    assert first.start_char == 0
    assert first.end_char == 7


def test_chunk_can_reference_parent_section():
    chunk = Chunk.create(
        document_id="doc-1",
        text="泡泡玛特的能力是 IP 运营。",
        chunk_index=0,
        parent_section_id="section-1",
        start_char=10,
        end_char=25,
    )

    assert chunk.parent_section_id == "section-1"
    assert chunk.start_char == 10
    assert chunk.end_char == 25


def test_evidence_span_create_has_source_location():
    span = EvidenceSpan.create(
        document_id="doc-1",
        parent_section_id="section-1",
        chunk_id="chunk-1",
        text="核心能力不是单纯卖潮玩。",
        span_index=0,
        start_char=42,
        end_char=54,
    )

    assert span.document_id == "doc-1"
    assert span.parent_section_id == "section-1"
    assert span.chunk_id == "chunk-1"
    assert span.start_char == 42
    assert span.end_char == 54
