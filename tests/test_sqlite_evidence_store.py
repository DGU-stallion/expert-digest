from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection
from expert_digest.storage.sqlite_store import (
    clear_evidence,
    list_evidence_spans,
    list_parent_sections,
    save_chunks,
    save_documents,
    save_evidence_spans,
    save_parent_sections,
)


def test_save_and_list_parent_sections(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="第一段\n\n第二段",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text="第一段",
        section_index=0,
        start_char=0,
        end_char=3,
    )

    save_documents(db_path, [document])
    assert save_parent_sections(db_path, [section]) == 1

    sections = list_parent_sections(db_path)
    assert sections == [section]


def test_save_and_list_evidence_spans(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="核心能力是 IP。",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text=document.content,
        section_index=0,
    )
    chunk = Chunk.create(
        document_id=document.id,
        parent_section_id=section.id,
        text=document.content,
        chunk_index=0,
    )
    span = EvidenceSpan.create(
        document_id=document.id,
        parent_section_id=section.id,
        chunk_id=chunk.id,
        text="核心能力是 IP。",
        span_index=0,
        start_char=0,
        end_char=8,
    )

    save_documents(db_path, [document])
    save_parent_sections(db_path, [section])
    save_chunks(db_path, [chunk])
    assert save_evidence_spans(db_path, [span]) == 1

    spans = list_evidence_spans(db_path)
    assert spans == [span]


def test_clear_evidence_removes_sections_spans_chunks_and_embeddings(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="核心能力是 IP。",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text=document.content,
        section_index=0,
    )
    chunk = Chunk.create(
        document_id=document.id,
        parent_section_id=section.id,
        text=document.content,
        chunk_index=0,
    )
    span = EvidenceSpan.create(
        document_id=document.id,
        parent_section_id=section.id,
        chunk_id=chunk.id,
        text=document.content,
        span_index=0,
    )

    save_documents(db_path, [document])
    save_parent_sections(db_path, [section])
    save_chunks(db_path, [chunk])
    save_evidence_spans(db_path, [span])

    assert clear_evidence(db_path) == {
        "parent_sections": 1,
        "chunks": 1,
        "evidence_spans": 1,
    }
    assert list_parent_sections(db_path) == []
    assert list_evidence_spans(db_path) == []
