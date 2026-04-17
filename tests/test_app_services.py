from pathlib import Path

import pytest

from expert_digest.app.services import (
    collect_data_overview,
    generate_handbook,
    import_documents,
)
from expert_digest.domain.models import Chunk, ChunkEmbedding, Document, Handbook


def test_collect_data_overview_counts_documents_chunks_and_embeddings(monkeypatch):
    doc_a = Document.create(
        author="黄彦臻",
        title="A",
        content="A",
        source="sample",
    )
    doc_b = Document.create(
        author="陈一鸣",
        title="B",
        content="B",
        source="sample",
    )
    chunk = Chunk.create(document_id=doc_a.id, text="片段", chunk_index=0)
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )

    monkeypatch.setattr(
        "expert_digest.app.services.list_documents",
        lambda *_a, **_k: [doc_a, doc_b],
    )
    monkeypatch.setattr(
        "expert_digest.app.services.list_chunks",
        lambda *_a, **_k: [chunk],
    )
    monkeypatch.setattr(
        "expert_digest.app.services.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )

    overview = collect_data_overview(
        db_path="data/processed/zhihu_huang.sqlite3",
        model="hash-bow-v1",
    )

    assert overview.document_count == 2
    assert overview.chunk_count == 1
    assert overview.embedding_count == 1
    assert overview.authors == ["陈一鸣", "黄彦臻"]


def test_import_documents_dispatches_jsonl_loader(monkeypatch):
    expected_path = Path("data/sample/articles.jsonl")
    expected_db = Path("data/processed/zhihu_huang.sqlite3")
    docs = [
        Document.create(
            author="黄彦臻",
            title="A",
            content="A",
            source="sample",
        )
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "expert_digest.app.services.load_jsonl_documents",
        lambda path: docs if Path(path) == expected_path else [],
    )

    def _fake_save_documents(db_path, documents):
        captured["db_path"] = Path(db_path)
        captured["documents"] = documents
        return len(documents)

    monkeypatch.setattr(
        "expert_digest.app.services.save_documents",
        _fake_save_documents,
    )

    saved = import_documents(
        kind="jsonl",
        source_path=expected_path,
        db_path=expected_db,
    )

    assert saved == 1
    assert captured["db_path"] == expected_db
    assert captured["documents"] == docs


def test_import_documents_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="unsupported import kind"):
        import_documents(
            kind="csv",
            source_path=Path("data/sample/articles.csv"),
            db_path=Path("data/processed/zhihu_huang.sqlite3"),
        )


def test_generate_handbook_uses_hybrid_with_default_llm_client(monkeypatch):
    captured: dict[str, object] = {}
    fake_handbook = Handbook(
        author="黄彦臻",
        title="黄彦臻学习手册",
        markdown="# 手册",
        source_document_ids=["doc-1"],
    )
    fake_llm_client = object()

    def _fake_create_default_handbook_llm_client(**kwargs):
        captured["llm_kwargs"] = kwargs
        return fake_llm_client

    monkeypatch.setattr(
        "expert_digest.app.services.create_default_handbook_llm_client",
        _fake_create_default_handbook_llm_client,
    )

    def _fake_build_handbook(**kwargs):
        captured["build_kwargs"] = kwargs
        return fake_handbook

    monkeypatch.setattr(
        "expert_digest.app.services.build_handbook",
        _fake_build_handbook,
    )
    monkeypatch.setattr(
        "expert_digest.app.services.write_handbook",
        lambda *, handbook, output_path: Path(output_path),
    )

    result = generate_handbook(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author=None,
        model="hash-bow-v1",
        top_k=3,
        max_themes=3,
        output_path=Path("data/outputs/handbook.md"),
        synthesis_mode="hybrid",
        ccswitch_db_path=Path("data/processed/mock_ccswitch.sqlite3"),
        llm_timeout=12,
        llm_max_tokens=600,
    )

    assert result.handbook == fake_handbook
    assert result.output_path == Path("data/outputs/handbook.md")
    assert captured["llm_kwargs"]["ccswitch_db_path"] == Path(
        "data/processed/mock_ccswitch.sqlite3"
    )
    assert captured["llm_kwargs"]["timeout_seconds"] == 12
    assert captured["llm_kwargs"]["max_output_tokens"] == 600
    synthesizer = captured["build_kwargs"]["synthesizer"]
    assert getattr(synthesizer, "_llm_client", None) is fake_llm_client


def test_generate_handbook_rejects_unsupported_synthesis_mode():
    with pytest.raises(ValueError, match="unsupported synthesis mode"):
        generate_handbook(
            db_path=Path("data/processed/zhihu_huang.sqlite3"),
            author=None,
            model="hash-bow-v1",
            top_k=3,
            max_themes=3,
            output_path=Path("data/outputs/handbook.md"),
            synthesis_mode="custom",
        )
