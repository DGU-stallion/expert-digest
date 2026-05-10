from pathlib import Path
from uuid import uuid4

import pytest

from expert_digest.app.services import (
    build_author_profile_snapshot,
    cluster_topics,
    collect_data_overview,
    generate_handbook,
    generate_skill_draft,
    import_documents,
    persist_uploaded_jsonl,
)
from expert_digest.domain.models import Chunk, ChunkEmbedding, Document, Handbook
from expert_digest.knowledge.topic_clusterer import TopicCluster


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


def test_generate_handbook_uses_pipeline(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _FakePipeline:
        def invoke(self, state):
            captured["state"] = state
            return {"handbook_markdown": "# Handbook\n\nContent here."}

    monkeypatch.setattr(
        "expert_digest.app.services.compile_pipeline",
        lambda: _FakePipeline(),
    )

    output_path = tmp_path / "handbook.md"
    result = generate_handbook(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author="黄彦臻",
        model="hash-bow-v1",
        top_k=3,
        max_themes=3,
        output_path=output_path,
    )

    assert result.handbook.author == "黄彦臻"
    assert result.handbook.markdown == "# Handbook\n\nContent here."
    assert result.output_path == output_path
    assert output_path.exists()


def test_generate_handbook_uses_pipeline_without_author(monkeypatch, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"handbook_markdown": "# Default Handbook\n"}

    monkeypatch.setattr(
        "expert_digest.app.services.compile_pipeline",
        lambda: _FakePipeline(),
    )

    output_path = tmp_path / "default_handbook.md"
    result = generate_handbook(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author=None,
        model="hash-bow-v1",
        top_k=3,
        max_themes=3,
        output_path=output_path,
    )

    assert result.handbook.author == "unknown"
    assert result.output_path == output_path


def test_persist_uploaded_jsonl_writes_uploaded_content():
    upload_dir = Path(".tmp") / f"test_uploaded_jsonl_{uuid4().hex}"
    payload = (
        '{"author":"测试作者","title":"测试标题","content":"测试内容","source":"sample"}\n'
    ).encode()
    saved_path = persist_uploaded_jsonl(
        filename="sample_upload.jsonl",
        content=payload,
        upload_dir=upload_dir,
    )
    assert saved_path.exists()
    assert saved_path.parent == upload_dir
    assert saved_path.read_bytes() == payload


def test_persist_uploaded_jsonl_sanitizes_filename_and_rejects_empty_name():
    upload_dir = Path(".tmp") / f"test_uploaded_jsonl_name_{uuid4().hex}"
    payload = b'{"author":"a","title":"t","content":"c","source":"s"}\n'
    saved_path = persist_uploaded_jsonl(
        filename="..\\unsafe.jsonl",
        content=payload,
        upload_dir=upload_dir,
    )
    assert saved_path.parent == upload_dir
    assert ".." not in saved_path.name

    with pytest.raises(ValueError, match="filename must not be empty"):
        persist_uploaded_jsonl(
            filename="  ",
            content=payload,
            upload_dir=upload_dir,
        )


def test_cluster_topics_returns_report(monkeypatch):
    fake_topics = [
        TopicCluster(
            topic_id="topic-1",
            label="主题1：测试",
            chunk_count=3,
            representative_chunk_ids=["c1"],
            representative_documents=[],
        )
    ]
    captured: dict[str, object] = {}

    def _fake_build_topic_clusters(**kwargs):
        captured.update(kwargs)
        return fake_topics

    monkeypatch.setattr(
        "expert_digest.app.services.build_topic_clusters",
        _fake_build_topic_clusters,
    )
    monkeypatch.setattr(
        "expert_digest.app.services.list_chunk_embeddings",
        lambda *_a, **_k: [
            ChunkEmbedding.create(
                chunk_id="c1",
                model="hash-bow-v1",
                vector=[1.0, 0.0],
            )
        ],
    )

    result = cluster_topics(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        model="hash-bow-v1",
        num_topics=2,
        top_docs=1,
    )

    assert result.topics == fake_topics
    assert result.report.topic_count == 1
    assert captured["num_topics"] == 2


def test_cluster_topics_rejects_unsupported_label_mode():
    with pytest.raises(ValueError, match="unsupported label_mode"):
        cluster_topics(
            db_path=Path("data/processed/zhihu_huang.sqlite3"),
            model="hash-bow-v1",
            label_mode="custom",
        )


def test_build_author_profile_snapshot_supports_export(monkeypatch, tmp_path):
    fake_profile = {
        "author": "黄彦臻",
        "document_count": 2,
        "source_document_ids": ["doc-1", "doc-2"],
        "focus_topics": ["供给需求"],
        "keywords": [{"keyword": "风险", "count": 3}],
        "reasoning_patterns": [{"pattern": "因为...所以...", "count": 2}],
    }

    monkeypatch.setattr(
        "expert_digest.app.services.build_author_profile",
        lambda **_kwargs: fake_profile,
    )

    output_path = tmp_path / "author_profile_test.json"
    result = build_author_profile_snapshot(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author="黄彦臻",
        output_path=output_path,
    )

    assert result.profile["author"] == "黄彦臻"
    assert result.output_path == output_path
    assert result.output_path.exists()


def test_generate_skill_draft_supports_default_output(monkeypatch, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"skill_markdown": "# SKILL: test\n", "documents": [{"id": "1"}]}

    monkeypatch.setattr(
        "expert_digest.app.services.compile_pipeline",
        lambda: _FakePipeline(),
    )

    output_path = tmp_path / "skill.md"
    result = generate_skill_draft(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author="黄彦臻",
        output_path=output_path,
    )

    assert result.profile["author"] == "黄彦臻"
    assert result.output_path == output_path
    assert result.output_path.exists()
