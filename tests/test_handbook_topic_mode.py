from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.generation.handbook_writer import (
    DeterministicThemeSynthesizer,
    build_handbook,
)
from expert_digest.knowledge.topic_clusterer import (
    TopicCluster,
    TopicRepresentativeDocument,
)


def _build_fixture():
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
        url="https://example.com/p1",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )
    return document, chunk, embedding


def test_build_handbook_supports_cluster_theme_source(monkeypatch):
    document, chunk, embedding = _build_fixture()

    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.get_documents_by_author",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunks",
        lambda *_a, **_k: [chunk],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.cluster_chunks_by_embeddings",
        lambda **_kwargs: [
            TopicCluster(
                topic_id="topic-1",
                label="主题1：测试主题",
                chunk_count=1,
                representative_chunk_ids=[chunk.id],
                representative_documents=[
                    TopicRepresentativeDocument(
                        document_id=document.id,
                        title=document.title,
                        author=document.author,
                        url=document.url,
                        score=0.9,
                        supporting_chunk_id=chunk.id,
                    )
                ],
            )
        ],
    )

    handbook = build_handbook(
        db_path="data/processed/unused.sqlite3",
        author="黄彦臻",
        theme_source="cluster",
        num_topics=1,
        top_k=1,
        max_themes=1,
        synthesizer=DeterministicThemeSynthesizer(),
    )

    assert "主题1：测试主题" in handbook.markdown
    assert "主题组织方式：cluster" in handbook.markdown


def test_build_handbook_cluster_falls_back_to_preset_when_no_topics(monkeypatch):
    document, chunk, embedding = _build_fixture()

    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.get_documents_by_author",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunks",
        lambda *_a, **_k: [chunk],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.cluster_chunks_by_embeddings",
        lambda **_kwargs: [],
    )

    handbook = build_handbook(
        db_path="data/processed/unused.sqlite3",
        author="黄彦臻",
        theme_source="cluster",
        num_topics=3,
        top_k=1,
        max_themes=1,
        synthesizer=DeterministicThemeSynthesizer(),
    )

    assert "核心能力与方法" in handbook.markdown


def test_build_handbook_rejects_unsupported_theme_source(monkeypatch):
    document, chunk, embedding = _build_fixture()

    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.get_documents_by_author",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunks",
        lambda *_a, **_k: [chunk],
    )
    monkeypatch.setattr(
        "expert_digest.generation.handbook_writer.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )

    try:
        build_handbook(
            db_path="data/processed/unused.sqlite3",
            theme_source="invalid",
        )
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "unsupported theme_source" in str(error)
