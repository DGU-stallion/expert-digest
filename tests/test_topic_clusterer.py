import pytest

from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.knowledge.topic_clusterer import (
    DeterministicTopicLabeler,
    LLMTopicLabeler,
    TopicCluster,
    build_topic_clusters,
    cluster_chunks_by_embeddings,
)


def _build_fixture():
    doc_a = Document.create(
        author="作者A",
        title="主题A 复盘",
        content="A",
        source="sample",
    )
    doc_b = Document.create(
        author="作者B",
        title="主题B 复盘",
        content="B",
        source="sample",
    )

    chunk_a1 = Chunk.create(document_id=doc_a.id, text="A1", chunk_index=0)
    chunk_a2 = Chunk.create(document_id=doc_a.id, text="A2", chunk_index=1)
    chunk_b1 = Chunk.create(document_id=doc_b.id, text="B1", chunk_index=0)
    chunk_b2 = Chunk.create(document_id=doc_b.id, text="B2", chunk_index=1)

    embeddings = [
        ChunkEmbedding.create(
            chunk_id=chunk_a1.id,
            model="hash-bow-v1",
            vector=[1.0, 0.0],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_a2.id,
            model="hash-bow-v1",
            vector=[0.9, 0.1],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_b1.id,
            model="hash-bow-v1",
            vector=[-1.0, 0.0],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_b2.id,
            model="hash-bow-v1",
            vector=[-0.9, -0.1],
        ),
    ]
    chunks_by_id = {
        chunk_a1.id: chunk_a1,
        chunk_a2.id: chunk_a2,
        chunk_b1.id: chunk_b1,
        chunk_b2.id: chunk_b2,
    }
    documents_by_id = {doc_a.id: doc_a, doc_b.id: doc_b}
    return chunks_by_id, documents_by_id, embeddings


def test_cluster_chunks_by_embeddings_returns_two_topics_for_clear_groups():
    chunks_by_id, documents_by_id, embeddings = _build_fixture()

    topics = cluster_chunks_by_embeddings(
        chunks_by_id=chunks_by_id,
        documents_by_id=documents_by_id,
        chunk_embeddings=embeddings,
        num_topics=2,
        top_docs_per_topic=1,
    )

    assert len(topics) == 2
    assert all(isinstance(topic, TopicCluster) for topic in topics)
    top_titles = {
        topic.representative_documents[0].title
        for topic in topics
        if topic.representative_documents
    }
    assert top_titles == {"主题A 复盘", "主题B 复盘"}


def test_cluster_chunks_by_embeddings_rejects_invalid_num_topics():
    chunks_by_id, documents_by_id, embeddings = _build_fixture()

    with pytest.raises(ValueError, match="num_topics must be > 0"):
        cluster_chunks_by_embeddings(
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
            chunk_embeddings=embeddings,
            num_topics=0,
        )


def test_build_topic_clusters_returns_empty_without_embeddings(monkeypatch):
    monkeypatch.setattr(
        "expert_digest.knowledge.topic_clusterer.list_documents",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "expert_digest.knowledge.topic_clusterer.list_chunks",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "expert_digest.knowledge.topic_clusterer.list_chunk_embeddings",
        lambda *_a, **_k: [],
    )

    topics = build_topic_clusters(
        db_path="data/processed/zhihu_huang.sqlite3",
        model="hash-bow-v1",
        num_topics=3,
    )

    assert topics == []


def test_cluster_chunks_by_embeddings_supports_custom_labeler():
    chunks_by_id, documents_by_id, embeddings = _build_fixture()

    class _StaticLabeler:
        def label_topic(self, *, topic: TopicCluster, topic_index: int) -> str:
            return f"固定主题{topic_index}"

    topics = cluster_chunks_by_embeddings(
        chunks_by_id=chunks_by_id,
        documents_by_id=documents_by_id,
        chunk_embeddings=embeddings,
        num_topics=2,
        top_docs_per_topic=1,
        labeler=_StaticLabeler(),
    )

    assert [topic.label for topic in topics] == ["固定主题1", "固定主题2"]


def test_llm_topic_labeler_uses_llm_result():
    topic = TopicCluster(
        topic_id="topic-1",
        label="主题1：占位",
        chunk_count=4,
        representative_chunk_ids=["c1"],
        representative_documents=[],
    )

    class _FakeLLMClient:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            assert "主题命名" in system_prompt
            assert "topic-1" in user_prompt
            return "价值投资与周期判断"

    labeler = LLMTopicLabeler(llm_client=_FakeLLMClient())
    label = labeler.label_topic(topic=topic, topic_index=1)

    assert label == "主题1：价值投资与周期判断"
    assert labeler.runtime_metadata()["fallback_used"] is False


def test_llm_topic_labeler_falls_back_when_client_missing():
    topic = TopicCluster(
        topic_id="topic-1",
        label="主题1：占位",
        chunk_count=4,
        representative_chunk_ids=["c1"],
        representative_documents=[],
    )

    fallback = DeterministicTopicLabeler()
    labeler = LLMTopicLabeler(llm_client=None, fallback=fallback)
    label = labeler.label_topic(topic=topic, topic_index=2)

    assert label == "主题2：未命名主题"
    assert labeler.runtime_metadata()["fallback_used"] is True
