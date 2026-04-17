from expert_digest.domain.models import ChunkEmbedding
from expert_digest.knowledge.topic_clusterer import (
    TopicCluster,
    TopicRepresentativeDocument,
)
from expert_digest.knowledge.topic_report import build_topic_report


def _build_topics() -> list[TopicCluster]:
    return [
        TopicCluster(
            topic_id="topic-1",
            label="主题1：价值投资",
            chunk_count=6,
            representative_chunk_ids=["c1", "c2"],
            representative_documents=[
                TopicRepresentativeDocument(
                    document_id="d1",
                    title="价值投资复盘",
                    author="黄彦臻",
                    url=None,
                    score=0.9,
                    supporting_chunk_id="c1",
                ),
                TopicRepresentativeDocument(
                    document_id="d2",
                    title="估值模型",
                    author="黄彦臻",
                    url=None,
                    score=0.8,
                    supporting_chunk_id="c2",
                ),
            ],
        ),
        TopicCluster(
            topic_id="topic-2",
            label="主题2：风险控制",
            chunk_count=4,
            representative_chunk_ids=["c3"],
            representative_documents=[
                TopicRepresentativeDocument(
                    document_id="d3",
                    title="风险预算",
                    author="黄彦臻",
                    url=None,
                    score=0.7,
                    supporting_chunk_id="c3",
                )
            ],
        ),
    ]


def test_build_topic_report_computes_global_and_proxy_metrics():
    topics = _build_topics()
    embeddings = [
        ChunkEmbedding.create(chunk_id="c1", model="hash-bow-v1", vector=[1.0, 0.0]),
        ChunkEmbedding.create(chunk_id="c2", model="hash-bow-v1", vector=[0.8, 0.2]),
        ChunkEmbedding.create(chunk_id="c3", model="hash-bow-v1", vector=[-1.0, 0.0]),
    ]

    report = build_topic_report(
        topics=topics,
        chunk_embeddings=embeddings,
        model="hash-bow-v1",
    )

    assert report.model == "hash-bow-v1"
    assert report.topic_count == 2
    assert report.total_chunks == 10
    assert report.largest_topic_ratio == 0.6
    assert report.mean_topic_size == 5.0
    assert report.mean_intra_similarity_proxy == 0.8
    assert report.mean_inter_topic_similarity_proxy == -1.0
    assert report.topics[0].topic_id == "topic-1"
    assert report.topics[0].representative_document_count == 2
    assert report.topics[0].lead_document_title == "价值投资复盘"


def test_build_topic_report_handles_empty_topics():
    report = build_topic_report(
        topics=[],
        chunk_embeddings=[],
        model="hash-bow-v1",
    )

    assert report.topic_count == 0
    assert report.total_chunks == 0
    assert report.largest_topic_ratio == 0.0
    assert report.mean_topic_size == 0.0
    assert report.mean_intra_similarity_proxy is None
    assert report.mean_inter_topic_similarity_proxy is None
    assert report.topics == []
