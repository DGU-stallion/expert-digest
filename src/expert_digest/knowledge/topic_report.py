"""Topic clustering report helpers for M6 evaluation output."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from expert_digest.domain.models import ChunkEmbedding
from expert_digest.knowledge.topic_clusterer import TopicCluster
from expert_digest.retrieval.retriever import cosine_similarity


@dataclass(frozen=True)
class TopicReportItem:
    topic_id: str
    label: str
    chunk_count: int
    representative_document_count: int
    mean_representative_score: float | None
    lead_document_title: str | None


@dataclass(frozen=True)
class TopicReport:
    model: str
    topic_count: int
    total_chunks: int
    largest_topic_ratio: float
    mean_topic_size: float
    mean_intra_similarity_proxy: float | None
    mean_inter_topic_similarity_proxy: float | None
    topics: list[TopicReportItem]


def build_topic_report(
    *,
    topics: list[TopicCluster],
    chunk_embeddings: list[ChunkEmbedding],
    model: str,
) -> TopicReport:
    topic_count = len(topics)
    total_chunks = sum(topic.chunk_count for topic in topics)
    largest_topic_ratio = (
        _round(max(topic.chunk_count for topic in topics) / total_chunks)
        if total_chunks > 0
        else 0.0
    )
    mean_topic_size = _round(total_chunks / topic_count) if topic_count > 0 else 0.0

    representative_scores = [
        item.score
        for topic in topics
        for item in topic.representative_documents
    ]
    mean_intra = (
        _round(sum(representative_scores) / len(representative_scores))
        if representative_scores
        else None
    )

    vectors_by_chunk = {
        embedding.chunk_id: embedding.vector for embedding in chunk_embeddings
    }
    lead_vectors = []
    for topic in topics:
        if not topic.representative_documents:
            continue
        lead_chunk_id = topic.representative_documents[0].supporting_chunk_id
        vector = vectors_by_chunk.get(lead_chunk_id)
        if vector is not None:
            lead_vectors.append(vector)
    inter_scores = [
        cosine_similarity(vec_a, vec_b)
        for vec_a, vec_b in combinations(lead_vectors, 2)
    ]
    mean_inter = (
        _round(sum(inter_scores) / len(inter_scores)) if inter_scores else None
    )

    topic_items: list[TopicReportItem] = []
    for topic in topics:
        scores = [item.score for item in topic.representative_documents]
        topic_items.append(
            TopicReportItem(
                topic_id=topic.topic_id,
                label=topic.label,
                chunk_count=topic.chunk_count,
                representative_document_count=len(topic.representative_documents),
                mean_representative_score=(
                    _round(sum(scores) / len(scores)) if scores else None
                ),
                lead_document_title=(
                    topic.representative_documents[0].title
                    if topic.representative_documents
                    else None
                ),
            )
        )

    return TopicReport(
        model=model,
        topic_count=topic_count,
        total_chunks=total_chunks,
        largest_topic_ratio=largest_topic_ratio,
        mean_topic_size=mean_topic_size,
        mean_intra_similarity_proxy=mean_intra,
        mean_inter_topic_similarity_proxy=mean_inter,
        topics=topic_items,
    )


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)
