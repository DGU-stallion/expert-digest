"""Topic clustering primitives for M6 knowledge enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.processing.embedder import DEFAULT_EMBEDDING_MODEL
from expert_digest.retrieval.retriever import cosine_similarity
from expert_digest.storage.sqlite_store import (
    list_chunk_embeddings,
    list_chunks,
    list_documents,
)


@dataclass(frozen=True)
class TopicRepresentativeDocument:
    document_id: str
    title: str
    author: str
    url: str | None
    score: float
    supporting_chunk_id: str


@dataclass(frozen=True)
class TopicCluster:
    topic_id: str
    label: str
    chunk_count: int
    representative_chunk_ids: list[str]
    representative_documents: list[TopicRepresentativeDocument]


def build_topic_clusters(
    *,
    db_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    num_topics: int = 3,
    top_docs_per_topic: int = 3,
    max_iter: int = 30,
) -> list[TopicCluster]:
    documents = list_documents(db_path)
    chunks = list_chunks(db_path)
    chunk_embeddings = list_chunk_embeddings(db_path, model=model)
    return cluster_chunks_by_embeddings(
        chunks_by_id={chunk.id: chunk for chunk in chunks},
        documents_by_id={document.id: document for document in documents},
        chunk_embeddings=chunk_embeddings,
        num_topics=num_topics,
        top_docs_per_topic=top_docs_per_topic,
        max_iter=max_iter,
    )


def cluster_chunks_by_embeddings(
    *,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
    chunk_embeddings: list[ChunkEmbedding],
    num_topics: int = 3,
    top_docs_per_topic: int = 3,
    max_iter: int = 30,
) -> list[TopicCluster]:
    if num_topics <= 0:
        raise ValueError("num_topics must be > 0")
    if top_docs_per_topic <= 0:
        raise ValueError("top_docs_per_topic must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    entries = []
    sorted_embeddings = sorted(
        chunk_embeddings,
        key=lambda item: (item.chunk_id, item.id),
    )
    for embedding in sorted_embeddings:
        chunk = chunks_by_id.get(embedding.chunk_id)
        if chunk is None:
            continue
        document = documents_by_id.get(chunk.document_id)
        if document is None:
            continue
        entries.append((chunk, document, embedding.vector))

    if not entries:
        return []

    vectors = [entry[2] for entry in entries]
    topic_count = min(num_topics, len(vectors))
    centroids = _initialize_centroids(vectors=vectors, topic_count=topic_count)

    assignments: list[int] | None = None
    for _ in range(max_iter):
        next_assignments = [
            _best_centroid_index(vector, centroids) for vector in vectors
        ]
        if assignments == next_assignments:
            break
        assignments = next_assignments
        centroids = _recompute_centroids(
            vectors=vectors,
            assignments=assignments,
            previous_centroids=centroids,
            topic_count=topic_count,
        )

    assert assignments is not None

    raw_clusters: list[dict[str, object]] = []
    for cluster_index in range(topic_count):
        cluster_entries = [
            entries[index]
            for index, assigned in enumerate(assignments)
            if assigned == cluster_index
        ]
        if not cluster_entries:
            continue

        centroid = centroids[cluster_index]
        chunk_scores: list[tuple[str, float, str]] = []
        doc_scores: dict[str, tuple[float, str]] = {}

        for chunk, document, vector in cluster_entries:
            score = cosine_similarity(vector, centroid)
            chunk_scores.append((chunk.id, score, document.id))
            current = doc_scores.get(document.id)
            if current is None or score > current[0]:
                doc_scores[document.id] = (score, chunk.id)

        representative_chunk_ids = [
            item[0]
            for item in sorted(chunk_scores, key=lambda item: item[1], reverse=True)[:5]
        ]

        representative_documents: list[TopicRepresentativeDocument] = []
        for document_id, (score, chunk_id) in sorted(
            doc_scores.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:top_docs_per_topic]:
            document = documents_by_id[document_id]
            representative_documents.append(
                TopicRepresentativeDocument(
                    document_id=document.id,
                    title=document.title,
                    author=document.author,
                    url=document.url,
                    score=score,
                    supporting_chunk_id=chunk_id,
                )
            )

        raw_clusters.append(
            {
                "chunk_count": len(cluster_entries),
                "representative_chunk_ids": representative_chunk_ids,
                "representative_documents": representative_documents,
                "mean_score": mean(score for _, score, _ in chunk_scores),
            }
        )

    ranked = sorted(
        raw_clusters,
        key=lambda item: (
            -int(item["chunk_count"]),
            -float(item["mean_score"]),
            item["representative_documents"][0].title
            if item["representative_documents"]
            else "",
        ),
    )

    topics: list[TopicCluster] = []
    for index, item in enumerate(ranked, start=1):
        representatives = item["representative_documents"]
        lead_title = representatives[0].title if representatives else "未命名主题"
        topics.append(
            TopicCluster(
                topic_id=f"topic-{index}",
                label=f"主题{index}：{_shorten_label(lead_title)}",
                chunk_count=int(item["chunk_count"]),
                representative_chunk_ids=list(item["representative_chunk_ids"]),
                representative_documents=list(representatives),
            )
        )
    return topics


def _initialize_centroids(
    *,
    vectors: list[list[float]],
    topic_count: int,
) -> list[list[float]]:
    if topic_count == 1:
        return [_normalize_vector(vectors[0][:])]

    selected_indexes = {
        min(len(vectors) - 1, int(round(i * (len(vectors) - 1) / (topic_count - 1))))
        for i in range(topic_count)
    }
    while len(selected_indexes) < topic_count:
        selected_indexes.add(len(selected_indexes))
    return [_normalize_vector(vectors[index][:]) for index in sorted(selected_indexes)]


def _best_centroid_index(vector: list[float], centroids: list[list[float]]) -> int:
    best_index = 0
    best_score = cosine_similarity(vector, centroids[0])
    for index in range(1, len(centroids)):
        score = cosine_similarity(vector, centroids[index])
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _recompute_centroids(
    *,
    vectors: list[list[float]],
    assignments: list[int],
    previous_centroids: list[list[float]],
    topic_count: int,
) -> list[list[float]]:
    dimensions = len(vectors[0])
    grouped: list[list[list[float]]] = [[] for _ in range(topic_count)]
    for index, assignment in enumerate(assignments):
        grouped[assignment].append(vectors[index])

    centroids: list[list[float]] = []
    for index in range(topic_count):
        group = grouped[index]
        if not group:
            centroids.append(previous_centroids[index])
            continue
        centroid = [0.0] * dimensions
        for vector in group:
            for dim in range(dimensions):
                centroid[dim] += vector[dim]
        centroids.append(_normalize_vector(centroid))
    return centroids


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _shorten_label(title: str, limit: int = 18) -> str:
    compact = " ".join(title.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
