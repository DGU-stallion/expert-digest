"""Topic clustering primitives for M6 knowledge enhancement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Protocol

from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.knowledge.community_detection import detect_communities
from expert_digest.knowledge.topic_graph import build_topic_graph
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


class TopicNamingLLMClient(Protocol):
    """Protocol for topic-label generation compatible with handbook llm client."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


class TopicLabeler(Protocol):
    """Protocol for topic label generation strategy."""

    def label_topic(self, *, topic: TopicCluster, topic_index: int) -> str: ...


class DeterministicTopicLabeler:
    """Stable fallback labeler based on representative document title."""

    def label_topic(self, *, topic: TopicCluster, topic_index: int) -> str:
        if topic.representative_documents:
            lead_title = topic.representative_documents[0].title
        else:
            lead_title = "未命名主题"
        return f"主题{topic_index}：{_shorten_label(lead_title)}"


class LLMTopicLabeler:
    """LLM-first topic labeler with deterministic fallback."""

    def __init__(
        self,
        *,
        llm_client: TopicNamingLLMClient | None,
        fallback: TopicLabeler | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback or DeterministicTopicLabeler()
        self._llm_attempts = 0
        self._llm_failures = 0
        self._last_error_reason: str | None = None

    def label_topic(self, *, topic: TopicCluster, topic_index: int) -> str:
        if self._llm_client is not None:
            self._llm_attempts += 1
            system_prompt, user_prompt = _build_topic_label_prompts(topic=topic)
            try:
                raw = self._llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ).strip()
                candidate = _extract_candidate_label(raw)
                if candidate:
                    return f"主题{topic_index}：{_shorten_label(candidate)}"
            except Exception:
                self._llm_failures += 1
                self._last_error_reason = "topic_label_generation_error"
        return self._fallback.label_topic(topic=topic, topic_index=topic_index)

    def runtime_metadata(self) -> dict[str, object]:
        if self._llm_client is None:
            return {
                "fallback_used": True,
                "error_reason": "llm_client_unavailable",
            }
        if self._llm_failures > 0:
            return {
                "fallback_used": True,
                "error_reason": (
                    self._last_error_reason or "topic_label_generation_error"
                ),
            }
        return {
            "fallback_used": False,
            "error_reason": None,
        }


def build_topic_clusters(
    *,
    db_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    num_topics: int = 3,
    top_docs_per_topic: int = 3,
    max_iter: int = 30,
    labeler: TopicLabeler | None = None,
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
        labeler=labeler,
    )


def cluster_chunks_by_embeddings(
    *,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
    chunk_embeddings: list[ChunkEmbedding],
    num_topics: int = 3,
    top_docs_per_topic: int = 3,
    max_iter: int = 30,
    labeler: TopicLabeler | None = None,
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

    entry_by_chunk_id: dict[str, tuple[Chunk, Document, list[float]]] = {
        chunk.id: (chunk, document, vector)
        for chunk, document, vector in entries
    }
    filtered_embeddings = [
        embedding
        for embedding in sorted_embeddings
        if embedding.chunk_id in entry_by_chunk_id
    ]
    graph = build_topic_graph(
        chunk_embeddings=filtered_embeddings,
        similarity_threshold=0.35,
        max_neighbors=8,
    )
    communities = detect_communities(graph=graph, max_iter=max_iter)
    if not communities:
        communities = [[chunk_id] for chunk_id in sorted(entry_by_chunk_id.keys())]
    communities = communities[: min(num_topics, len(communities))]

    raw_clusters: list[dict[str, object]] = []
    for community in communities:
        cluster_entries = [
            entry_by_chunk_id[chunk_id]
            for chunk_id in community
            if chunk_id in entry_by_chunk_id
        ]
        if not cluster_entries:
            continue

        vectors = [entry[2] for entry in cluster_entries]
        centroid = _normalize_vector(
            [
                sum(vector[dim] for vector in vectors)
                for dim in range(len(vectors[0]))
            ]
        )
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
    active_labeler = labeler or DeterministicTopicLabeler()
    fallback_labeler = DeterministicTopicLabeler()
    for index, item in enumerate(ranked, start=1):
        representatives = item["representative_documents"]
        topic = TopicCluster(
            topic_id=f"topic-{index}",
            label="",
            chunk_count=int(item["chunk_count"]),
            representative_chunk_ids=list(item["representative_chunk_ids"]),
            representative_documents=list(representatives),
        )
        label = active_labeler.label_topic(topic=topic, topic_index=index).strip()
        if not label:
            label = fallback_labeler.label_topic(topic=topic, topic_index=index)
        topics.append(replace(topic, label=label))
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


def _build_topic_label_prompts(*, topic: TopicCluster) -> tuple[str, str]:
    system_prompt = (
        "你是一个主题命名助手。请根据证据为聚类主题命名，"
        "输出一个简短中文短语，不要编造。"
    )
    representative_lines = [
        f"{index}. {item.title} | score={item.score:.4f}"
        for index, item in enumerate(topic.representative_documents, start=1)
    ]
    user_prompt = (
        f"topic_id: {topic.topic_id}\n"
        f"chunk_count: {topic.chunk_count}\n"
        "代表文档：\n"
        + ("\n".join(representative_lines) if representative_lines else "(无)")
        + "\n\n请只返回一个主题名（不超过12个字，不要解释）。"
    )
    return system_prompt, user_prompt


def _extract_candidate_label(raw: str) -> str:
    if not raw:
        return ""
    line = raw.splitlines()[0].strip().strip("`\"' ")
    if not line:
        return ""
    if "：" in line and line.startswith("主题"):
        line = line.split("：", maxsplit=1)[1].strip()
    if ":" in line and line.lower().startswith("topic"):
        line = line.split(":", maxsplit=1)[1].strip()
    return line
