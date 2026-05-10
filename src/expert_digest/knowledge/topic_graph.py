"""Graph construction helpers for topic community discovery."""

from __future__ import annotations

from expert_digest.domain.models import ChunkEmbedding
from expert_digest.retrieval.retriever import cosine_similarity


def build_topic_graph(
    *,
    chunk_embeddings: list[ChunkEmbedding],
    similarity_threshold: float = 0.35,
    max_neighbors: int = 8,
) -> dict[str, set[str]]:
    """Build an undirected similarity graph over chunk embeddings."""
    if similarity_threshold < -1.0 or similarity_threshold > 1.0:
        raise ValueError("similarity_threshold must be in [-1.0, 1.0]")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be > 0")

    sorted_embeddings = sorted(chunk_embeddings, key=lambda item: item.chunk_id)
    graph: dict[str, set[str]] = {
        item.chunk_id: set() for item in sorted_embeddings
    }
    if len(sorted_embeddings) <= 1:
        return graph

    for index, item in enumerate(sorted_embeddings):
        neighbors: list[tuple[str, float]] = []
        for other_index, other in enumerate(sorted_embeddings):
            if index == other_index:
                continue
            score = cosine_similarity(item.vector, other.vector)
            if score < similarity_threshold:
                continue
            neighbors.append((other.chunk_id, score))
        neighbors.sort(key=lambda pair: (-pair[1], pair[0]))
        for neighbor_id, _score in neighbors[:max_neighbors]:
            graph[item.chunk_id].add(neighbor_id)

    # Force undirected links to make community detection stable.
    for chunk_id, neighbors in list(graph.items()):
        for neighbor_id in neighbors:
            graph.setdefault(neighbor_id, set()).add(chunk_id)

    return graph
