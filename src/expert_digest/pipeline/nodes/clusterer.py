"""Content clustering node: embed chunks and run community-detection clustering."""

from __future__ import annotations

from dataclasses import asdict

from expert_digest.knowledge.topic_clusterer import (
    DeterministicTopicLabeler,
    TopicCluster,
    build_topic_clusters,
)
from expert_digest.pipeline.state import DigestState

_MIN_CLUSTER_SIZE_RATIO = 0.15


def run_cluster_content(state: DigestState) -> dict:
    """Run topic clustering on all chunk embeddings and filter to major topics.

    Loads chunks and their embeddings from the database, runs
    community-detection clustering, then drops small clusters
    (below 15% of the largest cluster size) to focus the handbook
    on major themes only.
    """
    db_path = state.get("db_path", "")
    if not db_path:
        return {"topic_clusters": []}

    all_clusters = build_topic_clusters(
        db_path=db_path,
        num_topics=12,
        top_docs_per_topic=5,
        max_iter=30,
        labeler=DeterministicTopicLabeler(),
    )

    if not all_clusters:
        return {"topic_clusters": []}

    # Keep only major clusters: those with >= 15% of the largest cluster size
    max_size = max(c.chunk_count for c in all_clusters)
    threshold = max(int(max_size * _MIN_CLUSTER_SIZE_RATIO), 2)
    major_clusters = [c for c in all_clusters if c.chunk_count >= threshold]

    return {"topic_clusters": _serialize_clusters(major_clusters)}


def _serialize_clusters(clusters: list[TopicCluster]) -> list[dict]:
    """Convert TopicCluster dataclasses to dicts for pipeline state."""
    result: list[dict] = []
    for c in clusters:
        d = asdict(c)
        d["representative_documents"] = [
            asdict(rd) for rd in c.representative_documents
        ]
        result.append(d)
    return result
