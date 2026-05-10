from expert_digest.domain.models import ChunkEmbedding
from expert_digest.knowledge.community_detection import detect_communities
from expert_digest.knowledge.topic_graph import build_topic_graph


def test_build_topic_graph_connects_high_similarity_nodes():
    embeddings = [
        ChunkEmbedding.create(
            chunk_id="a",
            model="hash-bow-v1",
            vector=[1.0, 0.0],
        ),
        ChunkEmbedding.create(
            chunk_id="b",
            model="hash-bow-v1",
            vector=[0.95, 0.05],
        ),
        ChunkEmbedding.create(
            chunk_id="c",
            model="hash-bow-v1",
            vector=[-1.0, 0.0],
        ),
    ]

    graph = build_topic_graph(
        chunk_embeddings=embeddings,
        similarity_threshold=0.5,
        max_neighbors=2,
    )

    assert graph["a"] == {"b"}
    assert graph["b"] == {"a"}
    assert graph["c"] == set()


def test_detect_communities_groups_connected_nodes():
    graph = {
        "a": {"b"},
        "b": {"a"},
        "c": set(),
    }

    communities = detect_communities(graph=graph, max_iter=20)
    normalized = sorted(sorted(items) for items in communities)

    assert normalized == [["a", "b"], ["c"]]
