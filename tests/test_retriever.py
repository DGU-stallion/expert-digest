from expert_digest.domain.models import ChunkEmbedding
from expert_digest.retrieval.retriever import cosine_similarity, rank_chunk_embeddings


def _embedding(chunk_id: str, vector: list[float]) -> ChunkEmbedding:
    return ChunkEmbedding.create(
        chunk_id=chunk_id,
        model="test-model",
        vector=vector,
    )


def test_cosine_similarity_returns_one_for_identical_vectors():
    score = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert round(score, 6) == 1.0


def test_rank_chunk_embeddings_returns_highest_scores_first():
    ranked = rank_chunk_embeddings(
        query_vector=[1.0, 0.0],
        chunk_embeddings=[
            _embedding("a", [1.0, 0.0]),
            _embedding("b", [0.8, 0.2]),
            _embedding("c", [0.0, 1.0]),
        ],
        top_k=2,
    )

    assert [item.chunk_id for item in ranked] == ["a", "b"]
    assert ranked[0].score >= ranked[1].score
