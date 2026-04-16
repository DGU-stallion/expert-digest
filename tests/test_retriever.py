from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.retrieval.retriever import (
    ScoredChunk,
    cosine_similarity,
    hydrate_scored_chunks,
    rank_chunk_embeddings,
)


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


def test_hydrate_scored_chunks_joins_chunk_and_document_fields():
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="正文",
        source="sample",
        url="https://example.com/p1",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )

    hydrated = hydrate_scored_chunks(
        [ScoredChunk(chunk_id=chunk.id, score=0.9)],
        chunks_by_id={chunk.id: chunk},
        documents_by_id={document.id: document},
    )

    assert len(hydrated) == 1
    assert hydrated[0].chunk_id == chunk.id
    assert hydrated[0].title == "泡泡玛特复盘"
    assert hydrated[0].author == "黄彦臻"
    assert hydrated[0].url == "https://example.com/p1"
