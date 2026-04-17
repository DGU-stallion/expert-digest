from expert_digest.domain.models import Chunk
from expert_digest.processing.embedder import (
    DEFAULT_EMBEDDING_DIM,
    embed_chunk,
    embed_text,
)


def test_embed_text_is_deterministic_and_has_expected_length():
    first = embed_text("泡泡玛特 估值 与 预期", dim=64)
    second = embed_text("泡泡玛特 估值 与 预期", dim=64)

    assert first == second
    assert len(first) == 64
    assert any(value != 0.0 for value in first)


def test_embed_chunk_generates_embedding_with_metadata():
    chunk = Chunk.create(
        document_id="doc-1",
        text="产业周期与流动性预期",
        chunk_index=0,
    )

    embedding = embed_chunk(chunk, model="hash-bow-v1", dim=DEFAULT_EMBEDDING_DIM)

    assert embedding.chunk_id == chunk.id
    assert embedding.model == "hash-bow-v1"
    assert embedding.dimensions == DEFAULT_EMBEDDING_DIM
    assert len(embedding.vector) == DEFAULT_EMBEDDING_DIM
