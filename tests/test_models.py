from expert_digest.domain.models import Chunk, ChunkEmbedding, Document


def test_document_create_derives_stable_id_from_content():
    first = Document.create(
        author="王老师",
        title="长期主义",
        content="真正的复利来自持续学习。",
        source="sample",
        url="https://example.com/long-term",
        created_at="2026-01-01",
    )
    second = Document.create(
        author="王老师",
        title="长期主义",
        content="真正的复利来自持续学习。",
        source="sample",
        url="https://example.com/long-term",
        created_at="2026-01-01",
    )

    assert first.id == second.id
    assert len(first.id) == 64


def test_document_to_dict_preserves_optional_fields():
    document = Document.create(
        author="王老师",
        title="长期主义",
        content="真正的复利来自持续学习。",
        source="sample",
    )

    assert document.to_dict() == {
        "id": document.id,
        "author": "王老师",
        "title": "长期主义",
        "content": "真正的复利来自持续学习。",
        "source": "sample",
        "url": None,
        "created_at": None,
    }


def test_chunk_create_derives_stable_id_from_document_position_and_text():
    chunk = Chunk.create(
        document_id="doc-1",
        text="第一段内容",
        chunk_index=0,
        start_char=0,
        end_char=5,
    )
    same_chunk = Chunk.create(
        document_id="doc-1",
        text="第一段内容",
        chunk_index=0,
        start_char=0,
        end_char=5,
    )

    assert chunk.id == same_chunk.id
    assert chunk.to_dict() == {
        "id": chunk.id,
        "document_id": "doc-1",
        "text": "第一段内容",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 5,
    }


def test_chunk_embedding_create_derives_stable_id_and_dimensions():
    first = ChunkEmbedding.create(
        chunk_id="chunk-1",
        model="hash-bow-v1",
        vector=[0.1, 0.2, 0.3],
    )
    second = ChunkEmbedding.create(
        chunk_id="chunk-1",
        model="hash-bow-v1",
        vector=[0.1, 0.2, 0.3],
    )

    assert first.id == second.id
    assert first.dimensions == 3
    assert first.to_dict() == {
        "id": first.id,
        "chunk_id": "chunk-1",
        "model": "hash-bow-v1",
        "vector": [0.1, 0.2, 0.3],
        "dimensions": 3,
    }
