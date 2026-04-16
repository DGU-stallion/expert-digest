import sqlite3

from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.storage.sqlite_store import (
    DEFAULT_DATABASE_PATH,
    clear_chunk_embeddings,
    clear_chunks,
    get_documents_by_author,
    list_chunk_embeddings,
    list_chunks_for_document,
    list_documents,
    save_chunk_embeddings,
    save_chunks,
    save_documents,
)


def _document(title: str, author: str = "刘老师") -> Document:
    return Document.create(
        author=author,
        title=title,
        content=f"{title} 的正文",
        source="sample",
        url=f"https://example.com/{title}",
    )


def test_save_documents_creates_database_and_preserves_document_fields(tmp_path):
    db_path = tmp_path / "nested" / "expert_digest.sqlite3"
    document = _document("第一篇")

    saved_count = save_documents(db_path, [document])
    documents = list_documents(db_path)

    assert saved_count == 1
    assert db_path.exists()
    assert documents == [document]


def test_save_documents_is_idempotent_for_same_document_id(tmp_path):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = _document("第一篇")

    save_documents(db_path, [document])
    save_documents(db_path, [document])

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1


def test_get_documents_by_author_returns_only_matching_documents(tmp_path):
    db_path = tmp_path / "expert_digest.sqlite3"
    matching = _document("第一篇", author="刘老师")
    other = _document("第二篇", author="张老师")
    save_documents(db_path, [matching, other])

    documents = get_documents_by_author(db_path, "刘老师")

    assert documents == [matching]


def test_default_database_path_points_to_processed_data():
    assert str(DEFAULT_DATABASE_PATH) == "data\\processed\\expert_digest.sqlite3"


def test_save_chunks_and_list_by_document(tmp_path):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = _document("第一篇")
    save_documents(db_path, [document])
    chunks = [
        Chunk.create(document_id=document.id, text="第一段", chunk_index=0),
        Chunk.create(document_id=document.id, text="第二段", chunk_index=1),
    ]

    saved_count = save_chunks(db_path, chunks)
    loaded = list_chunks_for_document(db_path, document.id)

    assert saved_count == 2
    assert loaded == chunks


def test_clear_chunks_removes_existing_rows(tmp_path):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = _document("第一篇")
    save_documents(db_path, [document])
    save_chunks(
        db_path,
        [
            Chunk.create(document_id=document.id, text="第一段", chunk_index=0),
            Chunk.create(document_id=document.id, text="第二段", chunk_index=1),
        ],
    )

    cleared = clear_chunks(db_path)

    assert cleared == 2
    assert list_chunks_for_document(db_path, document.id) == []


def test_save_and_clear_chunk_embeddings(tmp_path):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = _document("第一篇")
    chunk = Chunk.create(document_id=document.id, text="第一段", chunk_index=0)
    save_documents(db_path, [document])
    save_chunks(db_path, [chunk])
    embeddings = [
        ChunkEmbedding.create(
            chunk_id=chunk.id,
            model="hash-bow-v1",
            vector=[0.1, 0.2, 0.3],
        )
    ]

    saved = save_chunk_embeddings(db_path, embeddings)
    loaded = list_chunk_embeddings(db_path, model="hash-bow-v1")
    cleared = clear_chunk_embeddings(db_path, model="hash-bow-v1")

    assert saved == 1
    assert loaded == embeddings
    assert cleared == 1
    assert list_chunk_embeddings(db_path, model="hash-bow-v1") == []
