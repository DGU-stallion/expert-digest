from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.rag.answering import StructuredAnswer
from expert_digest.rag.query_service import answer_question
from expert_digest.retrieval.retriever import ScoredChunk


def test_answer_question_returns_refusal_when_no_embeddings(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_build_structured_answer(**kwargs):
        captured.update(kwargs)
        return StructuredAnswer(
            answer="拒答",
            evidence=[],
            recommended_original=[],
            uncertainty="无证据",
            refused=True,
        )

    monkeypatch.setattr(
        "expert_digest.rag.query_service.list_chunk_embeddings",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.build_structured_answer",
        _fake_build_structured_answer,
    )

    result = answer_question(
        question="泡泡玛特的核心能力是什么？",
        db_path="data/processed/zhihu_huang.sqlite3",
    )

    assert result.refused is True
    assert captured["question"] == "泡泡玛特的核心能力是什么？"
    assert captured["evidence_chunks"] == []


def test_answer_question_hydrates_ranked_chunks(monkeypatch):
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )
    captured: dict[str, object] = {}

    def _fake_build_structured_answer(**kwargs):
        captured.update(kwargs)
        return StructuredAnswer(
            answer="可回答",
            evidence=[],
            recommended_original=[],
            uncertainty="低",
            refused=False,
        )

    monkeypatch.setattr(
        "expert_digest.rag.query_service.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.embed_text",
        lambda *_a, **_k: [1.0, 0.0],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.rank_chunk_embeddings",
        lambda **_k: [ScoredChunk(chunk_id=chunk.id, score=0.91)],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.list_chunks",
        lambda *_a, **_k: [chunk],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr(
        "expert_digest.rag.query_service.build_structured_answer",
        _fake_build_structured_answer,
    )

    result = answer_question(
        question="泡泡玛特的核心能力是什么？",
        db_path="data/processed/zhihu_huang.sqlite3",
        top_k=1,
        max_evidence=2,
        min_top_score=0.1,
        min_avg_score=0.1,
    )

    assert result.refused is False
    evidence_chunks = captured["evidence_chunks"]
    assert len(evidence_chunks) == 1
    assert evidence_chunks[0].chunk_id == chunk.id
    assert evidence_chunks[0].score == 0.91
