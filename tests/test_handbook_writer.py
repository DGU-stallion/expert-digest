from __future__ import annotations

from pathlib import Path

from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.generation.handbook_writer import (
    HybridThemeSynthesizer,
    build_handbook,
    write_handbook,
)
from expert_digest.processing.embedder import embed_chunks
from expert_digest.retrieval.retriever import RetrievedChunk
from expert_digest.storage.sqlite_store import (
    clear_chunk_embeddings,
    clear_chunks,
    save_chunk_embeddings,
    save_chunks,
    save_documents,
)


def _prepare_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()

    documents = [
        Document.create(
            author="黄彦臻",
            title="泡泡玛特复盘",
            content="泡泡玛特的核心在于IP运营与预期管理。复盘强调目标与反馈闭环。",
            source="sample",
            url="https://example.com/p1",
        ),
        Document.create(
            author="黄彦臻",
            title="阅读路径的设计",
            content="学习路径要从低门槛到高难度，逐步建立上下文和概念密度。",
            source="sample",
            url="https://example.com/p2",
        ),
    ]
    save_documents(db_path, documents)
    clear_chunks(db_path)
    chunks = [
        Chunk.create(
            document_id=documents[0].id,
            text="泡泡玛特的核心在于IP运营与预期管理。",
            chunk_index=0,
        ),
        Chunk.create(
            document_id=documents[0].id,
            text="复盘强调目标与反馈闭环。",
            chunk_index=1,
        ),
        Chunk.create(
            document_id=documents[1].id,
            text="学习路径要从低门槛到高难度。",
            chunk_index=0,
        ),
        Chunk.create(
            document_id=documents[1].id,
            text="逐步建立上下文和概念密度。",
            chunk_index=1,
        ),
    ]
    save_chunks(db_path, chunks)
    clear_chunk_embeddings(db_path, model="hash-bow-v1")
    embeddings: list[ChunkEmbedding] = embed_chunks(chunks, model="hash-bow-v1")
    save_chunk_embeddings(db_path, embeddings)


def test_build_handbook_contains_required_sections_and_sources():
    db_path = Path("data/processed/test_handbook_writer.sqlite3")
    output_path = Path("data/outputs/test_handbook_writer.md")
    _prepare_db(db_path)
    if output_path.exists():
        output_path.unlink()

    handbook = build_handbook(
        db_path=db_path,
        author="黄彦臻",
        top_k=3,
        max_themes=3,
    )
    write_handbook(handbook, output_path=output_path)

    markdown = output_path.read_text(encoding="utf-8")
    assert handbook.author == "黄彦臻"
    assert len(handbook.source_document_ids) >= 2
    assert "## 专家内容总览" in markdown
    assert "## 核心主题初稿" in markdown
    assert "## 每个主题的核心观点" in markdown
    assert "## 推荐阅读路径" in markdown
    assert "## 原文索引" in markdown


def test_hybrid_theme_synthesizer_prefers_llm_when_client_available():
    class _FakeLLMClient:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return "LLM主题总结"

    synthesizer = HybridThemeSynthesizer(llm_client=_FakeLLMClient())
    summary = synthesizer.summarize_theme(
        theme_name="核心能力",
        question="作者的核心能力是什么？",
        evidence_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                score=0.9,
                document_id="d1",
                title="泡泡玛特复盘",
                author="黄彦臻",
                text="泡泡玛特的核心在于IP运营与预期管理。",
                url="https://example.com/p1",
            )
        ],
    )

    assert "LLM主题总结" in summary


def test_hybrid_theme_synthesizer_falls_back_when_llm_fails():
    class _FailingLLMClient:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("llm unavailable")

    synthesizer = HybridThemeSynthesizer(llm_client=_FailingLLMClient())
    summary = synthesizer.summarize_theme(
        theme_name="核心能力",
        question="作者的核心能力是什么？",
        evidence_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                score=0.9,
                document_id="d1",
                title="泡泡玛特复盘",
                author="黄彦臻",
                text="泡泡玛特的核心在于IP运营与预期管理。",
                url="https://example.com/p1",
            )
        ],
    )

    assert "泡泡玛特的核心在于IP运营与预期管理" in summary
