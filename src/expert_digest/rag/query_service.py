"""Shared retrieval pipeline for structured question answering."""

from __future__ import annotations

from pathlib import Path

from expert_digest.processing.embedder import DEFAULT_EMBEDDING_MODEL, embed_text
from expert_digest.rag.answering import StructuredAnswer, build_structured_answer
from expert_digest.retrieval.retriever import (
    hydrate_scored_chunks,
    rank_chunk_embeddings,
)
from expert_digest.storage.sqlite_store import (
    list_chunk_embeddings,
    list_chunks,
    list_documents,
)


def answer_question(
    *,
    question: str,
    db_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3,
    min_score: float = 0.05,
    min_top_score: float | None = None,
    min_avg_score: float = 0.03,
    max_evidence: int = 3,
) -> StructuredAnswer:
    """Run retrieval + deterministic answer composition for one question."""
    resolved_min_top_score = (
        min_top_score if min_top_score is not None else min_score
    )
    chunk_embeddings = list_chunk_embeddings(db_path, model=model)
    if not chunk_embeddings:
        return build_structured_answer(
            question=question,
            evidence_chunks=[],
            max_evidence=max_evidence,
            min_top_score=resolved_min_top_score,
            min_avg_score=min_avg_score,
        )

    query_vector = embed_text(question, dim=chunk_embeddings[0].dimensions)
    ranked = rank_chunk_embeddings(
        query_vector=query_vector,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )
    chunks = {chunk.id: chunk for chunk in list_chunks(db_path)}
    documents = {document.id: document for document in list_documents(db_path)}
    evidence_chunks = hydrate_scored_chunks(
        ranked,
        chunks_by_id=chunks,
        documents_by_id=documents,
    )

    return build_structured_answer(
        question=question,
        evidence_chunks=evidence_chunks,
        max_evidence=max_evidence,
        min_top_score=resolved_min_top_score,
        min_avg_score=min_avg_score,
    )
