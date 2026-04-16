"""Similarity retrieval primitives for chunk embeddings."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from expert_digest.domain.models import ChunkEmbedding


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compute cosine similarity, returning 0.0 for zero-norm vectors."""
    if len(first) != len(second):
        raise ValueError("vector length mismatch")
    dot = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot / (first_norm * second_norm)


def rank_chunk_embeddings(
    *,
    query_vector: list[float],
    chunk_embeddings: Iterable[ChunkEmbedding],
    top_k: int = 5,
) -> list[ScoredChunk]:
    """Rank chunk embeddings by cosine similarity and return top_k hits."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    scored = [
        ScoredChunk(
            chunk_id=embedding.chunk_id,
            score=cosine_similarity(query_vector, embedding.vector),
        )
        for embedding in chunk_embeddings
    ]
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    return ranked[:top_k]
