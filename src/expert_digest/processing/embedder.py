"""Deterministic local embedding helpers."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

from expert_digest.domain.models import Chunk, ChunkEmbedding

DEFAULT_EMBEDDING_MODEL = "hash-bow-v1"
DEFAULT_EMBEDDING_DIM = 256


def embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Embed text into a fixed-size vector using a hash bag-of-words strategy."""
    if dim <= 0:
        raise ValueError("dim must be > 0")

    vector = [0.0] * dim
    tokens = _tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _l2_normalize(vector)


def embed_chunk(
    chunk: Chunk,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> ChunkEmbedding:
    """Create one chunk embedding using the requested local embedding model."""
    if model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(f"unsupported embedding model: {model}")
    vector = embed_text(chunk.text, dim=dim)
    return ChunkEmbedding.create(
        chunk_id=chunk.id,
        model=model,
        vector=vector,
    )


def embed_chunks(
    chunks: Iterable[Chunk],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> list[ChunkEmbedding]:
    """Create embeddings for all chunks in stable iteration order."""
    return [embed_chunk(chunk, model=model, dim=dim) for chunk in chunks]


def _tokenize(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []

    latin_tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens = latin_tokens + cjk_tokens
    if tokens:
        return tokens
    return [char for char in normalized if not char.isspace()]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
