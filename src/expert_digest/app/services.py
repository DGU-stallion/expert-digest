"""Application-facing service helpers for the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from expert_digest.domain.models import Handbook
from expert_digest.generation.handbook_writer import (
    DeterministicThemeSynthesizer,
    HybridThemeSynthesizer,
    build_handbook,
    write_handbook,
)
from expert_digest.generation.llm_client import (
    DEFAULT_CCSWITCH_DB_PATH,
    create_default_handbook_llm_client,
)
from expert_digest.ingest.jsonl_loader import load_jsonl_documents
from expert_digest.ingest.markdown_loader import load_markdown_documents
from expert_digest.ingest.zhihu_loader import load_zhihu_documents
from expert_digest.processing.cleaner import clean_document
from expert_digest.processing.embedder import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    embed_chunks,
)
from expert_digest.processing.splitter import split_documents
from expert_digest.rag.answering import StructuredAnswer
from expert_digest.rag.query_service import answer_question
from expert_digest.storage.sqlite_store import (
    clear_chunk_embeddings,
    clear_chunks,
    list_chunk_embeddings,
    list_chunks,
    list_documents,
    save_chunk_embeddings,
    save_chunks,
    save_documents,
)


@dataclass(frozen=True)
class DataOverview:
    document_count: int
    chunk_count: int
    embedding_count: int
    authors: list[str]


@dataclass(frozen=True)
class HandbookResult:
    handbook: Handbook
    output_path: Path


def collect_data_overview(
    *,
    db_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> DataOverview:
    documents = list_documents(db_path)
    chunks = list_chunks(db_path)
    embeddings = list_chunk_embeddings(db_path, model=model)
    authors = sorted({document.author for document in documents})
    return DataOverview(
        document_count=len(documents),
        chunk_count=len(chunks),
        embedding_count=len(embeddings),
        authors=authors,
    )


def import_documents(
    *,
    kind: str,
    source_path: str | Path,
    db_path: str | Path,
) -> int:
    source = Path(source_path)
    loaders = {
        "jsonl": load_jsonl_documents,
        "markdown": load_markdown_documents,
        "zhihu": load_zhihu_documents,
    }
    loader = loaders.get(kind)
    if loader is None:
        raise ValueError(f"unsupported import kind: {kind}")
    documents = loader(source)
    return save_documents(db_path, documents)


def rebuild_chunks(
    *,
    db_path: str | Path,
    max_chars: int,
    min_chars: int,
) -> int:
    documents = list_documents(db_path)
    clear_chunks(db_path)
    cleaned_documents = [clean_document(document) for document in documents]
    chunks = split_documents(
        cleaned_documents,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    return save_chunks(db_path, chunks)


def rebuild_embeddings(
    *,
    db_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> int:
    chunks = list_chunks(db_path)
    clear_chunk_embeddings(db_path, model=model)
    embeddings = embed_chunks(chunks, model=model, dim=dim)
    return save_chunk_embeddings(db_path, embeddings)


def ask(
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
    return answer_question(
        question=question,
        db_path=db_path,
        model=model,
        top_k=top_k,
        min_score=min_score,
        min_top_score=min_top_score,
        min_avg_score=min_avg_score,
        max_evidence=max_evidence,
    )


def generate_handbook(
    *,
    db_path: str | Path,
    author: str | None,
    model: str,
    top_k: int,
    max_themes: int,
    output_path: str | Path,
    synthesis_mode: str = "hybrid",
    ccswitch_db_path: str | Path = DEFAULT_CCSWITCH_DB_PATH,
    llm_timeout: int = 30,
    llm_max_tokens: int = 700,
) -> HandbookResult:
    if synthesis_mode not in {"hybrid", "deterministic"}:
        raise ValueError(f"unsupported synthesis mode: {synthesis_mode}")

    if synthesis_mode == "hybrid":
        llm_client = create_default_handbook_llm_client(
            ccswitch_db_path=ccswitch_db_path,
            timeout_seconds=llm_timeout,
            max_output_tokens=llm_max_tokens,
        )
        synthesizer = HybridThemeSynthesizer(llm_client=llm_client)
    else:
        synthesizer = DeterministicThemeSynthesizer()

    handbook = build_handbook(
        db_path=db_path,
        author=author,
        model=model,
        top_k=top_k,
        max_themes=max_themes,
        synthesizer=synthesizer,
    )
    resolved_output_path = write_handbook(handbook=handbook, output_path=output_path)
    return HandbookResult(handbook=handbook, output_path=resolved_output_path)
