"""Service toolkit used by the M8 MCP server tools."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from expert_digest.app import services
from expert_digest.domain.models import Document
from expert_digest.knowledge.topic_clusterer import (
    DeterministicTopicLabeler,
    TopicCluster,
    cluster_chunks_by_embeddings,
)
from expert_digest.processing.embedder import DEFAULT_EMBEDDING_MODEL, embed_text
from expert_digest.rag.answering import StructuredAnswer, build_structured_answer
from expert_digest.rag.query_service import answer_question
from expert_digest.retrieval.retriever import (
    hydrate_scored_chunks,
    rank_chunk_embeddings,
)
from expert_digest.storage.sqlite_store import (
    get_documents_by_author,
    list_chunk_embeddings,
    list_chunks,
    list_documents,
)


@dataclass(frozen=True)
class MCPToolkit:
    """Adapter that maps project services to MCP-friendly tool outputs."""

    db_path: Path
    model: str = DEFAULT_EMBEDDING_MODEL
    output_dir: Path = Path("data/outputs")

    def ask_author(
        self,
        *,
        question: str,
        author_id: str | None = None,
        top_k: int = 3,
        max_evidence: int = 3,
        min_top_score: float = 0.05,
        min_avg_score: float = 0.03,
    ) -> dict[str, object]:
        if author_id:
            result = self._answer_question_for_author(
                question=question,
                author_id=author_id,
                top_k=top_k,
                max_evidence=max_evidence,
                min_top_score=min_top_score,
                min_avg_score=min_avg_score,
            )
        else:
            result = answer_question(
                question=question,
                db_path=self.db_path,
                model=self.model,
                top_k=top_k,
                max_evidence=max_evidence,
                min_top_score=min_top_score,
                min_avg_score=min_avg_score,
            )
        payload = asdict(result)
        payload["author_filter"] = author_id
        return payload

    def search_posts(
        self,
        *,
        query: str,
        author_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, object]:
        documents = self._resolve_documents(author_id=author_id)
        if not documents:
            return {
                "query": query,
                "author_filter": author_id,
                "hits": [],
            }

        retrieved = self._retrieve_chunks(
            query=query,
            documents=documents,
            top_k=max(top_k * 2, top_k),
        )
        hits: list[dict[str, object]] = []
        seen_documents: set[str] = set()
        for item in retrieved:
            if item.document_id in seen_documents:
                continue
            seen_documents.add(item.document_id)
            hits.append(
                {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "title": item.title,
                    "author": item.author,
                    "snippet": item.text,
                    "score": item.score,
                    "url": item.url,
                }
            )
            if len(hits) >= top_k:
                break
        return {
            "query": query,
            "author_filter": author_id,
            "hits": hits,
        }

    def recommend_readings(
        self,
        *,
        question: str,
        author_id: str | None = None,
        top_k: int = 3,
    ) -> dict[str, object]:
        answer_payload = self.ask_author(
            question=question,
            author_id=author_id,
            top_k=top_k,
        )
        evidence = answer_payload.get("evidence", [])
        evidence_documents: list[dict[str, object]] = []
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                evidence_documents.append(
                    {
                        "title": item.get("title"),
                        "author": item.get("author"),
                        "score": item.get("score"),
                        "url": item.get("url"),
                    }
                )
        return {
            "question": question,
            "author_filter": author_id,
            "recommended_original": answer_payload.get("recommended_original", []),
            "evidence_documents": evidence_documents,
            "uncertainty": answer_payload.get("uncertainty"),
            "refused": bool(answer_payload.get("refused", False)),
        }

    def list_topics(
        self,
        *,
        author_id: str | None = None,
        num_topics: int = 3,
        top_docs: int = 3,
        max_iter: int = 30,
    ) -> dict[str, object]:
        topics = self._build_topics_for_author(
            author_id=author_id,
            num_topics=num_topics,
            top_docs=top_docs,
            max_iter=max_iter,
        )
        return {
            "author_filter": author_id,
            "topics": [asdict(topic) for topic in topics],
        }

    def generate_handbook(
        self,
        *,
        author_id: str | None = None,
        output_path: str | Path | None = None,
        synthesis_mode: str = "deterministic",
        theme_source: str = "cluster",
        num_topics: int = 3,
        top_k: int = 3,
        max_themes: int = 4,
    ) -> dict[str, object]:
        resolved_output = (
            Path(output_path)
            if output_path is not None
            else self.output_dir / f"mcp_handbook_{_slug(author_id or 'all')}.md"
        )
        result = services.generate_handbook(
            db_path=self.db_path,
            author=author_id,
            model=self.model,
            top_k=top_k,
            max_themes=max_themes,
            output_path=resolved_output,
            synthesis_mode=synthesis_mode,
            theme_source=theme_source,
            num_topics=num_topics,
        )
        return {
            "author": result.handbook.author,
            "title": result.handbook.title,
            "output_path": result.output_path.as_posix(),
            "source_document_ids": result.handbook.source_document_ids,
            "synthesis_mode": synthesis_mode,
            "theme_source": theme_source,
        }

    def generate_skill(
        self,
        *,
        author_id: str | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, object]:
        resolved_output = (
            Path(output_path)
            if output_path is not None
            else self.output_dir / f"mcp_skill_{_slug(author_id or 'all')}.md"
        )
        result = services.generate_skill_draft(
            db_path=self.db_path,
            author=author_id,
            output_path=resolved_output,
        )
        return {
            "author": result.profile.get("author"),
            "document_count": result.profile.get("document_count"),
            "output_path": result.output_path.as_posix(),
            "markdown_preview": "\n".join(result.markdown.splitlines()[:24]),
        }

    def _answer_question_for_author(
        self,
        *,
        question: str,
        author_id: str,
        top_k: int,
        max_evidence: int,
        min_top_score: float,
        min_avg_score: float,
    ) -> StructuredAnswer:
        documents = self._resolve_documents(author_id=author_id)
        if not documents:
            return build_structured_answer(
                question=question,
                evidence_chunks=[],
                max_evidence=max_evidence,
                min_top_score=min_top_score,
                min_avg_score=min_avg_score,
            )
        retrieved = self._retrieve_chunks(
            query=question,
            documents=documents,
            top_k=top_k,
        )
        return build_structured_answer(
            question=question,
            evidence_chunks=retrieved,
            max_evidence=max_evidence,
            min_top_score=min_top_score,
            min_avg_score=min_avg_score,
        )

    def _resolve_documents(self, *, author_id: str | None) -> list[Document]:
        if author_id:
            return get_documents_by_author(self.db_path, author_id)
        return list_documents(self.db_path)

    def _retrieve_chunks(
        self,
        *,
        query: str,
        documents: list[Document],
        top_k: int,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        doc_ids = {document.id for document in documents}
        chunks = [
            chunk
            for chunk in list_chunks(self.db_path)
            if chunk.document_id in doc_ids
        ]
        if not chunks:
            return []
        chunk_ids = {chunk.id for chunk in chunks}
        embeddings = [
            item
            for item in list_chunk_embeddings(self.db_path, model=self.model)
            if item.chunk_id in chunk_ids
        ]
        if not embeddings:
            return []
        query_vector = embed_text(query, dim=embeddings[0].dimensions)
        ranked = rank_chunk_embeddings(
            query_vector=query_vector,
            chunk_embeddings=embeddings,
            top_k=top_k,
        )
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        documents_by_id = {document.id: document for document in documents}
        return hydrate_scored_chunks(
            ranked,
            chunks_by_id=chunks_by_id,
            documents_by_id=documents_by_id,
        )

    def _build_topics_for_author(
        self,
        *,
        author_id: str | None,
        num_topics: int,
        top_docs: int,
        max_iter: int,
    ) -> list[TopicCluster]:
        documents = self._resolve_documents(author_id=author_id)
        if not documents:
            return []
        doc_ids = {document.id for document in documents}
        chunks = [
            chunk
            for chunk in list_chunks(self.db_path)
            if chunk.document_id in doc_ids
        ]
        if not chunks:
            return []
        chunk_ids = {chunk.id for chunk in chunks}
        embeddings = [
            item
            for item in list_chunk_embeddings(self.db_path, model=self.model)
            if item.chunk_id in chunk_ids
        ]
        if not embeddings:
            return []
        return cluster_chunks_by_embeddings(
            chunks_by_id={chunk.id: chunk for chunk in chunks},
            documents_by_id={document.id: document for document in documents},
            chunk_embeddings=embeddings,
            num_topics=num_topics,
            top_docs_per_topic=top_docs,
            max_iter=max_iter,
            labeler=DeterministicTopicLabeler(),
        )


def _slug(value: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_")
    return compact.lower() or "default"
