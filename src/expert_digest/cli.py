"""Command line entry point for ExpertDigest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from expert_digest import __version__
from expert_digest.domain.models import Handbook
from expert_digest.generation.handbook_writer import (
    DeterministicThemeSynthesizer,
    HybridThemeSynthesizer,
    build_handbook,
    write_handbook,
)
from expert_digest.generation.llm_client import (
    DEFAULT_CCSWITCH_DB_PATH,
    AnthropicCompatibleClient,
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
    embed_text,
)
from expert_digest.processing.splitter import split_documents
from expert_digest.rag.answering import StructuredAnswer, build_structured_answer
from expert_digest.retrieval.retriever import (
    hydrate_scored_chunks,
    rank_chunk_embeddings,
)
from expert_digest.storage.sqlite_store import (
    DEFAULT_DATABASE_PATH,
    clear_chunk_embeddings,
    clear_chunks,
    get_documents_by_author,
    list_chunk_embeddings,
    list_chunks,
    list_documents,
    save_chunk_embeddings,
    save_chunks,
    save_documents,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ExpertDigest command line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "import-jsonl":
        documents = load_jsonl_documents(args.path)
        count = save_documents(args.db, documents)
        print(f"Imported {count} document(s) into {args.db}")
        return 0

    if args.command == "import-markdown":
        documents = load_markdown_documents(args.folder)
        count = save_documents(args.db, documents)
        print(f"Imported {count} document(s) into {args.db}")
        return 0

    if args.command == "import-zhihu":
        documents = load_zhihu_documents(args.path)
        count = save_documents(args.db, documents)
        print(f"Imported {count} document(s) into {args.db}")
        return 0

    if args.command == "build-chunks":
        documents = list_documents(args.db)
        cleaned_documents = [clean_document(document) for document in documents]
        chunks = split_documents(
            cleaned_documents,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        )
        count = save_chunks(args.db, chunks)
        print(
            f"Built {count} chunk(s) from {len(documents)} document(s) into {args.db}"
        )
        return 0

    if args.command == "rebuild-chunks":
        documents = list_documents(args.db)
        removed = clear_chunks(args.db)
        cleaned_documents = [clean_document(document) for document in documents]
        chunks = split_documents(
            cleaned_documents,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        )
        count = save_chunks(args.db, chunks)
        print(
            "Rebuilt "
            f"{count} chunk(s) from {len(documents)} document(s) into {args.db} "
            f"(cleared {removed} old chunk(s))"
        )
        return 0

    if args.command == "build-embeddings":
        chunks = list_chunks(args.db)
        embeddings = embed_chunks(
            chunks,
            model=args.model,
            dim=args.dim,
        )
        count = save_chunk_embeddings(args.db, embeddings)
        print(
            f"Embedded {count} chunk(s) with model {args.model} into {args.db}"
        )
        return 0

    if args.command == "rebuild-embeddings":
        chunks = list_chunks(args.db)
        removed = clear_chunk_embeddings(args.db, model=args.model)
        embeddings = embed_chunks(
            chunks,
            model=args.model,
            dim=args.dim,
        )
        count = save_chunk_embeddings(args.db, embeddings)
        print(
            "Rebuilt embeddings: "
            f"{count} chunk(s) with model {args.model} into {args.db} "
            f"(cleared {removed} old embedding(s))"
        )
        return 0

    if args.command == "search-chunks":
        chunk_embeddings = list_chunk_embeddings(args.db, model=args.model)
        if not chunk_embeddings:
            print(f"No embeddings found for model {args.model} in {args.db}")
            return 0

        query_vector = embed_text(
            args.query,
            dim=chunk_embeddings[0].dimensions,
        )
        ranked = rank_chunk_embeddings(
            query_vector=query_vector,
            chunk_embeddings=chunk_embeddings,
            top_k=args.top_k,
        )
        chunks = {chunk.id: chunk for chunk in list_chunks(args.db)}
        documents = {document.id: document for document in list_documents(args.db)}
        for item in ranked:
            chunk = chunks.get(item.chunk_id)
            if chunk is None:
                continue
            document = documents.get(chunk.document_id)
            title = document.title if document else "<unknown>"
            snippet = chunk.text.replace("\n", " ").strip()[:100]
            print(f"score={item.score:.4f}\t{title}\t{snippet}")
        return 0

    if args.command == "list-documents":
        documents = (
            get_documents_by_author(args.db, args.author)
            if args.author
            else list_documents(args.db)
        )
        for document in documents:
            url = f" {document.url}" if document.url else ""
            print(f"{document.id}\t{document.author}\t{document.title}{url}")
        return 0

    if args.command == "ask":
        chunk_embeddings = list_chunk_embeddings(args.db, model=args.model)
        min_top_score = (
            args.min_top_score if args.min_top_score is not None else args.min_score
        )
        if not chunk_embeddings:
            result = build_structured_answer(
                question=args.query,
                evidence_chunks=[],
                max_evidence=args.max_evidence,
                min_top_score=min_top_score,
                min_avg_score=args.min_avg_score,
            )
            _emit_structured_answer(result, output_format=args.format)
            return 0

        query_vector = embed_text(
            args.query,
            dim=chunk_embeddings[0].dimensions,
        )
        ranked = rank_chunk_embeddings(
            query_vector=query_vector,
            chunk_embeddings=chunk_embeddings,
            top_k=args.top_k,
        )
        chunks = {chunk.id: chunk for chunk in list_chunks(args.db)}
        documents = {document.id: document for document in list_documents(args.db)}
        evidence_chunks = hydrate_scored_chunks(
            ranked,
            chunks_by_id=chunks,
            documents_by_id=documents,
        )
        result = build_structured_answer(
            question=args.query,
            evidence_chunks=evidence_chunks,
            max_evidence=args.max_evidence,
            min_top_score=min_top_score,
            min_avg_score=args.min_avg_score,
        )
        _emit_structured_answer(result, output_format=args.format)
        return 0

    if args.command == "generate-handbook":
        llm_client: AnthropicCompatibleClient | None = None
        if args.synthesis_mode == "hybrid":
            llm_client = create_default_handbook_llm_client(
                ccswitch_db_path=args.ccswitch_db,
                timeout_seconds=args.llm_timeout,
                max_output_tokens=args.llm_max_tokens,
            )
            synthesizer = HybridThemeSynthesizer(llm_client=llm_client)
        else:
            synthesizer = DeterministicThemeSynthesizer()
        try:
            handbook = build_handbook(
                db_path=args.db,
                author=args.author,
                model=args.model,
                top_k=args.top_k,
                max_themes=args.max_themes,
                synthesizer=synthesizer,
            )
            output_path = write_handbook(handbook=handbook, output_path=args.output)
        except ValueError as error:
            print(f"Failed to generate handbook: {error}")
            return 1
        _emit_handbook_result(
            handbook=handbook,
            output_path=output_path,
            synthesis_mode=args.synthesis_mode,
            llm_client=llm_client,
            output_format=args.format,
        )
        return 0

    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expert-digest",
        description=f"ExpertDigest {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    import_jsonl = subparsers.add_parser("import-jsonl")
    import_jsonl.add_argument("path", type=Path)
    import_jsonl.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)

    import_markdown = subparsers.add_parser("import-markdown")
    import_markdown.add_argument("folder", type=Path)
    import_markdown.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)

    import_zhihu = subparsers.add_parser("import-zhihu")
    import_zhihu.add_argument("path", type=Path)
    import_zhihu.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)

    build_chunks = subparsers.add_parser("build-chunks")
    build_chunks.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    build_chunks.add_argument("--max-chars", type=int, default=1000)
    build_chunks.add_argument("--min-chars", type=int, default=1)

    rebuild_chunks = subparsers.add_parser("rebuild-chunks")
    rebuild_chunks.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    rebuild_chunks.add_argument("--max-chars", type=int, default=1000)
    rebuild_chunks.add_argument("--min-chars", type=int, default=1)

    build_embeddings = subparsers.add_parser("build-embeddings")
    build_embeddings.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    build_embeddings.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    build_embeddings.add_argument("--dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    rebuild_embeddings = subparsers.add_parser("rebuild-embeddings")
    rebuild_embeddings.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    rebuild_embeddings.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    rebuild_embeddings.add_argument("--dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    search_chunks = subparsers.add_parser("search-chunks")
    search_chunks.add_argument("query")
    search_chunks.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    search_chunks.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    search_chunks.add_argument("--top-k", type=int, default=5)

    list_parser = subparsers.add_parser("list-documents")
    list_parser.add_argument("--author")
    list_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    ask_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--min-score", type=float, default=0.05)
    ask_parser.add_argument("--min-top-score", type=float, default=None)
    ask_parser.add_argument("--min-avg-score", type=float, default=0.03)
    ask_parser.add_argument("--max-evidence", type=int, default=3)
    ask_parser.add_argument("--format", choices=["text", "json"], default="text")

    handbook_parser = subparsers.add_parser("generate-handbook")
    handbook_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    handbook_parser.add_argument("--author")
    handbook_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    handbook_parser.add_argument("--top-k", type=int, default=3)
    handbook_parser.add_argument("--max-themes", type=int, default=3)
    handbook_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/handbook.md"),
    )
    handbook_parser.add_argument(
        "--synthesis-mode",
        choices=["deterministic", "hybrid"],
        default="hybrid",
    )
    handbook_parser.add_argument(
        "--ccswitch-db",
        type=Path,
        default=DEFAULT_CCSWITCH_DB_PATH,
    )
    handbook_parser.add_argument("--llm-timeout", type=int, default=30)
    handbook_parser.add_argument("--llm-max-tokens", type=int, default=700)
    handbook_parser.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def _emit_structured_answer(result: StructuredAnswer, *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False))
        return
    _print_structured_answer(result)


def _print_structured_answer(result: StructuredAnswer) -> None:
    print(f"回答: {result.answer}")
    print("依据:")
    if not result.evidence:
        print("- (无)")
    else:
        for index, item in enumerate(result.evidence, start=1):
            print(
                f"- {index}. score={item.score:.4f} "
                f"{item.title} / {item.author} / {item.snippet}"
            )
    print("推荐原文:")
    if not result.recommended_original:
        print("- (无)")
    else:
        for index, label in enumerate(result.recommended_original, start=1):
            print(f"- {index}. {label}")
    print(f"不确定性: {result.uncertainty}")


def _emit_handbook_result(
    *,
    handbook: Handbook,
    output_path: Path,
    synthesis_mode: str,
    llm_client: AnthropicCompatibleClient | None,
    output_format: str,
) -> None:
    llm_enabled = llm_client is not None
    llm_provider = llm_client.provider if llm_client is not None else None
    llm_model = llm_client.model if llm_client is not None else None
    llm_base_url = llm_client.base_url if llm_client is not None else None

    if output_format == "json":
        payload = {
            "author": handbook.author,
            "title": handbook.title,
            "output_path": str(output_path),
            "source_document_ids": handbook.source_document_ids,
            "synthesis_mode": synthesis_mode,
            "llm_enabled": llm_enabled,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return

    print(
        f"Generated handbook for {handbook.author}: {output_path} "
        f"(sources={len(handbook.source_document_ids)}, "
        f"mode={synthesis_mode}, llm_enabled={llm_enabled})"
    )
