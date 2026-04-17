"""Command line entry point for ExpertDigest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from expert_digest import __version__
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
from expert_digest.retrieval.retriever import rank_chunk_embeddings
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

    return parser
