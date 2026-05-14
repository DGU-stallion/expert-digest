"""Command line entry point for ExpertDigest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import time
from datetime import datetime

from expert_digest import __version__
from expert_digest.generation.llm_client import AnthropicCompatibleClient
from expert_digest.ingest.jsonl_loader import load_jsonl_documents
from expert_digest.ingest.markdown_loader import load_markdown_documents
from expert_digest.ingest.zhihu_loader import load_zhihu_documents
from expert_digest.knowledge.topic_clusterer import (
    DeterministicTopicLabeler,
    LLMTopicLabeler,
    TopicCluster,
    build_topic_clusters,
)
from expert_digest.knowledge.topic_report import build_topic_report
from expert_digest.pipeline.graph import compile_handbook_pipeline, compile_pipeline, compile_skill_pipeline
from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import make_initial_state
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
    clear_chunks,
    get_documents_by_author,
    list_chunk_embeddings,
    list_chunks,
    list_documents,
    save_chunk_embeddings,
    save_chunks,
    save_documents,
)
from pathlib import Path

from expert_digest.wiki.analyzer import analyze_document
from expert_digest.wiki.evaluator import evaluate_wiki
from expert_digest.wiki.linter import lint_wiki
from expert_digest.wiki.retriever import search_wiki
from expert_digest.wiki.vault import WikiVault
from expert_digest.wiki.writer import write_analysis_to_vault


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ExpertDigest command line interface."""
    _load_env_dotenv()
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
        print(f"Embedded {count} chunk(s) with model {args.model} into {args.db}")
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

    if args.command == "build-wiki":
        vault = WikiVault(root=args.wiki_root)
        vault.initialize(
            expert_id=args.expert_id,
            expert_name=args.expert_name,
            purpose=args.purpose,
        )
        documents = list_documents(args.db)
        total = len(documents)
        start = time.time()
        print(f"[{datetime.now():%H:%M:%S}] Building wiki for {total} sources ...")
        print(f"  Vault root: {args.wiki_root}")
        print()
        written_sources = 0
        skipped = 0
        for index, document in enumerate(documents, start=1):
            source_path = Path(args.wiki_root) / "sources" / f"{document.id}.md"
            if source_path.exists():
                skipped += 1
                continue
            doc_dict = {
                "id": document.id,
                "title": document.title,
                "content": document.content or "",
                "author": document.author,
                "url": document.url,
            }
            analysis = analyze_document(doc_dict)
            write_analysis_to_vault(
                vault=vault,
                analysis=analysis,
            )
            written_sources += 1
            elapsed = time.time() - start
            eta = (elapsed / index) * (total - index) if index > 0 else 0
            _print_text_safely(
                f"  [{elapsed:7.1f}s] ({index}/{total}) "
                f"{document.title[:50]}  ~{eta:.0f}s left"
            )
        total_elapsed = time.time() - start
        _print_text_safely("")
        _print_text_safely(
            f"[{datetime.now():%H:%M:%S}] Built wiki: "
            f"sources={written_sources} (skipped={skipped}) in {total_elapsed:.0f}s "
            f"root={args.wiki_root}"
        )
        return 0

    if args.command == "search-wiki":
        hits = search_wiki(
            vault=WikiVault(root=args.wiki_root),
            query=args.query,
            top_k=args.top_k,
        )
        for hit in hits:
            print(
                f"score={hit.score:.2f}\t{hit.page.page_type}\t"
                f"{hit.page.title}\t{hit.page.path}\t"
                f"sources={','.join(hit.source_ids)}"
            )
        return 0

    if args.command == "eval-wiki":
        report = evaluate_wiki(
            vault=WikiVault(root=args.wiki_root),
            expected_source_count=args.expected_source_count,
        )
        _print_json_safely(asdict(report))
        return 0

    if args.command == "lint-wiki":
        report = lint_wiki(vault=WikiVault(root=args.wiki_root))
        _print_json_safely(asdict(report))
        return 0

    if args.command == "list-documents":
        documents = (
            get_documents_by_author(args.db, args.author)
            if args.author
            else list_documents(args.db)
        )
        for document in documents:
            url = f" {document.url}" if document.url else ""
            _print_text_safely(
                f"{document.id}\t{document.author}\t{document.title}{url}"
            )
        return 0

    if args.command == "cluster-topics":
        llm_client: AnthropicCompatibleClient | None = None
        topic_labeler = DeterministicTopicLabeler()
        if args.label_mode == "llm":
            _load_pipeline_env()
            llm_client = require_fast_client()
            topic_labeler = LLMTopicLabeler(llm_client=llm_client)

        topics = build_topic_clusters(
            db_path=args.db,
            model=args.model,
            num_topics=args.num_topics,
            top_docs_per_topic=args.top_docs,
            max_iter=args.max_iter,
            labeler=topic_labeler,
        )
        chunk_embeddings = list_chunk_embeddings(args.db, model=args.model)
        report = build_topic_report(
            topics=topics,
            chunk_embeddings=chunk_embeddings,
            model=args.model,
        )
        metadata_fn = getattr(topic_labeler, "runtime_metadata", None)
        runtime: dict[str, object] = {}
        if callable(metadata_fn):
            raw = metadata_fn()
            if isinstance(raw, dict):
                runtime = raw
        payload = {
            "topics": [asdict(topic) for topic in topics],
            "report": asdict(report),
            "label_mode": args.label_mode,
            "fallback_used": bool(runtime.get("fallback_used", False)),
            "error_reason": runtime.get("error_reason"),
            "llm_provider": getattr(llm_client, "provider", None),
            "llm_model": getattr(llm_client, "model", None),
        }
        if args.report_output is not None:
            _save_run_metadata(payload=payload, output_path=args.report_output)
        _emit_topic_clusters(
            topics=topics,
            output_format=args.format,
            metadata={key: value for key, value in payload.items() if key != "topics"},
        )
        return 0

    if args.command == "generate-handbook":
        _load_pipeline_env()
        state = make_initial_state(
            db_path=args.db,
            wiki_root=args.wiki_root or "",
            author=args.author or "",
            output_dir=str(args.output.parent) if args.output else "data/outputs",
        )
        pipeline = compile_handbook_pipeline()
        try:
            result = _run_pipeline_with_progress(
                state=state, pipeline=pipeline, label="handbook"
            )
        except (RuntimeError, ValueError) as error:
            print(f"Failed to generate handbook: {error}")
            return 1
        handbook_md = result.get("handbook_markdown", "")
        if not handbook_md.strip():
            doc_count = len(result.get("documents", []))
            print(
                f"Pipeline completed: loaded {doc_count} documents, "
                "handbook output is empty (stub phase)."
            )
            return 0
        output_path = args.output or Path("data/outputs/handbook.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(handbook_md, encoding="utf-8")
        print(f"Generated handbook via pipeline: {output_path}")
        return 0

    if args.command == "generate-skill":
        _load_pipeline_env()
        state = make_initial_state(
            db_path=args.db,
            wiki_root=args.wiki_root or "",
            author=args.author or "",
            output_dir=str(args.output.parent) if args.output else "data/outputs",
        )
        pipeline = compile_skill_pipeline()
        try:
            result = _run_pipeline_with_progress(
                state=state, pipeline=pipeline, label="skill"
            )
        except RuntimeError as error:
            print(f"Failed to generate SKILL: {error}")
            return 1
        skill_md = result.get("skill_markdown", "")
        if not skill_md.strip():
            doc_count = len(result.get("documents", []))
            print(
                f"Pipeline completed: loaded {doc_count} documents, "
                "SKILL output is empty (stub phase)."
            )
            return 0
        output_path = args.output or Path("data/outputs/skill.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(skill_md, encoding="utf-8")
        print(f"Generated SKILL via pipeline: {output_path}")
        return 0

    parser.print_help()
    return 0


def _run_pipeline_with_progress(
    *,
    state: dict,
    pipeline: object,
    label: str,
) -> dict:
    """Run the pipeline with per-node progress printed to stdout.

    Uses ``stream()`` to emit a line per executed node with elapsed time,
    giving the user visibility into long-running handbook/skill generation.
    """
    _STAGE_LABELS = {
        "entry": "load data",
        "cluster_content": "topic clustering",
        "analyze_content": "content analysis (LLM)",
        "analyze_expression": "expression analysis (LLM)",
        "assess_quality": "quality check",
        "route_to_products": "route",
        "handbook_pipeline": "handbook generation (LLM)",
        "skill_pipeline": "skill generation (LLM)",
        "output_handbook": "save handbook",
        "output_skill": "save skill",
    }  # fmt: skip

    start = time.time()
    print(f"[{datetime.now():%H:%M:%S}] Starting {label}...")
    print(
        "  Stages: load  cluster  analyze(LLM)  express(LLM)  "
        "quality  handbook(LLM)  skill(LLM)"
    )
    print()

    result = dict(state)
    try:
        for event in pipeline.stream(state):
            for node_name, output in event.items():
                elapsed = time.time() - start
                stage = _STAGE_LABELS.get(node_name, node_name)
                print(f"  [{elapsed:7.1f}s] {stage}")
                if isinstance(output, dict):
                    result.update(output)
    except (RuntimeError, ValueError) as exc:
        print(f"[{time.time() - start:7.1f}s] FAILED: {exc}")
        raise

    total = time.time() - start
    print(f"\n[{datetime.now():%H:%M:%S}] {label} done ({total:.0f}s total)")
    return result


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

    wiki_parser = subparsers.add_parser("build-wiki")
    wiki_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    wiki_parser.add_argument(
        "--wiki-root", type=Path, default=Path("data/wiki/default")
    )
    wiki_parser.add_argument("--expert-id", default="default")
    wiki_parser.add_argument("--expert-name", default="unknown")
    wiki_parser.add_argument("--purpose", default="沉淀专家公开内容。")

    search_wiki_parser = subparsers.add_parser("search-wiki")
    search_wiki_parser.add_argument("query")
    search_wiki_parser.add_argument(
        "--wiki-root",
        type=Path,
        default=Path("data/wiki/default"),
    )
    search_wiki_parser.add_argument("--top-k", type=int, default=5)

    eval_wiki_parser = subparsers.add_parser("eval-wiki")
    eval_wiki_parser.add_argument(
        "--wiki-root",
        type=Path,
        default=Path("data/wiki/default"),
    )
    eval_wiki_parser.add_argument("--expected-source-count", type=int, default=0)

    lint_wiki_parser = subparsers.add_parser("lint-wiki")
    lint_wiki_parser.add_argument(
        "--wiki-root",
        type=Path,
        default=Path("data/wiki/default"),
    )

    list_parser = subparsers.add_parser("list-documents")
    list_parser.add_argument("--author")
    list_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)

    cluster_parser = subparsers.add_parser("cluster-topics")
    cluster_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    cluster_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    cluster_parser.add_argument("--num-topics", type=int, default=3)
    cluster_parser.add_argument("--top-docs", type=int, default=3)
    cluster_parser.add_argument("--max-iter", type=int, default=30)
    cluster_parser.add_argument(
        "--label-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
    )
    cluster_parser.add_argument("--llm-timeout", type=int, default=20)
    cluster_parser.add_argument("--report-output", type=Path, default=None)
    cluster_parser.add_argument("--format", choices=["text", "json"], default="text")

    hb_parser = subparsers.add_parser("generate-handbook")
    hb_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    hb_parser.add_argument("--author", default=None)
    hb_parser.add_argument("--wiki-root", type=Path, default=None)
    hb_parser.add_argument(
        "--output", type=Path, default=Path("data/outputs/handbook.md")
    )

    sk_parser = subparsers.add_parser("generate-skill")
    sk_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    sk_parser.add_argument("--author", default=None)
    sk_parser.add_argument("--wiki-root", type=Path, default=None)
    sk_parser.add_argument(
        "--output", type=Path, default=Path("data/outputs/skill.md")
    )

    return parser


def _save_run_metadata(*, payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_env_dotenv() -> None:
    """Load KEY=VALUE pairs from .env file into process environment variables.

    Minimal implementation — no ``python-dotenv`` dependency required.
    Skips comment (``#``) and blank lines.  Does **not** override variables
    already present in ``os.environ``.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _load_pipeline_env() -> None:
    """Require pipeline LLM provider env vars, raising a clear error if missing.

    Expected variables (set them in ``.env`` or in the shell):

    * ``PIPELINE_FAST_BASE_URL`` / ``PIPELINE_FAST_API_KEY`` / ``PIPELINE_FAST_MODEL``
    * ``PIPELINE_REASONING_BASE_URL`` / ``PIPELINE_REASONING_API_KEY`` /
      ``PIPELINE_REASONING_MODEL``
    """
    _REQUIRED_PIPELINE_VARS = (
        "PIPELINE_FAST_BASE_URL",
        "PIPELINE_FAST_API_KEY",
        "PIPELINE_FAST_MODEL",
        "PIPELINE_REASONING_BASE_URL",
        "PIPELINE_REASONING_API_KEY",
        "PIPELINE_REASONING_MODEL",
    )
    missing = [var for var in _REQUIRED_PIPELINE_VARS if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            "Pipeline LLM environment variables are not set.\n"
            "Missing: " + ", ".join(missing) + "\n\n"
            "Create a .env file in the project root. Example:\n"
            "  PIPELINE_FAST_BASE_URL=https://api.deepseek.com/anthropic\n"
            "  PIPELINE_FAST_API_KEY=sk-your-key-here\n"
            "  PIPELINE_FAST_MODEL=deepseek-v4-flash\n"
            "  PIPELINE_REASONING_BASE_URL=https://api.deepseek.com/anthropic\n"
            "  PIPELINE_REASONING_API_KEY=sk-your-key-here\n"
            "  PIPELINE_REASONING_MODEL=deepseek-v4-pro\n\n"
            "Or copy .env.example:\n"
            "  cp .env.example .env\n"
            "  # then edit .env with your real API keys"
        )


def _emit_topic_clusters(
    *,
    topics: list[TopicCluster],
    output_format: str,
    metadata: dict[str, object] | None = None,
) -> None:
    metadata = metadata or {}
    if output_format == "json":
        payload = {"topics": [asdict(topic) for topic in topics]}
        payload.update(metadata)
        _print_json_safely(payload)
        return
    _print_topic_clusters(topics, metadata=metadata)


def _print_json_safely(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for non-UTF8 Windows consoles (e.g., GBK) that cannot print
        # some Unicode code points returned by source documents.
        ascii_text = json.dumps(payload, ensure_ascii=True)
        stream = sys.stdout
        stream.write(f"{ascii_text}\n")


def _print_text_safely(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "backslashreplace").decode("ascii"))
        sys.stdout.write("\n")


def _print_topic_clusters(
    topics: list[TopicCluster], *, metadata: dict[str, object] | None = None
) -> None:
    if not topics:
        print("No topic clusters generated.")
        return
    for index, topic in enumerate(topics, start=1):
        print(f"Topic {index}: {topic.label} (chunks={topic.chunk_count})")
        if not topic.representative_documents:
            print("- representative docs: (无)")
            continue
        for doc_index, document in enumerate(topic.representative_documents, start=1):
            print(
                f"- {doc_index}. score={document.score:.4f} "
                f"{document.title} / {document.author}"
            )
    report = metadata.get("report")
    if isinstance(report, dict):
        print(
            "Report: "
            f"topic_count={report.get('topic_count')} "
            f"largest_topic_ratio={report.get('largest_topic_ratio')} "
            f"mean_intra={report.get('mean_intra_similarity_proxy')} "
            f"mean_inter={report.get('mean_inter_topic_similarity_proxy')}"
        )
    if metadata and metadata.get("label_mode") == "llm":
        print(
            "Labeling metadata: "
            f"fallback_used={metadata.get('fallback_used', False)} "
            f"error_reason={metadata.get('error_reason')} "
            f"provider={metadata.get('llm_provider')}"
        )
