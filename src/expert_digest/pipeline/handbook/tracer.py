"""Evidence trace builder: maps handbook content back to source documents via clusters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from expert_digest.pipeline.state import ChapterPlan, DigestState


def run_build_trace(state: DigestState) -> dict:
    """Build trace mapping from chapters back to source documents.

    Uses topic_clusters (when available) for richer trace linking,
    falling back to LLM-extracted themes when clusters are absent.
    """
    chapters = state.get("chapters", [])
    chapter_plan = state.get("chapter_plan", [])
    themes = state.get("themes", [])
    clusters = state.get("topic_clusters", [])
    documents = state.get("documents", [])
    output_dir = Path(state.get("output_dir", "data/outputs"))

    plan_map: dict[str, ChapterPlan] = {p.title: p for p in chapter_plan}
    doc_map = {d.get("id", ""): d for d in documents}

    trace: dict[str, object] = {
        "author": state.get("author", ""),
        "generated_at": datetime.now(UTC).isoformat(),
        "total_chapters": len(chapters),
        "total_documents": len(documents),
        "total_clusters": len(clusters),
        "chapters": [],
    }

    # Build cluster label → document IDs mapping for fast lookup
    cluster_sources: dict[str, list[str]] = {}
    for c in clusters:
        label = c.get("label", "")
        rep_docs = c.get("representative_documents", [])
        cluster_sources[label] = [
            d.get("document_id", "") for d in rep_docs if d.get("document_id")
        ]

    chapter_entries: list[dict] = []
    for chapter in chapters:
        plan = plan_map.get(chapter.title)
        target_themes = plan.target_themes if plan else []

        source_doc_ids: set[str] = set()
        matched_cluster_labels: list[str] = []

        # Try cluster-based matching first
        for t_label in target_themes:
            for c_label, doc_ids in cluster_sources.items():
                if t_label in c_label or c_label in t_label:
                    source_doc_ids.update(doc_ids)
                    matched_cluster_labels.append(c_label)

        # Fallback: LLM theme matching
        if not source_doc_ids:
            for t_label in target_themes:
                for theme in themes:
                    if theme.label == t_label:
                        source_doc_ids.update(theme.source_document_ids)

        entry: dict = {
            "title": chapter.title,
            "section_count": chapter.section_count,
            "content_length": len(chapter.content),
            "target_themes": target_themes,
            "matched_clusters": matched_cluster_labels,
            "source_documents": [
                {"id": did, "title": doc_map.get(did, {}).get("title", "")}
                for did in source_doc_ids
                if did in doc_map
            ],
        }
        chapter_entries.append(entry)

    trace["chapters"] = chapter_entries

    # Write trace sidecar
    trace_path = output_dir / "handbook.trace.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    return {}
