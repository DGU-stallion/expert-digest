"""Evidence trace builder: maps handbook content back to source documents."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from expert_digest.pipeline.state import ChapterPlan, DigestState


def run_build_trace(state: DigestState) -> dict:
    """Build trace mapping from chapters/themes back to source documents.

    Produces a trace dictionary stored in state for downstream output.
    Does NOT require an LLM call — builds deterministically from existing data.
    """
    chapters = state.get("chapters", [])
    chapter_plan = state.get("chapter_plan", [])
    themes = state.get("themes", [])
    documents = state.get("documents", [])
    output_dir = Path(state.get("output_dir", "data/outputs"))

    # Map chapter titles to their plans (for theme linkage)
    plan_map: dict[str, ChapterPlan] = {}
    for plan in chapter_plan:
        plan_map[plan.title] = plan

    # Build trace entries
    doc_map = {d.get("id", ""): d for d in documents}
    trace: dict[str, object] = {
        "author": state.get("author", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_chapters": len(chapters),
        "total_documents": len(documents),
        "chapters": [],
    }

    chapter_entries: list[dict] = []
    for chapter in chapters:
        plan = plan_map.get(chapter.title)
        entry: dict = {
            "title": chapter.title,
            "section_count": chapter.section_count,
            "content_length": len(chapter.content),
            "target_themes": plan.target_themes if plan else [],
        }

        # Collect source documents referenced by the chapter's target themes
        source_doc_ids: set[str] = set()
        for theme_label in entry["target_themes"]:
            for theme in themes:
                if theme.label == theme_label:
                    source_doc_ids.update(theme.source_document_ids)

        entry["source_documents"] = [
            {
                "id": did,
                "title": doc_map.get(did, {}).get("title", ""),
            }
            for did in source_doc_ids
            if did in doc_map
        ]
        chapter_entries.append(entry)

    trace["chapters"] = chapter_entries

    # Write trace file as side effect
    trace_path = output_dir / "handbook.trace.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # Non-critical — trace enriches but doesn't block output

    return {}  # Trace written to file; state unchanged
