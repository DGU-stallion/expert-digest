"""Chapter drafting node with LLM writing, using full-document context."""

from __future__ import annotations

import json
from pathlib import Path

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import ChapterDraft, ChapterPlan, DigestState

_WRITER_SYSTEM_PROMPT = """\
你是一位教育内容作家。你的任务是根据作者的文章原文，为学习手册撰写一个章节。

要求：
1. 基于提供的原文材料深度撰写，不要编造数据和具体数字
2. 用中文撰写，风格清晰、结构化，每章至少800字
3. 适当引用作者的原话或案例作为论据，注明来自哪篇文章
4. 每个小节用 ## 标题分隔，小节的论点要层层递进
5. 内容要有深度，展示作者的独特见解和思维框架
6. 本章是学习路径的一部分，注意与前序章节的知识衔接

必须只输出 markdown 格式的章节内容，不要输出任何多余文本。"""


def _build_writer_prompt(
    plan: ChapterPlan,
    context_texts: list[str],
    previous_chapters: list[str],
    review_feedback: list[str] | None = None,
) -> str:
    prompt = (
        f"请撰写以下章节：\n\n"
        f"## 章节标题\n{plan.title}\n\n"
        f"## 学习目的\n{plan.purpose}\n\n"
        f"## 目标主题\n"
        + "\n".join(f"- {t}" for t in plan.target_themes)
        + "\n\n"
    )

    if previous_chapters:
        prompt += (
            "## 前序章节（本章应承接这些章节的知识）\n"
            + "\n".join(f"- {c}" for c in previous_chapters)
            + "\n\n"
        )

    prompt += (
        "## 参考原文材料\n"
        + "\n\n---\n".join(context_texts)
    )

    if review_feedback:
        prompt += (
            "\n\n## 上一轮评审意见\n"
            + "\n".join(f"- {issue}" for issue in review_feedback)
            + "\n\n请根据以上意见修改章节内容，确保这些问题已解决。"
        )

    return prompt


def _extract_wiki_source_ids(wiki_pages: list[dict]) -> dict[str, set[str]]:
    """Build a mapping from topic title → set of source document IDs.

    Uses wiki topic pages to map topic labels to their source document IDs,
    so chapter drafting reads full documents rather than chunk fragments.
    """
    topic_to_ids: dict[str, set[str]] = {}
    for page in wiki_pages:
        if page.get("page_type") != "topic":
            continue
        label = page.get("title", "")
        ids: set[str] = set()
        for s in page.get("sources", []):
            sid = s.get("source_id", "")
            if sid:
                ids.add(sid)
        if label and ids:
            topic_to_ids[label] = ids
    return topic_to_ids


def _gather_context_for_chapter(
    plan: ChapterPlan,
    state: DigestState,
    max_docs: int = 6,
    max_chars_per_doc: int = 4000,
) -> list[str]:
    """Gather full document texts relevant to a chapter.

    Matches chapter target themes to:
    1. Wiki topic pages → source document IDs → full document text
    2. Topic clusters → representative chunk content (fallback)
    3. Direct document excerpts (last resort)
    """
    documents = state.get("documents", [])
    wiki_pages = state.get("wiki_pages", [])
    doc_map = {d.get("id", ""): d for d in documents}
    doc_texts: list[str] = []
    seen_ids: set[str] = set()

    # Strategy 1: Match via wiki topic pages
    if wiki_pages:
        topic_to_ids = _extract_wiki_source_ids(wiki_pages)
        for theme in plan.target_themes:
            for topic_label, source_ids in topic_to_ids.items():
                if theme in topic_label or topic_label in theme:
                    for sid in source_ids:
                        if sid in seen_ids or len(doc_texts) >= max_docs:
                            continue
                        doc = doc_map.get(sid)
                        if doc:
                            title = doc.get("title", "")
                            content = doc.get("content", "")
                            if not content:
                                continue
                            excerpt = (
                                content[:max_chars_per_doc]
                                if len(content) > max_chars_per_doc
                                else content
                            )
                            doc_texts.append(f"### {title}\n{excerpt}")
                            seen_ids.add(sid)

    # Strategy 2: Match via topic clusters (fallback, no wiki)
    if not doc_texts:
        clusters = state.get("topic_clusters", [])
        for cluster in clusters:
            label = cluster.get("label", "")
            if any(t in label for t in plan.target_themes) or any(
                t in cluster.get("label", "") for t in plan.target_themes
            ):
                for d in cluster.get("representative_documents", []):
                    doc_id = d.get("document_id", "")
                    if doc_id in seen_ids or len(doc_texts) >= max_docs:
                        continue
                    doc = doc_map.get(doc_id)
                    if doc:
                        title = doc.get("title", "")
                        content = doc.get("content", "")
                        if not content:
                            continue
                        excerpt = (
                            content[:max_chars_per_doc]
                            if len(content) > max_chars_per_doc
                            else content
                        )
                        doc_texts.append(f"### {title}\n{excerpt}")
                        seen_ids.add(doc_id)

    # Strategy 3: Direct document excerpts (last resort)
    if not doc_texts:
        for doc in documents[:max_docs]:
            title = doc.get("title", "").strip()
            content = doc.get("content", "").strip()
            excerpt = (
                content[:max_chars_per_doc]
                if len(content) > max_chars_per_doc
                else content
            )
            if excerpt:
                doc_texts.append(f"### {title}\n{excerpt}")

    return doc_texts


# ── Chapter cache helpers ─────────────────────────────────────────────

_CACHE_DIR = ".handbook_cache"


def _cache_dir(state: DigestState) -> Path:
    """Resolve the chapter cache directory from pipeline state."""
    output_dir = Path(state.get("output_dir", "data/outputs"))
    cache = output_dir / _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _cache_path(cache: Path, index: int) -> Path:
    return cache / f"chapter_{index}.json"


def _load_chapter_cache(state: DigestState) -> tuple[dict[str, ChapterDraft], dict[str, bool], dict[str, list[str]]]:
    """Load all cached chapters and their pass/issues state.

    Returns:
        (chapter_map, passed_map, issues_map)
        chapter_map: title → ChapterDraft
        passed_map: title → bool
        issues_map: title → list[str]
    """
    cache_dir = _cache_dir(state)
    chapters: dict[str, ChapterDraft] = {}
    passed: dict[str, bool] = {}
    issues_map: dict[str, list[str]] = {}

    for f in sorted(cache_dir.glob("chapter_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            title = data.get("title", "")
            content = data.get("content", "")
            if title and content:
                chapters[title] = ChapterDraft(
                    title=title,
                    content=content,
                    section_count=data.get("section_count", content.count("\n##")),
                )
                passed[title] = data.get("passed", False)
                issues_map[title] = data.get("issues", [])
        except (json.JSONDecodeError, OSError):
            continue

    return chapters, passed, issues_map


def _save_chapter_cache(state: DigestState, index: int, chapter: ChapterDraft,
                        passed: bool = False, issues: list[str] | None = None) -> None:
    """Save a single chapter to disk cache, overwriting any previous entry at same index."""
    cache = _cache_dir(state)
    data = {
        "title": chapter.title,
        "content": chapter.content,
        "section_count": chapter.section_count,
        "passed": passed,
        "issues": issues or [],
    }
    try:
        _cache_path(cache, index).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def save_chapter_cache(state: DigestState, index: int, chapter: ChapterDraft) -> None:
    """Public helper for reviewer to update pass/issues in cache."""
    cache = _cache_dir(state)
    path = _cache_path(cache, index)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["passed"] = True
        data["issues"] = []
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


# ── Writer ─────────────────────────────────────────────────────────────

def run_draft_chapters(state: DigestState) -> dict:
    """Draft or rewrite chapters with incremental disk cache.

    On first run: generate all chapters, saving each to disk immediately.
    On restart: load cached chapters, only generate missing or unpassed ones.
    On rewrite: only regenerate chapters with review feedback.
    """
    chapter_plan = state.get("chapter_plan", [])
    review_results = state.get("review_results", [])

    if not chapter_plan:
        return {}

    feedback_map: dict[str, list[str]] = {}
    for rr in review_results:
        if rr.issues:
            feedback_map[rr.chapter_title] = rr.issues

    # Load disk cache — allows recovery after interrupted runs
    cached_chapters, passed_map, issues_map = _load_chapter_cache(state)
    is_restart = len(cached_chapters) > 0
    is_rewrite = len(review_results) > 0

    if is_restart and not is_rewrite:
        print(f"  [handbook] draft_chapters: {len(cached_chapters)}/{len(chapter_plan)} in cache, resuming...")
    elif is_rewrite:
        need = [p.title for p in chapter_plan if p.title in feedback_map]
        print(f"  [handbook] rewrite_chapters: {len(need)}/{len(chapter_plan)} need rewrite: {', '.join(need)}")

    llm = require_reasoning_client()
    chapters: list[ChapterDraft] = []
    previous_chapters: list[str] = []

    for i, plan in enumerate(chapter_plan, 1):
        feedback = feedback_map.get(plan.title, None)
        idx = i - 1  # 0-based index for cache

        # Decision: do we need to generate this chapter?
        cached = cached_chapters.get(plan.title)

        # Generate if: (a) not in cache, OR (b) has pending feedback, OR (c) didn't pass
        needs_generation = False
        skip_reason = None

        if cached is None:
            needs_generation = True
        elif plan.title in feedback_map:
            needs_generation = True
            skip_reason = "feedback"
        elif not passed_map.get(plan.title, False):
            needs_generation = True
            skip_reason = "not passed"

        if not needs_generation:
            chapters.append(cached)
            previous_chapters.append(f"{plan.title}：{plan.purpose}")
            if is_rewrite:
                print(f"  [handbook]   keep    {i}/{len(chapter_plan)}: {plan.title}")
            continue

        if is_rewrite:
            print(f"  [handbook]   rewrite {i}/{len(chapter_plan)}: {plan.title}")
        elif skip_reason == "not passed":
            print(f"  [handbook]   resume  {i}/{len(chapter_plan)}: {plan.title}")
        else:
            print(f"  [handbook]   draft   {i}/{len(chapter_plan)}: {plan.title}")

        context_texts = _gather_context_for_chapter(plan, state)
        user_prompt = _build_writer_prompt(
            plan, context_texts, list(previous_chapters), feedback
        )
        raw = llm.generate(
            system_prompt=_WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        content = raw.strip()
        section_count = content.count("\n##")
        chapter = ChapterDraft(
            title=plan.title,
            content=content,
            section_count=section_count,
        )
        chapters.append(chapter)
        previous_chapters.append(f"{plan.title}：{plan.purpose}")

        # Persist immediately after generation
        _save_chapter_cache(state, idx, chapter)

    return {"chapters": chapters}
