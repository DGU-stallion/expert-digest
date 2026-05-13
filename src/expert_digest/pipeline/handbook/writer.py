"""Chapter drafting node with LLM writing, using full-document context."""

from __future__ import annotations

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


def run_draft_chapters(state: DigestState) -> dict:
    """Draft all chapters with full-document context and cross-chapter awareness."""
    chapter_plan = state.get("chapter_plan", [])
    review_results = state.get("review_results", [])

    if not chapter_plan:
        return {}

    feedback_map: dict[str, list[str]] = {}
    for rr in review_results:
        if rr.issues:
            feedback_map[rr.chapter_title] = rr.issues

    llm = require_reasoning_client()
    chapters: list[ChapterDraft] = []
    previous_chapters: list[str] = []

    for plan in chapter_plan:
        context_texts = _gather_context_for_chapter(plan, state)
        feedback = feedback_map.get(plan.title, None)
        user_prompt = _build_writer_prompt(
            plan, context_texts, list(previous_chapters), feedback
        )
        raw = llm.generate(
            system_prompt=_WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        content = raw.strip()
        section_count = content.count("\n##")
        chapters.append(
            ChapterDraft(
                title=plan.title,
                content=content,
                section_count=section_count,
            )
        )
        previous_chapters.append(f"{plan.title}：{plan.purpose}")

    return {"chapters": chapters}
