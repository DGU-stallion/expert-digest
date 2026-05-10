"""Chapter drafting node with LLM writing and rewrite feedback support."""

from __future__ import annotations

from pathlib import Path

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import ChapterDraft, ChapterPlan, DigestState
from expert_digest.storage.sqlite_store import list_chunks

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
    """Build the user prompt for a single chapter draft."""
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


def _gather_context_for_chapter(
    plan: ChapterPlan,
    state: DigestState,
    max_chunks_per_chapter: int = 15,
    max_chars_per_chunk: int = 2000,
) -> list[str]:
    """Gather chunk texts relevant to a chapter from topic clusters and documents.

    Prefers cluster representative chunks when topic_clusters are available.
    Falls back to document excerpts when clusters are missing.
    """
    clusters = state.get("topic_clusters", [])
    documents = state.get("documents", [])
    chunk_texts: list[str] = []

    # Try to match chapter target_themes to cluster labels
    if clusters:
        doc_id_to_doc = {d.get("id", ""): d for d in documents}
        matched_chunk_ids: set[str] = set()
        for cluster in clusters:
            label = cluster.get("label", "")
            if any(t in label for t in plan.target_themes) or any(
                t in cluster.get("label", "") for t in plan.target_themes
            ):
                for cid in cluster.get("representative_chunk_ids", [])[:8]:
                    matched_chunk_ids.add(cid)

        if matched_chunk_ids:
            db_path = state.get("db_path", "")
            if db_path and Path(db_path).exists():
                all_chunks = list_chunks(db_path)
                chunk_map = {c.id: c for c in all_chunks}
                for cid in matched_chunk_ids:
                    if cid in chunk_map:
                        text = chunk_map[cid].text.strip()
                        if len(text) > max_chars_per_chunk:
                            text = text[:max_chars_per_chunk] + "..."
                        doc_title = ""
                        if chunk_map[cid].document_id in doc_id_to_doc:
                            doc_title = doc_id_to_doc[chunk_map[cid].document_id].get(
                                "title", ""
                            )
                        header = f"### {doc_title}" if doc_title else "### 原文片段"
                        chunk_texts.append(f"{header}\n{text}")

    # Fallback: use document excerpts
    if not chunk_texts:
        for doc in documents[:8]:
            title = doc.get("title", "").strip()
            content = doc.get("content", "").strip()
            excerpt = content[:max_chars_per_chunk] if len(content) > max_chars_per_chunk else content
            if excerpt:
                chunk_texts.append(f"### {title}\n{excerpt}")

    return chunk_texts[:max_chunks_per_chapter]


def run_draft_chapters(state: DigestState) -> dict:
    """Draft all chapters with cluster-informed context and cross-chapter awareness."""
    chapter_plan = state.get("chapter_plan", [])
    review_results = state.get("review_results", [])

    if not chapter_plan:
        return {}

    # Collect review feedback per chapter for rewrite pass
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
