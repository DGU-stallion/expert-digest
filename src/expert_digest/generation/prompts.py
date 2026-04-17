"""Prompt builders for future LLM-based handbook synthesis."""

from __future__ import annotations

from expert_digest.retrieval.retriever import RetrievedChunk


def build_theme_summary_prompts(
    *,
    theme_name: str,
    question: str,
    evidence_chunks: list[RetrievedChunk],
) -> tuple[str, str]:
    system_prompt = (
        "你是一个知识蒸馏助手。请基于给定证据总结主题核心观点，"
        "不得虚构事实，若证据不足要明确不确定性。"
    )
    evidence_lines = [
        (
            f"{index}. score={item.score:.4f} | {item.title} | "
            f"{item.text.strip().replace(chr(10), ' ')}"
        )
        for index, item in enumerate(evidence_chunks, start=1)
    ]
    user_prompt = (
        f"主题：{theme_name}\n"
        f"问题：{question}\n"
        "证据：\n"
        + "\n".join(evidence_lines)
        + "\n\n请输出 2-4 句中文总结，突出可行动信息。"
    )
    return system_prompt, user_prompt
