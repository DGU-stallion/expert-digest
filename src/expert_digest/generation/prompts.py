"""Prompt builders for future LLM-based handbook synthesis."""

from __future__ import annotations

from pathlib import Path

from expert_digest.retrieval.retriever import RetrievedChunk

DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "prompts.yaml"
)

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个知识蒸馏助手。请基于给定证据总结主题核心观点，"
    "不得虚构事实，若证据不足要明确不确定性。"
)
_DEFAULT_OUTPUT_INSTRUCTION = (
    "请输出 6-10 句中文总结，按“结论-证据-分歧-行动建议”组织，避免空话。"
)


def build_theme_summary_prompts(
    *,
    theme_name: str,
    question: str,
    evidence_chunks: list[RetrievedChunk],
) -> tuple[str, str]:
    templates = _load_prompt_templates(DEFAULT_PROMPTS_PATH)
    system_prompt = templates.get(
        "theme_summary.system_prompt",
        _DEFAULT_SYSTEM_PROMPT,
    )
    output_instruction = templates.get(
        "theme_summary.output_instruction",
        _DEFAULT_OUTPUT_INSTRUCTION,
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
        + f"\n\n{output_instruction}"
    )
    return system_prompt, user_prompt


def _load_prompt_templates(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    current_section: str | None = None
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            continue
        if current_section is None or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[f"{current_section}.{key}"] = value
    return parsed
