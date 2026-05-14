"""Expression DNA encoding into role-playing rules via LLM."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import DigestState

_EXPRESSER_SYSTEM_PROMPT = """\
你是一位角色设计专家。你的任务是根据作者的表达特征和智识背景，设计一组角色扮演规则和身份卡。

已有表达DNA分析数据：
- 句式特点、高频短语、确定性频谱、引用习惯
- 智识谱系
- 关键判断

请输出：
1. **角色扮演规则**（5-8条）：让AI能模仿作者的语气、思维方式和表达习惯的规则
2. **身份卡**：作者的背景、能力边界
3. **表达DNA描述**：对作者表达风格的流畅自然描述（供SKILL.md使用）

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "role_rules": [
    "规则1：直接以作者的身份回应",
    "规则2：结论先行，再用论据展开"
  ],
  "identity_card": "我是谁：...我的核心经历：...",
  "expression_dna_description": "作者表达风格的流畅描述"
}"""


def _build_expresser_prompt(state: DigestState) -> str:
    dna = state.get("expression_dna")
    genealogy = state.get("intellectual_genealogy", "")
    decisions = state.get("key_decisions", [])

    parts: list[str] = []
    if dna:
        parts.append("## 表达DNA")
        parts.append(f"句式特点：{dna.sentence_patterns}")
        parts.append(f"高频短语：{dna.high_frequency_phrases}")
        parts.append(f"确定性频谱：{dna.certainty_spectrum}")
        parts.append(f"引用习惯：{dna.citation_habits}")
    if genealogy:
        parts.append(f"## 智识谱系\n{genealogy}")
    if decisions:
        parts.append("## 关键判断")
        for d in decisions[:5]:
            parts.append(f"- 议题：{d.get('topic', '')}，立场：{d.get('position', '')}")

    return "\n".join(parts)


def _parse_expresser_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_encode_expression(state: DigestState) -> dict:
    """Encode expression DNA into role-playing rules for SKILL.md."""
    dna = state.get("expression_dna")
    if dna is None:
        return {"role_rules": ""}

    print("  [skill] encode_expression: LLM style encoding...")
    llm = require_fast_client()
    user_prompt = _build_expresser_prompt(state)
    raw = llm.generate(
        system_prompt=_EXPRESSER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    parsed = _parse_expresser_json(raw)

    role_rules_raw = parsed.get("role_rules", [])
    if isinstance(role_rules_raw, list):
        role_rules_lines = [str(r) for r in role_rules_raw if isinstance(r, str) and r.strip()]
        role_rules_text = "\n".join(role_rules_lines)
    else:
        role_rules_text = str(role_rules_raw)

    identity_card = str(parsed.get("identity_card", ""))
    expression_desc = str(parsed.get("expression_dna_description", ""))

    # Build the full role_rules section markdown
    sections = []
    if role_rules_text:
        sections.append("## 角色扮演规则\n" + role_rules_text)
    if identity_card:
        sections.append("## 身份卡\n" + identity_card)
    if expression_desc:
        sections.append("## 表达DNA\n" + expression_desc)

    return {"role_rules": "\n\n".join(sections)}
