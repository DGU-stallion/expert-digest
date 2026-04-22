from expert_digest.knowledge.skill_writer import (
    build_skill_markdown_from_profile,
    render_skill_filename,
)


def _sample_profile() -> dict[str, object]:
    return {
        "author": "黄彦臻",
        "document_count": 5,
        "focus_topics": ["供给需求", "风险控制"],
        "keywords": [{"keyword": "风险", "count": 4}, {"keyword": "供给", "count": 3}],
        "reasoning_patterns": [
            {"pattern": "因为...所以...", "count": 3},
            {"pattern": "如果...那么...", "count": 2},
        ],
    }


def test_build_skill_markdown_from_profile_contains_required_sections():
    class _FakeLLMClient:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return """
{
  "style_principles": ["先结论后证据", "保持结构化表达"],
  "response_flow": ["复述问题边界", "给出结论与依据", "输出行动建议"],
  "risk_guardrails": ["证据不足时明确不下结论", "高风险场景不提供决策指令"],
  "refusal_policy": ["超出证据范围拒答并说明原因", "拒绝伪造引用"]
}
""".strip()

    content = build_skill_markdown_from_profile(
        _sample_profile(),
        llm_client=_FakeLLMClient(),
    )

    assert "# SKILL: 黄彦臻风格助理" in content
    assert "## 风格原则" in content
    assert "## 回答流程" in content
    assert "## 风险守则" in content
    assert "## 拒答策略" in content


def test_render_skill_filename_sanitizes_author_name():
    filename = render_skill_filename(author="黄 彦臻 / test")
    assert filename.endswith("_skill.md")
    assert " " not in filename
    assert "/" not in filename
