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
    content = build_skill_markdown_from_profile(_sample_profile())

    assert "# SKILL: 黄彦臻风格助理" in content
    assert "## 规则" in content
    assert "## 引用约束" in content
    assert "## 拒答规则" in content
    assert "必须基于 RAG 检索证据" in content


def test_render_skill_filename_sanitizes_author_name():
    filename = render_skill_filename(author="黄 彦臻 / test")
    assert filename.endswith("_skill.md")
    assert " " not in filename
    assert "/" not in filename
