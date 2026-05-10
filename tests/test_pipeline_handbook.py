"""Tests for Phase 3 Handbook generation pipeline nodes."""

from __future__ import annotations

from expert_digest.pipeline.handbook.editor import _assemble_raw
from expert_digest.pipeline.handbook.graph import build_handbook_subgraph
from expert_digest.pipeline.handbook.planner import _parse_chapter_plan
from expert_digest.pipeline.handbook.reviewer import (
    _parse_review,
    all_chapters_pass,
)
from expert_digest.pipeline.handbook.tracer import run_build_trace
from expert_digest.pipeline.handbook.writer import (
    _build_writer_prompt,
    _gather_context_for_chapter,
)
from expert_digest.pipeline.state import (
    ChapterDraft,
    ChapterPlan,
    ReviewResult,
    Theme,
    make_initial_state,
)


class TestChapterPlanParsing:
    """Tests for the planner node's JSON parsing."""

    def test_parse_valid_json(self):
        raw = '[{"title": "t1", "purpose": "p1", "target_themes": ["th1"], "estimated_sections": 4}]'
        result = _parse_chapter_plan(raw)
        assert len(result) == 1
        assert result[0].title == "t1"
        assert result[0].purpose == "p1"
        assert result[0].target_themes == ["th1"]
        assert result[0].estimated_sections == 4

    def test_parse_json_with_fence(self):
        raw = "```json\n[{\"title\": \"t1\", \"purpose\": \"p1\"}]\n```"
        result = _parse_chapter_plan(raw)
        assert len(result) == 1
        assert result[0].title == "t1"

    def test_parse_json_with_fence_no_lang(self):
        raw = "```\n[{\"title\": \"t1\", \"purpose\": \"p1\"}]\n```"
        result = _parse_chapter_plan(raw)
        assert len(result) == 1

    def test_parse_invalid_json_returns_empty(self):
        assert _parse_chapter_plan("not json") == []

    def test_parse_non_list_returns_empty(self):
        assert _parse_chapter_plan("{}") == []

    def test_parse_skips_invalid_items(self):
        raw = (
            '[{"title": "t1", "purpose": "p1"}, '
            '{"title": "", "purpose": "p2"}, '
            '{"title": "t3", "purpose": ""}, '
            '{"title": "t4", "purpose": "p4", "estimated_sections": 99}]'
        )
        result = _parse_chapter_plan(raw)
        assert len(result) == 2
        assert result[0].title == "t1"
        assert result[1].title == "t4"
        assert result[1].estimated_sections == 10  # clamped

    def test_parse_respects_max_8(self):
        items = [{"title": f"t{i}", "purpose": f"p{i}"} for i in range(12)]
        raw = str(items).replace("'", '"')
        result = _parse_chapter_plan(raw)
        assert len(result) == 8

    def test_parse_handles_non_dict_items(self):
        raw = '[{"title": "t1", "purpose": "p1"}, "not a dict"]'
        result = _parse_chapter_plan(raw)
        assert len(result) == 1

    def test_parse_handles_estimated_non_int(self):
        raw = '[{"title": "t1", "purpose": "p1", "estimated_sections": "lots"}]'
        result = _parse_chapter_plan(raw)
        assert result[0].estimated_sections == 3  # default

    def test_parse_handles_target_themes_non_list(self):
        raw = '[{"title": "t1", "purpose": "p1", "target_themes": "not a list"}]'
        result = _parse_chapter_plan(raw)
        assert result[0].target_themes == []


class TestWriterUtilities:
    """Tests for the writer node's utility functions."""

    def test_build_writer_prompt_includes_plan(self):
        plan = ChapterPlan(title="测试章节", purpose="学习目的", target_themes=["t1"])
        texts = ["### doc1\ncontent1"]
        prompt = _build_writer_prompt(plan, texts, [])
        assert "测试章节" in prompt
        assert "学习目的" in prompt
        assert "content1" in prompt

    def test_build_writer_prompt_includes_feedback(self):
        plan = ChapterPlan(title="测试", purpose="目的")
        prompt = _build_writer_prompt(plan, [], [], review_feedback=["缺少案例", "结构不清晰"])
        assert "缺少案例" in prompt
        assert "结构不清晰" in prompt

    def test_build_writer_prompt_no_feedback(self):
        plan = ChapterPlan(title="测试", purpose="目的")
        prompt = _build_writer_prompt(plan, [], [])
        assert "上一轮评审意见" not in prompt

    def test_find_relevant_docs_theme_match(self):
        state = make_initial_state()
        state["themes"] = [Theme(label="t1", summary="s1", source_document_ids=["d1"])]
        state["documents"] = [{"id": "d1", "title": "doc1", "content": "c1"}, {"id": "d2", "title": "doc2", "content": "c2"}]
        plan = ChapterPlan(title="测试", purpose="目的", target_themes=["t1"])
        texts = _gather_context_for_chapter(plan, state)
        assert len(texts) >= 1
        assert any("doc1" in t for t in texts)

    def test_find_relevant_docs_fallback(self):
        state = make_initial_state()
        state["documents"] = [{"id": "d1", "title": "doc1", "content": "c1"}]
        plan = ChapterPlan(title="测试", purpose="目的", target_themes=["unmatched"])
        texts = _gather_context_for_chapter(plan, state)
        assert len(texts) >= 1
        assert any("doc1" in t for t in texts)

    def test_find_relevant_docs_empty_docs(self):
        state = make_initial_state()
        plan = ChapterPlan(title="测试", purpose="目的")
        texts = _gather_context_for_chapter(plan, state)
        assert texts == []


class TestReviewer:
    """Tests for the reviewer node."""

    def test_parse_review_pass(self):
        raw = '{"passed": true, "issues": []}'
        result = _parse_review(raw)
        assert result["passed"] is True
        assert result["issues"] == []

    def test_parse_review_fail_with_issues(self):
        raw = '{"passed": false, "issues": ["缺少事实依据", "深度不够"]}'
        result = _parse_review(raw)
        assert result["passed"] is False
        assert len(result["issues"]) == 2

    def test_parse_review_with_fence(self):
        raw = "```\n{\"passed\": true}\n```"
        result = _parse_review(raw)
        assert result["passed"] is True

    def test_parse_review_invalid_json(self):
        result = _parse_review("not json")
        assert result["passed"] is False
        assert len(result["issues"]) == 1

    def test_parse_review_non_dict(self):
        result = _parse_review("[1, 2, 3]")
        assert result["passed"] is False

    def test_all_chapters_pass_proceed_when_all_pass(self):
        state = make_initial_state()
        state["review_results"] = [
            ReviewResult(chapter_title="c1", passed=True, issues=[]),
            ReviewResult(chapter_title="c2", passed=True, issues=[]),
        ]
        state["chapters"] = [ChapterDraft(title="c1", content="x"), ChapterDraft(title="c2", content="y")]
        assert all_chapters_pass(state) == "proceed"

    def test_all_chapters_pass_rewrite_when_any_fails(self):
        state = make_initial_state()
        state["review_results"] = [
            ReviewResult(chapter_title="c1", passed=True),
            ReviewResult(chapter_title="c2", passed=False, issues=["缺少深度"]),
        ]
        state["chapters"] = [ChapterDraft(title="c1", content="x"), ChapterDraft(title="c2", content="y")]
        assert all_chapters_pass(state) == "rewrite"

    def test_all_chapters_pass_proceed_on_empty_results(self):
        state = make_initial_state()
        assert all_chapters_pass(state) == "proceed"

    def test_all_chapters_pass_rewrite_on_any_fail_second_round(self):
        """Even on second round, if any chapter still fails → rewrite."""
        state = make_initial_state()
        state["chapters"] = [ChapterDraft(title="c1", content="x")]
        state["review_results"] = [
            ReviewResult(chapter_title="c1", passed=False, issues=["still bad"]),
        ]
        assert all_chapters_pass(state) == "rewrite"


class TestHandbookGraph:
    """Tests for the handbook subgraph structure."""

    def test_build_returns_stategraph(self):
        graph = build_handbook_subgraph()
        assert "StateGraph" in type(graph).__name__

    def test_compile_and_invoke(self):
        graph = build_handbook_subgraph().compile()
        state = make_initial_state(author="test")
        result = graph.invoke(state)
        assert result["author"] == "test"

    def test_has_rewrite_conditional_edge(self):
        """Verify the review→draft conditional edge exists."""
        graph = build_handbook_subgraph()
        # The conditional edge should map "rewrite" → "draft_chapters"
        # and "proceed" → "coherence_pass". Verify by checking the
        # compile doesn't complain about missing targets.
        compiled = graph.compile()
        state = make_initial_state(author="test")
        result = compiled.invoke(state)
        assert isinstance(result, dict)


class TestEditor:
    """Tests for the coherence pass editor."""

    def test_build_editor_prompt_includes_all_chapters(self):
        chapters = [
            {"title": "第一章", "content": "内容1"},
            {"title": "第二章", "content": "内容2"},
        ]
        prompt = _assemble_raw(chapters, "作者甲")
        assert "第一章" in prompt
        assert "第二章" in prompt
        assert "内容1" in prompt
        assert "作者甲" in prompt

    def test_editor_prompt_includes_transition(self):
        chapters = [{"title": "A", "content": "a"}, {"title": "B", "content": "b"}]
        prompt = _assemble_raw(chapters, "test")
        # Should have a separator between chapters
        assert "---" in prompt


class TestTracer:
    """Tests for the trace builder."""

    def test_build_trace_produces_file(self, tmp_path):
        state = make_initial_state(
            author="test",
            output_dir=str(tmp_path / "outputs"),
        )
        state["chapters"] = [
            ChapterDraft(title="c1", content="hello", section_count=1),
        ]
        state["chapter_plan"] = [
            ChapterPlan(title="c1", purpose="p1", target_themes=["t1"]),
        ]
        # Need themes + documents for theme→doc mapping
        state["themes"] = [Theme(label="t1", summary="s1", source_document_ids=["d1"])]
        state["documents"] = [{"id": "d1", "title": "doc1", "content": "c1"}]

        run_build_trace(state)
        trace_file = tmp_path / "outputs" / "handbook.trace.json"
        assert trace_file.exists()
        content = trace_file.read_text(encoding="utf-8")
        assert "test" in content
        assert "c1" in content

    def test_build_trace_handles_no_chapters(self):
        state = make_initial_state()
        result = run_build_trace(state)
        assert result == {}
