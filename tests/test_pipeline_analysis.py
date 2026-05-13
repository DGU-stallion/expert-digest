"""Tests for Phase 2 analysis pipeline nodes."""

from __future__ import annotations

from expert_digest.pipeline.nodes.analyzer import (
    _build_themes,
    _parse_analysis_json,
)
from expert_digest.pipeline.nodes.expression import (
    _ensure_str_list,
    _parse_expression_json,
)
from expert_digest.pipeline.nodes.quality import (
    run_assess_quality,
    should_retry_analysis,
)
from expert_digest.pipeline.state import DigestState, Theme, make_initial_state


class TestAnalyzerParsing:
    """Tests for the analyzer node's JSON parsing and theme building."""

    def test_parse_valid_json(self):
        raw = '{"themes": [], "concepts": ["c1"], "thinking_patterns": ["p1"]}'
        result = _parse_analysis_json(raw)
        assert result["concepts"] == ["c1"]

    def test_parse_json_with_markdown_fence(self):
        raw = "```json\n{\"themes\": [], \"concepts\": [\"c1\"]}\n```"
        result = _parse_analysis_json(raw)
        assert result["concepts"] == ["c1"]

    def test_parse_json_with_markdown_fence_no_lang(self):
        raw = "```\n{\"themes\": []}\n```"
        result = _parse_analysis_json(raw)
        assert result == {"themes": []}

    def test_parse_invalid_json_returns_empty(self):
        result = _parse_analysis_json("not json")
        assert result == {}

    def test_parse_json_array_returns_empty(self):
        result = _parse_analysis_json("[1, 2, 3]")
        assert result == {}

    def test_build_themes_empty_input(self):
        result = _build_themes([], [])
        assert result == []

    def test_build_themes_skips_invalid_items(self):
        raw = [
            {"label": "t1", "summary": "s1"},
            {"label": "", "summary": "s2"},
            {"label": "t3", "summary": ""},
            {"label": "t4", "summary": "s4", "source_titles": ["doc1"], "related_concepts": ["c1"]},
        ]
        docs = [{"id": "id1", "title": "doc1"}]
        themes = _build_themes(raw, docs)
        assert len(themes) == 2
        assert themes[0].label == "t1"
        assert themes[0].summary == "s1"
        assert themes[1].label == "t4"
        assert themes[1].source_document_ids == ["id1"]
        assert themes[1].related_concepts == ["c1"]

    def test_build_themes_respects_max_6(self):
        raw = [{"label": f"t{i}", "summary": f"s{i}"} for i in range(10)]
        themes = _build_themes(raw, [])
        assert len(themes) == 6

    def test_build_themes_handles_non_list_source_titles(self):
        raw = [{"label": "t1", "summary": "s1", "source_titles": "not a list"}]
        themes = _build_themes(raw, [])
        assert len(themes) == 1
        assert themes[0].source_document_ids == []


class TestExpressionParsing:
    """Tests for the expression node's parsing utilities."""

    def test_parse_valid_json(self):
        raw = '{"sentence_patterns": ["sp1"], "intellectual_genealogy": "ig"}'
        result = _parse_expression_json(raw)
        assert result["sentence_patterns"] == ["sp1"]
        assert result["intellectual_genealogy"] == "ig"

    def test_parse_json_with_fence(self):
        raw = "```\n{\"high_frequency_phrases\": [\"a\", \"b\"]}\n```"
        result = _parse_expression_json(raw)
        assert result["high_frequency_phrases"] == ["a", "b"]

    def test_parse_invalid_returns_empty(self):
        assert _parse_expression_json("bad") == {}

    def test_ensure_str_list_with_list(self):
        assert _ensure_str_list(["a", "b"]) == ["a", "b"]

    def test_ensure_str_list_filters_non_strings(self):
        assert _ensure_str_list(["a", 1, None, "b"]) == ["a", "b"]

    def test_ensure_str_list_with_non_list(self):
        assert _ensure_str_list("hello") == []

    def test_ensure_str_list_empty(self):
        assert _ensure_str_list([]) == []


class TestQualityNode:
    """Tests for the quality check node."""

    def test_assess_quality_passes_with_good_data(self):
        state = make_initial_state()
        state["themes"] = [Theme(label="t", summary="s") for _ in range(3)]
        state["concepts"] = [f"c{i}" for i in range(8)]
        state["thinking_patterns"] = [f"p{i}" for i in range(3)]
        result = run_assess_quality(state)
        assert result == {}  # no issues

    def test_assess_quality_reports_insufficient(self):
        state = make_initial_state()
        state["themes"] = [Theme(label="t", summary="s")]
        state["concepts"] = []
        state["thinking_patterns"] = []
        result = run_assess_quality(state)
        assert result["current_round"] == 1
        # Errors are recorded even on retry for traceability
        assert len(result["errors"]) == 1
        assert "insufficient_themes" in result["errors"][0].message

    def test_assess_quality_records_error_when_max_rounds_reached(self):
        state = make_initial_state(max_rounds=1)
        state["current_round"] = 1
        state["themes"] = []
        state["concepts"] = []
        state["thinking_patterns"] = []
        result = run_assess_quality(state)
        assert len(result["errors"]) == 1
        assert "max_retry_rounds_exceeded" in result["errors"][0].message

    def test_should_retry_with_good_data(self):
        state: DigestState = {
            **make_initial_state(),
            "themes": [Theme(label="t", summary="s") for _ in range(3)],
            "concepts": [f"c{i}" for i in range(5)],
            "thinking_patterns": ["p1", "p2"],
        }
        assert should_retry_analysis(state) == "proceed"

    def test_should_retry_with_bad_data(self):
        state = make_initial_state()
        state["themes"] = []
        state["concepts"] = []
        state["thinking_patterns"] = []
        assert should_retry_analysis(state) == "retry"

    def test_should_retry_gives_up_after_max_rounds(self):
        state = make_initial_state(max_rounds=2)
        state["current_round"] = 2
        state["themes"] = []
        state["concepts"] = []
        state["thinking_patterns"] = []
        assert should_retry_analysis(state) == "proceed"


class TestLoaderLogic:
    """Tests for loader behavior."""

    def test_loader_returns_empty_without_db(self):
        from expert_digest.pipeline.nodes.loader import run_load_data

        state = make_initial_state()
        result = run_load_data(state)
        assert result["documents"] == []

    def test_loader_returns_empty_with_missing_db(self):
        from expert_digest.pipeline.nodes.loader import run_load_data

        state = make_initial_state(db_path="/nonexistent/test.sqlite3")
        result = run_load_data(state)
        assert result["documents"] == []
