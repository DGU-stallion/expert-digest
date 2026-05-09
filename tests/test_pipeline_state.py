"""Tests for pipeline state types and initial state factory."""

from __future__ import annotations

from expert_digest.pipeline.state import (
    DigestState,
    ExpressionDNA,
    MentalModel,
    PipelineError,
    Theme,
    make_initial_state,
)


def test_make_initial_state_has_all_keys() -> None:
    state = make_initial_state(
        db_path="/tmp/db",
        wiki_root="/tmp/wiki",
        author="test_author",
    )
    assert state["author"] == "test_author"
    assert state["wiki_root"] == "/tmp/wiki"
    assert state["db_path"] == "/tmp/db"
    assert state["documents"] == []
    assert state["themes"] == []
    assert state["concepts"] == []
    assert state["thinking_patterns"] == []
    assert state["expression_dna"] is None
    assert state["intellectual_genealogy"] == ""
    assert state["key_decisions"] == []
    assert state["chapter_plan"] == []
    assert state["chapters"] == []
    assert state["review_results"] == []
    assert state["handbook_markdown"] == ""
    assert state["mental_models"] == []
    assert state["decision_heuristics"] == []
    assert state["values_antipatterns"] == {}
    assert state["honest_boundaries"] == []
    assert state["skill_markdown"] == ""
    assert state["max_rounds"] == 3
    assert state["current_round"] == 0
    assert state["errors"] == []
    assert state["output_dir"] == "data/outputs"


def test_make_initial_state_custom_output_dir() -> None:
    state = make_initial_state(output_dir="custom/output")
    assert state["output_dir"] == "custom/output"


def test_make_initial_state_custom_max_rounds() -> None:
    state = make_initial_state(max_rounds=5)
    assert state["max_rounds"] == 5


def test_theme_dataclass() -> None:
    theme = Theme(label="test", summary="a test theme")
    assert theme.label == "test"
    assert theme.summary == "a test theme"
    assert theme.source_document_ids == []
    assert theme.related_concepts == []


def test_theme_with_sources() -> None:
    theme = Theme(
        label="value",
        summary="value investing",
        source_document_ids=["doc1", "doc2"],
        related_concepts=["moat", "margin of safety"],
    )
    assert len(theme.source_document_ids) == 2
    assert len(theme.related_concepts) == 2


def test_expression_dna_defaults() -> None:
    dna = ExpressionDNA()
    assert dna.sentence_patterns == []
    assert dna.high_frequency_phrases == []
    assert dna.certainty_spectrum == []
    assert dna.citation_habits == ""


def test_mental_model_dataclass() -> None:
    model = MentalModel(
        name="test_model",
        summary="a mental model",
        evidence_snippet="source text",
        application="apply to X",
        limitation="fails when Y",
    )
    assert model.name == "test_model"
    assert model.evidence_snippet == "source text"
    assert model.application == "apply to X"


def test_pipeline_error_dataclass() -> None:
    err = PipelineError(node="analyzer", message="analysis failed")
    assert err.node == "analyzer"
    assert err.message == "analysis failed"


def test_state_dict_compliance() -> None:
    """Verify DigestState is a valid TypedDict that can store all field types."""
    state: DigestState = {
        "db_path": "/db",
        "wiki_root": "/wiki",
        "author": "author",
        "documents": [{"id": "d1", "title": "test"}],
        "themes": [Theme(label="t", summary="s")],
        "concepts": ["c1"],
        "thinking_patterns": ["p1"],
        "expression_dna": None,
        "intellectual_genealogy": "g",
        "key_decisions": [{"decision": "d1"}],
        "chapter_plan": [],
        "chapters": [],
        "review_results": [],
        "handbook_markdown": "# Handbook",
        "mental_models": [],
        "decision_heuristics": [],
        "values_antipatterns": {"key": "value"},
        "honest_boundaries": ["boundary"],
        "skill_markdown": "# SKILL",
        "max_rounds": 3,
        "current_round": 0,
        "errors": [PipelineError(node="n", message="m")],
        "output_dir": "out",
    }
    assert state["themes"][0].label == "t"
    assert state["key_decisions"][0]["decision"] == "d1"
    assert state["errors"][0].message == "m"
