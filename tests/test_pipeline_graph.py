"""Tests for LangGraph pipeline graph structure and execution."""

from __future__ import annotations

from expert_digest.pipeline.graph import build_main_graph, compile_pipeline
from expert_digest.pipeline.handbook.graph import build_handbook_subgraph
from expert_digest.pipeline.skill.graph import build_skill_subgraph
from expert_digest.pipeline.state import make_initial_state


class TestMainGraph:
    """Tests for the main pipeline graph."""

    def test_build_returns_stategraph(self):
        graph = build_main_graph()
        assert "StateGraph" in type(graph).__name__

    def test_compile_returns_compiled_graph(self):
        compiled = compile_pipeline()
        assert "CompiledStateGraph" in type(compiled).__name__

    def test_invoke_empty_state_returns_all_keys(self):
        pipeline = compile_pipeline()
        state = make_initial_state(author="test")
        result = pipeline.invoke(state)
        assert isinstance(result, dict)
        assert result["author"] == "test"

    def test_invoke_preserves_string_fields(self):
        pipeline = compile_pipeline()
        state = make_initial_state(author="黄彦臻", wiki_root="/data/wiki")
        result = pipeline.invoke(state)
        assert result["author"] == "黄彦臻"
        assert result["wiki_root"] == "/data/wiki"

    def test_invoke_state_passes_through_all_fields(self):
        """Verify stub nodes pass state through without dropping keys."""
        pipeline = compile_pipeline()
        state = make_initial_state(author="test")
        result = pipeline.invoke(state)
        expected_keys = {
            "db_path", "wiki_root", "author", "documents",
            "themes", "concepts",
            "thinking_patterns", "expression_dna", "intellectual_genealogy",
            "key_decisions", "chapter_plan", "chapters", "review_results",
            "handbook_markdown", "mental_models", "decision_heuristics",
            "values_antipatterns", "honest_boundaries", "skill_markdown",
            "max_rounds", "current_round", "errors", "output_dir",
        }
        assert set(result.keys()) == expected_keys


class TestHandbookSubgraph:
    """Tests for the handbook generation subgraph."""

    def test_build_returns_stategraph(self):
        graph = build_handbook_subgraph()
        assert "StateGraph" in type(graph).__name__

    def test_compile_and_invoke(self):
        graph = build_handbook_subgraph().compile()
        state = make_initial_state(author="test")
        result = graph.invoke(state)
        assert result["author"] == "test"


class TestSkillSubgraph:
    """Tests for the SKILL distillation subgraph."""

    def test_build_returns_stategraph(self):
        graph = build_skill_subgraph()
        assert "StateGraph" in type(graph).__name__

    def test_compile_and_invoke(self):
        graph = build_skill_subgraph().compile()
        state = make_initial_state(author="test")
        result = graph.invoke(state)
        assert result["author"] == "test"


class TestCLIPipelineCommands:
    """Tests for the CLI pipeline entry points."""

    def test_generate_handbook_pipeline_help(self):
        import subprocess
        import sys

        code = (
            "from expert_digest.cli import main; "
            "main(['generate-handbook-pipeline', '--help'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--author" in (result.stdout + result.stderr)

    def test_generate_skill_pipeline_help(self):
        import subprocess
        import sys

        code = (
            "from expert_digest.cli import main; "
            "main(['generate-skill-pipeline', '--help'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--author" in (result.stdout + result.stderr)
