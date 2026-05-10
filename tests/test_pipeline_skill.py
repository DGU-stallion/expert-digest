"""Tests for Phase 4 SKILL distillation pipeline nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from expert_digest.pipeline.skill.builder import run_build_skill_md
from expert_digest.pipeline.skill.expresser import (
    _build_expresser_prompt,
    _parse_expresser_json,
    run_encode_expression,
)
from expert_digest.pipeline.skill.graph import build_skill_subgraph
from expert_digest.pipeline.skill.protocol import (
    _build_protocol_prompt,
    run_design_protocol,
)
from expert_digest.pipeline.skill.thinker import (
    _build_mental_models,
    _ensure_strings,
    _parse_thinker_json,
    run_extract_mental_models,
)
from expert_digest.pipeline.skill.verifier import run_verify_skill, skill_verdict
from expert_digest.pipeline.state import (
    ExpressionDNA,
    MentalModel,
    Theme,
    make_initial_state,
)


# ── Thinker (mental model extraction) ────────────────────────────────────


class TestThinkerParsing:
    """Tests for the thinker node's JSON parsing."""

    def test_parse_valid_json(self):
        raw = '{"mental_models": [{"name": "m1", "summary": "s1"}], "decision_heuristics": ["h1"]}'
        result = _parse_thinker_json(raw)
        assert len(result["mental_models"]) == 1
        assert result["mental_models"][0]["name"] == "m1"
        assert result["decision_heuristics"] == ["h1"]

    def test_parse_json_with_fence(self):
        raw = '```json\n{"mental_models": [{"name": "m1", "summary": "s1"}]}\n```'
        result = _parse_thinker_json(raw)
        assert len(result["mental_models"]) == 1

    def test_parse_json_with_fence_no_lang(self):
        raw = '```\n{"mental_models": [{"name": "m1", "summary": "s1"}]}\n```'
        result = _parse_thinker_json(raw)
        assert len(result["mental_models"]) == 1

    def test_parse_invalid_json_returns_empty(self):
        assert _parse_thinker_json("not json") == {}

    def test_parse_non_dict_returns_empty(self):
        assert _parse_thinker_json("[]") == {}


class TestBuildMentalModels:
    """Tests for _build_mental_models."""

    def test_build_valid_models(self):
        raw = [
            {"name": "m1", "summary": "s1", "evidence_snippet": "e1", "application": "a1", "limitation": "l1"},
            {"name": "m2", "summary": "s2"},
        ]
        models = _build_mental_models(raw)
        assert len(models) == 2
        assert models[0].name == "m1"
        assert models[0].summary == "s1"
        assert models[0].evidence_snippet == "e1"
        assert models[0].application == "a1"
        assert models[0].limitation == "l1"
        assert models[1].name == "m2"
        assert models[1].evidence_snippet == ""

    def test_skips_items_missing_name_or_summary(self):
        raw = [
            {"name": "m1", "summary": "s1"},
            {"name": "", "summary": "s2"},  # empty name
            {"name": "m3", "summary": ""},  # empty summary
            {},  # missing both
            "not a dict",
        ]
        models = _build_mental_models(raw)
        assert len(models) == 1
        assert models[0].name == "m1"

    def test_respects_max_7(self):
        raw = [{"name": f"m{i}", "summary": f"s{i}"} for i in range(10)]
        models = _build_mental_models(raw)
        assert len(models) == 7

    def test_empty_list(self):
        assert _build_mental_models([]) == []

    def test_none_values_truncated(self):
        raw = [{"name": "  m1  ", "summary": "  s1  "}]
        models = _build_mental_models(raw)
        assert models[0].name == "m1"
        assert models[0].summary == "s1"


class TestEnsureStrings:
    """Tests for _ensure_strings."""

    def test_valid_list(self):
        assert _ensure_strings(["a", "b"]) == ["a", "b"]

    def test_filters_and_strips(self):
        assert _ensure_strings(["  a  ", "", "b", "  "]) == ["a", "b"]

    def test_filters_non_strings(self):
        assert _ensure_strings(["a", 1, None, "b"]) == ["a", "b"]

    def test_non_list_returns_empty(self):
        assert _ensure_strings("not a list") == []

    def test_empty_list(self):
        assert _ensure_strings([]) == []


class TestThinkerNode:
    """Tests for the thinker node entry point."""

    def test_early_return_when_no_themes(self):
        state = make_initial_state()
        result = run_extract_mental_models(state)
        assert result["mental_models"] == []
        assert result["decision_heuristics"] == []
        assert result["values_antipatterns"] == {}
        assert result["honest_boundaries"] == []

    @patch("expert_digest.pipeline.skill.thinker.require_reasoning_client")
    def test_with_mock_llm(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"mental_models": [{"name": "模型1", "summary": "概要1"}],'
            '"decision_heuristics": ["启发式1"],'
            '"values_antipatterns": {"pursues": ["追1"], "opposes": ["反1"]},'
            '"honest_boundaries": ["边界1"]}'
        )
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["themes"] = [Theme(label="t1", summary="s1")]
        result = run_extract_mental_models(state)
        assert len(result["mental_models"]) == 1
        assert result["mental_models"][0].name == "模型1"
        assert result["decision_heuristics"] == ["启发式1"]
        assert result["values_antipatterns"]["pursues"] == ["追1"]
        assert result["honest_boundaries"] == ["边界1"]

    @patch("expert_digest.pipeline.skill.thinker.require_reasoning_client")
    def test_handles_broken_llm_output(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = "broken response"
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["themes"] = [Theme(label="t1", summary="s1")]
        result = run_extract_mental_models(state)
        assert result["mental_models"] == []
        assert result["decision_heuristics"] == []
        assert result["honest_boundaries"] == []


# ── Expresser (expression encoding) ──────────────────────────────────────


class TestExpresserParsing:
    """Tests for the expresser node's JSON parsing."""

    def test_parse_valid_json(self):
        raw = '{"role_rules": ["规则1", "规则2"], "identity_card": "id", "expression_dna_description": "desc"}'
        result = _parse_expresser_json(raw)
        assert result["role_rules"] == ["规则1", "规则2"]
        assert result["identity_card"] == "id"

    def test_parse_with_fence(self):
        raw = '```json\n{"role_rules": ["规则1"]}\n```'
        result = _parse_expresser_json(raw)
        assert result["role_rules"] == ["规则1"]

    def test_parse_invalid_returns_empty(self):
        assert _parse_expresser_json("garbage") == {}


class TestExpresserPrompt:
    """Tests for the expresser prompt builder."""

    def test_build_prompt_includes_dna(self):
        state = make_initial_state()
        state["expression_dna"] = ExpressionDNA(
            sentence_patterns=["结论先行"],
            high_frequency_phrases=["其实"],
            certainty_spectrum=["不确定", "确定"],
            citation_habits="引日常",
        )
        state["intellectual_genealogy"] = "智识谱系内容"
        state["key_decisions"] = [{"topic": "议题1", "position": "立场1"}]
        prompt = _build_expresser_prompt(state)
        assert "结论先行" in prompt
        assert "其实" in prompt
        assert "智识谱系内容" in prompt
        assert "议题1" in prompt

    def test_build_prompt_empty_state(self):
        state = make_initial_state()
        prompt = _build_expresser_prompt(state)
        assert prompt == ""


class TestExpresserNode:
    """Tests for the expresser node entry point."""

    def test_early_return_when_no_dna(self):
        state = make_initial_state()
        result = run_encode_expression(state)
        assert result["role_rules"] == ""

    @patch("expert_digest.pipeline.skill.expresser.require_fast_client")
    def test_with_mock_llm(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"role_rules": ["规则1：直接回应", "规则2：结论先行"],'
            '"identity_card": "我是谁：作者",'
            '"expression_dna_description": "风格流畅"}'
        )
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["expression_dna"] = ExpressionDNA()
        result = run_encode_expression(state)
        assert "角色扮演规则" in result["role_rules"]
        assert "规则1：直接回应" in result["role_rules"]
        assert "身份卡" in result["role_rules"]
        assert "表达DNA" in result["role_rules"]

    @patch("expert_digest.pipeline.skill.expresser.require_fast_client")
    def test_handles_broken_llm_output(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = "broken"
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["expression_dna"] = ExpressionDNA()
        result = run_encode_expression(state)
        assert result["role_rules"] == ""

    @patch("expert_digest.pipeline.skill.expresser.require_fast_client")
    def test_handles_role_rules_as_string(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = '{"role_rules": "single string rule"}'
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["expression_dna"] = ExpressionDNA()
        result = run_encode_expression(state)
        assert "single string rule" in result["role_rules"]


# ── Protocol (Agentic Protocol design) ───────────────────────────────────


class TestProtocolPrompt:
    """Tests for the protocol prompt builder."""

    def test_build_prompt_includes_models_heuristics_themes(self):
        state = make_initial_state()
        state["mental_models"] = [MentalModel(name="模型1", summary="概要1")]
        state["decision_heuristics"] = ["启发式1"]
        state["themes"] = [Theme(label="主题1", summary="摘要1")]
        prompt = _build_protocol_prompt(state)
        assert "模型1" in prompt
        assert "概要1" in prompt
        assert "启发式1" in prompt
        assert "主题1" in prompt

    def test_build_prompt_empty_state(self):
        state = make_initial_state()
        prompt = _build_protocol_prompt(state)
        assert prompt == ""


class TestProtocolNode:
    """Tests for the protocol node entry point."""

    def test_early_return_when_no_models(self):
        state = make_initial_state()
        result = run_design_protocol(state)
        assert result["protocol_steps"] == ""

    @patch("expert_digest.pipeline.skill.protocol.require_fast_client")
    def test_with_mock_llm(self, mock_require):
        mock_client = MagicMock()
        mock_client.generate.return_value = "## 问题分类\n| 类型 | 策略 |\n| --- | --- |\n"
        mock_require.return_value = mock_client

        state = make_initial_state()
        state["mental_models"] = [MentalModel(name="模型1", summary="概要1")]
        result = run_design_protocol(state)
        assert "问题分类" in result["protocol_steps"]
        assert "| 类型 | 策略 |" in result["protocol_steps"]


# ── Builder (SKILL.md assembly) ──────────────────────────────────────────


class TestBuilder:
    """Tests for the SKILL.md builder (deterministic assembly)."""

    def test_build_full_skill_md(self):
        state = make_initial_state(author="测试作者")
        state["themes"] = [Theme(label="投资", summary="投资分析")]
        state["role_rules"] = "## 角色扮演规则\n规则1"
        state["protocol_steps"] = "## 回答步骤\n步骤1"
        state["mental_models"] = [
            MentalModel(name="模型1", summary="概要1", evidence_snippet="证据1", application="应用1", limitation="局限1"),
        ]
        state["decision_heuristics"] = ["启发式1", "启发式2"]
        state["values_antipatterns"] = {
            "pursues": ["追求1"],
            "opposes": ["反对1"],
            "tensions": ["张力1"],
        }
        state["honest_boundaries"] = ["边界1", "边界2"]
        state["documents"] = [{"title": "文1", "url": "https://example.com/1"}]

        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "测试作者" in md
        assert "投资" in md
        assert "角色扮演规则" in md
        assert "回答工作流" in md
        assert "核心心智模型" in md
        assert "决策启发式" in md
        assert "价值观与反模式" in md
        assert "诚实边界" in md
        assert "推荐阅读" in md

    def test_build_empty_state(self):
        state = make_initial_state(author="空作者")
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "空作者" in md
        # Optional sections should be absent
        assert "心智模型" not in md
        assert "角色扮演规则" not in md  # no role_rules content

    def test_build_no_themes(self):
        state = make_initial_state(author="作者")
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert md.strip() == "# 作者"  # bare minimum

    def test_build_values_empty_dict(self):
        state = make_initial_state(author="作者")
        state["themes"] = [Theme(label="t1", summary="s1")]
        state["values_antipatterns"] = {}
        state["honest_boundaries"] = []
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "价值观与反模式" not in md
        assert "诚实边界" not in md

    def test_build_none_values_dict(self):
        state = make_initial_state(author="作者")
        state["themes"] = [Theme(label="t1", summary="s1")]
        state["values_antipatterns"] = {"pursues": [], "opposes": [], "tensions": []}
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "价值观与反模式" not in md  # all empty lists

    def test_build_documents_limit_10(self):
        state = make_initial_state(author="作者")
        state["themes"] = [Theme(label="t1", summary="s1")]
        state["documents"] = [{"title": f"文{i}", "url": f"https://ex.com/{i}"} for i in range(15)]
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        # Only first 10 should appear
        for i in range(10):
            assert f"文{i}" in md
        assert "文10" not in md

    def test_build_documents_without_url(self):
        state = make_initial_state(author="作者")
        state["themes"] = [Theme(label="t1", summary="s1")]
        state["documents"] = [{"title": "无链接文"}]
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "无链接文" in md

    def test_build_heuristics_numbered(self):
        state = make_initial_state(author="作者")
        state["themes"] = [Theme(label="t1", summary="s1")]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        result = run_build_skill_md(state)
        md = result.get("skill_markdown", "")
        assert "1. h1" in md
        assert "2. h2" in md
        assert "3. h3" in md


# ── Verifier (quality checks) ────────────────────────────────────────────


class TestVerifier:
    """Tests for the SKILL quality verifier."""

    def _make_valid_skill_md(self) -> str:
        """Build a SKILL.md string long enough to pass the 500-char length check."""
        lines = [
            "# 作者 · 投资",
            "",
            "## 角色扮演规则",
            "规则内容。直接以作者身份回应，结论先行，再用论据展开。",
            "",
            "## 回答工作流（Agentic Protocol）",
            "协议步骤内容。第一步分类问题，第二步研究分析，第三步输出回答。",
            "",
            "## 核心心智模型",
            "### 模型1: 四阶段框架",
            "一句话：市场定价分四个阶段。",
            "证据：原文片段。",
            "应用：分析股票时先判断阶段。",
            "局限：不适用于所有市场环境。",
            "",
            "### 模型2: 低线性关联",
            "一句话：事物间的关联往往被高估。",
            "证据：原文片段。",
            "",
            "### 模型3: 胜率-赔率匹配",
            "一句话：高胜率低赔率 vs 低胜率高赔率。",
            "证据：原文片段。",
            "",
            "## 决策启发式",
            "1. 不赌趋势的力量",
            "2. 关注结构性机会而不是短期波动",
            "3. 赔率思维优先于胜率思维",
            "4. 独立思考不盲从市场共识",
            "",
            "## 价值观与反模式",
            "### 追求",
            "- 独立判断能力",
            "- 赔率思维和结构性机会",
            "- 长期主义视角",
            "",
            "### 反对",
            "- 跟风和短期博弈",
            "- 无框架的主观判断",
            "",
            "### 内在张力",
            "- 独立判断与市场验证之间的平衡",
            "",
            "## 诚实边界",
            "- 不能预测短期市场走势",
            "- 部分判断基于未公开的个人经验",
            "- 公开表达与私下操作可能有差异",
            "- 信息截止到调研时间点",
            "- 不具备行业内部信息优势",
            "",
            "## 推荐阅读",
            "- [相关文章](https://example.com)",
            "",
        ]
        md = "\n".join(lines)
        assert len(md) > 500, f"test content too short: {len(md)}"
        return md

    def test_pass_valid_skill(self):
        """A well-formed SKILL.md passes all three tests."""
        md = self._make_valid_skill_md()
        state = make_initial_state()
        state["skill_markdown"] = md
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2"), MentalModel(name="m3", summary="s3")]
        state["honest_boundaries"] = ["b1", "b2", "b3"]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        state["role_rules"] = "规则内容"
        state["protocol_steps"] = "协议内容"
        state["values_antipatterns"] = {"pursues": ["追求1"]}
        result = run_verify_skill(state)
        assert result["_skill_verified"] is True

    def test_fail_empty_markdown(self):
        state = make_initial_state()
        state["skill_markdown"] = ""
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False

    def test_fail_missing_required_sections(self):
        md = "## 只有标题\n"
        state = make_initial_state()
        state["skill_markdown"] = md
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2"), MentalModel(name="m3", summary="s3")]
        state["honest_boundaries"] = ["b1", "b2", "b3"]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        state["role_rules"] = "规则"
        state["protocol_steps"] = "协议"
        state["values_antipatterns"] = {"pursues": ["追求1"]}
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False
        # Should mention missing sections
        error_msg = result.get("errors", [])[0].message
        assert "心智模型" in error_msg or "诚实边界" in error_msg

    def test_fail_insufficient_depth(self):
        md = "## 核心心智模型\n内容\n## 角色扮演规则\n内容\n## 回答工作流（Agentic Protocol）\n内容\n## 诚实边界\n内容\n## 决策启发式\n内容\n"
        state = make_initial_state()
        state["skill_markdown"] = md
        # Only 2 models (need 3)
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2")]
        state["honest_boundaries"] = ["b1", "b2"]
        state["decision_heuristics"] = ["h1", "h2"]
        state["role_rules"] = "规则"
        state["protocol_steps"] = "协议"
        state["values_antipatterns"] = {"pursues": ["追求1"]}
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False

    def test_fail_template_variables(self):
        md = "## 核心心智模型\n{{变量}}"
        state = make_initial_state()
        state["skill_markdown"] = md
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2"), MentalModel(name="m3", summary="s3")]
        state["honest_boundaries"] = ["b1", "b2", "b3"]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        state["role_rules"] = "规则"
        state["protocol_steps"] = "协议"
        state["values_antipatterns"] = {"pursues": ["追求1"]}
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False

    def test_fail_too_short(self):
        md = "# 标题"
        state = make_initial_state()
        state["skill_markdown"] = md
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2"), MentalModel(name="m3", summary="s3")]
        state["honest_boundaries"] = ["b1", "b2", "b3"]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        state["role_rules"] = "规则"
        state["protocol_steps"] = "协议"
        state["values_antipatterns"] = {"pursues": ["追求1"]}
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False

    def test_missing_values_antipatterns(self):
        md = "## 核心心智模型\n内容\n## 角色扮演规则\n内容\n## 回答工作流（Agentic Protocol）\n内容\n## 诚实边界\n内容\n## 决策启发式\n内容\n"
        state = make_initial_state()
        state["skill_markdown"] = md
        state["mental_models"] = [MentalModel(name="m1", summary="s1"), MentalModel(name="m2", summary="s2"), MentalModel(name="m3", summary="s3")]
        state["honest_boundaries"] = ["b1", "b2", "b3"]
        state["decision_heuristics"] = ["h1", "h2", "h3"]
        state["role_rules"] = "规则"
        state["protocol_steps"] = "协议"
        state["values_antipatterns"] = {}  # empty dict
        result = run_verify_skill(state)
        assert result["_skill_verified"] is False


class TestSkillVerdict:
    """Tests for the verifier routing function."""

    def test_always_routes_to_output(self):
        state = make_initial_state()
        assert skill_verdict(state) == "output"


# ── Skill Subgraph ───────────────────────────────────────────────────────


class TestSkillGraph:
    """Tests for the skill subgraph structure."""

    def test_build_returns_stategraph(self):
        graph = build_skill_subgraph()
        assert "StateGraph" in type(graph).__name__

    def test_compile_and_invoke(self):
        graph = build_skill_subgraph().compile()
        state = make_initial_state(author="test")
        result = graph.invoke(state)
        assert result["author"] == "test"
        assert result["_skill_verified"] is False

    def test_verify_node_always_routes_to_end(self):
        """The verify node's conditional edge always maps to output."""
        graph = build_skill_subgraph().compile()
        state = make_initial_state(author="test")
        # skill_verdict always returns "output", which maps to END
        result = graph.invoke(state)
        assert isinstance(result, dict)
        assert result["_skill_verified"] is False
