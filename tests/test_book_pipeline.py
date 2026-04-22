from expert_digest.generation.book_pipeline import (
    BookPipeline,
    ChapterPlanItem,
    build_chapter_markdowns,
)


def test_build_chapter_markdowns_uses_llm_pipeline():
    class _FakeLLMClient:
        def __init__(self):
            self.calls = 0

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return (
                    '[{"title":"第一章：方法论","objective":"理解方法论骨架"},'
                    '{"title":"第二章：策略执行","objective":"把框架转成动作"}]'
                )
            return "本章采用“结论-证据-行动”三段结构展开。"

    sections = [
        {"name": "方法论与认知", "summary": "先判断框架，再确定动作。"},
        {"name": "股票与交易", "summary": "强调仓位纪律与风险约束。"},
    ]
    pipeline = BookPipeline(llm_client=_FakeLLMClient())
    chapter_markdowns = build_chapter_markdowns(
        pipeline=pipeline,
        sections=sections,
    )

    assert len(chapter_markdowns) == 2
    assert chapter_markdowns[0].title == "第一章：方法论"
    assert "结论-证据-行动" in chapter_markdowns[0].body


def test_book_pipeline_raises_chapter_plan_failed_on_invalid_json():
    class _BadPlanLLM:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return "not-json"

    pipeline = BookPipeline(llm_client=_BadPlanLLM())
    sections = [{"name": "方法论", "summary": "摘要"}]

    try:
        build_chapter_markdowns(pipeline=pipeline, sections=sections)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert str(error) == "chapter_plan_failed"


def test_book_pipeline_raises_chapter_draft_failed_on_empty_text():
    class _BadDraftLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return '[{"title":"第一章：方法论","objective":"理解方法论骨架"}]'
            return "   "

    pipeline = BookPipeline(llm_client=_BadDraftLLM())
    sections = [{"name": "方法论", "summary": "摘要"}]

    try:
        build_chapter_markdowns(pipeline=pipeline, sections=sections)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert str(error) == "chapter_draft_failed"


def test_book_pipeline_rejects_empty_plan_item_title():
    class _BadPlanItemLLM:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return '[{"title":"", "objective":"理解方法论骨架"}]'

    pipeline = BookPipeline(llm_client=_BadPlanItemLLM())
    sections = [{"name": "方法论", "summary": "摘要"}]

    try:
        build_chapter_markdowns(pipeline=pipeline, sections=sections)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert str(error) == "chapter_plan_failed"
