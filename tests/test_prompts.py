from expert_digest.generation.prompts import build_theme_summary_prompts
from expert_digest.retrieval.retriever import RetrievedChunk


def _sample_evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            document_id="doc-1",
            title="标题A",
            author="作者A",
            text="这是证据正文。",
            url="https://example.com/a",
        )
    ]


def test_build_theme_summary_prompts_uses_defaults_when_config_missing(
    monkeypatch, tmp_path
):
    missing_path = tmp_path / "missing-prompts.yaml"
    monkeypatch.setattr(
        "expert_digest.generation.prompts.DEFAULT_PROMPTS_PATH",
        missing_path,
    )

    system_prompt, user_prompt = build_theme_summary_prompts(
        theme_name="核心能力",
        question="作者的核心能力是什么？",
        evidence_chunks=_sample_evidence(),
    )

    assert "知识蒸馏助手" in system_prompt
    assert "请输出 6-10 句中文总结" in user_prompt


def test_build_theme_summary_prompts_reads_simple_yaml_override(monkeypatch, tmp_path):
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(
        "\n".join(
            [
                "theme_summary:",
                '  system_prompt: "系统提示-覆盖版"',
                '  output_instruction: "输出要求-覆盖版"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "expert_digest.generation.prompts.DEFAULT_PROMPTS_PATH",
        prompts_path,
    )

    system_prompt, user_prompt = build_theme_summary_prompts(
        theme_name="核心能力",
        question="作者的核心能力是什么？",
        evidence_chunks=_sample_evidence(),
    )

    assert system_prompt == "系统提示-覆盖版"
    assert user_prompt.endswith("输出要求-覆盖版")
