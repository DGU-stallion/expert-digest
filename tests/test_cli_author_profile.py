import json
from pathlib import Path

from expert_digest.cli import main


def _fake_profile_dict() -> dict[str, object]:
    return {
        "author": "黄彦臻",
        "document_count": 2,
        "source_document_ids": ["doc-1", "doc-2"],
        "focus_topics": ["供给需求", "风险控制"],
        "keywords": [
            {"keyword": "风险", "count": 3},
            {"keyword": "供给", "count": 2},
        ],
        "reasoning_patterns": [
            {"pattern": "因为...所以...", "count": 2},
            {"pattern": "如果...那么...", "count": 1},
        ],
    }


def test_cli_build_author_profile_supports_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: _fake_profile_dict(),
    )

    exit_code = main(["build-author-profile", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["author"] == "黄彦臻"
    assert payload["keywords"][0]["keyword"] == "风险"


def test_cli_build_author_profile_can_save_output(monkeypatch, capsys):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: _fake_profile_dict(),
    )

    def _fake_save_run_metadata(*, payload, output_path):
        captured["payload"] = payload
        captured["output_path"] = output_path

    monkeypatch.setattr("expert_digest.cli._save_run_metadata", _fake_save_run_metadata)

    exit_code = main(
        [
            "build-author-profile",
            "--format",
            "json",
            "--output",
            "data/outputs/author_profile.json",
        ]
    )
    _ = capsys.readouterr().out

    assert exit_code == 0
    assert captured["output_path"] == Path("data/outputs/author_profile.json")
    assert captured["payload"]["document_count"] == 2


def test_cli_build_author_profile_returns_error_on_empty_documents(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("no documents available")),
    )

    exit_code = main(["build-author-profile"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to build author profile" in output


def test_cli_generate_skill_draft_writes_output(monkeypatch, capsys, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"skill_markdown": "# SKILL: test\n", "documents": [{"id": "1"}]}

    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: _FakePipeline(),
    )

    output_path = tmp_path / "test_skill.md"
    exit_code = main(
        [
            "generate-skill-draft",
            "--output",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "SKILL" in output


def test_cli_generate_skill_draft_returns_error_when_profile_missing(
    monkeypatch, capsys
):
    class _BrokenPipeline:
        def invoke(self, state):
            raise RuntimeError("no documents available")

    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: _BrokenPipeline(),
    )

    exit_code = main(["generate-skill-draft"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "no documents available" in output


def test_cli_generate_skill_draft_handles_runtime_error_from_llm(
    monkeypatch, capsys
):
    class _BrokenPipeline:
        def invoke(self, state):
            raise RuntimeError("http_error 429")

    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: _BrokenPipeline(),
    )

    exit_code = main(["generate-skill-draft"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "http_error 429" in output


def test_cli_generate_skill_draft_fails_quality_gate(monkeypatch, capsys):
    class _EmptyPipeline:
        def invoke(self, state):
            return {"skill_markdown": "", "documents": []}

    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: _EmptyPipeline(),
    )

    exit_code = main(["generate-skill-draft"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "empty" in output.lower()


def test_cli_generate_skill_draft_fails_without_gemini_flash(monkeypatch, capsys):
    class _BrokenPipeline:
        def invoke(self, state):
            raise RuntimeError("llm_client_unavailable")

    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: _BrokenPipeline(),
    )

    exit_code = main(["generate-skill-draft"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "llm_client_unavailable" in output
