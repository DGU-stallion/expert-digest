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


def test_cli_generate_skill_draft_writes_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: _fake_profile_dict(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.build_skill_markdown_from_profile",
        lambda profile: "# SKILL: 黄彦臻风格助理\n",
    )

    exit_code = main(
        [
            "generate-skill-draft",
            "--output",
            "data/outputs/test_skill.md",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Generated skill draft" in output


def test_cli_generate_skill_draft_returns_error_when_profile_missing(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("no documents available")),
    )

    exit_code = main(["generate-skill-draft"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to generate skill draft" in output


def test_cli_generate_skill_draft_fails_quality_gate(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.evaluate_wiki",
        lambda **_kwargs: type(
            "Report",
            (),
            {"traceability_ratio": 1.0, "coverage_ratio": 1.0},
        )(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.lint_wiki",
        lambda **_kwargs: type("Lint", (), {"issue_count": 50})(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.build_author_profile",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not build profile")
        ),
    )

    exit_code = main(
        [
            "generate-skill-draft",
            "--wiki-root-for-quality",
            "data/wiki/huang_pass1b",
            "--expected-source-count-for-quality",
            "824",
            "--max-lint-issues-for-quality",
            "20",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed quality gate" in output
