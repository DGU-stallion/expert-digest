from pathlib import Path

from expert_digest.cli import main


def test_cli_run_mcp_server_invokes_runner(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_mcp_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("expert_digest.cli.run_mcp_server", _fake_run_mcp_server)

    exit_code = main(
        [
            "run-mcp-server",
            "--db",
            "data/processed/mcp.sqlite3",
            "--model",
            "hash-bow-v1",
            "--output-dir",
            "data/outputs",
            "--transport",
            "stdio",
        ]
    )

    assert exit_code == 0
    assert captured["db_path"] == Path("data/processed/mcp.sqlite3")
    assert captured["model"] == "hash-bow-v1"
    assert captured["output_dir"] == Path("data/outputs")
    assert captured["transport"] == "stdio"


def test_cli_run_mcp_server_returns_error_on_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.run_mcp_server",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("dependency missing")),
    )

    exit_code = main(["run-mcp-server"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to start MCP server" in output
