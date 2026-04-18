import builtins
import sys
import types
from pathlib import Path

import pytest

from expert_digest.mcp import server


def test_create_fastmcp_registers_expected_tools(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeToolkit:
        def ask_author(self, **kwargs):
            calls.append(("ask_author", kwargs))
            return {"tool": "ask_author"}

        def search_posts(self, **kwargs):
            calls.append(("search_posts", kwargs))
            return {"tool": "search_posts"}

        def recommend_readings(self, **kwargs):
            calls.append(("recommend_readings", kwargs))
            return {"tool": "recommend_readings"}

        def list_topics(self, **kwargs):
            calls.append(("list_topics", kwargs))
            return {"tool": "list_topics"}

        def generate_handbook(self, **kwargs):
            calls.append(("generate_handbook", kwargs))
            return {"tool": "generate_handbook"}

        def generate_skill(self, **kwargs):
            calls.append(("generate_skill", kwargs))
            return {"tool": "generate_skill"}

    class _FakeMCP:
        def __init__(self, name: str):
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self):
            def _decorator(fn):
                self.tools[fn.__name__] = fn
                return fn

            return _decorator

        def run(self, *, transport: str):
            return transport

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = _FakeMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)

    mcp = server._create_fastmcp(toolkit=_FakeToolkit())

    assert mcp.name == "expert-digest"
    assert set(mcp.tools.keys()) == {
        "ask_author",
        "search_posts",
        "recommend_readings",
        "list_topics",
        "generate_handbook",
        "generate_skill",
    }

    assert mcp.tools["ask_author"]("问题", author_id="A") == {"tool": "ask_author"}
    assert mcp.tools["search_posts"]("查询", top_k=2) == {"tool": "search_posts"}
    assert mcp.tools["recommend_readings"]("问", top_k=2) == {
        "tool": "recommend_readings"
    }
    assert mcp.tools["list_topics"]("A", num_topics=2) == {"tool": "list_topics"}
    assert mcp.tools["generate_handbook"]("A") == {"tool": "generate_handbook"}
    assert mcp.tools["generate_skill"]("A") == {"tool": "generate_skill"}

    assert calls


def test_create_fastmcp_raises_runtime_error_when_dependency_missing(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mcp.server.fastmcp":
            raise ImportError("forced-missing-mcp")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class _DummyToolkit:
        pass

    with pytest.raises(RuntimeError, match="MCP dependency is missing"):
        server._create_fastmcp(toolkit=_DummyToolkit())


def test_run_mcp_server_passes_toolkit_and_transport(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeMCP:
        def run(self, *, transport: str):
            captured["transport"] = transport

    def _fake_create_fastmcp(*, toolkit):
        captured["toolkit"] = toolkit
        return _FakeMCP()

    monkeypatch.setattr(
        "expert_digest.mcp.server._create_fastmcp",
        _fake_create_fastmcp,
    )

    server.run_mcp_server(
        db_path=Path("data/processed/mock.sqlite3"),
        model="hash-bow-v1",
        output_dir=Path("data/outputs"),
        transport="sse",
    )

    toolkit = captured["toolkit"]
    assert toolkit.db_path == Path("data/processed/mock.sqlite3")
    assert toolkit.model == "hash-bow-v1"
    assert toolkit.output_dir == Path("data/outputs")
    assert captured["transport"] == "sse"
