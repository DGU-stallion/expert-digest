from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from urllib.error import HTTPError

from expert_digest.generation.llm_client import (
    AnthropicCompatibleClient,
    GeminiCompatibleClient,
    OpenAICompatibleClient,
    _post_json,
    create_anthropic_client_from_mapping,
    create_default_handbook_llm_client,
    create_gemini_client_from_mapping,
    create_openai_client_from_mapping,
    load_ccswitch_provider_env,
)


def test_load_ccswitch_provider_env_reads_current_provider_config(monkeypatch):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE providers (
            id TEXT PRIMARY KEY,
            app_type TEXT,
            name TEXT,
            settings_config TEXT,
            is_current BOOLEAN
        )
        """
    )
    connection.execute(
        """
        INSERT INTO providers (id, app_type, name, settings_config, is_current)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "provider-1",
            "gemini",
            "Google Official",
            json.dumps(
                {
                    "env": {
                        "GEMINI_API_KEY": "test-key",
                        "GEMINI_MODEL": "gemini-2.5-flash",
                    }
                },
                ensure_ascii=False,
            ),
            1,
        ),
    )
    connection.commit()
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.sqlite3.connect",
        lambda _path: connection,
    )

    env = load_ccswitch_provider_env(
        db_path=Path(__file__),
        app_type="gemini",
        provider_name="Google Official",
    )
    assert env is not None
    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["GEMINI_MODEL"] == "gemini-2.5-flash"


def test_create_anthropic_client_from_mapping_returns_none_on_missing_token():
    client = create_anthropic_client_from_mapping(
        {
            "ANTHROPIC_BASE_URL": "https://example.com",
            "ANTHROPIC_MODEL": "model-a",
        }
    )
    assert client is None


def test_create_gemini_client_from_mapping_uses_default_model():
    client = create_gemini_client_from_mapping({"GEMINI_API_KEY": "key-123"})
    assert client is not None
    assert client.model == "gemini-2.5-flash"


def test_anthropic_compatible_client_generates_text(monkeypatch):
    client = AnthropicCompatibleClient(
        base_url="https://example.com",
        api_key="test-token",
        model="model-a",
    )

    def _fake_post_json(*, url, payload, headers, timeout_seconds):
        assert url.endswith("/v1/messages")
        assert payload["model"] == "model-a"
        assert "x-api-key" in headers
        return {
            "content": [
                {"type": "text", "text": "LLM主题总结"},
            ]
        }

    monkeypatch.setattr(
        "expert_digest.generation.llm_client._post_json",
        _fake_post_json,
    )
    result = client.generate(system_prompt="sys", user_prompt="user")
    assert result == "LLM主题总结"


def test_gemini_compatible_client_generates_text(monkeypatch):
    client = GeminiCompatibleClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-key",
        model="gemini-2.5-flash",
    )

    def _fake_post_json(*, url, payload, headers, timeout_seconds):
        assert "models/gemini-2.5-flash:generateContent" in url
        assert "key=gemini-key" in url
        assert payload["contents"][0]["parts"][0]["text"] == "user"
        assert payload["system_instruction"]["parts"][0]["text"] == "sys"
        assert headers["Content-Type"] == "application/json"
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini主题总结"}],
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "expert_digest.generation.llm_client._post_json",
        _fake_post_json,
    )
    result = client.generate(system_prompt="sys", user_prompt="user")
    assert result == "Gemini主题总结"


def test_create_openai_client_from_mapping_reads_openai_keys():
    client = create_openai_client_from_mapping(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "cherry-key",
            "OPENAI_MODEL": "gemini-2.5-flash",
        }
    )
    assert client is not None
    assert client.base_url == "http://127.0.0.1:8000/v1"
    assert client.api_key == "cherry-key"
    assert client.model == "gemini-2.5-flash"


def test_openai_compatible_client_generates_text(monkeypatch):
    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:8000/v1",
        api_key="cherry-key",
        model="gemini-2.5-flash",
    )

    def _fake_post_json(*, url, payload, headers, timeout_seconds):
        assert url.endswith("/v1/chat/completions")
        assert payload["model"] == "gemini-2.5-flash"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert headers["Authorization"] == "Bearer cherry-key"
        return {
            "choices": [
                {
                    "message": {
                        "content": "OpenAI兼容Gemini主题总结",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "expert_digest.generation.llm_client._post_json",
        _fake_post_json,
    )
    result = client.generate(system_prompt="sys", user_prompt="user")
    assert result == "OpenAI兼容Gemini主题总结"


def test_create_default_handbook_llm_client_prefers_ccswitch_google(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def _fake_load_ccswitch_provider_env(*, db_path, app_type, provider_name):
        calls.append((app_type, provider_name))
        if app_type == "gemini" and provider_name == "Google Official":
            return {
                "GEMINI_API_KEY": "token-from-google",
                "GEMINI_MODEL": "gemini-2.5-flash",
            }
        return None

    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        _fake_load_ccswitch_provider_env,
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [],
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://integrate.api.nvidia.com")
    monkeypatch.setenv("ANTHROPIC_MODEL", "minimaxai/minimax-m2.7")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-from-nvidia")

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert isinstance(client, GeminiCompatibleClient)
    assert client.api_key == "token-from-google"
    assert client.provider == "google"
    assert all(provider_name != "Nvidia" for _app_type, provider_name in calls)


def test_create_default_handbook_llm_client_ignores_nvidia_env(monkeypatch):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [],
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://integrate.api.nvidia.com")
    monkeypatch.setenv("ANTHROPIC_MODEL", "minimaxai/minimax-m2.7")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-from-nvidia")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert client is None


def test_create_default_handbook_llm_client_supports_cherry_openai_mapping(
    monkeypatch,
):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "cherry-key",
            "OPENAI_MODEL": "gemini-2.5-flash",
        },
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [],
    )

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "gemini-2.5-flash"


def test_create_default_handbook_llm_client_scans_non_current_provider_envs(
    monkeypatch,
):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [
            (
                "claude",
                "Nvidia",
                {
                    "ANTHROPIC_BASE_URL": "https://integrate.api.nvidia.com",
                    "ANTHROPIC_AUTH_TOKEN": "token-from-nvidia",
                    "ANTHROPIC_MODEL": "minimaxai/minimax-m2.7",
                },
            ),
            (
                "claude",
                "Cherry Studio",
                {
                    "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                    "OPENAI_API_KEY": "cherry-key",
                    "OPENAI_MODEL": "gemini-2.5-flash",
                },
            ),
        ],
    )

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "Cherry Studio"


def test_create_default_handbook_llm_client_prefers_env_openai_over_scan(
    monkeypatch,
):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [
            (
                "claude",
                "Zhipu GLM",
                {
                    "ANTHROPIC_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
                    "ANTHROPIC_AUTH_TOKEN": "zhipu-token",
                    "ANTHROPIC_MODEL": "glm-4.7-flash",
                },
            )
        ],
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "cherry-key")
    monkeypatch.setenv("OPENAI_MODEL", "gemini-2.5-flash")

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "env"


def test_create_default_handbook_llm_client_prefers_env_gemini_over_ccswitch_openai(
    monkeypatch,
):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "cherry-key",
            "OPENAI_MODEL": "gemini-2.5-flash",
        },
    )
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_all_ccswitch_provider_envs",
        lambda **_kwargs: [],
    )
    monkeypatch.setenv("GEMINI_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert isinstance(client, GeminiCompatibleClient)
    assert client.provider == "env"


def test_post_json_retries_on_503_then_succeeds(monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    calls = {"count": 0}

    def _fake_urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                url="https://example.com",
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=io.BytesIO(
                    b'{"error":{"code":503,"message":"Please retry in 1s."}}'
                ),
            )
        return _FakeResponse()

    monkeypatch.setattr("expert_digest.generation.llm_client.urlopen", _fake_urlopen)
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.time.sleep",
        lambda *_a: None,
    )

    payload = _post_json(
        url="https://example.com",
        payload={"x": 1},
        headers={"Content-Type": "application/json"},
        timeout_seconds=3,
    )

    assert payload == {"ok": True}
    assert calls["count"] == 2


def test_post_json_does_not_retry_on_400(monkeypatch):
    def _fake_urlopen(_request, timeout):
        raise HTTPError(
            url="https://example.com",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"code":400,"message":"invalid"}}'),
        )

    monkeypatch.setattr("expert_digest.generation.llm_client.urlopen", _fake_urlopen)
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.time.sleep",
        lambda *_a: None,
    )

    try:
        _post_json(
            url="https://example.com",
            payload={"x": 1},
            headers={"Content-Type": "application/json"},
            timeout_seconds=3,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as error:
        assert "http_error 400" in str(error)
