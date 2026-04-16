from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from expert_digest.generation.llm_client import (
    AnthropicCompatibleClient,
    create_anthropic_client_from_mapping,
    create_default_handbook_llm_client,
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
            "claude",
            "Nvidia",
            json.dumps(
                {
                    "env": {
                        "ANTHROPIC_BASE_URL": "https://integrate.api.nvidia.com",
                        "ANTHROPIC_MODEL": "stepfun-ai/step-3.5-flash",
                        "ANTHROPIC_AUTH_TOKEN": "test-token",
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
        app_type="claude",
        provider_name="Nvidia",
    )
    assert env is not None
    assert env["ANTHROPIC_BASE_URL"] == "https://integrate.api.nvidia.com"
    assert env["ANTHROPIC_MODEL"] == "stepfun-ai/step-3.5-flash"


def test_create_anthropic_client_from_mapping_returns_none_on_missing_token():
    client = create_anthropic_client_from_mapping(
        {
            "ANTHROPIC_BASE_URL": "https://integrate.api.nvidia.com",
            "ANTHROPIC_MODEL": "stepfun-ai/step-3.5-flash",
        }
    )
    assert client is None


def test_anthropic_compatible_client_generates_text(monkeypatch):
    client = AnthropicCompatibleClient(
        base_url="https://integrate.api.nvidia.com",
        api_key="test-token",
        model="stepfun-ai/step-3.5-flash",
    )

    def _fake_post_json(*, url, payload, headers, timeout_seconds):
        assert url.endswith("/v1/messages")
        assert payload["model"] == "stepfun-ai/step-3.5-flash"
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


def test_create_default_handbook_llm_client_prefers_ccswitch(monkeypatch):
    monkeypatch.setattr(
        "expert_digest.generation.llm_client.load_ccswitch_provider_env",
        lambda **_kwargs: {
            "ANTHROPIC_BASE_URL": "https://integrate.api.nvidia.com",
            "ANTHROPIC_MODEL": "stepfun-ai/step-3.5-flash",
            "ANTHROPIC_AUTH_TOKEN": "token-from-ccswitch",
        },
    )
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    client = create_default_handbook_llm_client(
        ccswitch_db_path=Path("data/processed/does_not_matter.sqlite3")
    )
    assert client is not None
    assert client.api_key == "token-from-ccswitch"
