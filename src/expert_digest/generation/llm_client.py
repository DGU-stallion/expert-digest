"""LLM client adapters for handbook generation."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_CCSWITCH_DB_PATH = Path.home() / ".cc-switch" / "cc-switch.db"


@dataclass(frozen=True)
class AnthropicCompatibleClient:
    """Minimal Anthropic-compatible client for text generation."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 30
    max_output_tokens: int = 700

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/v1/messages"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        response = _post_json(
            url=url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
        )
        text = _extract_text_content(response)
        if not text:
            raise ValueError("empty llm text response")
        return text


def load_ccswitch_provider_env(
    *,
    db_path: str | Path = DEFAULT_CCSWITCH_DB_PATH,
    app_type: str = "claude",
    provider_name: str | None = None,
) -> dict[str, str] | None:
    """Load env-like config mapping from cc-switch provider records."""
    path = Path(db_path)
    if not path.exists():
        return None

    query = (
        "SELECT settings_config "
        "FROM providers "
        "WHERE app_type = ? "
        + ("AND name = ? " if provider_name else "")
        + "ORDER BY is_current DESC, name "
        "LIMIT 1"
    )
    params: tuple[str, ...] = (
        (app_type, provider_name) if provider_name else (app_type,)
    )

    with sqlite3.connect(path) as connection:
        row = connection.execute(query, params).fetchone()
    if not row or not row[0]:
        return None

    try:
        config = json.loads(row[0])
    except json.JSONDecodeError:
        return None

    if not isinstance(config, dict):
        return None
    env = config.get("env")
    if not isinstance(env, dict):
        return None

    result: dict[str, str] = {}
    for key, value in env.items():
        if isinstance(key, str) and isinstance(value, str):
            result[key] = value
    return result or None


def create_anthropic_client_from_mapping(
    mapping: Mapping[str, str],
    *,
    timeout_seconds: int = 30,
    max_output_tokens: int = 700,
) -> AnthropicCompatibleClient | None:
    base_url = (
        mapping.get("ANTHROPIC_BASE_URL")
        or mapping.get("ANTHROPIC_API_BASE")
        or mapping.get("BASE_URL")
    )
    api_key = (
        mapping.get("ANTHROPIC_AUTH_TOKEN")
        or mapping.get("ANTHROPIC_API_KEY")
        or mapping.get("API_KEY")
    )
    model = (
        mapping.get("ANTHROPIC_MODEL")
        or mapping.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
        or mapping.get("MODEL")
    )
    if not base_url or not api_key or not model:
        return None

    return AnthropicCompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )


def create_default_handbook_llm_client(
    *,
    ccswitch_db_path: str | Path = DEFAULT_CCSWITCH_DB_PATH,
    timeout_seconds: int = 30,
    max_output_tokens: int = 700,
) -> AnthropicCompatibleClient | None:
    """Create default client, preferring cc-switch provider configuration."""
    ccswitch_env = load_ccswitch_provider_env(
        db_path=ccswitch_db_path,
        app_type="claude",
        provider_name="Nvidia",
    )
    if ccswitch_env:
        client = create_anthropic_client_from_mapping(
            ccswitch_env,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if client:
            return client

    env_client = create_anthropic_client_from_mapping(
        os.environ,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    return env_client


def _extract_text_content(response: dict[str, object]) -> str:
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _post_json(
    *,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"http_error {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"network_error: {error}") from error
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("invalid_llm_response_type")
    return parsed
