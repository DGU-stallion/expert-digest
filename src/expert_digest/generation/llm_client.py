"""LLM client adapters for handbook generation."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

DEFAULT_LLM_PROVIDER_DB_PATH = Path.home() / ".cc-switch" / "cc-switch.db"
# Backward-compatible alias used by existing callers/tests.
DEFAULT_CCSWITCH_DB_PATH = DEFAULT_LLM_PROVIDER_DB_PATH
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_MAX_HTTP_RETRY = 4
_RETRYABLE_HTTP_CODES = {429, 503}
_RETRY_SECONDS_RE = re.compile(
    r'"retryDelay"\s*:\s*"(?P<delay>\d+)s"|retry in (?P<retry>[0-9]+(?:\.[0-9]+)?)s',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnthropicCompatibleClient:
    """Minimal Anthropic-compatible client for text generation."""

    base_url: str
    api_key: str
    model: str
    provider: str | None = None
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


@dataclass(frozen=True)
class GeminiCompatibleClient:
    """Minimal Gemini REST client for text generation."""

    base_url: str
    api_key: str
    model: str = DEFAULT_GEMINI_MODEL
    provider: str | None = None
    timeout_seconds: int = 30
    max_output_tokens: int = 700

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        model = self.model.removeprefix("models/")
        base = self.base_url.rstrip("/")
        url = (
            f"{base}/models/{model}:generateContent"
            f"?key={quote_plus(self.api_key)}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        response = _post_json(
            url=url,
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout_seconds=self.timeout_seconds,
        )
        text = _extract_gemini_text_content(response)
        if not text:
            raise ValueError("empty llm text response")
        return text


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible client for text generation."""

    base_url: str
    api_key: str
    model: str
    provider: str | None = None
    timeout_seconds: int = 30
    max_output_tokens: int = 700

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = _post_json(
            url=url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
        )
        text = _extract_openai_text_content(response)
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


def load_all_ccswitch_provider_envs(
    *,
    db_path: str | Path = DEFAULT_CCSWITCH_DB_PATH,
    app_types: tuple[str, ...] = ("gemini", "claude", "codex"),
) -> list[tuple[str, str, dict[str, str]]]:
    """Load provider env mappings ordered by app type and current flag."""
    path = Path(db_path)
    if not path.exists():
        return []

    placeholders = ",".join("?" for _ in app_types)
    query = (
        "SELECT app_type, name, settings_config "
        "FROM providers "
        f"WHERE app_type IN ({placeholders}) "
        "ORDER BY CASE app_type "
        "WHEN 'gemini' THEN 0 "
        "WHEN 'claude' THEN 1 "
        "WHEN 'codex' THEN 2 "
        "ELSE 3 END, "
        "is_current DESC, name"
    )
    rows: list[tuple[str, str, str]] = []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(query, app_types).fetchall()

    results: list[tuple[str, str, dict[str, str]]] = []
    for app_type, provider_name, raw_config in rows:
        if not raw_config:
            continue
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError:
            continue
        if not isinstance(config, dict):
            continue
        env = config.get("env")
        if not isinstance(env, dict):
            continue
        normalized: dict[str, str] = {}
        for key, value in env.items():
            if isinstance(key, str) and isinstance(value, str):
                normalized[key] = value
        if normalized:
            results.append((str(app_type), str(provider_name), normalized))
    return results


def create_anthropic_client_from_mapping(
    mapping: Mapping[str, str],
    *,
    provider: str | None = None,
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
        provider=provider,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )


def create_gemini_client_from_mapping(
    mapping: Mapping[str, str],
    *,
    provider: str | None = None,
    timeout_seconds: int = 30,
    max_output_tokens: int = 700,
) -> GeminiCompatibleClient | None:
    api_key = (
        mapping.get("GEMINI_API_KEY")
        or mapping.get("GOOGLE_API_KEY")
        or mapping.get("API_KEY")
    )
    if not api_key:
        return None

    model = (
        mapping.get("GEMINI_MODEL")
        or mapping.get("GOOGLE_MODEL")
        or mapping.get("MODEL")
        or DEFAULT_GEMINI_MODEL
    )
    base_url = (
        mapping.get("GEMINI_BASE_URL")
        or mapping.get("GOOGLE_BASE_URL")
        or mapping.get("BASE_URL")
        or DEFAULT_GEMINI_BASE_URL
    )

    return GeminiCompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )


def create_openai_client_from_mapping(
    mapping: Mapping[str, str],
    *,
    provider: str | None = None,
    timeout_seconds: int = 30,
    max_output_tokens: int = 700,
) -> OpenAICompatibleClient | None:
    base_url = (
        mapping.get("OPENAI_BASE_URL")
        or mapping.get("OPENAI_API_BASE")
        or mapping.get("BASE_URL")
    )
    api_key = (
        mapping.get("OPENAI_API_KEY")
        or mapping.get("API_KEY")
    )
    model = (
        mapping.get("OPENAI_MODEL")
        or mapping.get("MODEL")
        or DEFAULT_GEMINI_MODEL
    )
    if not base_url or not api_key:
        return None
    if "openai" not in base_url.lower() and "v1" not in base_url.lower():
        return None

    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )


def create_default_handbook_llm_client(
    *,
    ccswitch_db_path: str | Path = DEFAULT_LLM_PROVIDER_DB_PATH,
    timeout_seconds: int = 30,
    max_output_tokens: int = 700,
) -> (
    AnthropicCompatibleClient
    | GeminiCompatibleClient
    | OpenAICompatibleClient
    | None
):
    """Create default client, preferring Google/Gemini provider configuration."""
    # Highest priority: explicit process environment.
    # This allows direct Google API usage (GEMINI_API_KEY) without relying on
    # Cherry Studio/OpenAI-compatible forwarding.
    env_gemini_client = create_gemini_client_from_mapping(
        os.environ,
        provider="env",
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    if env_gemini_client and not _looks_like_nvidia(
        env_gemini_client.base_url,
        env_gemini_client.model,
    ):
        return env_gemini_client

    env_openai_client = create_openai_client_from_mapping(
        os.environ,
        provider="env",
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    if env_openai_client and not _looks_like_nvidia(
        env_openai_client.base_url,
        env_openai_client.model,
    ):
        return env_openai_client

    env_anthropic_client = create_anthropic_client_from_mapping(
        os.environ,
        provider="env",
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    if env_anthropic_client and not _looks_like_nvidia(
        env_anthropic_client.base_url,
        env_anthropic_client.model,
    ):
        return env_anthropic_client

    candidates = [
        ("gemini", "Google Official", "google"),
        ("gemini", "Google", "google"),
        ("gemini", None, "google"),
        ("claude", "Google Official", "google"),
        ("claude", "Google", "google"),
        ("claude", "Gemini", "google"),
    ]
    for app_type, provider_name, provider_tag in candidates:
        provider_env = load_ccswitch_provider_env(
            db_path=ccswitch_db_path,
            app_type=app_type,
            provider_name=provider_name,
        )
        if not provider_env:
            continue
        openai_client = create_openai_client_from_mapping(
            provider_env,
            provider=provider_tag,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if openai_client and not _looks_like_nvidia(
            provider_tag,
            openai_client.base_url,
            openai_client.model,
        ):
            return openai_client
        gemini_client = create_gemini_client_from_mapping(
            provider_env,
            provider=provider_tag,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if gemini_client and not _looks_like_nvidia(
            provider_tag,
            gemini_client.base_url,
            gemini_client.model,
        ):
            return gemini_client
        anthropic_client = create_anthropic_client_from_mapping(
            provider_env,
            provider=provider_tag,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if anthropic_client and not _looks_like_nvidia(
            provider_tag,
            anthropic_client.base_url,
            anthropic_client.model,
        ):
            return anthropic_client

    for _app_type, provider_name, provider_env in load_all_ccswitch_provider_envs(
        db_path=ccswitch_db_path,
    ):
        openai_client = create_openai_client_from_mapping(
            provider_env,
            provider=provider_name,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if openai_client and not _looks_like_nvidia(
            provider_name,
            openai_client.base_url,
            openai_client.model,
        ):
            return openai_client

        gemini_client = create_gemini_client_from_mapping(
            provider_env,
            provider=provider_name,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if gemini_client and not _looks_like_nvidia(
            provider_name,
            gemini_client.base_url,
            gemini_client.model,
        ):
            return gemini_client

        anthropic_client = create_anthropic_client_from_mapping(
            provider_env,
            provider=provider_name,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        if anthropic_client and not _looks_like_nvidia(
            provider_name,
            anthropic_client.base_url,
            anthropic_client.model,
        ):
            return anthropic_client
    return None


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


def _extract_gemini_text_content(response: dict[str, object]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        candidate_parts = content.get("parts")
        if not isinstance(candidate_parts, list):
            continue
        for part in candidate_parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_openai_text_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts).strip()


def _looks_like_nvidia(*parts: str | None) -> bool:
    merged = " ".join(part for part in parts if isinstance(part, str)).lower()
    return "nvidia" in merged


def _post_json(
    *,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, _MAX_HTTP_RETRY + 1):
        request = Request(
            url,
            data=body_bytes,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="ignore")
            if (
                error.code in _RETRYABLE_HTTP_CODES
                and attempt < _MAX_HTTP_RETRY
            ):
                delay = _resolve_retry_delay_seconds(body=body, attempt=attempt)
                time.sleep(delay)
                continue
            raise RuntimeError(f"http_error {error.code}: {body}") from error
        except URLError as error:
            if attempt < _MAX_HTTP_RETRY:
                time.sleep(min(2**attempt, 10))
                continue
            raise RuntimeError(f"network_error: {error}") from error
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("invalid_llm_response_type")
        return parsed
    raise RuntimeError("llm_request_timeout")


def _resolve_retry_delay_seconds(*, body: str, attempt: int) -> float:
    match = _RETRY_SECONDS_RE.search(body)
    if match is not None:
        value = match.group("delay") or match.group("retry")
        try:
            parsed = float(value)
            return min(max(parsed, 1.0), 30.0)
        except (TypeError, ValueError):
            pass
    return float(min(2**attempt, 10))
