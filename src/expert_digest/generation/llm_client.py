"""LLM client adapter for Anthropic-compatible API (DeepSeek)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_LLM_PROVIDER_DB_PATH = Path.home() / ".cc-switch" / "cc-switch.db"
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
        except (URLError, TimeoutError) as error:
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
