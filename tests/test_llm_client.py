from __future__ import annotations

import io
from urllib.error import HTTPError

from expert_digest.generation.llm_client import (
    AnthropicCompatibleClient,
    _post_json,
)


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
                {"type": "text", "text": "LLM topic summary"},
            ]
        }

    monkeypatch.setattr(
        "expert_digest.generation.llm_client._post_json",
        _fake_post_json,
    )
    result = client.generate(system_prompt="sys", user_prompt="user")
    assert result == "LLM topic summary"


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
