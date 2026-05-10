"""LLM client helpers for pipeline nodes.

Two provider tiers:
- fast: DeepSeek v4-flash for straightforward tasks (analysis, formatting)
- reasoning: DeepSeek v4-pro for complex reasoning (synthesis, editing, writing)

Configured via environment variables:
  PIPELINE_FAST_BASE_URL / PIPELINE_FAST_API_KEY / PIPELINE_FAST_MODEL
  PIPELINE_REASONING_BASE_URL / PIPELINE_REASONING_API_KEY / PIPELINE_REASONING_MODEL
"""

from __future__ import annotations

import os

from expert_digest.generation.llm_client import AnthropicCompatibleClient

_PREFIX_FAST = "PIPELINE_FAST_"
_PREFIX_REASONING = "PIPELINE_REASONING_"
_DEFAULT_TIMEOUT = 120
_DEFAULT_FAST_MAX_TOKENS = 8192
_DEFAULT_REASONING_MAX_TOKENS = 16384


def _create_client(prefix: str, max_output_tokens: int) -> AnthropicCompatibleClient | None:
    base_url = os.environ.get(f"{prefix}BASE_URL")
    api_key = os.environ.get(f"{prefix}API_KEY")
    model = os.environ.get(f"{prefix}MODEL")
    if not all([base_url, api_key, model]):
        return None
    return AnthropicCompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=_DEFAULT_TIMEOUT,
        max_output_tokens=max_output_tokens,
    )


def create_fast_client() -> AnthropicCompatibleClient | None:
    """Create the fast LLM client (DeepSeek v4-flash)."""
    return _create_client(_PREFIX_FAST, _DEFAULT_FAST_MAX_TOKENS)


def create_reasoning_client() -> AnthropicCompatibleClient | None:
    """Create the reasoning LLM client (DeepSeek v4-pro)."""
    return _create_client(_PREFIX_REASONING, _DEFAULT_REASONING_MAX_TOKENS)


def require_fast_client() -> AnthropicCompatibleClient:
    client = create_fast_client()
    if client is None:
        raise RuntimeError(
            "fast llm client not configured: "
            "set PIPELINE_FAST_BASE_URL, PIPELINE_FAST_API_KEY, PIPELINE_FAST_MODEL"
        )
    return client


def require_reasoning_client() -> AnthropicCompatibleClient:
    client = create_reasoning_client()
    if client is None:
        raise RuntimeError(
            "reasoning llm client not configured: "
            "set PIPELINE_REASONING_BASE_URL, "
            "PIPELINE_REASONING_API_KEY, PIPELINE_REASONING_MODEL"
        )
    return client
