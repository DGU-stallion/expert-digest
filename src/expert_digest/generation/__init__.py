"""Learning handbook generation primitives."""

from expert_digest.generation.handbook_writer import (
    DeterministicThemeSynthesizer,
    HybridThemeSynthesizer,
    build_handbook,
    write_handbook,
)
from expert_digest.generation.llm_client import (
    DEFAULT_CCSWITCH_DB_PATH,
    AnthropicCompatibleClient,
    create_default_handbook_llm_client,
)

__all__ = [
    "DeterministicThemeSynthesizer",
    "HybridThemeSynthesizer",
    "AnthropicCompatibleClient",
    "DEFAULT_CCSWITCH_DB_PATH",
    "create_default_handbook_llm_client",
    "build_handbook",
    "write_handbook",
]
