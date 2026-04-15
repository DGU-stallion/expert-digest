"""Command line entry point for ExpertDigest."""

from expert_digest import __version__


def main() -> None:
    """Print a minimal startup message."""
    print(f"ExpertDigest {__version__}")
