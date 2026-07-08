"""Unit tests for atoll core behavior."""

from atoll import greet


def test_greet_uses_name() -> None:
    """The greeting includes the provided name."""
    assert greet("Codex") == "Hello, Codex!"


def test_greet_falls_back_for_blank_name() -> None:
    """Blank names fall back to a stable default."""
    assert greet("   ") == "Hello, World!"
