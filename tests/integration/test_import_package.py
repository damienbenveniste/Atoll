"""Import smoke tests for atoll."""

import atoll


def test_package_exports_greet() -> None:
    """The generated package exposes its public greeting helper."""
    assert atoll.greet("Agent") == "Hello, Agent!"
