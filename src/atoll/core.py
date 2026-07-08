"""Core behavior for atoll."""


def greet(name: str = "World") -> str:
    """Return a friendly greeting for `name`."""
    clean_name = name.strip() or "World"
    return f"Hello, {clean_name}!"
