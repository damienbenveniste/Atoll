"""Small public helpers retained for package import smoke tests."""


def greet(name: str = "World") -> str:
    """Return a deterministic greeting with whitespace-only names normalized.

    This helper is intentionally simple; it gives generated-project smoke tests a
    stable public function without pulling in the CLI or analysis stack.
    """
    clean_name = name.strip() or "World"
    return f"Hello, {clean_name}!"
