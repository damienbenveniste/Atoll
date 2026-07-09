"""Imported typed helpers used by the method-region fixture."""


def twice(value: int) -> int:
    """Double one integer without changing Python integer semantics."""
    return value * 2
