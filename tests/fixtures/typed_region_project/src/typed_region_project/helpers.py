"""Imported typed helpers used by the method-region fixture."""

from __future__ import annotations


def twice(value: int) -> int:
    """Double one integer without changing Python integer semantics."""
    return value * 2


class Payload:
    """Nominal value used to exercise constant-time specialization guards."""

    value: int

    def __init__(self, value: int) -> None:
        """Store one typed payload value."""
        self.value = value
