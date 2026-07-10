"""Generic function fixture with one statically closed internal call."""

from __future__ import annotations


def pair_value[T](value: T) -> tuple[T, T]:
    """Keep the public generic implementation available as guarded fallback."""
    return value, value


def pair_int(value: int) -> tuple[int, int]:
    """Provide a same-module call site that closes ``T`` to ``int``."""
    return pair_value(value)
