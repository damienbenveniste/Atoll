"""Small typed records used by the ranking fixture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class User:
    """Fixture user record."""

    activity: float


@dataclass(frozen=True, slots=True)
class Event:
    """Fixture event record."""

    name: str


@dataclass(frozen=True, slots=True)
class Score:
    """Fixture score record."""

    value: float
