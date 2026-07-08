"""Fixture module with clean functions and one dynamic residue function."""

from __future__ import annotations

from app.types import Event, Score, User

DEFAULT_WEIGHT = 1.5


def normalize_features(xs: list[float]) -> list[float]:
    """Normalize feature values by their total."""
    total = sum(xs)
    if total == 0:
        return xs
    return [x / total for x in xs]


def score_user(user: User, events: list[Event]) -> Score:
    """Score one user from activity and event count."""
    features = normalize_features([float(len(events)), user.activity])
    return Score(value=sum(features) * DEFAULT_WEIGHT)


def rank_candidates(users: list[User], events: list[Event]) -> list[Score]:
    """Rank candidate users by score."""
    return [score_user(user, events) for user in users]


def debug_dump(obj: object) -> object:
    """Dynamic residue that must not poison the whole module."""
    return getattr(obj, input("field: "))
