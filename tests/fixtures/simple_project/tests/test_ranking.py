"""Fixture tests for ranking behavior."""

from app.ranking import rank_candidates
from app.types import Event, User


def test_rank_candidates_returns_scores() -> None:
    """The fixture project has executable tests for later trial mode."""
    scores = rank_candidates([User(activity=2.0)], [Event(name="view")])
    assert len(scores) == 1
