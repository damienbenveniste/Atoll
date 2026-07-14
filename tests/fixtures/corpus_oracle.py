"""Canonical installed-wheel oracle for the local corpus lifecycle fixture."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Protocol, cast


class _Score(Protocol):
    value: float


class _UserFactory(Protocol):
    def __call__(self, *, activity: float) -> object: ...


class _EventFactory(Protocol):
    def __call__(self, *, name: str) -> object: ...


class _RankCandidates(Protocol):
    def __call__(self, users: list[object], events: list[object]) -> list[_Score]: ...


def main() -> int:
    """Emit deterministic behavior and import-path evidence as one JSON object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--corrupt-compiled", action="store_true")
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    if args.fail:
        raise RuntimeError("fixture baseline oracle failure")
    app = importlib.import_module("app")
    ranking = importlib.import_module("app.ranking")
    types = importlib.import_module("app.types")
    user = cast(_UserFactory, types.User)
    event = cast(_EventFactory, types.Event)
    rank_candidates = cast(_RankCandidates, ranking.rank_candidates)
    scores = rank_candidates(
        [user(activity=2.0), user(activity=4.0)],
        [event(name="view"), event(name="click")],
    )
    values = [score.value for score in scores]
    source = Path(cast(str, ranking.__file__)).read_text(encoding="utf-8")
    if args.corrupt_compiled and "# BEGIN ATOLL TYPED REGIONS" in source:
        values.append(-1.0)
    payload = {
        "canonical": {"scores": values},
        "imports": [
            str(Path(cast(str, app.__file__)).resolve()),
            str(Path(cast(str, ranking.__file__)).resolve()),
        ],
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
