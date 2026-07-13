"""Deterministic SQLAlchemy declarative-construction workload."""

from __future__ import annotations

from types import ModuleType

import sqlalchemy
from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_BATCH = 30_000


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Construct mapped declarative instances without database I/O."""

    class Base(DeclarativeBase):
        pass

    class Record(Base):
        __tablename__ = "corpus_record"

        identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
        label: Mapped[str] = mapped_column(String(32))
        score: Mapped[int] = mapped_column(Integer)
        active: Mapped[bool] = mapped_column(Boolean)

    constructions = _BATCH * repetitions
    checksum = 0
    for index in range(constructions):
        value = seed + index
        record = Record(
            identifier=value,
            label=f"record-{value % 101}",
            score=value % 997,
            active=value % 3 == 0,
        )
        checksum += record.identifier + record.score + len(record.label) + int(record.active)
    return {
        "checksum": checksum,
        "columns": len(Record.__table__.columns),
        "constructions": constructions,
    }, (sqlalchemy,)
