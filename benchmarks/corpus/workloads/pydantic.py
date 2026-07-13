"""Deterministic Pydantic model-validation workload."""

from __future__ import annotations

from types import ModuleType

import pydantic
from pydantic import BaseModel, ConfigDict, Field

_BATCH = 20_000


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Validate typed records and summarize stable field values."""

    class Record(BaseModel):
        model_config = ConfigDict(extra="forbid")

        identifier: int = Field(ge=0)
        label: str = Field(min_length=3)
        weights: tuple[int, int, int]
        active: bool

    checksum = 0
    total = _BATCH * repetitions
    for index in range(total):
        value = seed + index
        record = Record.model_validate(
            {
                "identifier": value,
                "label": f"item-{value % 97:02d}",
                "weights": (value % 11, value % 13, value % 17),
                "active": value % 2 == 0,
            }
        )
        checksum += record.identifier + sum(record.weights) + int(record.active)
    return {"checksum": checksum, "validated": total}, (pydantic,)
