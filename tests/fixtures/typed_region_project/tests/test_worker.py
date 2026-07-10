"""Behavior tests shipped with the typed-region fixture."""

from __future__ import annotations

import asyncio
import inspect
import pickle
from collections.abc import Callable
from typing import cast

from typed_region_project.generic_functions import pair_int, pair_value
from typed_region_project.helpers import Payload
from typed_region_project.worker import (
    UNSAFE_IDENTITY_CLASS,
    UNSAFE_IDENTITY_INSTANCE,
    DynamicWorker,
    IntPairer,
    OptionalPairer,
    Pairer,
    PayloadPairer,
    ScaleModel,
    UnsafeIdentityWorker,
    Worker,
)

SCALE_RESULT = 23
PARSED_RESULT = 7
ADJUSTED_RESULT = 6
DYNAMIC_RESULT = 11
ASYNC_FIRST_RESULT = 4
ASYNC_SENT_RESULT = 8
ASYNC_THROWN_RESULT = 2
MODEL_RESULT = 18
MODEL_DEFAULT_FACTOR = 2
MODEL_CHILD_RESULT = 20
UNSAFE_SQUARE_RESULT = 16
_PICKLE_LOADS_NAME = "loads"


def test_worker_callable_shapes_and_results() -> None:
    """The fixture defines stable behavior for compiled and fallback routing."""
    worker = Worker(3)

    assert worker.scale(5) == SCALE_RESULT
    assert Worker.parse("7") == PARSED_RESULT
    assert Worker.adjust(4) == ADJUSTED_RESULT
    assert list(worker.values(3)) == [3, 3, 5]
    assert asyncio.run(worker.score(5)) == SCALE_RESULT
    asyncio.run(_assert_async_generator_protocol(worker))
    assert inspect.isgeneratorfunction(Worker.values)
    assert inspect.iscoroutinefunction(Worker.score)
    assert inspect.isasyncgenfunction(Worker.exchange)


def test_dynamic_worker_remains_interpreted() -> None:
    """Custom attribute behavior stays authoritative in the source class."""
    worker = DynamicWorker()

    assert worker.calculate(4) == DYNAMIC_RESULT


def test_atomic_class_metadata_behavior_and_pickle_round_trip() -> None:
    """The source fixture defines the class identity compiled wheels must preserve."""
    signature = inspect.signature(ScaleModel)
    model = ScaleModel("fixture", 3)

    assert ScaleModel.__module__ == "typed_region_project.worker"
    assert ScaleModel.__qualname__ == "ScaleModel"
    assert ScaleModel.__doc__ == "Safe non-generic class used to verify atomic class compilation."
    assert ScaleModel.__annotations__ == {"name": "str", "factor": "int"}
    assert signature.parameters["name"].annotation == "str"
    assert signature.parameters["factor"].annotation == "int"
    assert signature.parameters["factor"].default == MODEL_DEFAULT_FACTOR
    assert signature.return_annotation == "None"
    assert model.apply(6) == MODEL_RESULT
    assert model.describe() == "fixture:3"
    assert isinstance(model, ScaleModel)
    assert issubclass(type(model), ScaleModel)

    class InterpretedScaleModel(ScaleModel):
        pass

    interpreted = InterpretedScaleModel("child", 4)
    assert isinstance(interpreted, ScaleModel)
    assert issubclass(InterpretedScaleModel, ScaleModel)
    assert interpreted.apply(5) == MODEL_CHILD_RESULT
    restored = _pickle_round_trip(model)
    assert isinstance(restored, ScaleModel)
    assert restored.describe() == "fixture:3"


def test_dynamic_owner_downgrades_to_method_level_without_replacing_class() -> None:
    """Unsafe class identity remains source-owned while safe methods still work."""
    worker = UnsafeIdentityWorker(5)

    assert UNSAFE_IDENTITY_CLASS is UnsafeIdentityWorker
    assert type(UNSAFE_IDENTITY_INSTANCE) is UnsafeIdentityWorker
    assert worker.square(4) == UNSAFE_SQUARE_RESULT


def test_generic_fallbacks_cover_subclass_and_closed_call_specializations() -> None:
    """The fixture accepts both specialized and deliberately incompatible values."""
    assert Pairer[str]().pair("base") == ("base", "base")
    assert IntPairer().pair(4) == (4, 4)
    assert cast(Pairer[object], IntPairer()).pair("fallback") == (
        "fallback",
        "fallback",
    )
    payload = Payload(3)
    assert PayloadPairer().maybe_pair(payload) == (payload, payload)
    assert PayloadPairer().maybe_pair(None) == (None, None)
    assert cast(OptionalPairer[object], PayloadPairer()).maybe_pair(3) == (3, 3)
    assert pair_int(5) == (5, 5)
    assert pair_value("fallback") == ("fallback", "fallback")


async def _assert_async_generator_protocol(worker: Worker) -> None:
    generator = worker.exchange(1)

    assert await anext(generator) == ASYNC_FIRST_RESULT
    assert await generator.asend(5) == ASYNC_SENT_RESULT
    assert await generator.athrow(ValueError("fixture")) == ASYNC_THROWN_RESULT
    await generator.aclose()
    assert worker.closed is True


def _pickle_round_trip[T](value: T) -> T:
    loads = cast(Callable[[bytes], object], getattr(pickle, _PICKLE_LOADS_NAME))
    return cast(T, loads(pickle.dumps(value)))
