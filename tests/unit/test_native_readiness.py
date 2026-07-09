"""Tests for native-readiness scoring of generated sidecars."""

from atoll.analysis.native_readiness import NativeReadiness, analyze_native_readiness

EXPECTED_GENERATED_FUNCTIONS = 2
EXPECTED_MISSING_EXPORT_SCORE = 20
EXPECTED_LOOP_COUNT = 5


def test_typed_loop_kernel_is_native_eligible() -> None:
    """Pure typed loops with primitive work are eligible for native compilation."""
    readiness = analyze_native_readiness(
        "sample",
        "score",
        "\n".join(
            [
                "def helper(value: int) -> int:",
                "    return value * 2",
                "",
                "def score(values: list[int]) -> int:",
                "    total = 0",
                "    for value in values:",
                "        total += helper(value) + values[0]",
                "    return total",
                "",
            ]
        ),
    )

    assert readiness == NativeReadiness(
        source_module="sample",
        symbol="score",
        eligible=True,
        score=100,
        function_count=2,
        any_typed_functions=(),
        boxed_typed_functions=(),
        dynamic_dependencies=(),
        loop_count=1,
        native_operation_count=3,
        reasons=(),
    )


def test_any_signature_is_rejected() -> None:
    """Any-typed signatures are hard blockers even when the body has work."""
    readiness = analyze_native_readiness(
        "sample",
        "normalize",
        "\n".join(
            [
                "import typing",
                "",
                "def normalize(value: typing.Any) -> int:",
                "    return value[0] + value[1] + value[2] + value[3]",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.any_typed_functions == ("normalize",)
    assert readiness.boxed_typed_functions == ()
    assert readiness.reasons == ("Any annotations: normalize",)


def test_boxed_signature_is_rejected() -> None:
    """Non-builtin boxed annotations are rejected for sidecar-native kernels."""
    readiness = analyze_native_readiness(
        "sample",
        "total",
        "\n".join(
            [
                "def total(values: Series) -> int:",
                "    return values[0] + values[1] + values[2] + values[3]",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.any_typed_functions == ()
    assert readiness.boxed_typed_functions == ("total",)
    assert readiness.reasons == ("boxed annotations: total",)


def test_dynamic_getattr_dependency_is_rejected() -> None:
    """Runtime references to top-level getattr aliases block native readiness."""
    readiness = analyze_native_readiness(
        "sample",
        "root",
        "\n".join(
            [
                "import math",
                "",
                "sqrt = getattr(math, 'sqrt')",
                "",
                "def root(values: list[float]) -> float:",
                "    total = 0.0",
                "    for value in values:",
                "        total += sqrt(value)",
                "    return total",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.dynamic_dependencies == ("sqrt",)
    assert readiness.loop_count == 1
    assert readiness.reasons == ("dynamic getattr dependencies: sqrt",)


def test_trivial_external_call_helper_is_rejected() -> None:
    """A typed helper that only delegates to an external call has no native signal."""
    readiness = analyze_native_readiness(
        "sample",
        "slugify",
        "\n".join(
            [
                "def slugify(value: str) -> str:",
                "    return external_slugify(value)",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.loop_count == 0
    assert readiness.native_operation_count == 0
    assert readiness.reasons == ("no repeated/native work signal",)


def test_missing_annotations_reject_all_generated_functions() -> None:
    """Every generated dependency must carry complete annotations."""
    readiness = analyze_native_readiness(
        "sample",
        "score",
        "\n".join(
            [
                "def helper(value):",
                "    return value + 1",
                "",
                "def score(value: int) -> int:",
                "    return helper(value) + value + value + value",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.function_count == EXPECTED_GENERATED_FUNCTIONS
    assert readiness.reasons == ("missing annotations: helper",)


def test_missing_export_is_rejected_and_penalized() -> None:
    """A requested export absent from generated source is never eligible."""
    readiness = analyze_native_readiness(
        "sample",
        "missing",
        "def helper(value: int) -> int:\n    return value\n",
    )

    assert readiness.eligible is False
    assert readiness.score == EXPECTED_MISSING_EXPORT_SCORE
    assert readiness.reasons == (
        "exported symbol not generated: missing",
        "no repeated/native work signal",
    )


def test_string_annotations_and_extended_arguments_are_analyzed() -> None:
    """Forward strings, positional-only, variadic, and keyword annotations are parsed."""
    readiness = analyze_native_readiness(
        "sample",
        "kernel",
        "\n".join(
            [
                "def kernel(value: 'typing.Any', /, *items: int, option: int, "
                "**kwargs: int) -> 'None':",
                "    return value[0] + value[1] + value[2] + value[3]",
                "",
            ]
        ),
    )

    assert readiness.eligible is False
    assert readiness.any_typed_functions == ("kernel",)
    assert readiness.boxed_typed_functions == ()

    invalid_forward_reference = analyze_native_readiness(
        "sample",
        "kernel",
        "def kernel(value: 'invalid[') -> int:\n    return value + value + value + value\n",
    )
    assert invalid_forward_reference.boxed_typed_functions == ()


def test_dynamic_binding_analysis_skips_nested_scopes() -> None:
    """Tuple and annotated aliases only block functions that load them at runtime."""
    readiness = analyze_native_readiness(
        "sample",
        "kernel",
        "\n".join(
            [
                "first, second = getattr(runtime, 'pair')",
                "third: object = getattr(runtime, 'third')",
                "",
                "def kernel(values: list[int]) -> int:",
                "    def nested() -> object:",
                "        return second",
                "    async def nested_async() -> object:",
                "        return third",
                "    class Hidden:",
                "        dependency = third",
                "    total = 0",
                "    for value in values:",
                "        total += first(value)",
                "    return total",
                "",
            ]
        ),
    )

    assert readiness.dynamic_dependencies == ("first",)


def test_async_loops_and_comprehensions_count_as_repeated_work() -> None:
    """All supported loop and comprehension forms contribute native-work signal."""
    readiness = analyze_native_readiness(
        "sample",
        "kernel",
        "\n".join(
            [
                "async def kernel(values: list[int]) -> int:",
                "    total = 0",
                "    while total < 1:",
                "        total += 1",
                "    async for value in stream():",
                "        total += value",
                "    selected = {value for value in values}",
                "    indexed = {value: value for value in selected}",
                "    return total + sum(value for value in indexed)",
                "",
            ]
        ),
    )

    assert readiness.eligible is True
    assert readiness.loop_count == EXPECTED_LOOP_COUNT
