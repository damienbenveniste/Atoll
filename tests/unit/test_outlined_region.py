from __future__ import annotations

import asyncio
import builtins
import inspect
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import cast

import pytest

from atoll.generation.outlined_region import OutlinedRegionGeneration, generate_outlined_region
from atoll.models import (
    BindingKind,
    BindingTarget,
    ExecutionKind,
    ModuleId,
    RegionMember,
    SymbolId,
    TypedRegion,
)

_SHA256_HEX_LENGTH = 64
_EXPECTED_AUGMENTED_RESULT = 18
_EXPECTED_BLOCK_TOTAL = 9


def _member(
    source_text: str,
    *,
    qualname: str = "worker",
    owner_class: str | None = None,
    execution_kind: ExecutionKind = "coroutine",
    binding_kind: BindingKind | None = None,
) -> RegionMember:
    return RegionMember(
        id=SymbolId(module="sample.worker", qualname=qualname),
        kind="method" if owner_class is not None else "function",
        owner_class=owner_class,
        binding_kind=(
            binding_kind
            if binding_kind is not None
            else "instance_method"
            if owner_class is not None
            else "module"
        ),
        execution_kind=execution_kind,
        source_text=dedent(source_text),
        type_parameters=(),
        type_parameter_records=(),
        scope_type_parameters=(),
        scope_type_parameter_records=(),
        parameters=(),
        return_annotation=None,
    )


def _binding(member: RegionMember) -> BindingTarget:
    return BindingTarget(
        source=member.id,
        compiled_name=member.id.qualname.replace(".", "__"),
        kind=member.binding_kind,
        owner_class=member.owner_class,
        execution_kind=member.execution_kind,
    )


def _region(member: RegionMember) -> TypedRegion:
    binding = _binding(member)
    return TypedRegion(
        id="sample.worker::region",
        source_module=ModuleId(name="sample.worker", path=Path("sample/worker.py")),
        members=(member,),
        dependencies=(),
        type_bindings=(),
        bindings=(binding,),
        decisions=(),
        source_hash="0" * 64,
    )


def _generate(member: RegionMember, tmp_path: Path) -> OutlinedRegionGeneration:
    return generate_outlined_region(
        _region(member),
        member.id,
        _binding(member),
        output_path=tmp_path / "outlined.py",
    )


def _native_from_source(source_text: str) -> SimpleNamespace:
    namespace: dict[str, object] = {}
    _execute_source(source_text, namespace, namespace)
    helper_names = tuple(name for name in namespace if name.startswith("_worker__outlined_"))
    helpers = {name: namespace[name] for name in helper_names}
    return SimpleNamespace(**helpers)


def _shell(
    generation: OutlinedRegionGeneration,
    native: object,
    globals_namespace: dict[str, object],
) -> Callable[..., object]:
    locals_namespace: dict[str, object] = {}
    _execute_source(generation.shell.factory_source, globals_namespace, locals_namespace)
    factory = cast(
        Callable[[object], Callable[..., object]],
        locals_namespace[generation.shell.factory_name],
    )
    return factory(native)


def _run(awaitable: object) -> object:
    return asyncio.run(cast(Coroutine[object, object, object], awaitable))


def _execute_source(
    source_text: str,
    globals_namespace: dict[str, object],
    locals_namespace: dict[str, object],
) -> None:
    executor = cast(
        Callable[[str, dict[str, object], dict[str, object]], None],
        vars(builtins)["exec"],
    )
    executor(source_text, globals_namespace, locals_namespace)


def test_coroutine_shell_rewrites_live_outs_and_control_tag(tmp_path: Path) -> None:
    member = _member(
        """
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            audit(total)
            await checkpoint()
            result = total * 2
            return result
        """
    )

    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {"audit": seen.append, "checkpoint": checkpoint, "RuntimeError": RuntimeError},
    )

    expected_result = 18
    assert len(generation.source_hash) == _SHA256_HEX_LENGTH
    assert inspect.iscoroutinefunction(shell)
    assert generation.helper_names == generation.shell.helper_names
    assert "'continue'" in generation.source_text
    assert "if _atoll_outlined_control_" in generation.shell.factory_source
    assert _run(cast(Callable[[list[int]], object], shell)([1, 2])) == expected_result
    assert seen == [9]
    assert tmp_path.joinpath("outlined.py").read_text(encoding="utf-8") == generation.source_text


def test_runtime_globals_are_passed_as_explicit_helper_arguments(tmp_path: Path) -> None:
    member = _member(
        """
        async def worker(values):
            start = transform(BASE + len(values))
            total = start + 1
            final = transform(total)
            audit(final)
            await checkpoint()
            return final
        """
    )

    generation = _generate(member, tmp_path)
    assert "(_atoll_resolve_global, values)" in generation.source_text
    assert "_atoll_resolve_global('BASE')" in generation.source_text
    assert "_atoll_resolve_global('transform')" in generation.source_text
    assert f"{generation.helper_names[0]}(_atoll_resolve_global, values)" in (
        generation.shell.factory_source
    )

    native = _native_from_source(generation.source_text)
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    def transform(value: int) -> int:
        return value * 2

    shell = _shell(
        generation,
        native,
        {
            "BASE": 3,
            "audit": seen.append,
            "checkpoint": checkpoint,
            "len": len,
            "transform": transform,
            "RuntimeError": RuntimeError,
        },
    )

    expected_result = 22
    assert _run(cast(Callable[[list[int]], object], shell)([1, 2])) == expected_result
    assert seen == [22]


def test_global_reads_remain_late_bound_within_native_block(tmp_path: Path) -> None:
    """A call that rebinds a source global affects later reads in the same helper."""
    member = _member(
        """
        async def worker():
            first = G + 1
            mutate()
            second = G + 2
            total = first + second
            await checkpoint()
            return second
        """
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    globals_namespace: dict[str, object] = {"G": 1}

    def mutate() -> None:
        globals_namespace["G"] = 10

    async def checkpoint() -> None:
        return None

    globals_namespace.update({"checkpoint": checkpoint, "mutate": mutate})
    shell = _shell(generation, native, globals_namespace)

    expected_rebound_value = 12
    assert _run(cast(Callable[[], object], shell)()) == expected_rebound_value


def test_augmented_assignment_after_await_receives_shell_local(tmp_path: Path) -> None:
    """A helper receives the prior shell value read implicitly by augmented assignment."""
    member = _member(
        """
        async def worker(total):
            await checkpoint()
            total += 2
            result = total * 3
            audit(result)
            await checkpoint()
            return result
        """
    )
    generation = _generate(member, tmp_path)
    native_namespace: dict[str, object] = {}
    _execute_source(generation.source_text, native_namespace, native_namespace)
    native = SimpleNamespace(
        **{generation.helper_names[0]: native_namespace[generation.helper_names[0]]}
    )
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {"audit": seen.append, "checkpoint": checkpoint},
    )

    assert "_atoll_resolve_global, total" in generation.source_text
    assert _run(cast(Callable[[int], object], shell)(4)) == _EXPECTED_AUGMENTED_RESULT
    assert seen == [_EXPECTED_AUGMENTED_RESULT]


def test_self_referential_assignment_after_await_receives_shell_local(tmp_path: Path) -> None:
    """Ordinary assignment RHS loads are captured before their target store."""
    member = _member(
        """
        async def worker(total):
            await checkpoint()
            total = total + 2
            result = total * 3
            audit(result)
            await checkpoint()
            return result
        """
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {"audit": seen.append, "checkpoint": checkpoint},
    )

    assert "total" in generation.plan.blocks[0].live_ins
    assert _run(cast(Callable[[int], object], shell)(4)) == _EXPECTED_AUGMENTED_RESULT
    assert seen == [_EXPECTED_AUGMENTED_RESULT]


def test_same_line_suspension_restores_live_out_by_column(tmp_path: Path) -> None:
    """Semicolon-separated suspension statements retain precise block coordinates."""
    member = _member(
        "async def worker(values): start = len(values) + 1; "
        "doubled = start * 2; total = doubled + 3; await checkpoint(); "
        "result = total * 2; return result"
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)

    async def checkpoint() -> None:
        return None

    shell = _shell(generation, native, {"checkpoint": checkpoint, "len": len})

    assert generation.plan.blocks[0].live_outs == ("total",)
    assert _run(cast(Callable[[list[int]], object], shell)([1, 2])) == _EXPECTED_AUGMENTED_RESULT


def test_generator_shell_preserves_yield_protocol(tmp_path: Path) -> None:
    member = _member(
        """
        def worker(values):
            for value in values:
                audit(value + 1)
            yield "ready"
            return "done"
        """,
        execution_kind="generator",
    )

    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []
    shell = _shell(generation, native, {"audit": seen.append, "RuntimeError": RuntimeError})

    assert inspect.isgeneratorfunction(shell)
    iterator = cast(Callable[[list[int]], Generator[str, None, str]], shell)([2, 4])
    assert next(iterator) == "ready"
    with pytest.raises(StopIteration) as stopped:
        next(iterator)
    assert stopped.value.value == "done"
    assert seen == [3, 5]


def test_async_generator_shell_preserves_throw_send_close_and_finally(tmp_path: Path) -> None:
    """Async-generator suspension and cleanup remain in the Python shell."""
    member = _member(
        """
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            try:
                sent = yield total
                yield sent
            except ValueError:
                yield -1
            finally:
                audit(total)
        """,
        execution_kind="async_generator",
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []
    shell = _shell(generation, native, {"audit": seen.append, "len": len})

    async def exercise() -> tuple[int, int, int]:
        stream = cast(
            AsyncGenerator[int, int],
            cast(Callable[[list[int]], object], shell)([1, 2]),
        )
        first = await anext(stream)
        second = await stream.asend(7)
        handled = await stream.athrow(ValueError("fixture"))
        await stream.aclose()
        return first, second, handled

    assert inspect.isasyncgenfunction(shell)
    assert asyncio.run(exercise()) == (9, 7, -1)
    assert seen == [9]


def test_nested_control_block_rewrites_inside_original_ast_body(tmp_path: Path) -> None:
    member = _member(
        """
        async def worker(flag, values):
            if flag:
                for value in values:
                    audit(value + 1)
            else:
                await checkpoint()
            return "done"
        """
    )

    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {"audit": seen.append, "checkpoint": checkpoint, "RuntimeError": RuntimeError},
    )

    assert "if flag:" in generation.shell.factory_source
    assert f"{generation.helper_names[0]}(_atoll_resolve_global, values)" in (
        generation.shell.factory_source
    )
    assert _run(cast(Callable[[bool, list[int]], object], shell)(True, [3, 4])) == "done"
    assert _run(cast(Callable[[bool, list[int]], object], shell)(False, [3, 4])) == "done"
    assert seen == [4, 5]


def test_control_assignment_uses_incoming_parameter_on_false_branch(tmp_path: Path) -> None:
    """Definite-assignment analysis passes a parameter used after a partial branch store."""
    member = _member(
        """
        async def worker(x, flag):
            if flag:
                x = 10
            first = x + 1
            second = first * 2
            audit(second)
            await checkpoint()
            return "done"
        """
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)
    seen: list[int] = []

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {"audit": seen.append, "checkpoint": checkpoint},
    )

    expected_false = 8
    expected_true = 22
    assert _run(cast(Callable[[int, bool], object], shell)(3, False)) == "done"
    assert _run(cast(Callable[[int, bool], object], shell)(3, True)) == "done"
    assert seen == [expected_false, expected_true]


def test_shell_restores_helper_value_before_later_delete(tmp_path: Path) -> None:
    """A shell-owned delete sees the local created by the preceding native block."""
    member = _member(
        """
        async def worker():
            x = 1
            first = x + 1
            x = first + 2
            audit(x)
            await checkpoint()
            del x
            return "done"
        """
    )
    generation = _generate(member, tmp_path)
    native = _native_from_source(generation.source_text)

    async def checkpoint() -> None:
        return None

    def audit(value: int) -> int:
        return value

    shell = _shell(
        generation,
        native,
        {"audit": audit, "checkpoint": checkpoint},
    )

    assert generation.plan.blocks[0].live_outs == ("x",)
    assert _run(cast(Callable[[], object], shell)()) == "done"


def test_method_shell_uses_owner_class_and_does_not_re_evaluate_defaults(tmp_path: Path) -> None:
    member = _member(
        """
        async def score(self, token: object = make_default()):
            total = self.bias + len(self.values)
            adjusted = total + 1
            self.record(adjusted)
            await checkpoint()
            return token, adjusted
        """,
        qualname="Worker.score",
        owner_class="Worker",
    )

    generation = _generate(member, tmp_path)
    assert "class Worker:" in generation.shell.factory_source
    assert "Worker.__dict__['score']" in generation.shell.factory_source
    assert "make_default" not in generation.shell.factory_source
    assert "token=None" in generation.shell.factory_source

    native_namespace: dict[str, object] = {}
    _execute_source(generation.source_text, native_namespace, native_namespace)
    native = SimpleNamespace(
        **{generation.helper_names[0]: native_namespace[generation.helper_names[0]]}
    )

    calls = 0

    def make_default() -> object:
        nonlocal calls
        calls += 1
        return object()

    async def checkpoint() -> None:
        return None

    shell = _shell(
        generation,
        native,
        {
            "checkpoint": checkpoint,
            "len": len,
            "make_default": make_default,
            "RuntimeError": RuntimeError,
        },
    )

    def record(_value: int) -> None:
        return None

    worker = SimpleNamespace(
        bias=2,
        values=[5, 6],
        record=record,
    )

    assert calls == 0
    expected_result = (None, 5)
    assert _run(cast(Callable[[object], object], shell)(worker)) == expected_result
    assert calls == 0


def test_staticmethod_descriptor_is_allowed_and_removed_only_from_private_shell(
    tmp_path: Path,
) -> None:
    """Recognized descriptors are reapplied by the shim instead of run in the factory."""
    member = _member(
        """
        @staticmethod
        async def score(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            await checkpoint()
            return total
        """,
        qualname="Worker.score",
        owner_class="Worker",
        binding_kind="staticmethod",
    )
    generation = _generate(member, tmp_path)
    native_namespace: dict[str, object] = {}
    _execute_source(generation.source_text, native_namespace, native_namespace)
    native = SimpleNamespace(
        **{generation.helper_names[0]: native_namespace[generation.helper_names[0]]}
    )

    async def checkpoint() -> None:
        return None

    shell = _shell(generation, native, {"checkpoint": checkpoint, "len": len})

    assert "staticmethod" not in generation.shell.factory_source
    assert inspect.iscoroutinefunction(shell)
    assert _run(cast(Callable[[list[int]], object], shell)([1, 2])) == _EXPECTED_BLOCK_TOTAL


@pytest.mark.parametrize(
    "decorator",
    ["@decorator", "@decorator()", "@custom.staticmethod"],
)
def test_unknown_decorator_is_rejected(tmp_path: Path, decorator: str) -> None:
    """Outlined shells never bypass source decorators that may replace a callable."""
    member = _member(
        f"""
        {decorator}
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            await checkpoint()
            return total
        """
    )

    with pytest.raises(ValueError, match="unknown decorator"):
        _generate(member, tmp_path)


@pytest.mark.parametrize(
    "source_text",
    [
        """
        async def worker(values):
            super().method()
            await checkpoint()
        """,
        """
        async def worker(values):
            token = __class__
            await checkpoint()
            return token
        """,
    ],
)
def test_blockers_reject_super_and_class_dependencies(source_text: str, tmp_path: Path) -> None:
    member = _member(source_text)

    with pytest.raises(ValueError, match="rejects"):
        _generate(member, tmp_path)


def test_rejects_binding_mismatch(tmp_path: Path) -> None:
    member = _member(
        """
        async def worker(values):
            total = 0
            for value in values:
                total += value
            await checkpoint()
            return total
        """
    )
    region = _region(member)
    binding = replace(_binding(member), source=SymbolId(module="sample.worker", qualname="other"))

    with pytest.raises(ValueError, match="binding does not target"):
        generate_outlined_region(region, member.id, binding, output_path=tmp_path / "outlined.py")


def test_generation_rejects_member_level_planner_evidence(tmp_path: Path) -> None:
    """Cells and nested scopes prevent every otherwise local block from outlining."""
    member = _member(
        """
        async def worker(values):
            callback = lambda value: value
            start = len(values) + 1
            doubled = start * 2
            total = callback(doubled)
            await checkpoint()
            return total
        """
    )

    with pytest.raises(ValueError, match="suspension planner"):
        _generate(member, tmp_path)


def test_generation_rejects_callable_without_credible_block(tmp_path: Path) -> None:
    """A suspension shell without synchronous work remains entirely interpreted."""
    member = _member(
        """
        async def worker():
            await checkpoint()
        """
    )

    with pytest.raises(ValueError, match="at least one eligible"):
        _generate(member, tmp_path)


def test_generation_validates_selected_member_and_binding_shape(tmp_path: Path) -> None:
    """Selection, backend shape, and descriptor evidence must agree exactly."""
    member = _member(
        """
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            await checkpoint()
            return total
        """
    )
    region = _region(member)
    binding = _binding(member)
    output_path = tmp_path / "outlined.py"

    with pytest.raises(ValueError, match="exactly one selected member"):
        generate_outlined_region(
            region,
            SymbolId("sample.worker", "missing"),
            binding,
            output_path=output_path,
        )

    sync_member = _member(
        """
        def worker(values):
            return len(values)
        """,
        execution_kind="sync",
    )
    with pytest.raises(ValueError, match="coroutine or generator"):
        _generate(sync_member, tmp_path)

    with pytest.raises(ValueError, match="not promised"):
        generate_outlined_region(
            replace(region, bindings=()),
            member.id,
            binding,
            output_path=output_path,
        )

    wrong_execution = replace(binding, execution_kind="generator")
    with pytest.raises(ValueError, match="execution kind"):
        generate_outlined_region(
            replace(region, bindings=(wrong_execution,)),
            member.id,
            wrong_execution,
            output_path=output_path,
        )

    wrong_kind = replace(binding, kind="staticmethod")
    with pytest.raises(ValueError, match="kind or owner"):
        generate_outlined_region(
            replace(region, bindings=(wrong_kind,)),
            member.id,
            wrong_kind,
            output_path=output_path,
        )


def test_generation_rejects_descriptor_source_mismatch(tmp_path: Path) -> None:
    """Descriptor metadata is insufficient when source syntax does not declare it."""
    member = _member(
        """
        async def score(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            await checkpoint()
            return total
        """,
        qualname="Worker.score",
        owner_class="Worker",
        binding_kind="staticmethod",
    )

    with pytest.raises(ValueError, match="descriptor decorators"):
        _generate(member, tmp_path)


def test_generation_rejects_ambiguous_source_and_generated_name_collision(
    tmp_path: Path,
) -> None:
    """Ambiguous declarations and private shell-name collisions fail before writing."""
    ambiguous = _member(
        """
        async def worker(values):
            await checkpoint()

        async def other(values):
            await checkpoint()
        """
    )
    with pytest.raises(ValueError, match="exactly one callable declaration"):
        _generate(ambiguous, tmp_path)

    collision = _member(
        """
        async def worker(_atoll_native, values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            await checkpoint()
            return total
        """
    )
    with pytest.raises(ValueError, match="generated names collide"):
        _generate(collision, tmp_path)


def test_other_suspension_members_do_not_poison_selected_outline(tmp_path: Path) -> None:
    """A directed root can be outlined while another async member stays interpreted."""
    member = _member(
        """
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            audit(total)
            await checkpoint()
            return total
        """
    )
    interpreted = _member(
        """
        async def helper(value):
            await checkpoint()
            return value
        """,
        qualname="helper",
    )
    region = replace(_region(member), members=(member, interpreted))

    generation = generate_outlined_region(
        region,
        member.id,
        _binding(member),
        output_path=tmp_path / "outlined.py",
    )

    assert generation.member.id == member.id
    assert generation.helper_names


def test_generation_is_stable_for_identical_inputs(tmp_path: Path) -> None:
    member = _member(
        """
        async def worker(values):
            start = len(values) + 1
            doubled = start * 2
            total = doubled + 3
            audit(total)
            await checkpoint()
            return total
        """
    )

    first = _generate(member, tmp_path / "first")
    second = _generate(member, tmp_path / "second")

    assert first.source_text == second.source_text
    assert first.source_hash == second.source_hash
    assert first.shell.factory_source == second.shell.factory_source
    assert first.helper_names == second.helper_names
