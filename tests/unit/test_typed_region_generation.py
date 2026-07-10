"""Tests for preserved typed-method source generation."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.typed_region import (
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import ModuleId, ModuleScan


def _scan(tmp_path: Path) -> ModuleScan:
    source_path = tmp_path / "worker.py"
    source_path.write_text(
        """from __future__ import annotations

from collections.abc import Iterator

LIMIT = 3

class Worker:
    bias: int

    def scale(self, value: int) -> int:
        return value * self.bias + LIMIT

    @staticmethod
    def parse(value: str) -> int:
        return int(value)

    @classmethod
    def adjust(cls, value: int) -> int:
        return value + 1

    def values(self, limit: int) -> Iterator[int]:
        for value in range(limit):
            yield self.scale(value)

    async def score(self, value: int) -> int:
        return self.scale(value)

    async def stream(self, limit: int) -> Iterator[int]:
        for value in range(limit):
            yield value
""",
        encoding="utf-8",
    )
    return enrich_island_analysis(scan_module(ModuleId(name="worker", path=source_path)))


def test_generate_typed_methods_preserves_annotations_and_shapes(tmp_path: Path) -> None:
    """Methods become top-level backend inputs without type erasure."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    selected = tuple(
        member.id
        for member in region.members
        if member.id.qualname
        in {
            "Worker.scale",
            "Worker.parse",
            "Worker.adjust",
            "Worker.values",
            "Worker.score",
        }
    )
    output = tmp_path / "generated.py"

    generation = generate_typed_method_region(
        scan,
        region,
        selected,
        output_path=output,
    )

    source = generation.source_text
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert output.read_text(encoding="utf-8") == source
    assert "importlib" not in source
    assert "Any" not in source
    assert "LIMIT = 3" in source
    assert "from collections.abc import Iterator" in source
    assert "from worker import Worker" not in source
    assert set(functions) == {
        "Worker__scale",
        "Worker__parse",
        "Worker__adjust",
        "Worker__values",
        "Worker__score",
    }
    value_annotation = functions["Worker__scale"].args.args[1].annotation
    return_annotation = functions["Worker__scale"].returns
    assert value_annotation is not None
    assert return_annotation is not None
    assert ast.unparse(value_annotation) == "int"
    assert ast.unparse(return_annotation) == "int"
    assert isinstance(functions["Worker__score"], ast.AsyncFunctionDef)
    assert any(isinstance(node, ast.Yield) for node in ast.walk(functions["Worker__values"]))
    assert all(not function.decorator_list for function in functions.values())
    assert [binding.compiled_name for binding in generation.bindings] == list(functions)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """Equivalent region input produces byte-identical source and hashes."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    selected = (
        next(member.id for member in region.members if member.id.qualname == "Worker.scale"),
    )

    first = generate_typed_method_region(
        scan,
        region,
        selected,
        output_path=tmp_path / "first.py",
    )
    second = generate_typed_method_region(
        scan,
        region,
        selected,
        output_path=tmp_path / "second.py",
    )

    assert first.source_text == second.source_text
    assert first.source_hash == second.source_hash


def test_atomic_class_generation_preserves_declaration_and_mypyc_contract(
    tmp_path: Path,
) -> None:
    """A safe complete class becomes one native class binding without method extraction."""
    source_path = tmp_path / "counter.py"
    source_path.write_text(
        """class Counter:
    \"\"\"Store and increment one integer.\"\"\"

    value: int

    def __init__(self, value: int) -> None:
        self.value = value

    def increment(self, amount: int) -> int:
        self.value += amount
        return self.value
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="counter", path=source_path)))
    region = next(region for region in scan.typed_regions if region.atomic_class)
    owner = next(member.id for member in region.members if member.kind == "class")

    generation = generate_typed_method_region(
        scan,
        region,
        (owner,),
        output_path=tmp_path / "generated_counter.py",
    )
    cython_generation = generate_typed_method_region(
        scan,
        region,
        (owner,),
        output_path=tmp_path / "generated_counter_cython.py",
        options=TypedRegionGenerationOptions(backend="cython"),
    )

    tree = ast.parse(generation.source_text)
    generated_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert generated_class.name == "Counter"
    assert ast.unparse(generated_class.decorator_list[0]) == (
        "mypyc_attr(allow_interpreted_subclasses=True, serializable=True)"
    )
    assert "if TYPE_CHECKING:\n    from mypy_extensions import mypyc_attr" in (
        generation.source_text
    )
    assert {node.name for node in generated_class.body if isinstance(node, ast.FunctionDef)} == {
        "__init__",
        "increment",
    }
    assert "Protocol" not in generation.source_text
    assert generation.bindings == region.bindings
    assert generation.bindings[0].kind == "class"
    assert generation.bindings[0].compiled_name == "Counter"
    assert cython_generation.backend == "cython"
    assert "mypyc_attr" not in cython_generation.source_text
    assert cython_generation.bindings == generation.bindings


def test_generation_rejects_async_generators(tmp_path: Path) -> None:
    """Async generators remain reserved for the Cython backend milestone."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    stream = next(member.id for member in region.members if member.id.qualname == "Worker.stream")

    with pytest.raises(ValueError, match="async_generator"):
        generate_typed_method_region(
            scan,
            region,
            (stream,),
            output_path=tmp_path / "generated.py",
        )


def test_cython_generation_preserves_async_generator_source(tmp_path: Path) -> None:
    """Cython units retain async-generator syntax without mypyc owner facades."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    stream = next(member.id for member in region.members if member.id.qualname == "Worker.stream")

    generation = generate_typed_method_region(
        scan,
        region,
        (stream,),
        output_path=tmp_path / "generated.py",
        options=TypedRegionGenerationOptions(backend="cython"),
    )

    assert generation.backend == "cython"
    assert "async def Worker__stream" in generation.source_text
    assert "yield value" in generation.source_text
    assert "__atoll_execution_kinds__ = {'Worker__stream': 'async_generator'}" in (
        generation.source_text
    )
    assert "Protocol" not in generation.source_text
    assert "_AtollWorker" not in generation.source_text


def test_generation_preserves_ordinary_module_functions(tmp_path: Path) -> None:
    """Unspecialized module functions lower without owner facades or type erasure."""
    source_path = tmp_path / "ordinary_functions.py"
    source_path.write_text(
        """from collections.abc import Iterator

LIMIT = 4

def scale(value: int) -> int:
    return value * LIMIT

async def score(value: int) -> int:
    return value + LIMIT

def values(limit: int) -> Iterator[int]:
    for value in range(limit):
        yield scale(value)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(
        scan_module(ModuleId(name="ordinary_functions", path=source_path))
    )
    selected_generations = {
        member.id.qualname: generate_typed_method_region(
            scan,
            region,
            (member.id,),
            output_path=tmp_path / f"{member.id.qualname}.py",
        )
        for region in scan.typed_regions
        for member in region.members
        if member.id.qualname in {"scale", "score", "values"}
    }

    assert set(selected_generations) == {"scale", "score", "values"}
    scale_generation = selected_generations["scale"]
    score_generation = selected_generations["score"]
    values_generation = selected_generations["values"]
    scale_tree = ast.parse(scale_generation.source_text)
    score_tree = ast.parse(score_generation.source_text)
    values_tree = ast.parse(values_generation.source_text)
    scale_function = next(node for node in scale_tree.body if isinstance(node, ast.FunctionDef))
    score_function = next(
        node for node in score_tree.body if isinstance(node, ast.AsyncFunctionDef)
    )
    values_function = next(node for node in values_tree.body if isinstance(node, ast.FunctionDef))

    assert scale_generation.bindings[0].kind == "module"
    assert score_generation.bindings[0].kind == "module"
    assert values_generation.bindings[0].kind == "module"
    assert scale_function.name == "scale"
    assert score_function.name == "score"
    assert values_function.name == "values"
    scale_annotation = scale_function.args.args[0].annotation
    scale_return = scale_function.returns
    assert scale_annotation is not None
    assert scale_return is not None
    assert ast.unparse(scale_annotation) == "int"
    assert ast.unparse(scale_return) == "int"
    assert isinstance(score_function, ast.AsyncFunctionDef)
    assert any(isinstance(node, ast.Yield) for node in ast.walk(values_function))
    assert "Protocol" not in scale_generation.source_text
    assert "_Atoll" not in values_generation.source_text


def test_cython_generation_preserves_ordinary_async_generator_function(
    tmp_path: Path,
) -> None:
    """The Cython backend accepts ordinary module async generators."""
    source_path = tmp_path / "ordinary_async_generator.py"
    source_path.write_text(
        """from collections.abc import AsyncIterator

async def stream(limit: int) -> AsyncIterator[int]:
    for value in range(limit):
        yield value
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(
        scan_module(ModuleId(name="ordinary_async_generator", path=source_path))
    )
    region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "stream" for member in region.members)
    )
    stream = next(member.id for member in region.members if member.id.qualname == "stream")

    with pytest.raises(ValueError, match="async_generator"):
        generate_typed_method_region(
            scan,
            region,
            (stream,),
            output_path=tmp_path / "generated_mypyc.py",
        )

    generation = generate_typed_method_region(
        scan,
        region,
        (stream,),
        output_path=tmp_path / "generated_cython.py",
        options=TypedRegionGenerationOptions(backend="cython"),
    )

    assert generation.backend == "cython"
    assert generation.bindings[0].kind == "module"
    assert "async def stream" in generation.source_text
    assert "yield value" in generation.source_text
    assert "Protocol" not in generation.source_text


def test_generation_validates_member_selection(tmp_path: Path) -> None:
    """Empty, duplicate, foreign, and non-method selections fail before writes."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    scale = next(member.id for member in region.members if member.id.qualname == "Worker.scale")
    owner = next(member.id for member in region.members if member.id.qualname == "Worker")
    foreign = replace(scale, qualname="Worker.missing")
    output = tmp_path / "generated.py"

    with pytest.raises(ValueError, match="at least one"):
        generate_typed_method_region(
            scan,
            region,
            (),
            output_path=output,
        )
    with pytest.raises(ValueError, match="duplicate"):
        generate_typed_method_region(
            scan,
            region,
            (scale, scale),
            output_path=output,
        )
    with pytest.raises(ValueError, match="outside typed region"):
        generate_typed_method_region(
            scan,
            region,
            (foreign,),
            output_path=output,
        )
    with pytest.raises(ValueError, match="unsupported typed-region binding"):
        generate_typed_method_region(
            scan,
            region,
            (owner,),
            output_path=output,
        )
    assert not output.exists()


def test_generation_rejects_unknown_decorators_and_missing_receivers(tmp_path: Path) -> None:
    """Lowering refuses declaration shapes that cannot retain method semantics."""
    scan = _scan(tmp_path)
    region = next(region for region in scan.typed_regions if region.atomic_class)
    scale = next(member for member in region.members if member.id.qualname == "Worker.scale")
    adjust = next(member for member in region.members if member.id.qualname == "Worker.adjust")
    decorated = replace(scale, source_text="@custom\n" + dedent(scale.source_text))
    no_self = replace(
        scale,
        source_text="def scale(*, value: int) -> int:\n    return value\n",
    )
    no_cls = replace(
        adjust,
        source_text=("@classmethod\ndef adjust(*, value: int) -> int:\n    return value\n"),
    )
    no_owner = replace(scale, owner_class=None)

    with pytest.raises(ValueError, match="unsupported method decorator"):
        generate_typed_method_region(
            scan,
            replace(region, members=(decorated,)),
            (decorated.id,),
            output_path=tmp_path / "decorated.py",
        )
    with pytest.raises(ValueError, match="no self parameter"):
        generate_typed_method_region(
            scan,
            replace(region, members=(no_self,)),
            (no_self.id,),
            output_path=tmp_path / "no_self.py",
        )
    with pytest.raises(ValueError, match="no cls parameter"):
        generate_typed_method_region(
            scan,
            replace(region, members=(no_cls,)),
            (no_cls.id,),
            output_path=tmp_path / "no_cls.py",
        )
    with pytest.raises(ValueError, match="no owner class"):
        generate_typed_method_region(
            scan,
            replace(region, members=(no_owner,)),
            (no_owner.id,),
            output_path=tmp_path / "no_owner.py",
        )


def test_generation_resolves_relative_imports_and_owner_annotations(tmp_path: Path) -> None:
    """Package imports and explicit receiver types retain their source meaning."""
    package = tmp_path / "pkg"
    package.mkdir()
    source_path = package / "worker.py"
    source_path.write_text(
        """from __future__ import annotations

import typing
from typing import Self

from .helpers import twice as double

class Worker:
    def scale(self: "Worker", value: typing.SupportsInt) -> int:
        return double(int(value))

    def copy(self: typing.Self) -> typing.Self:
        return self

    @classmethod
    def create(cls: type[Self], value: int) -> int:
        return value
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="pkg.worker", path=source_path)))
    region = next(region for region in scan.typed_regions if region.atomic_class)
    selected = tuple(member.id for member in region.members if member.kind == "method")

    generation = generate_typed_method_region(
        scan,
        region,
        selected,
        output_path=tmp_path / "generated.py",
    )

    tree = ast.parse(generation.source_text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scale_receiver = functions["Worker__scale"].args.args[0].annotation
    copy_receiver = functions["Worker__copy"].args.args[0].annotation
    create_receiver = functions["Worker__create"].args.args[0].annotation
    copy_return = functions["Worker__copy"].returns
    assert scale_receiver is not None
    assert copy_receiver is not None
    assert create_receiver is not None
    assert copy_return is not None
    assert "from pkg.helpers import twice as double" in generation.source_text
    assert ast.unparse(scale_receiver) == "_AtollWorker"
    assert ast.unparse(copy_receiver) == "_AtollWorker"
    assert ast.unparse(create_receiver) == "type[_AtollWorker]"
    assert ast.unparse(copy_return) == "_AtollWorker"


def test_generation_rejects_relative_imports_outside_the_package(tmp_path: Path) -> None:
    """A generated top-level unit never guesses an invalid package boundary."""
    source_path = tmp_path / "worker.py"
    source_path.write_text(
        """from ..helpers import twice

class Worker:
    def scale(self, value: int) -> int:
        return twice(value)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="worker", path=source_path)))
    region = next(region for region in scan.typed_regions if region.atomic_class)
    scale = next(member.id for member in region.members if member.id.qualname == "Worker.scale")

    with pytest.raises(ValueError, match="escapes package"):
        generate_typed_method_region(
            scan,
            region,
            (scale,),
            output_path=tmp_path / "generated.py",
        )


def test_generation_specializes_generic_method_only_for_concrete_subclass(
    tmp_path: Path,
) -> None:
    """A subclass specialization rewrites annotations but retains generic source IR."""
    source_path = tmp_path / "generic_worker.py"
    source_path.write_text(
        """class Pairer[T]:
    def normalize(self, value: T) -> T:
        return value

    def pair(self, value: T) -> tuple[T, T]:
        normalized = self.normalize(value)
        return normalized, normalized

class IntPairer(Pairer[int]):
    pass
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="generic_worker", path=source_path)))
    region = next(region for region in scan.typed_regions if region.specializations)
    specialization = next(
        item
        for item in region.specializations
        if item.target_owner_class == "IntPairer" and item.source_member.qualname == "Pairer.pair"
    )
    output = tmp_path / "specialized.py"

    generation = generate_typed_method_region(
        scan,
        region,
        (specialization.source_member,),
        output_path=output,
        options=TypedRegionGenerationOptions(specialization=specialization),
    )

    binding = generation.bindings[0]
    tree = ast.parse(generation.source_text)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == binding.compiled_name
    )
    value_annotation = function.args.args[1].annotation
    assert value_annotation is not None
    assert function.returns is not None
    assert ast.unparse(value_annotation) == "int"
    assert ast.unparse(function.returns) == "tuple[int, int]"
    assert function.type_params == []
    assert "class _AtollIntPairer(Protocol)" in generation.source_text
    assert "def normalize(self, value: int) -> int" in generation.source_text
    assert "Any" not in generation.source_text
    assert binding.owner_class == "Pairer"
    assert binding.target_owner_class == "IntPairer"
    assert binding.guards == specialization.guards
    assert generation.specialization == specialization
    original = next(
        member for member in region.members if member.id == specialization.source_member
    )
    assert "value: T" in original.source_text


def test_generation_specializes_closed_generic_function_without_owner_facade(
    tmp_path: Path,
) -> None:
    """A closed call produces a module binding with concrete annotations and guards."""
    source_path = tmp_path / "generic_function.py"
    source_path.write_text(
        """def pair_value[T](value: T) -> tuple[T, T]:
    return value, value

def pair_int(value: int) -> tuple[int, int]:
    return pair_value(value)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="generic_function", path=source_path)))
    region = next(region for region in scan.typed_regions if region.specializations)
    specialization = next(item for item in region.specializations if item.origin == "closed_call")

    generation = generate_typed_method_region(
        scan,
        region,
        (specialization.source_member,),
        output_path=tmp_path / "specialized_function.py",
        options=TypedRegionGenerationOptions(specialization=specialization),
    )

    binding = generation.bindings[0]
    tree = ast.parse(generation.source_text)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == binding.compiled_name
    )
    assert binding.kind == "module"
    assert binding.owner_class is None
    assert binding.target_owner_class is None
    assert binding.guards[0].parameter_name == "value"
    assert function.args.args[0].annotation is not None
    assert function.returns is not None
    assert ast.unparse(function.args.args[0].annotation) == "int"
    assert ast.unparse(function.returns) == "tuple[int, int]"
    assert "Protocol" not in generation.source_text
