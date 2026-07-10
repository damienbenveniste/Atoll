"""Tests for preserved typed-method source generation."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.typed_region import generate_typed_method_region
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
        backend="cython",
    )

    assert generation.backend == "cython"
    assert "async def Worker__stream" in generation.source_text
    assert "yield value" in generation.source_text
    assert "Protocol" not in generation.source_text
    assert "_AtollWorker" not in generation.source_text


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
