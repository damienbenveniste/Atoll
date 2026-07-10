"""Tests for AST scanning of fixture modules."""

from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.models import ModuleId, SymbolRecord

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXPECTED_OUTER_ARG_COUNT = 4


def _scan_ranking_module() -> dict[str, SymbolRecord]:
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    scan = scan_module(module)
    return {symbol.id.qualname: symbol for symbol in scan.symbols}


def test_scan_extracts_top_level_functions() -> None:
    """The ranking fixture exposes four top-level function symbols."""
    symbols = _scan_ranking_module()

    assert tuple(symbols) == (
        "normalize_features",
        "score_user",
        "rank_candidates",
        "debug_dump",
    )


def test_scan_records_imports_and_constants() -> None:
    """Imports and simple constants are available for later sidecar generation."""
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    scan = scan_module(module)

    assert [record.imported_names for record in scan.imports] == [
        ("annotations",),
        ("Event", "Score", "User"),
    ]
    assert [(record.name, record.kind) for record in scan.constants] == [
        ("DEFAULT_WEIGHT", "literal_constant")
    ]


def test_scan_records_symbol_references_and_globals() -> None:
    """Referenced names distinguish helper calls and global constants."""
    symbols = _scan_ranking_module()
    score_user = symbols["score_user"]

    assert "normalize_features" in score_user.referenced_names
    assert "DEFAULT_WEIGHT" in score_user.uses_globals
    assert "features" in score_user.local_names


def test_scan_keeps_annotation_names_out_of_runtime_globals(tmp_path: Path) -> None:
    """Type-only names are copied for sidecars without becoming runtime dependencies."""
    module_path = tmp_path / "typed.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def identity(value: T) -> T:",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="typed", path=module_path))
    identity = scan.symbols[0]

    assert "T" in identity.referenced_names
    assert identity.uses_globals == ()


def test_scan_records_import_aliases_constants_classes_and_methods(tmp_path: Path) -> None:
    """Scanner records V1 facts across imports, constants, classes, and methods."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import math as m",
                "import xml.etree.ElementTree",
                "from .local import value as local_value",
                "from .everything import *",
                "",
                "NEGATIVE = -1",
                "PAIR = (1, 'a')",
                "DYNAMIC = make_value()",
                "UNKNOWN = BASE + 1",
                "TYPED: int = 3",
                "",
                "if True:",
                "    pass",
                "",
                "def outer(x: int, *items: str, flag: bool = True, **kwargs: str) -> int:",
                "    def nested() -> int:",
                "        return x",
                "    class Inner:",
                "        pass",
                "    return eval('1')",
                "",
                "class Dynamic(metaclass=Meta):",
                "    def __getattr__(self, name: str) -> object:",
                "        return name",
                "",
                "class Worker(Base):",
                "    @classmethod",
                "    def make(cls, value: int) -> Worker:",
                "        return cls()",
                "",
                "    @staticmethod",
                "    def helper(value: int) -> int:",
                "        return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="sample", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert [record.imported_names for record in scan.imports] == [
        ("annotations",),
        ("m",),
        ("xml",),
        ("local_value",),
        (),
    ]
    assert [(record.name, record.kind) for record in scan.constants] == [
        ("NEGATIVE", "literal_constant"),
        ("PAIR", "literal_constant"),
        ("DYNAMIC", "runtime_dynamic"),
        ("UNKNOWN", "unknown"),
        ("TYPED", "literal_constant"),
    ]
    assert scan.top_level_statement_lines == (13,)
    assert symbols["outer"].arg_count == EXPECTED_OUTER_ARG_COUNT
    assert symbols["outer"].annotated_arg_count == EXPECTED_OUTER_ARG_COUNT
    assert {blocker.code for blocker in symbols["outer"].blockers} == {
        "DYN_EVAL",
        "NESTED_SYMBOL",
    }
    assert {blocker.code for blocker in symbols["Dynamic"].blockers} == {"DYN_CLASS_MONKEYPATCH"}
    assert symbols["Worker"].uses_globals == ("Base",)
    assert symbols["Worker.make"].decorators == ("classmethod",)
    assert symbols["Worker.helper"].decorators == ("staticmethod",)


def test_scan_populates_extended_function_class_and_method_facts(tmp_path: Path) -> None:
    """Extended symbol facts preserve source-level callable and class shape."""
    module_path = tmp_path / "extended.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import typing as typ",
                "",
                "class Base:",
                "    pass",
                "",
                "def fallback() -> int:",
                "    return 1",
                "",
                "class Box[T](Base):",
                "    count: int = 0",
                "    label: typ.ClassVar[str] = 'box'",
                "",
                "    def method(self, item: list[T], value: int = fallback()) -> T:",
                "        return helper(item)",
                "",
                "    @staticmethod",
                "    def util(value: typ.Any) -> None:",
                "        return None",
                "",
                "    @classmethod",
                "    async def build(cls, value: dict[str, T]) -> Box[T]:",
                "        return cls()",
                "",
                "def outer[T](first: T, /, second: int = 1, *items: str, "
                "flag: bool = True, **kwargs: object) -> T:",
                "    def nested() -> int:",
                "        yield 1",
                "        hidden()",
                "    return first",
                "",
                "def shaped[T: (str, bytes), *Ts, **P](value: T) -> T:",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="extended", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert symbols["Box"].binding_kind == "class"
    assert symbols["Box"].execution_kind == "class"
    assert symbols["Box"].type_parameters == ("T",)
    assert symbols["Box"].base_names == ("Base",)
    assert [
        (field.name, field.annotation, field.default_source, field.class_variable)
        for field in symbols["Box"].fields
    ] == [
        ("count", "int", "0", False),
        ("label", "typ.ClassVar[str]", "'box'", True),
    ]

    method = symbols["Box.method"]
    assert method.owner_class == "Box"
    assert method.binding_kind == "instance_method"
    assert method.execution_kind == "sync"
    assert method.return_annotation == "T"
    assert method.called_names == ("fallback", "helper")
    assert method.called_paths == ("fallback", "helper")
    assert method.annotation_names == ("list", "T", "int")
    assert [
        (parameter.name, parameter.kind, parameter.annotation, parameter.default_source)
        for parameter in method.parameters
    ] == [
        ("self", "positional", None, None),
        ("item", "positional", "list[T]", None),
        ("value", "positional", "int", "fallback()"),
    ]

    assert symbols["Box.util"].binding_kind == "staticmethod"
    assert symbols["Box.util"].declaration_start_lineno is not None
    assert symbols["Box.util"].declaration_start_lineno < symbols["Box.util"].lineno
    assert symbols["Box.util"].has_any_annotation is True
    assert symbols["Box.util"].annotation_names == ("typ.Any", "typ", "None")
    assert symbols["Box.build"].binding_kind == "classmethod"
    assert symbols["Box.build"].execution_kind == "coroutine"
    assert symbols["Box.build"].return_annotation == "Box[T]"

    outer = symbols["outer"]
    assert outer.owner_class is None
    assert outer.binding_kind == "module"
    assert outer.execution_kind == "sync"
    assert outer.type_parameters == ("T",)
    assert [
        (record.name, record.kind, record.declaration) for record in outer.type_parameter_records
    ] == [
        ("T", "type_var", "T"),
    ]
    assert outer.called_names == ()
    assert outer.called_paths == ()
    assert [
        (parameter.name, parameter.kind, parameter.annotation, parameter.default_source)
        for parameter in outer.parameters
    ] == [
        ("first", "positional_only", "T", None),
        ("second", "positional", "int", "1"),
        ("items", "vararg", "str", None),
        ("flag", "keyword_only", "bool", "True"),
        ("kwargs", "kwarg", "object", None),
    ]
    assert [
        (record.name, record.kind, record.declaration)
        for record in symbols["shaped"].type_parameter_records
    ] == [
        ("T", "type_var", "T: (str, bytes)"),
        ("Ts", "type_var_tuple", "*Ts"),
        ("P", "param_spec", "**P"),
    ]


def test_qualified_descriptor_decorator_is_not_treated_as_builtin(
    tmp_path: Path,
) -> None:
    """A custom attribute named staticmethod remains an unknown instance decorator."""
    module_path = tmp_path / "qualified_decorator.py"
    module_path.write_text(
        """class Worker:
    @custom.staticmethod
    async def score(values: list[int]) -> int:
        return len(values)
""",
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="qualified_decorator", path=module_path))
    method = next(symbol for symbol in scan.symbols if symbol.id.qualname == "Worker.score")

    assert method.binding_kind == "instance_method"
    assert method.decorators == ("custom.staticmethod",)


def test_scan_detects_execution_kinds_without_nested_symbol_bodies(tmp_path: Path) -> None:
    """Generator and coroutine shape belongs only to the scanned symbol body."""
    module_path = tmp_path / "execution.py"
    module_path.write_text(
        "\n".join(
            [
                "def sync() -> int:",
                "    def nested():",
                "        yield 1",
                "    return 1",
                "",
                "def generator() -> int:",
                "    yield from range(3)",
                "",
                "async def coroutine() -> int:",
                "    return 1",
                "",
                "async def async_generator() -> int:",
                "    yield 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="execution", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert symbols["sync"].execution_kind == "sync"
    assert symbols["generator"].execution_kind == "generator"
    assert symbols["coroutine"].execution_kind == "coroutine"
    assert symbols["async_generator"].execution_kind == "async_generator"


def test_scan_treats_exception_targets_as_coroutine_locals(tmp_path: Path) -> None:
    """Exception aliases stay local in async task handlers."""
    module_path = tmp_path / "tracked_task.py"
    module_path.write_text(
        "\n".join(
            [
                "async def _run_tracked_task(task):",
                "    try:",
                "        await task",
                "    except Exception as exc:",
                "        record_failure(task, exc)",
                "        return exc",
                "    return None",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="tracked_task", path=module_path))
    tracked_task = scan.symbols[0]

    assert tracked_task.execution_kind == "coroutine"
    assert "exc" in tracked_task.local_names
    assert "exc" in tracked_task.referenced_names
    assert "exc" not in tracked_task.uses_globals
    assert tracked_task.uses_globals == ("record_failure",)


def test_scan_records_ordered_call_sites_suspensions_and_runtime_imports(
    tmp_path: Path,
) -> None:
    """Scanner facts retain call and suspension syntax without runtime values."""
    module_path = tmp_path / "scanner_facts.py"
    module_path.write_text(
        "\n".join(
            [
                "async def consume(worker, source):",
                "    import json as json_lib",
                "    from collections import deque",
                "    start()",
                "    await worker.fetch()",
                "    await wrap(inner())",
                "    async for item in source.stream():",
                "        handle(item)",
                "    async with worker.lock():",
                "        yield item",
                "    return",
                "",
                "def produce():",
                "    yield from fallback()",
                "    yield transform()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="scanner_facts", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    consume = symbols["consume"]
    assert [
        (call.target, call.invocation_mode, call.lineno, call.col_offset)
        for call in consume.call_sites
    ] == [
        ("start", "ordinary", 4, 4),
        ("worker.fetch", "awaited", 5, 10),
        ("wrap", "awaited", 6, 10),
        ("inner", "ordinary", 6, 15),
        ("source.stream", "async_iteration", 7, 22),
        ("handle", "ordinary", 8, 8),
        ("worker.lock", "ordinary", 9, 15),
    ]
    assert [(record.imported_names, record.lineno) for record in consume.runtime_imports] == [
        (("json_lib",), 2),
        (("deque",), 3),
    ]
    assert [(point.kind, point.lineno) for point in consume.suspension_points] == [
        ("await", 5),
        ("await", 6),
        ("async_for", 7),
        ("async_with", 9),
        ("yield", 10),
    ]

    produce = symbols["produce"]
    assert [(call.target, call.invocation_mode, call.lineno) for call in produce.call_sites] == [
        ("fallback", "ordinary", 14),
        ("transform", "ordinary", 15),
    ]
    assert [(point.kind, point.lineno) for point in produce.suspension_points] == [
        ("yield_from", 14),
        ("yield", 15),
    ]


def test_scan_collects_non_name_store_lexical_bindings(tmp_path: Path) -> None:
    """Imports and structural pattern captures bind locals without false globals."""
    module_path = tmp_path / "binding_forms.py"
    module_path.write_text(
        "\n".join(
            [
                "def classify(payload):",
                "    import json as json_lib",
                "    from collections import Counter",
                "    match payload:",
                "        case {'items': [first, *rest], 'meta': meta, **remaining}:",
                "            counter = Counter(rest)",
                "            return json_lib.dumps((first, meta, remaining, counter))",
                "        case Counter() as fallback:",
                "            return fallback",
                "    return None",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="binding_forms", path=module_path))
    classify = scan.symbols[0]

    assert {
        "Counter",
        "fallback",
        "first",
        "json_lib",
        "meta",
        "remaining",
        "rest",
    }.issubset(classify.local_names)
    assert "Counter" in classify.referenced_names
    assert "json_lib" in classify.referenced_names
    assert classify.uses_globals == ()


def test_scan_preserves_nested_expression_scope_bindings(tmp_path: Path) -> None:
    """Comprehension and lambda binders do not leak into enclosing locals."""
    module_path = tmp_path / "nested_expression_scopes.py"
    module_path.write_text(
        "\n".join(
            [
                "def summarize(values):",
                (
                    "    callbacks = [lambda item, offset=scale: transform(item, offset) "
                    "for item in values]"
                ),
                "    pairs = {name: transform(name, scale) for name in values}",
                "    unique = {transform(item, scale) for item in values if predicate(item)}",
                "    stream = (transform(item, scale) for item in values if predicate(item))",
                "    async def nested():",
                "        return await async_transform(values)",
                "    return callbacks, pairs, unique, stream, nested",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="nested_expression_scopes", path=module_path))
    summarize = scan.symbols[0]

    assert "item" not in summarize.local_names
    assert "name" not in summarize.local_names
    assert "item" not in summarize.uses_globals
    assert "name" not in summarize.uses_globals
    assert "nested" in summarize.local_names
    assert summarize.uses_globals == ("predicate", "scale", "transform")


def test_scan_has_any_annotation_detects_real_any_import_forms(tmp_path: Path) -> None:
    """Any detection recognizes typing aliases, strings, and qualified references."""
    module_path = tmp_path / "any_annotations.py"
    typing_any_import = f"from typing import {chr(65)}ny as TypingAny"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                typing_any_import,
                "import typing as typ",
                "import typing_extensions as tx",
                "",
                "class LocalAny:",
                "    pass",
                "",
                "def direct(value: TypingAny) -> int:",
                "    return 1",
                "",
                "def qualified(value: typ.Any) -> int:",
                "    return 1",
                "",
                "def extension(value: tx.Any) -> int:",
                "    return 1",
                "",
                "def stringed(value: 'list[typ.Any]') -> int:",
                "    return 1",
                "",
                "def local(value: LocalAny) -> int:",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="any_annotations", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert symbols["direct"].has_any_annotation is True
    assert symbols["direct"].any_annotation_sources == ("TypingAny",)
    assert symbols["qualified"].has_any_annotation is True
    assert symbols["extension"].has_any_annotation is True
    assert symbols["stringed"].has_any_annotation is True
    assert symbols["local"].has_any_annotation is False
    assert symbols["local"].any_annotation_sources == ()


def test_scan_retains_legacy_typevars_as_scope_parameters(tmp_path: Path) -> None:
    """Imported TypeVar factories produce explicit scope bindings for users."""
    module_path = tmp_path / "legacy_typevars.py"
    module_path.write_text(
        "\n".join(
            [
                "from typing import Generic, TypeVar as TV",
                "import typing_extensions as tx",
                "",
                "T = TV('T')",
                "P = tx.ParamSpec('P')",
                "",
                "def identity(value: T) -> T:",
                "    return value",
                "",
                "class Box(Generic[T]):",
                "    def get(self, value: T) -> T:",
                "        return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="legacy_typevars", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert symbols["identity"].type_parameters == ()
    assert symbols["identity"].scope_type_parameters == ("T",)
    assert [
        (record.name, record.kind, record.declaration)
        for record in symbols["identity"].scope_type_parameter_records
    ] == [("T", "type_var", "T = TV('T')")]
    assert symbols["Box"].scope_type_parameters == ("T",)
    assert symbols["Box.get"].scope_type_parameters == ("T",)


def test_scan_raises_for_syntax_errors(tmp_path: Path) -> None:
    """Syntax errors are surfaced to the caller during scanning."""
    module_path = tmp_path / "broken.py"
    module_path.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(SyntaxError):
        scan_module(ModuleId(name="broken", path=module_path))
