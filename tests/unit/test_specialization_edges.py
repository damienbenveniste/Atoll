"""Edge tests for conservative generic-specialization parsing and inference."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from atoll.analysis import typed_regions as specialization_analysis
from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import (
    ModuleId,
    ModuleScan,
    ParameterRecord,
    RegionSpecialization,
    SymbolRecord,
    TypeBinding,
)


def _private(name: str) -> object:
    return vars(specialization_analysis)[name]


_annotation_expression = cast(Callable[[str], ast.expr | None], _private("_annotation_expression"))
_guard_shape = cast(
    Callable[[str, frozenset[str]], tuple[tuple[str, ...], bool] | None],
    _private("_guard_shape"),
)
_annotation_has_forbidden_path = cast(
    Callable[[str, frozenset[str]], bool],
    _private("_annotation_has_forbidden_path"),
)
_annotation_uses_any = cast(
    Callable[[str, tuple[str, ...]], bool],
    _private("_annotation_uses_any"),
)
_substitute_annotation = cast(
    Callable[[str, tuple[tuple[str, str], ...]], str],
    _private("_substitute_annotation"),
)
_infer_from_parameter_annotation = cast(
    Callable[[str, str, tuple[str, ...]], tuple[tuple[str, str], ...] | None],
    _private("_infer_from_parameter_annotation"),
)
_argument_annotation = cast(
    Callable[[ast.expr, dict[str, str]], str | None],
    _private("_argument_annotation"),
)
_semantic_any_paths = cast(
    Callable[[ModuleScan], frozenset[str]],
    _private("_semantic_any_paths"),
)
_concrete_base_substitution = cast(
    Callable[
        [str, dict[str, SymbolRecord], frozenset[str]],
        tuple[SymbolRecord | None, tuple[tuple[str, str], ...] | None],
    ],
    _private("_concrete_base_substitution"),
)
_parameter_type_binding = cast(
    Callable[[SymbolRecord, ParameterRecord, tuple[str, ...]], TypeBinding],
    _private("_parameter_type_binding"),
)
_is_concrete = cast(
    Callable[[str, tuple[str, ...], tuple[str, ...]], bool],
    _private("_is_concrete"),
)
_local_bound_names = cast(
    Callable[[ast.FunctionDef | ast.AsyncFunctionDef], set[str]],
    _private("_local_bound_names"),
)
_ambiguous_module_bindings = cast(
    Callable[[ast.Module], frozenset[str]],
    _private("_ambiguous_module_bindings"),
)


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _specializations(tmp_path: Path, source: str) -> tuple[RegionSpecialization, ...]:
    path = tmp_path / "specialization_edges.py"
    path.write_text(source, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name="specialization_edges", path=path)))
    return tuple(item for region in scan.typed_regions for item in region.specializations)


def test_annotation_guard_parser_handles_optional_forbidden_and_malformed_inputs() -> None:
    """Only nominal, None, and union forms produce constant-time guard shapes."""
    forbidden = frozenset({"typing.Any", "DynamicValue"})

    assert _annotation_expression("[") is None
    assert _annotation_expression("'['") is None
    assert _guard_shape("typing.Optional[int]", forbidden) == (("int",), True)
    assert _guard_shape("None", forbidden) == ((), True)
    assert _guard_shape("list[int]", forbidden) is None
    assert _guard_shape("typing.Any", forbidden) is None
    assert _guard_shape("DynamicValue", forbidden) is None
    assert _guard_shape("[", forbidden) is None
    assert _annotation_has_forbidden_path("typing.Any | int", forbidden) is True
    assert _annotation_has_forbidden_path("[", forbidden) is True
    assert _annotation_uses_any("[", ("T",)) is False
    assert _substitute_annotation("[", (("T", "int"),)) == "["


def test_closed_call_annotation_inference_covers_literals_names_and_constructors() -> None:
    """Closed calls infer only statically visible nominal argument annotations."""
    assert _argument_annotation(_expression("None"), {}) == "None"
    assert _argument_annotation(_expression("-1"), {}) == "int"
    assert _argument_annotation(_expression("value"), {"value": "Payload"}) == "Payload"
    assert _argument_annotation(_expression("missing"), {}) is None
    assert _argument_annotation(_expression("Payload()"), {}) == "Payload"
    assert _argument_annotation(_expression("models.Payload()"), {}) == "models.Payload"
    assert _argument_annotation(_expression("Payload(1)"), {}) is None
    assert _argument_annotation(_expression("value + 1"), {}) is None
    assert _infer_from_parameter_annotation("T | None", "int", ("T",)) == (("T", "int"),)
    assert _infer_from_parameter_annotation("T | None", "None", ("T",)) is None
    assert _infer_from_parameter_annotation("tuple[T]", "int", ("T",)) is None
    assert _infer_from_parameter_annotation("T | U", "int", ("T", "U")) is None
    assert _infer_from_parameter_annotation("[", "int", ("T",)) is None


def test_semantic_any_paths_include_module_and_direct_aliases(tmp_path: Path) -> None:
    """Both module aliases and direct Any aliases are rejected as runtime types."""
    path = tmp_path / "typing_aliases.py"
    any_import = f"from typing import {chr(65)}ny as DynamicValue"
    path.write_text(
        f"""import typing as tp
import typing_extensions as te
{any_import}

def identity(value: int) -> int:
    return value
""",
        encoding="utf-8",
    )
    scan = scan_module(ModuleId(name="typing_aliases", path=path))
    invalid_import = replace(scan.imports[0], source_text="[")

    paths = _semantic_any_paths(replace(scan, imports=(*scan.imports, invalid_import)))

    assert {"tp.Any", "te.Any", "DynamicValue"} <= paths


def test_base_substitution_rejects_malformed_arity_and_non_typevars(tmp_path: Path) -> None:
    """Malformed bases, wrong arity, containers, and ParamSpec bases stay generic."""
    path = tmp_path / "generic_bases.py"
    path.write_text(
        """class Pair[T, U]:
    pass

class Plain:
    pass

class CallableBox[**P]:
    pass
""",
        encoding="utf-8",
    )
    scan = scan_module(ModuleId(name="generic_bases", path=path))
    classes = {symbol.id.qualname: symbol for symbol in scan.symbols if symbol.kind == "class"}
    forbidden = frozenset[str]()

    assert _concrete_base_substitution("Pair[", classes, forbidden) == (None, None)
    assert _concrete_base_substitution("Pair", classes, forbidden) == (None, None)
    assert _concrete_base_substitution("Missing[int]", classes, forbidden) == (None, None)
    assert _concrete_base_substitution("Pair[int]", classes, forbidden) == (None, None)
    assert _concrete_base_substitution("Pair[list[int], str]", classes, forbidden) == (
        None,
        None,
    )
    assert _concrete_base_substitution("Plain[int]", classes, forbidden) == (None, None)
    assert _concrete_base_substitution("CallableBox[int]", classes, forbidden) == (None, None)
    base, substitutions = _concrete_base_substitution("Pair[int, str]", classes, forbidden)
    assert base == classes["Pair"]
    assert substitutions == (("T", "int"), ("U", "str"))


def test_private_type_binding_and_scope_helpers_reject_invalid_inputs(tmp_path: Path) -> None:
    """Invalid annotations and ambiguous local bindings fail conservatively."""
    path = tmp_path / "binding_helpers.py"
    path.write_text("def value(raw: int) -> int:\n    return raw\n", encoding="utf-8")
    symbol = scan_module(ModuleId(name="binding_helpers", path=path)).symbols[0]
    missing_annotation = ParameterRecord(
        name="raw",
        kind="positional",
        annotation=None,
        default_source=None,
    )

    with pytest.raises(ValueError, match="requires an annotation"):
        _parameter_type_binding(symbol, missing_annotation, ())
    assert _is_concrete("[", (), ()) is False

    function = ast.parse(
        """async def use(value: int, *items: int, **values: int) -> int:
    import os as local_os
    from pathlib import Path as LocalPath
    class LocalClass:
        pass
    def nested() -> None:
        pass
    async def async_nested() -> None:
        pass
    assigned = value
    return assigned
"""
    ).body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    assert {
        "value",
        "items",
        "values",
        "local_os",
        "LocalPath",
        "LocalClass",
        "nested",
        "async_nested",
        "assigned",
    } <= _local_bound_names(function)

    ambiguous = _ambiguous_module_bindings(
        ast.parse(
            """import first as repeated
repeated: int = 1
left = right = 1
def duplicate() -> None:
    pass
def duplicate() -> None:
    pass
"""
        )
    )
    assert {"repeated", "duplicate"} <= ambiguous


def test_closed_call_and_subclass_rejection_paths_remain_interpreted(tmp_path: Path) -> None:
    """Invalid calls and non-constant guard shapes never produce specializations."""
    specializations = _specializations(
        tmp_path,
        """def too_many[T](value: T) -> T:
    return value

TOO_MANY = too_many(1, 2)

def expanded[T](value: T) -> T:
    return value

VALUES = {"value": 1}
EXPANDED = expanded(**VALUES)

def wrong_keyword[T](value: T) -> T:
    return value

WRONG_KEYWORD = wrong_keyword(other=1)

def missing[T](value: T) -> T:
    return value

MISSING = missing()

def variadic[T](*values: T) -> T:
    return values[0]

VARIADIC = variadic(1)

def conflicting[T](left: T, right: T) -> T:
    return left

CONFLICTING = conflicting(1, "value")

def return_only[T]() -> T:
    raise RuntimeError

RETURN_ONLY = return_only()

def ambiguous[T](value: T) -> T:
    return value

ambiguous = ambiguous
AMBIGUOUS = ambiguous(1)

class Dynamic:
    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)

    def identity[T](self, value: T) -> T:
        return value

    def use(self, value: int) -> int:
        return self.identity(value)

class GenericBox[T]:
    def default(self, value: T = 1) -> T:
        return value

    def many(self, *values: T) -> T:
        return values[0]

    def container(self, values: list[T]) -> list[T]:
        return values

class IntBox(GenericBox[int]):
    pass
""",
    )

    assert specializations == ()
