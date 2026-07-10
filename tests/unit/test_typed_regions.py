"""Tests for backend-neutral typed-region formation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.typed_regions import build_directed_region_slice
from atoll.models import ModuleId, ModuleScan, RegionSpecialization, SymbolId


def _scan_source(tmp_path: Path, name: str, lines: list[str]) -> ModuleScan:
    module_path = tmp_path / f"{name}.py"
    module_path.write_text("\n".join(lines), encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId(name=name, path=module_path)))


def _specializations(
    tmp_path: Path, name: str, lines: list[str]
) -> tuple[RegionSpecialization, ...]:
    scan = _scan_source(tmp_path, name, lines)
    return tuple(
        specialization for region in scan.typed_regions for specialization in region.specializations
    )


def test_directed_region_slice_keeps_runtime_boundaries_outside_native_unit(
    tmp_path: Path,
) -> None:
    """A hot root expands only through dependencies proven to require one unit."""
    scan = _scan_source(
        tmp_path,
        "directed_slice",
        [
            "def helper(value: int) -> int:",
            "    return value + 1",
            "",
            "def hot(value: int) -> int:",
            "    return helper(value)",
            "",
        ],
    )
    region = next(
        item
        for item in scan.typed_regions
        if {member.id.qualname for member in item.members} == {"helper", "hot"}
    )
    hot = next(member.id for member in region.members if member.id.qualname == "hot")
    helper = next(member.id for member in region.members if member.id.qualname == "helper")

    sliced = build_directed_region_slice(region, hot)
    repeated = build_directed_region_slice(region, hot)

    assert sliced == repeated
    assert tuple(member.id for member in sliced.members) == (hot,)
    assert sliced.atomic_class is False
    dependency = next(item for item in sliced.dependencies if item.dst == helper)
    assert dependency.invocation_mode == "ordinary"
    assert dependency.requires_same_unit is False

    required_region = replace(
        region,
        dependencies=tuple(
            replace(item, requires_same_unit=True) if item.dst == helper else item
            for item in region.dependencies
        ),
    )
    required_slice = build_directed_region_slice(required_region, hot)
    assert {member.id for member in required_slice.members} == {hot, helper}


def test_directed_region_slice_rejects_invalid_roots_dependencies_and_bindings(
    tmp_path: Path,
) -> None:
    """Malformed directed plans fail before backend lowering or runtime binding."""
    scan = _scan_source(
        tmp_path,
        "invalid_directed_slice",
        [
            "def helper(value: int) -> int:",
            "    return value + 1",
            "",
            "def hot(value: int) -> int:",
            "    return helper(value)",
            "",
            "class Worker:",
            "    def __init__(self, value: int) -> None:",
            "        self.value = value",
            "",
        ],
    )
    function_region = next(
        region
        for region in scan.typed_regions
        if {member.id.qualname for member in region.members} == {"helper", "hot"}
    )
    hot = next(member.id for member in function_region.members if member.id.qualname == "hot")
    helper = next(member.id for member in function_region.members if member.id.qualname == "helper")

    with pytest.raises(ValueError, match="root is outside region"):
        build_directed_region_slice(
            function_region,
            SymbolId("invalid_directed_slice", "missing"),
        )

    missing_dependency = replace(
        function_region,
        dependencies=tuple(
            replace(
                dependency,
                dst=SymbolId("invalid_directed_slice", "missing"),
                requires_same_unit=True,
            )
            if dependency.dst == helper
            else dependency
            for dependency in function_region.dependencies
        ),
    )
    with pytest.raises(ValueError, match="required same-unit dependency is outside region"):
        build_directed_region_slice(missing_dependency, hot)

    hot_binding = next(binding for binding in function_region.bindings if binding.source == hot)
    duplicate_binding_region = replace(
        function_region,
        bindings=(*function_region.bindings, hot_binding),
    )
    with pytest.raises(ValueError, match="must have exactly one binding"):
        build_directed_region_slice(duplicate_binding_region, hot)

    class_region = next(region for region in scan.typed_regions if region.atomic_class)
    constructor = next(
        member.id for member in class_region.members if member.id.qualname == "Worker.__init__"
    )
    with pytest.raises(ValueError, match="not independently bindable"):
        build_directed_region_slice(class_region, constructor)


def test_typed_regions_group_safe_class_methods_atomically(tmp_path: Path) -> None:
    """A safe typed class and its connected methods form one atomic region."""
    module_path = tmp_path / "regions.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "class Worker:",
                "    bias: int",
                "",
                "    def __init__(self, bias: int) -> None:",
                "        self.bias = bias",
                "",
                "    def adjust(self, value: int) -> int:",
                "        return value + self.bias",
                "",
                "    async def score(self, value: int) -> int:",
                "        return self.adjust(value)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    first = enrich_island_analysis(scan_module(ModuleId(name="regions", path=module_path)))
    second = enrich_island_analysis(scan_module(ModuleId(name="regions", path=module_path)))

    class_region = next(region for region in first.typed_regions if region.atomic_class)
    assert tuple(member.id.qualname for member in class_region.members) == (
        "Worker",
        "Worker.__init__",
        "Worker.adjust",
        "Worker.score",
    )
    assert tuple(binding.source.qualname for binding in class_region.bindings) == ("Worker",)
    assert {member.execution_kind for member in class_region.members} == {
        "class",
        "sync",
        "coroutine",
    }
    assert any(
        dependency.kind == "calls_method"
        and getattr(dependency.dst, "qualname", None) == "Worker.adjust"
        for dependency in class_region.dependencies
    )
    assert class_region.id == second.typed_regions[0].id
    assert class_region.source_hash == second.typed_regions[0].source_hash

    score = next(
        member.id for member in class_region.members if member.id.qualname == "Worker.score"
    )
    score_slice = build_directed_region_slice(class_region, score)
    assert tuple(binding.source for binding in score_slice.bindings) == (score,)
    assert score_slice.bindings[0].kind == "instance_method"
    assert score_slice.bindings[0].execution_kind == "coroutine"


def test_dynamic_class_downgrades_to_method_regions(tmp_path: Path) -> None:
    """Unsafe class identity does not prevent an independent safe method region."""
    module_path = tmp_path / "dynamic.py"
    module_path.write_text(
        "\n".join(
            [
                "class Dynamic:",
                "    def safe(self, value: int) -> int:",
                "        return value * value",
                "",
                "    def __getattr__(self, name: str) -> object:",
                "        raise AttributeError(name)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = enrich_island_analysis(scan_module(ModuleId(name="dynamic", path=module_path)))

    safe_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "Dynamic.safe" for member in region.members)
    )
    assert safe_region.atomic_class is False
    assert tuple(binding.source.qualname for binding in safe_region.bindings) == ("Dynamic.safe",)
    assert safe_region.bindings[0].kind == "instance_method"
    assert any(
        decision.target == "dynamic::Dynamic"
        and decision.action == "fallback"
        and "class remains interpreted" in decision.reason
        for decision in safe_region.decisions
    )


def test_module_time_instance_downgrades_class_but_keeps_safe_methods(
    tmp_path: Path,
) -> None:
    """A pre-shim instance keeps the source class identity and permits method binding."""
    scan = _scan_source(
        tmp_path,
        "module_instance",
        [
            "class Worker:",
            "    value: int",
            "",
            "    def __init__(self, value: int) -> None:",
            "        self.value = value",
            "",
            "    def double(self) -> int:",
            "        return self.value * 2",
            "",
            "WORKER = Worker(3)",
            "",
        ],
    )

    assert not any(region.atomic_class for region in scan.typed_regions)
    method_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "Worker.double" for member in region.members)
    )
    assert tuple(binding.source.qualname for binding in method_region.bindings) == (
        "Worker.double",
    )
    assert any(
        decision.target == "module_instance::Worker"
        and decision.action == "fallback"
        and "module-time code retains" in decision.reason
        for decision in method_region.decisions
    )


def test_registration_decorator_keeps_original_class_for_method_binding(tmp_path: Path) -> None:
    """A class decorator prevents replacement without poisoning an eligible method."""
    scan = _scan_source(
        tmp_path,
        "registered_class",
        [
            "REGISTRY: list[type[object]] = []",
            "",
            "def register(cls: type[object]) -> type[object]:",
            "    REGISTRY.append(cls)",
            "    return cls",
            "",
            "@register",
            "class Worker:",
            "    def double(self, value: int) -> int:",
            "        return value * 2",
            "",
        ],
    )

    method_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "Worker.double" for member in region.members)
    )
    assert method_region.atomic_class is False
    assert method_region.bindings[0].kind == "instance_method"
    assert any(
        decision.target == "registered_class::Worker"
        and decision.action == "fallback"
        and "decorators may register" in decision.reason
        for decision in method_region.decisions
    )


@pytest.mark.parametrize(
    ("name", "owner", "lines", "reason"),
    [
        (
            "source_inheritance",
            "Child",
            [
                "class Base:",
                "    def value(self) -> int:",
                "        return 1",
                "",
                "class Child(Base):",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
            ],
            "source inheritance is outside",
        ),
        (
            "class_body_effect",
            "Worker",
            [
                "def create() -> int:",
                "    return 1",
                "",
                "class Worker:",
                "    value: int = create()",
                "",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
            ],
            "class body contains executable statements",
        ),
        (
            "method_default_effect",
            "Worker",
            [
                "def create() -> int:",
                "    return 1",
                "",
                "class Worker:",
                "    def double(self, value: int = create()) -> int:",
                "        return value * 2",
                "",
            ],
            "nonliteral default",
        ),
        (
            "post_class_import",
            "Worker",
            [
                "class Worker:",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
                "import math",
                "",
            ],
            "later import can expose",
        ),
        (
            "post_class_import_from",
            "Worker",
            [
                "class Worker:",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
                "from math import sqrt",
                "",
            ],
            "later import can expose",
        ),
        (
            "post_class_call",
            "Worker",
            [
                "class Factory:",
                "    pass",
                "",
                "class Worker:",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
                "FACTORY = Factory()",
                "",
            ],
            "later call can expose",
        ),
        (
            "method_runtime_decorator",
            "Worker",
            [
                "class Decorators:",
                "    @staticmethod",
                "    def cache(value: object) -> object:",
                "        return value",
                "",
                "class Worker:",
                "    def increment(self, value: int) -> int:",
                "        return value + 1",
                "",
                "    @Decorators.cache",
                "    def double(self, value: int) -> int:",
                "        return value * 2",
                "",
            ],
            "runtime decorator",
        ),
    ],
)
def test_atomic_class_safety_rejects_identity_and_definition_hazards(
    tmp_path: Path,
    name: str,
    owner: str,
    lines: list[str],
    reason: str,
) -> None:
    """Class replacement is rejected before copied declarations can execute twice."""
    scan = _scan_source(tmp_path, name, lines)

    assert not any(
        region.atomic_class and any(member.id.qualname == owner for member in region.members)
        for region in scan.typed_regions
    )
    assert any(
        any(member.owner_class == owner for member in region.members)
        and decision.action == "fallback"
        and reason in decision.reason
        for region in scan.typed_regions
        for decision in region.decisions
    )


def test_function_body_only_class_reference_does_not_escape_during_import(tmp_path: Path) -> None:
    """A global class lookup inside a later function observes the post-shim binding."""
    scan = _scan_source(
        tmp_path,
        "deferred_class_reference",
        [
            "class Worker:",
            "    def double(self, value: int) -> int:",
            "        return value * 2",
            "",
            "def deferred(value: int) -> int:",
            "    worker_type = Worker",
            "    return worker_type().double(value)",
            "",
        ],
    )

    assert any(
        region.atomic_class and any(member.id.qualname == "Worker" for member in region.members)
        for region in scan.typed_regions
    )


def test_deferred_import_after_class_does_not_escape_during_module_load(tmp_path: Path) -> None:
    """Imports inside later function bodies run only after the class shim is installed."""
    scan = _scan_source(
        tmp_path,
        "deferred_import",
        [
            "class Worker:",
            "    def double(self, value: int) -> int:",
            "        return value * 2",
            "",
            "async def deferred(value: int) -> int:",
            "    import math",
            "    return int(math.sqrt(value))",
            "",
        ],
    )

    assert any(
        region.atomic_class and any(member.id.qualname == "Worker" for member in region.members)
        for region in scan.typed_regions
    )


def test_dynamic_method_global_prevents_atomic_class_region(tmp_path: Path) -> None:
    """Atomic classes cannot depend on state omitted from their generated module."""
    scan = _scan_source(
        tmp_path,
        "dynamic_class_global",
        [
            "def load_factor() -> int:",
            "    return 2",
            "",
            "FACTOR = load_factor()",
            "",
            "class Worker:",
            "    def apply(self, value: int) -> int:",
            "        return value * FACTOR",
            "",
        ],
    )

    worker = next(symbol for symbol in scan.symbols if symbol.id.qualname == "Worker.apply")
    assert {blocker.code for blocker in worker.blockers} == {"DYNAMIC_GLOBAL_DEP"}
    assert not any(
        region.atomic_class and any(member.id.qualname == "Worker" for member in region.members)
        for region in scan.typed_regions
    )


def test_unannotated_staticmethod_becomes_non_atomic_boxed_region(tmp_path: Path) -> None:
    """Incomplete static methods remain visible without making their class atomic."""
    module_path = tmp_path / "static_method.py"
    module_path.write_text(
        "\n".join(
            [
                "class Worker:",
                "    @staticmethod",
                "    def parse(value) -> int:",
                "        return int(value)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = enrich_island_analysis(scan_module(ModuleId(name="static_method", path=module_path)))

    assert len(scan.typed_regions) == 1
    region = scan.typed_regions[0]
    assert region.atomic_class is False
    assert tuple(member.id.qualname for member in region.members) == ("Worker.parse",)
    decision = next(item for item in region.decisions if item.target.endswith("::Worker.parse"))
    assert decision.action == "box"
    assert decision.reason == "source callable has incomplete annotations"


def test_region_loss_ledger_keeps_explicit_any_and_generics_visible(tmp_path: Path) -> None:
    """Source Any and unresolved generic parameters are recorded rather than erased."""
    module_path = tmp_path / "typing_regions.py"
    typing_any_import = f"from typing import {chr(65)}ny as TypingAny"
    module_path.write_text(
        "\n".join(
            [
                "import typing",
                typing_any_import,
                "",
                "def dynamic(value: typing.Any) -> typing.Any:",
                "    return value",
                "",
                "def aliased(value: TypingAny) -> TypingAny:",
                "    return value",
                "",
                "class Namespace:",
                "    pass",
                "",
                "def local(value: Namespace.Any) -> Namespace.Any:",
                "    return value",
                "",
                "def identity[T](value: T) -> T:",
                "    return value",
                "",
                "class GenericBox[T]:",
                "    @staticmethod",
                "    def identity(value: T) -> T:",
                "        return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = enrich_island_analysis(scan_module(ModuleId(name="typing_regions", path=module_path)))
    decisions = {
        decision.target: decision for region in scan.typed_regions for decision in region.decisions
    }

    assert decisions["typing_regions::dynamic"].action == "box"
    assert decisions["typing_regions::aliased"].action == "box"
    assert decisions["typing_regions::local"].action == "preserve"
    assert decisions["typing_regions::identity"].action == "fallback"
    identity_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "identity" for member in region.members)
    )
    assert identity_region.members[0].type_parameters == ("T",)
    assert identity_region.members[0].parameters[0].annotation == "T"
    aliased_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "aliased" for member in region.members)
    )
    assert all(
        not binding.concrete
        for binding in aliased_region.type_bindings
        if binding.source in {"parameter", "return"}
    )
    local_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "local" for member in region.members)
    )
    assert all(
        binding.concrete
        for binding in local_region.type_bindings
        if binding.source in {"parameter", "return"}
    )
    generic_region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "GenericBox" for member in region.members)
    )
    generic_method = next(
        member for member in generic_region.members if member.id.qualname == "GenericBox.identity"
    )
    assert generic_method.type_parameters == ()
    assert generic_method.scope_type_parameters == ("T",)
    assert generic_method.source_text.lstrip().startswith("@staticmethod")
    assert (
        next(
            decision
            for decision in generic_region.decisions
            if decision.target == "typing_regions::GenericBox.identity"
        ).action
        == "fallback"
    )
    value_binding = next(
        binding
        for binding in generic_region.type_bindings
        if binding.name == "GenericBox.identity.value"
    )
    assert value_binding.concrete is False
    assert any(
        dependency.kind == "annotation" and dependency.role == "typing"
        for dependency in generic_region.dependencies
    )


def test_direct_concrete_subclass_specializes_method_without_rewriting_base(
    tmp_path: Path,
) -> None:
    """A non-generic direct subclass adds concrete evidence while the base stays generic."""
    lines = [
        "class GenericBox[T]:",
        "    def echo(self, value: T) -> T:",
        "        return value",
        "",
        "class IntBox(GenericBox[int]):",
        "    pass",
        "",
    ]
    first = _scan_source(tmp_path, "subclass_specialization", lines)
    second = _scan_source(tmp_path, "subclass_specialization", lines)

    region = next(
        region
        for region in first.typed_regions
        if any(member.id.qualname == "GenericBox.echo" for member in region.members)
    )
    specialization = region.specializations[0]
    second_specialization = next(
        specialization
        for region in second.typed_regions
        for specialization in region.specializations
        if specialization.source_member.qualname == "GenericBox.echo"
    )

    assert specialization.id == second_specialization.id
    assert specialization.origin == "concrete_subclass"
    assert specialization.source_owner_class == "GenericBox"
    assert specialization.target_owner_class == "IntBox"
    assert specialization.substitutions == (("T", "int"),)
    assert specialization.guards[0].parameter_name == "value"
    assert specialization.guards[0].positional_index == 1
    assert specialization.guards[0].nominal_type_paths == ("int",)
    assert specialization.guards[0].allow_none is False
    assert all(binding.concrete for binding in specialization.type_bindings)
    assert {binding.annotation for binding in specialization.type_bindings} == {"int"}
    assert (
        next(
            decision
            for decision in region.decisions
            if decision.target == "subclass_specialization::GenericBox.echo"
        ).action
        == "fallback"
    )
    assert (
        next(
            binding for binding in region.type_bindings if binding.name == "GenericBox.echo.value"
        ).annotation
        == "T"
    )


def test_subclass_specialization_supports_union_none_and_nominal_guards(
    tmp_path: Path,
) -> None:
    """Scalar, None, and nominal union annotations produce constant-time guards."""
    specializations = _specializations(
        tmp_path,
        "union_nominal",
        [
            "class Payload:",
            "    pass",
            "",
            "class GenericBox[T]:",
            "    def maybe(self, value: T | None) -> T | None:",
            "        return value",
            "",
            "class PayloadBox(GenericBox[Payload]):",
            "    pass",
            "",
        ],
    )

    specialization = specializations[0]

    assert specialization.substitutions == (("T", "Payload"),)
    assert specialization.guards[0].annotation == "Payload | None"
    assert specialization.guards[0].nominal_type_paths == ("Payload",)
    assert specialization.guards[0].allow_none is True
    assert {binding.annotation for binding in specialization.type_bindings} == {"Payload | None"}


def test_subclass_specialization_rejects_containers_and_unresolved_generics(
    tmp_path: Path,
) -> None:
    """Container and still-generic subclass evidence is not treated as concrete."""
    specializations = _specializations(
        tmp_path,
        "rejected_subclasses",
        [
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class ListBox(GenericBox[list[int]]):",
            "    pass",
            "",
            "class OtherBox[U](GenericBox[U]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_subclass_specialization_rejects_generic_field_usage(tmp_path: Path) -> None:
    """A method that reads a generic field is not specialized from subclass evidence."""
    specializations = _specializations(
        tmp_path,
        "generic_field_usage",
        [
            "class GenericBox[T]:",
            "    stored: T",
            "",
            "    def get(self) -> T:",
            "        return self.stored",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_top_level_closed_call_specializes_generic_function(tmp_path: Path) -> None:
    """A single agreeing direct call infers a concrete top-level function binding."""
    specializations = _specializations(
        tmp_path,
        "closed_call",
        [
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "RESULT = identity(1)",
            "",
        ],
    )

    specialization = specializations[0]

    assert specialization.origin == "closed_call"
    assert specialization.source_member.qualname == "identity"
    assert specialization.target_owner_class is None
    assert specialization.substitutions == (("T", "int"),)
    assert specialization.guards[0].positional_index == 0
    assert specialization.guards[0].nominal_type_paths == ("int",)
    assert all(binding.concrete for binding in specialization.type_bindings)
    assert {binding.annotation for binding in specialization.type_bindings} == {"int"}
    assert all("Any" not in binding.annotation for binding in specialization.type_bindings)


def test_closed_call_infers_nominal_enclosing_parameter(tmp_path: Path) -> None:
    """Concrete enclosing parameter annotations can close a direct generic call."""
    specializations = _specializations(
        tmp_path,
        "nominal_call",
        [
            "class Payload:",
            "    pass",
            "",
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "def use(payload: Payload) -> Payload:",
            "    return identity(payload)",
            "",
        ],
    )

    specialization = next(
        specialization
        for specialization in specializations
        if specialization.source_member.qualname == "identity"
    )

    assert specialization.substitutions == (("T", "Payload"),)
    assert specialization.guards[0].nominal_type_paths == ("Payload",)
    assert {binding.annotation for binding in specialization.type_bindings} == {"Payload"}


def test_conflicting_closed_calls_leave_generic_function_interpreted(tmp_path: Path) -> None:
    """Disagreeing direct calls do not produce a specialization."""
    specializations = _specializations(
        tmp_path,
        "conflicting_calls",
        [
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "FIRST = identity(1)",
            "SECOND = identity('x')",
            "",
        ],
    )

    assert specializations == ()


def test_unresolved_generic_enclosing_call_is_not_specialized(tmp_path: Path) -> None:
    """An enclosing generic parameter is unresolved closed-call evidence."""
    specializations = _specializations(
        tmp_path,
        "unresolved_enclosing_call",
        [
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "def use[U](value: U) -> U:",
            "    return identity(value)",
            "",
        ],
    )

    assert specializations == ()


def test_subclass_specialization_preserves_overrides_dunders_and_dynamic_targets(
    tmp_path: Path,
) -> None:
    """Specialization never replaces subclass behavior or unsafe class machinery."""
    specializations = _specializations(
        tmp_path,
        "unsafe_specialization_targets",
        [
            "class GenericBox[T]:",
            "    def __init__(self, value: T) -> None:",
            "        self.value = value",
            "",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class OverridingBox(GenericBox[int]):",
            "    def echo(self, value: int) -> int:",
            "        return value + 1",
            "",
            "class DynamicBox(GenericBox[int]):",
            "    def __getattr__(self, name: str) -> object:",
            "        raise AttributeError(name)",
            "",
        ],
    )

    assert specializations == ()


def test_subclass_specialization_rejects_semantic_any_alias(tmp_path: Path) -> None:
    """An imported Any alias cannot be misclassified as a nominal runtime class."""
    any_import = f"from typing import {chr(65)}ny as DynamicValue"
    specializations = _specializations(
        tmp_path,
        "any_specialization",
        [
            any_import,
            "",
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class DynamicBox(GenericBox[DynamicValue]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_subclass_specialization_accepts_quoted_typing_union_guard(tmp_path: Path) -> None:
    """Quoted Union and Optional forms still lower to constant-time nominal checks."""
    specializations = _specializations(
        tmp_path,
        "typing_union_specialization",
        [
            "from typing import Union",
            "",
            "class GenericBox[T]:",
            "    def echo(self, value: 'Union[T, None]') -> 'Union[T, None]':",
            "        return value",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    specialization = specializations[0]
    assert specialization.guards[0].annotation == "Union[int, None]"
    assert specialization.guards[0].nominal_type_paths == ("int",)
    assert specialization.guards[0].allow_none is True


def test_subclass_specialization_requires_every_method_typevar_to_close(
    tmp_path: Path,
) -> None:
    """A method-scoped TypeVar cannot disappear when only the owner TypeVar is bound."""
    specializations = _specializations(
        tmp_path,
        "partially_closed_method",
        [
            "class GenericBox[T]:",
            "    def choose[U](self, left: T, right: U) -> tuple[T, U]:",
            "        return left, right",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_legacy_generic_base_can_specialize_a_direct_concrete_subclass(
    tmp_path: Path,
) -> None:
    """Legacy Generic and TypeVar declarations retain the same direct-binding contract."""
    specializations = _specializations(
        tmp_path,
        "legacy_generic_specialization",
        [
            "from typing import Generic, TypeVar",
            "",
            "T = TypeVar('T')",
            "",
            "class GenericBox(Generic[T]):",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    specialization = specializations[0]
    assert specialization.substitutions == (("T", "int"),)
    assert specialization.target_owner_class == "IntBox"


def test_closed_generic_method_call_specializes_on_its_source_owner(tmp_path: Path) -> None:
    """A typed self.method call can close a method-scoped TypeVar without rebinding a base."""
    specializations = _specializations(
        tmp_path,
        "closed_method_call",
        [
            "class Worker:",
            "    def identity[T](self, value: T) -> T:",
            "        return value",
            "",
            "    def use(self, value: int) -> int:",
            "        return self.identity(value)",
            "",
        ],
    )

    specialization = next(
        item for item in specializations if item.source_member.qualname == "Worker.identity"
    )
    assert specialization.origin == "closed_call"
    assert specialization.target_owner_class == "Worker"
    assert specialization.substitutions == (("T", "int"),)
    assert specialization.guards[0].positional_index == 1


def test_closed_call_ignores_a_shadowed_generic_function_name(tmp_path: Path) -> None:
    """A local binding with the same name cannot provide false specialization evidence."""
    specializations = _specializations(
        tmp_path,
        "shadowed_closed_call",
        [
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "def use(value: int) -> int:",
            "    identity = lambda item: item + 1",
            "    return identity(value)",
            "",
        ],
    )

    assert specializations == ()


def test_transitive_generic_subclass_resolves_ancestor_substitutions(tmp_path: Path) -> None:
    """Concrete arguments flow through a safe same-module generic inheritance chain."""
    specializations = _specializations(
        tmp_path,
        "transitive_specialization",
        [
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class MiddleBox[T](GenericBox[T]):",
            "    pass",
            "",
            "class IntBox(MiddleBox[int]):",
            "    pass",
            "",
        ],
    )

    specialization = next(
        item for item in specializations if item.source_member.qualname == "GenericBox.echo"
    )
    assert specialization.target_owner_class == "IntBox"
    assert specialization.substitutions == (("T", "int"),)


def test_transitive_specialization_preserves_intermediate_override(tmp_path: Path) -> None:
    """An intermediate generic override hides the ancestor implementation."""
    specializations = _specializations(
        tmp_path,
        "transitive_override",
        [
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class MiddleBox[T](GenericBox[T]):",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class IntBox(MiddleBox[int]):",
            "    pass",
            "",
        ],
    )

    assert {item.source_member.qualname for item in specializations} == {"MiddleBox.echo"}


def test_specialization_models_reject_incomplete_runtime_evidence(tmp_path: Path) -> None:
    """Malformed guards and non-concrete specialization records fail immediately."""
    specialization = _specializations(
        tmp_path,
        "specialization_validation",
        [
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )[0]
    guard = specialization.guards[0]

    with pytest.raises(ValueError, match="parameter_name"):
        replace(guard, parameter_name=" ")
    with pytest.raises(ValueError, match="non-negative"):
        replace(guard, positional_index=-1)
    with pytest.raises(ValueError, match="annotation"):
        replace(guard, annotation="")
    with pytest.raises(ValueError, match="name a type"):
        replace(guard, nominal_type_paths=(), allow_none=False)
    with pytest.raises(ValueError, match="nominal paths"):
        replace(guard, nominal_type_paths=("",))
    with pytest.raises(ValueError, match="id must be non-empty"):
        replace(specialization, id="")
    with pytest.raises(ValueError, match="at least one substitution"):
        replace(specialization, substitutions=())
    with pytest.raises(ValueError, match="substitutions must be non-empty"):
        replace(specialization, substitutions=(("T", ""),))
    with pytest.raises(ValueError, match="unique TypeVars"):
        replace(specialization, substitutions=(("T", "int"), ("T", "str")))
    with pytest.raises(ValueError, match="type bindings must be concrete"):
        replace(
            specialization,
            type_bindings=(replace(specialization.type_bindings[0], concrete=False),),
        )
    with pytest.raises(ValueError, match="distinct source and target"):
        replace(specialization, target_owner_class="GenericBox")
    with pytest.raises(ValueError, match="retain its source owner"):
        replace(specialization, origin="closed_call")


def test_multiple_generic_bases_preserve_python_mro_precedence(tmp_path: Path) -> None:
    """Only the first visible generic implementation may bind on a target subclass."""
    specializations = _specializations(
        tmp_path,
        "multiple_generic_bases",
        [
            "class First[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class Second[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class Final(First[int], Second[int]):",
            "    pass",
            "",
        ],
    )

    assert {item.source_member.qualname for item in specializations} == {"First.echo"}


def test_runtime_typevar_use_keeps_legacy_generic_method_interpreted(tmp_path: Path) -> None:
    """A TypeVar used as a runtime value is not an annotation-only substitution."""
    specializations = _specializations(
        tmp_path,
        "runtime_typevar",
        [
            "from typing import Generic, TypeVar, cast",
            "",
            "T = TypeVar('T')",
            "",
            "class GenericBox(Generic[T]):",
            "    def echo(self, value: T) -> T:",
            "        return cast(T, value)",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_qualified_external_base_does_not_match_local_generic_basename(tmp_path: Path) -> None:
    """Only an unqualified same-module base can provide specialization evidence."""
    specializations = _specializations(
        tmp_path,
        "qualified_external_base",
        [
            "import external",
            "",
            "class GenericBox[T]:",
            "    def echo(self, value: T) -> T:",
            "        return value",
            "",
            "class ExternalBox(external.GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()


def test_shadowed_class_receiver_does_not_close_generic_method(tmp_path: Path) -> None:
    """A capitalized local parameter cannot resolve to a same-module class."""
    specializations = _specializations(
        tmp_path,
        "shadowed_class_receiver",
        [
            "class Worker:",
            "    @staticmethod",
            "    def identity[T](value: T) -> T:",
            "        return value",
            "",
            "class Other:",
            "    pass",
            "",
            "def use(Worker: type[Other], value: int) -> int:",
            "    return Worker.identity(value)",
            "",
        ],
    )

    assert specializations == ()


def test_rebound_module_class_does_not_close_generic_method(tmp_path: Path) -> None:
    """A reassigned class name cannot resolve calls to its earlier definition."""
    specializations = _specializations(
        tmp_path,
        "rebound_class_receiver",
        [
            "class Worker:",
            "    @staticmethod",
            "    def identity[T](value: T) -> T:",
            "        return value",
            "",
            "class Other:",
            "    pass",
            "",
            "Worker = Other",
            "RESULT = Worker.identity(1)",
            "",
        ],
    )

    assert specializations == ()


def test_duplicate_call_arguments_do_not_close_generic_function(tmp_path: Path) -> None:
    """A call that Python rejects cannot provide valid TypeVar inference evidence."""
    specializations = _specializations(
        tmp_path,
        "duplicate_call_arguments",
        [
            "def identity[T](value: T) -> T:",
            "    return value",
            "",
            "RESULT = identity(1, value=2)",
            "",
        ],
    )

    assert specializations == ()


def test_literal_getattr_of_generic_field_remains_interpreted(tmp_path: Path) -> None:
    """A dynamic spelling of a generic field read receives the same field blocker."""
    specializations = _specializations(
        tmp_path,
        "generic_field_getattr",
        [
            "class GenericBox[T]:",
            "    stored: T",
            "",
            "    def get(self) -> T:",
            "        return getattr(self, 'stored')",
            "",
            "class IntBox(GenericBox[int]):",
            "    pass",
            "",
        ],
    )

    assert specializations == ()
