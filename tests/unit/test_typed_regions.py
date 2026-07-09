"""Tests for backend-neutral typed-region formation."""

from __future__ import annotations

from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import ModuleId


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


def test_unannotated_staticmethod_prevents_atomic_class_region(tmp_path: Path) -> None:
    """A static method's first argument must be annotated because it is not bound."""
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

    assert scan.typed_regions == ()


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
