"""Tests for fixed-width Cython direct-call-chain generation."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation import call_chain
from atoll.generation.call_chain import (
    CALL_CHAIN_GENERATOR_VERSION,
    CallChainGenerationRequest,
    generate_call_chain_kernel,
)
from atoll.models import ModuleId, ModuleScan, RegionMember, SymbolId, TypedRegion
from atoll.native_optimization.call_chains import (
    CallChainFieldBinding,
    CallChainPlan,
    analyze_call_chain_scan,
)


def _private(name: str) -> object:
    return getattr(call_chain, name)


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "chain_generation.py"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId("chain_generation", path)))


def _case(
    tmp_path: Path, source: str, root: str = "root"
) -> tuple[ModuleScan, CallChainPlan, TypedRegion]:
    scan = _scan(tmp_path, source)
    plan = next(item for item in analyze_call_chain_scan(scan).plans if item.root.qualname == root)
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)
    return scan, plan, region


def _request(
    tmp_path: Path,
    scan: ModuleScan,
    plan: CallChainPlan,
    region: TypedRegion,
) -> CallChainGenerationRequest:
    return CallChainGenerationRequest(
        scan=scan,
        region=region,
        plan=plan,
        width_proof=plan.scalar_plan.width_proofs[0],
        logical_module="_atoll_chain_test",
        output_path=tmp_path / "chain.pyx",
    )


def test_call_chain_generation_emits_inline_helpers_and_positional_calls(tmp_path: Path) -> None:
    """Defaults and keywords become explicit unboxed calls inside one Cython unit."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int, increment: int = 3, *, scale: int = 2) -> int:
            return (value + increment) * scale

        def root(value: int, *, scale: int = 5) -> int:
            return leaf(value, scale=scale) + leaf(value, 4, scale=3)
        """,
    )
    plan = next(
        item for item in analyze_call_chain_scan(scan).plans if item.root.qualname == "root"
    )
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)
    output = tmp_path / "chain.pyx"

    generated = generate_call_chain_kernel(
        CallChainGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            width_proof=plan.scalar_plan.width_proofs[0],
            logical_module="_atoll_chain_test",
            output_path=output,
        )
    )

    source = generated.generation.source_text
    assert f"# atoll call chain {CALL_CHAIN_GENERATOR_VERSION}" in source
    assert "cdef inline int32_t _atoll_inline_leaf_i32(" in source
    assert "_atoll_inline_leaf_i32(value, 3, scale)" in source
    assert "_atoll_inline_leaf_i32(value, 4, 3)" in source
    assert generated.generation.selected_members == (plan.root, *plan.helpers)
    assert generated.generation.bindings[0].source == plan.root
    assert output.read_text(encoding="utf-8") == source


def test_call_chain_generation_passes_guarded_instance_fields_as_scalars(tmp_path: Path) -> None:
    """Instance helpers receive direct fields as private fixed-width arguments."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def step(self, value: int) -> int:
                return value * self.factor + 1

            def run(self, value: int) -> int:
                return self.step(value) + self.step(value + 1)
        """,
    )
    plan = next(
        item
        for item in analyze_call_chain_scan(scan).plans
        if item.root.qualname == "Arithmetic.run"
    )
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)

    generated = generate_call_chain_kernel(
        CallChainGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            width_proof=plan.scalar_plan.width_proofs[0],
            logical_module="_atoll_instance_chain",
            output_path=tmp_path / "instance_chain.pyx",
        )
    )

    source = generated.generation.source_text
    assert "cdef inline int32_t _atoll_inline_Arithmetic_step_i32(" in source
    assert "int32_t _atoll_field_factor, int32_t value" in source
    assert "_atoll_inline_Arithmetic_step_i32(_atoll_field_factor, value)" in source
    assert "return _atoll_chain_Arithmetic_run_i32_native(self.factor, value)" in source
    assert generated.generation.bindings[0].kind == "instance_method"


def test_call_chain_generation_rejects_staged_provenance_drift(tmp_path: Path) -> None:
    """Generation fails before writing when its region, proof, members, or source drift."""
    scan, plan, region = _case(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )
    request = _request(tmp_path, scan, plan, region)

    with pytest.raises(ValueError, match=r"require a \.pyx output path"):
        generate_call_chain_kernel(replace(request, output_path=tmp_path / "chain.py"))
    with pytest.raises(ValueError, match="region differs"):
        generate_call_chain_kernel(replace(request, region=replace(region, id="different")))
    with pytest.raises(ValueError, match="width proof was not authorized"):
        generate_call_chain_kernel(
            replace(
                request,
                width_proof=replace(request.width_proof, explicit_modular_width=8),
            )
        )
    with pytest.raises(ValueError, match="member is absent"):
        generate_call_chain_kernel(
            replace(request, region=replace(region, members=region.members[:1]))
        )
    drifted_root = replace(region.members[-1], source_text=region.members[-1].source_text + "\n")
    with pytest.raises(ValueError, match="source differs"):
        generate_call_chain_kernel(
            replace(request, region=replace(region, members=(*region.members[:-1], drifted_root)))
        )


def test_call_chain_generation_renders_locals_and_static_binding(tmp_path: Path) -> None:
    """Local scalar state and docstrings lower through a static-method chain."""
    scan, plan, region = _case(
        tmp_path,
        """
        class Arithmetic:
            @staticmethod
            def leaf(value: int) -> int:
                \"\"\"Apply a scalar adjustment.\"\"\"
                return value + 7

            @staticmethod
            def root(value: int, rounds: int = 3) -> int:
                \"\"\"Accumulate direct native helper calls.\"\"\"
                total = 0
                for offset in range(rounds):
                    total += Arithmetic.leaf(value + offset)
                return total
        """,
        root="Arithmetic.root",
    )

    generated = generate_call_chain_kernel(_request(tmp_path, scan, plan, region))
    source = generated.generation.source_text

    assert "cdef int32_t offset, total" in source
    assert "Apply a scalar adjustment" not in source
    assert generated.generation.bindings[0].kind == "staticmethod"


def test_call_chain_generation_rejects_generated_call_topology_drift(tmp_path: Path) -> None:
    """A hash-updated staged declaration still cannot invent an unproven callee."""
    scan, plan, region = _case(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )
    root = next(member for member in region.members if member.id == plan.root)
    drifted_root = replace(root, source_text=root.source_text.replace("leaf(value)", "abs(value)"))
    drifted_members = tuple(
        drifted_root if member.id == root.id else member for member in region.members
    )
    digest = hashlib.sha256(drifted_root.source_text.encode("utf-8")).hexdigest()
    hashes = tuple(
        (member_id, digest if member_id == root.id else source_hash)
        for member_id, source_hash in plan.source_hashes
    )
    request = _request(
        tmp_path,
        scan,
        replace(plan, source_hashes=hashes),
        replace(region, members=drifted_members),
    )

    with pytest.raises(ValueError, match="unresolved generated call-chain target"):
        generate_call_chain_kernel(request)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("self.step(value) + self.unknown", "unproven generated receiver attribute"),
        ("self.step(value) + (1 if self else 0)", "retains opaque self use"),
    ],
)
def test_call_chain_generation_rejects_unproven_instance_uses_after_drift(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    """Staged instance source cannot add fields or opaque receiver behavior."""
    scan, plan, region = _case(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def step(self, value: int) -> int:
                return value * self.factor

            def root(self, value: int) -> int:
                return self.step(value)
        """,
        root="Arithmetic.root",
    )
    root = next(member for member in region.members if member.id == plan.root)
    drifted_root = replace(
        root, source_text=root.source_text.replace("self.step(value)", replacement)
    )
    drifted_members = tuple(
        drifted_root if member.id == root.id else member for member in region.members
    )
    digest = hashlib.sha256(drifted_root.source_text.encode("utf-8")).hexdigest()
    hashes = tuple(
        (member_id, digest if member_id == root.id else source_hash)
        for member_id, source_hash in plan.source_hashes
    )
    request = _request(
        tmp_path,
        scan,
        replace(plan, source_hashes=hashes),
        replace(region, members=drifted_members),
    )

    with pytest.raises(ValueError, match=message):
        generate_call_chain_kernel(request)


def test_call_chain_generation_rejects_noncall_rewriter_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The call rewriter rejects malformed generic-visitor output."""
    _scan_result, plan, region = _case(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )
    root = next(member for member in region.members if member.id == plan.root)
    factory = cast(
        Callable[
            [
                dict[SymbolId, RegionMember],
                dict[SymbolId, ast.FunctionDef],
                dict[SymbolId, str],
                tuple[CallChainFieldBinding, ...],
                RegionMember,
            ],
            ast.NodeTransformer,
        ],
        _private("_NativeCallRewriter"),
    )
    rewriter = factory({}, {}, {}, (), root)
    expression = ast.parse("range(1)", mode="eval").body
    assert isinstance(expression, ast.Call)

    def invalid_generic_visit(_node: ast.AST) -> ast.Name:
        return ast.Name(id="not_a_call", ctx=ast.Load())

    monkeypatch.setattr(rewriter, "generic_visit", invalid_generic_visit)

    with pytest.raises(TypeError, match="non-call expression"):
        rewriter.visit(expression)


def test_generated_field_substitution_rejects_invalid_generic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated attributes recurse normally and reject non-expression results."""
    factory = cast(
        Callable[[tuple[CallChainFieldBinding, ...]], ast.NodeTransformer],
        _private("_GeneratedFieldReadSubstitution"),
    )
    transformer = factory(())
    expression = ast.parse("other.value", mode="eval").body
    assert isinstance(expression, ast.Attribute)
    assert transformer.visit(expression) is expression

    def invalid_generic_visit(_node: ast.AST) -> object:
        return object()

    monkeypatch.setattr(transformer, "generic_visit", invalid_generic_visit)

    with pytest.raises(TypeError, match="non-expression"):
        transformer.visit(expression)


def test_call_chain_generation_rejects_nonfunction_rewriter_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed call rewriter result fails before generated source is written."""
    scan, plan, region = _case(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )

    def invalid_rewriter(_self: object, _node: ast.AST) -> ast.Constant:
        return ast.Constant(value=1)

    monkeypatch.setattr(_private("_NativeCallRewriter"), "visit", invalid_rewriter)

    with pytest.raises(TypeError, match="non-function"):
        generate_call_chain_kernel(_request(tmp_path, scan, plan, region))


def test_call_chain_generation_rejects_non_ast_field_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed instance-field transformer result fails closed."""
    scan, plan, region = _case(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def step(self, value: int) -> int:
                return value * self.factor

            def root(self, value: int) -> int:
                return self.step(value)
        """,
        root="Arithmetic.root",
    )

    def invalid_field_rewrite(_self: object, _node: ast.AST) -> object:
        return object()

    monkeypatch.setattr(
        _private("_GeneratedFieldReadSubstitution"),
        "visit",
        invalid_field_rewrite,
    )

    with pytest.raises(TypeError, match="non-AST value"):
        generate_call_chain_kernel(_request(tmp_path, scan, plan, region))
