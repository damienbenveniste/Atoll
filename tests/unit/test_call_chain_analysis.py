"""Tests for conservative same-module scalar call-chain planning."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import ModuleId, ModuleScan, RegionMember
from atoll.native_optimization import call_chains
from atoll.native_optimization.call_chains import (
    CallChainFieldBinding,
    analyze_call_chain_scan,
    bind_call_arguments,
    call_chain_runtime_guards,
    call_chain_scalar_parameters,
)
from atoll.native_optimization.models import (
    CallableCodeIdentityGuardPayload,
    DirectFieldGuardPayload,
)

EXPECTED_SAME_LINE_EDGE_COUNT = 2


def _private(name: str) -> object:
    return getattr(call_chains, name)


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "call_chain.py"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId("call_chain", path)))


def _rejection(scan: ModuleScan, qualname: str) -> tuple[str, str]:
    rejection = next(
        item for item in analyze_call_chain_scan(scan).rejections if item.root.qualname == qualname
    )
    return rejection.code, rejection.message


def test_call_chain_analysis_proves_acyclic_helpers_and_stable_guards(tmp_path: Path) -> None:
    """Pure nested helpers become one fixed-width root plan in leaves-first order."""
    scan = _scan(
        tmp_path,
        """
        def affine(value: int, scale: int = 3, *, bias: int = 1) -> int:
            return value * scale + bias

        def squared(value: int) -> int:
            return affine(value, scale=2, bias=3) * affine(value)

        def pipeline(value: int, *, rounds: int = 2) -> int:
            return squared(value) * rounds + affine(value, bias=5)
        """,
    )

    first = analyze_call_chain_scan(scan)
    second = analyze_call_chain_scan(scan)
    plan = next(item for item in first.plans if item.root.qualname == "pipeline")

    assert first == second
    assert plan.id.startswith("call-chain-")
    assert tuple(helper.qualname for helper in plan.helpers) == ("affine", "squared")
    assert {edge.callee.qualname for edge in plan.edges} == {"affine", "squared"}
    assert [proof.native.width for proof in plan.scalar_plan.width_proofs] == [32, 64]
    assert all(guard.kind == "callable-code-identity" for guard in plan.callable_guards)


def test_call_chain_analysis_rejects_cycles_and_opaque_calls(tmp_path: Path) -> None:
    """Recursion and unresolved dynamic calls poison only their candidate roots."""
    scan = _scan(
        tmp_path,
        """
        def first(value: int) -> int:
            return second(value)

        def second(value: int) -> int:
            return first(value)

        def leaf(value: int) -> int:
            return value + 1

        def opaque(value: int) -> int:
            return leaf(value) + abs(value)
        """,
    )

    result = analyze_call_chain_scan(scan)
    rejected = {item.root.qualname: item.code for item in result.rejections}

    assert rejected["first"] == "recursive-chain"
    assert rejected["second"] == "recursive-chain"
    assert rejected["opaque"] == "opaque-call"


def test_call_chain_analysis_supports_direct_staticmethod_helpers(tmp_path: Path) -> None:
    """Static methods can fuse explicit same-class helper dispatch."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            @staticmethod
            def step(value: int, factor: int = 3) -> int:
                return value * factor + 1

            @staticmethod
            def run(value: int) -> int:
                return Arithmetic.step(value, factor=5) + Arithmetic.step(value)
        """,
    )

    result = analyze_call_chain_scan(scan)
    plan = next(item for item in result.plans if item.root.qualname == "Arithmetic.run")

    assert tuple(helper.qualname for helper in plan.helpers) == ("Arithmetic.step",)
    payload = plan.callable_guards[0].payload
    assert isinstance(payload, CallableCodeIdentityGuardPayload)
    assert payload.callable_qualname == "Arithmetic.step"


def test_call_chain_analysis_preserves_method_lexical_indentation(tmp_path: Path) -> None:
    """Unindented text inside a method docstring cannot break declaration parsing."""
    scan = _scan(
        tmp_path,
        '''
        class Arithmetic:
            @staticmethod
            def step(value: int) -> int:
                """Return the next value.

Example text intentionally starts in column zero.
                """
                return value + 1

            @staticmethod
            def run(value: int) -> int:
                return Arithmetic.step(value) * 2
        ''',
    )

    result = analyze_call_chain_scan(scan)

    assert any(plan.root.qualname == "Arithmetic.run" for plan in result.plans)
    assert not any(rejection.code == "unparseable-source" for rejection in result.rejections)


def test_call_chain_analysis_preserves_same_line_call_site_identity(tmp_path: Path) -> None:
    """Two direct calls on one line retain distinct source spans and profile IDs."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value) + leaf(value + 1)
        """,
    )

    plan = next(
        item for item in analyze_call_chain_scan(scan).plans if item.root.qualname == "root"
    )

    assert len(plan.edges) == EXPECTED_SAME_LINE_EDGE_COUNT
    assert plan.edges[0].lineno == plan.edges[1].lineno
    assert plan.edges[0].col_offset != plan.edges[1].col_offset


def test_call_chain_analysis_proves_instance_fields_and_exact_owner(tmp_path: Path) -> None:
    """Instance chains convert retained int fields into guarded scalar proof inputs."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def step(self, value: int, bias: int = 1) -> int:
                return value * self.factor + bias

            def run(self, value: int, rounds: int = 3) -> int:
                total = 0
                for offset in range(rounds):
                    total += self.step(value + offset)
                return total
        """,
    )

    plan = next(
        item
        for item in analyze_call_chain_scan(scan).plans
        if item.root.qualname == "Arithmetic.run"
    )
    guards = call_chain_runtime_guards(plan, plan.scalar_plan.width_proofs[0])

    assert tuple(item.field_name for item in plan.field_bindings) == ("factor",)
    assert plan.field_bindings[0].synthetic_name == "_atoll_field_factor"
    assert plan.receiver_guards[0].kind == "exact-type"
    direct = next(guard.payload for guard in guards if guard.kind == "direct-field")
    assert isinstance(direct, DirectFieldGuardPayload)
    assert direct.owner_subject == "self"
    assert direct.owner_type_qualname == "Arithmetic"
    assert direct.minimum is not None
    assert direct.maximum is not None


@pytest.mark.parametrize(
    ("source", "root", "code", "message"),
    [
        (
            """
            class Arithmetic:
                @staticmethod
                def leaf(value: int) -> int:
                    return value + 1

                @classmethod
                def root(cls, value: int) -> int:
                    return cls.leaf(value)
            """,
            "Arithmetic.root",
            "unsupported-root",
            "module, staticmethod, or instance-method",
        ),
        (
            """
            def leaf(value: str) -> int:
                return 1

            def root(value: int) -> int:
                return leaf(value)
            """,
            "root",
            "unsupported-helper",
            "exact int parameters",
        ),
        (
            """
            class Arithmetic:
                @classmethod
                def leaf(cls, value: int) -> int:
                    return value + 1

            def root(value: int) -> int:
                return Arithmetic.leaf(value)
            """,
            "root",
            "unsupported-helper",
            "unsupported binding classmethod",
        ),
        (
            """
            def leaf(value: int) -> int:
                adjusted = value + 1
                return adjusted

            def root(value: int) -> int:
                return leaf(value)
            """,
            "root",
            "unsupported-helper",
            "one pure return expression",
        ),
        (
            """
            def leaf(value: int) -> int:
                return value + 1

            def root(value: int) -> int:
                return leaf(value) / 2
            """,
            "root",
            "unproven-root",
            "exact-int argument domain",
        ),
        (
            """
            class Arithmetic:
                factor: int

                def __getattr__(self, name: str) -> object:
                    return 1

                def leaf(self, value: int) -> int:
                    return value * self.factor

                def root(self, value: int) -> int:
                    return self.leaf(value)
            """,
            "Arithmetic.root",
            "unsupported-root",
            "custom attribute hooks",
        ),
        (
            """
            class Arithmetic:
                factor: int

                def leaf(self, value: int) -> int:
                    return value * self.unknown

                def root(self, value: int) -> int:
                    return self.leaf(value)
            """,
            "Arithmetic.root",
            "unsupported-root",
            "not one retained int field",
        ),
        (
            """
            class Arithmetic:
                def leaf(self, value: int) -> int:
                    return value + 1

                def root(self, value: int) -> int:
                    return Arithmetic.leaf(self, value)
            """,
            "Arithmetic.root",
            "ambiguous-call",
            "requires direct self dispatch",
        ),
    ],
)
def test_call_chain_analysis_reports_conservative_rejections(
    tmp_path: Path,
    source: str,
    root: str,
    code: str,
    message: str,
) -> None:
    """Unsupported roots and helpers retain one stable fallback explanation."""
    actual_code, actual_message = _rejection(_scan(tmp_path, source), root)

    assert actual_code == code
    assert message in actual_message


@pytest.mark.parametrize(
    ("source", "root", "code", "message"),
    [
        (
            """
            BIAS = 7

            def leaf(value: int) -> int:
                return value + BIAS

            def root(value: int) -> int:
                return leaf(value)
            """,
            "root",
            "opaque-call",
            "mutable module constant(s): BIAS",
        ),
        (
            """
            def leaf(value: int, increment: int) -> int:
                return value

            def root(value: int) -> int:
                return leaf(value, 1)
            """,
            "root",
            "unsupported-helper",
            "ignores scalar parameter(s): increment",
        ),
        (
            """
            class DynamicMeta(type):
                pass

            class Arithmetic(metaclass=DynamicMeta):
                @staticmethod
                def leaf(value: int) -> int:
                    return value + 1

                @staticmethod
                def root(value: int) -> int:
                    return Arithmetic.leaf(value)
            """,
            "Arithmetic.root",
            "unsupported-root",
            "may customize metaclass dispatch",
        ),
    ],
)
def test_call_chain_analysis_rejects_unstable_specialization_inputs(
    tmp_path: Path,
    source: str,
    root: str,
    code: str,
    message: str,
) -> None:
    """Constants, unused arguments, and metaclasses remain interpreted."""
    actual_code, actual_message = _rejection(_scan(tmp_path, source), root)

    assert actual_code == code
    assert message in actual_message


@pytest.mark.parametrize(
    ("helper_signature", "call", "message"),
    [
        ("value: int, *rest: int", "leaf(value)", "variadic parameters"),
        ("value: int, other: int", "leaf(value)", "omits required argument other"),
        ("value: int, other: int = DEFAULT", "leaf(value)", "non-literal default"),
        ("value: int, /", "leaf(value=value)", "positional-only parameter value"),
        ("value: int", "leaf(value, unknown=1)", "unknown keyword unknown"),
        ("value: int", "leaf(value, value=1)", "dynamic keywords"),
        ("value: int", "leaf(value, **{})", "dynamic keywords"),
        ("value: int", "leaf(value, 1)", "cannot be bound statically"),
    ],
)
def test_call_chain_analysis_rejects_ambiguous_argument_binding(
    tmp_path: Path,
    helper_signature: str,
    call: str,
    message: str,
) -> None:
    """Native helper calls require an exact positional and keyword binding."""
    helper_term = "rest[0]" if "*rest" in helper_signature else "other"
    if "other" not in helper_signature and "*rest" not in helper_signature:
        helper_term = "1"
    source = f"""
        DEFAULT = 2

        def leaf({helper_signature}) -> int:
            return value + {helper_term}

        def root(value: int) -> int:
            return {call}
    """

    code, actual_message = _rejection(_scan(tmp_path, source), "root")

    assert code == "ambiguous-call"
    assert message in actual_message


def test_call_chain_analysis_rejects_missing_retained_call_site(tmp_path: Path) -> None:
    """A source call without its exact retained span cannot enter a fused chain."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value) + leaf(value + 1)
        """,
    )
    region = scan.typed_regions[0]
    root = next(member for member in region.members if member.id.qualname == "root")
    members = tuple(
        replace(member, call_sites=member.call_sites[:1]) if member.id == root.id else member
        for member in region.members
    )
    drifted = replace(scan, typed_regions=(replace(region, members=members),))

    code, message = _rejection(drifted, "root")

    assert code == "ambiguous-call"
    assert "is not retained" in message


def test_call_chain_analysis_supports_positional_only_instance_receiver(tmp_path: Path) -> None:
    """A conventional positional-only self receiver is removed from the scalar proof."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def leaf(self, /, value: int) -> int:
                return value * self.factor

            def root(self, /, value: int) -> int:
                return self.leaf(value)
        """,
    )

    plan = next(
        item
        for item in analyze_call_chain_scan(scan).plans
        if item.root.qualname == "Arithmetic.root"
    )

    assert tuple(parameter.name for parameter in plan.scalar_plan.width_proofs[0].parameters) == (
        "_atoll_field_factor",
        "value",
    )


def test_call_chain_analysis_rejects_instance_owner_missing_from_scan(tmp_path: Path) -> None:
    """Instance chains require the exact owner class to remain in the staged scan."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            def leaf(self, value: int) -> int:
                return value + 1

            def root(self, value: int) -> int:
                return self.leaf(value)
        """,
    )
    without_owner = replace(
        scan, symbols=tuple(item for item in scan.symbols if item.kind != "class")
    )

    code, message = _rejection(without_owner, "Arithmetic.root")

    assert code == "unsupported-root"
    assert "no retained owner class" in message


def test_call_chain_scalar_parameters_rejects_nonstandard_receiver(tmp_path: Path) -> None:
    """Instance metadata without a conventional self receiver is rejected defensively."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            def root(self, value: int) -> int:
                return value
        """,
    )
    member = next(
        item for item in scan.typed_regions[0].members if item.id.qualname.endswith("root")
    )
    malformed = replace(
        member,
        parameters=(replace(member.parameters[0], name="receiver"), *member.parameters[1:]),
    )

    with pytest.raises(Exception, match="no conventional self receiver"):
        call_chain_scalar_parameters(malformed)


def test_bind_call_arguments_rejects_starred_and_missing_calls() -> None:
    """The direct binder rejects starred arguments and required keyword-only gaps."""
    node = ast.parse("def leaf(value: int, *, scale: int) -> int: return value * scale").body[0]
    starred_call = ast.parse("leaf(*(1,))", mode="eval").body
    missing_call = ast.parse("leaf(1)", mode="eval").body
    assert isinstance(node, ast.FunctionDef)
    assert isinstance(starred_call, ast.Call)
    assert isinstance(missing_call, ast.Call)

    with pytest.raises(Exception, match="cannot be bound statically"):
        bind_call_arguments(node, starred_call)
    with pytest.raises(Exception, match="omits required argument scale"):
        bind_call_arguments(node, missing_call)


def test_call_chain_private_parsers_reject_malformed_members(tmp_path: Path) -> None:
    """Malformed retained declarations fail before a backend sees generated source."""
    scan = _scan(tmp_path, "def root(value: int) -> int:\n    return value")
    member = scan.typed_regions[0].members[0]
    message = "does not contain one synchronous declaration"
    callable_node = cast(Callable[[RegionMember], ast.FunctionDef], _private("_callable_node"))

    with pytest.raises(Exception, match=message):
        callable_node(replace(member, source_text="VALUE = 1"))
    with pytest.raises(Exception, match=message):
        callable_node(
            replace(
                member,
                source_text="def first() -> int: return 1\ndef second() -> int: return 2",
            )
        )


def test_direct_field_substitution_rejects_unknown_and_malformed_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof AST transformer fails closed for unknown fields and invalid rewrites."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def leaf(self, value: int) -> int:
                return value * self.factor

            def root(self, value: int) -> int:
                return self.leaf(value)
        """,
    )
    plan = next(
        item
        for item in analyze_call_chain_scan(scan).plans
        if item.root.qualname == "Arithmetic.root"
    )
    factory = cast(
        Callable[[tuple[CallChainFieldBinding, ...]], ast.NodeTransformer],
        _private("_DirectFieldSubstitution"),
    )
    transformer = factory(plan.field_bindings)
    unknown = ast.parse("self.unknown", mode="eval").body
    nested = ast.parse("other.value", mode="eval").body
    assert isinstance(unknown, ast.Attribute)
    assert isinstance(nested, ast.Attribute)

    with pytest.raises(Exception, match="unsupported receiver attribute"):
        transformer.visit(unknown)
    assert transformer.visit(nested) is nested

    def invalid_generic_visit(_node: ast.AST) -> object:
        return object()

    monkeypatch.setattr(transformer, "generic_visit", invalid_generic_visit)
    with pytest.raises(TypeError, match="non-expression"):
        transformer.visit(nested)


def test_call_chain_analysis_defensively_rejects_invalid_transformer_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected transformer return types never reach scalar proof generation."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )

    def invalid_substitution(_self: object, _node: ast.AST) -> object:
        return object()

    monkeypatch.setattr(_private("_NameSubstitution"), "visit", invalid_substitution)
    with pytest.raises(TypeError, match="substitution produced a non-expression"):
        analyze_call_chain_scan(scan)


def test_call_chain_analysis_rejects_roots_without_inliner_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained call fact is insufficient when inlining records no helper."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )

    def identity_visit(_self: object, node: ast.AST) -> ast.AST:
        return node

    monkeypatch.setattr(_private("_CallInliner"), "visit", identity_visit)

    code, message = _rejection(scan, "root")

    assert code == "ambiguous-call"
    assert "no resolvable direct helper" in message


def test_call_chain_analysis_rejects_nonfunction_inliner_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed root AST returned by the inliner raises a defensive type error."""
    scan = _scan(
        tmp_path,
        """
        def leaf(value: int) -> int:
            return value + 1

        def root(value: int) -> int:
            return leaf(value)
        """,
    )

    def invalid_inliner(_self: object, _node: ast.AST) -> ast.Constant:
        return ast.Constant(value=1)

    monkeypatch.setattr(_private("_CallInliner"), "visit", invalid_inliner)

    with pytest.raises(TypeError, match="non-function root"):
        analyze_call_chain_scan(scan)


def test_call_chain_analysis_rejects_nonfunction_field_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed instance-field rewrite cannot enter scalar analysis."""
    scan = _scan(
        tmp_path,
        """
        class Arithmetic:
            factor: int

            def leaf(self, value: int) -> int:
                return value * self.factor

            def root(self, value: int) -> int:
                return self.leaf(value)
        """,
    )

    def invalid_field_rewrite(_self: object, _node: ast.AST) -> ast.Constant:
        return ast.Constant(value=1)

    monkeypatch.setattr(
        _private("_DirectFieldSubstitution"),
        "visit",
        invalid_field_rewrite,
    )

    with pytest.raises(TypeError, match="non-function root"):
        analyze_call_chain_scan(scan)


def test_receiver_rewrite_requires_a_conventional_receiver() -> None:
    """The direct receiver rewriter rejects declarations without self."""
    node = ast.parse("def root(value: int) -> int: return value").body[0]
    assert isinstance(node, ast.FunctionDef)

    replace_receiver = cast(
        Callable[[ast.FunctionDef, tuple[CallChainFieldBinding, ...]], None],
        _private("_replace_receiver_with_fields"),
    )
    with pytest.raises(Exception, match="no conventional self receiver"):
        replace_receiver(node, ())
