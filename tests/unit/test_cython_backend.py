"""Tests for Cython backend protocol conformance and deterministic decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends import cython as cython_backend_module
from atoll.backends.base import CompilerBackend, UnsupportedBackendRegionError
from atoll.backends.cython import CYTHON_BACKEND, CythonBackend
from atoll.models import (
    BackendCompileContext,
    BackendLoweringRequest,
    BindingTarget,
    CompilationUnit,
    CompiledRegionVariant,
    ModuleId,
    SymbolId,
    TypedRegion,
)

SHA256_HEX_LENGTH = 64


def _regions(tmp_path: Path, module_name: str, source: str) -> tuple[TypedRegion, ...]:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name=module_name, path=module_path)))
    return scan.typed_regions


def _region_containing(
    regions: Iterable[TypedRegion],
    *qualnames: str,
) -> TypedRegion:
    requested = set(qualnames)
    return next(
        region
        for region in regions
        if requested <= {member.id.qualname for member in region.members}
    )


def _member(region: TypedRegion, qualname: str) -> SymbolId:
    return next(member.id for member in region.members if member.id.qualname == qualname)


def test_cython_backend_structurally_conforms_to_compiler_backend() -> None:
    """The concrete adapter and singleton satisfy the runtime protocol."""
    backend = CythonBackend()

    assert isinstance(backend, CompilerBackend)
    assert isinstance(CYTHON_BACKEND, CompilerBackend)
    assert backend.name == "cython"


def test_cython_assesses_typed_functions_methods_and_async_generators(tmp_path: Path) -> None:
    """Cython accepts typed callable members and closed atomic classes."""
    regions = _regions(
        tmp_path,
        "cython_shapes",
        """
        from collections.abc import AsyncIterator, Iterator

        def typed(value: int) -> int:
            return value + 1

        def values(limit: int) -> Iterator[int]:
            for value in range(limit):
                yield value

        async def fetch(value: int) -> int:
            return value

        async def stream(limit: int) -> AsyncIterator[int]:
            for value in range(limit):
                yield value

        class Worker:
            def scale(self, value: int) -> int:
                return value * 2

            @staticmethod
            def parse(value: str) -> int:
                return int(value)

            @classmethod
            def build(cls, value: int) -> int:
                return value
        """,
    )
    backend = CythonBackend()

    function_assessments = {
        member.id.qualname: backend.assess(region)
        for region in regions
        for member in region.members
        if member.id.qualname in {"typed", "values", "fetch", "stream"}
    }
    assert function_assessments["typed"].status == "supported"
    assert "typed_function" in function_assessments["typed"].capabilities
    assert "generator" in function_assessments["values"].capabilities
    assert "coroutine" in function_assessments["fetch"].capabilities
    assert "async_generator" in function_assessments["stream"].capabilities

    worker_region = _region_containing(regions, "Worker", "Worker.scale", "Worker.parse")
    worker_assessment = backend.assess(worker_region)

    assert worker_assessment.status == "supported"
    assert _member(worker_region, "Worker") in worker_assessment.supported_members
    assert _member(worker_region, "Worker.scale") in worker_assessment.supported_members
    assert _member(worker_region, "Worker.parse") in worker_assessment.supported_members
    assert _member(worker_region, "Worker.build") in worker_assessment.supported_members
    assert "native_class" in worker_assessment.capabilities
    assert set(worker_assessment.capabilities) >= {
        "instance_method",
        "staticmethod",
        "classmethod",
    }
    assert worker_assessment.reasons == ()


def test_cython_supports_boxed_any_and_unresolved_generic_callables(tmp_path: Path) -> None:
    """Whole-callable Cython retains Python semantics for dynamically typed inputs."""
    regions = _regions(
        tmp_path,
        "cython_fallbacks",
        """
        import typing

        def dynamic(value: typing.Any) -> typing.Any:
            return value

        def identity[T](value: T) -> T:
            return value

        def accepted(value: int) -> int:
            return value
        """,
    )
    backend = CythonBackend()

    dynamic = backend.assess(_region_containing(regions, "dynamic"))
    identity = backend.assess(_region_containing(regions, "identity"))

    assert dynamic.status == "supported"
    assert dynamic.supported_members == (SymbolId("cython_fallbacks", "dynamic"),)
    assert dynamic.reasons == ()
    assert identity.status == "supported"
    assert identity.supported_members == (SymbolId("cython_fallbacks", "identity"),)
    assert identity.reasons == ()


def test_cython_rejects_boxed_member_from_atomic_class(tmp_path: Path) -> None:
    """Boxed semantics require method rebinding on the original Python class."""
    region = _region_containing(
        _regions(
            tmp_path,
            "cython_atomic_box",
            """
            class Worker:
                def scale(self, value: int) -> int:
                    return value * 2
            """,
        ),
        "Worker",
        "Worker.scale",
    )
    scale = _member(region, "Worker.scale")
    boxed_region = replace(
        region,
        decisions=tuple(
            replace(decision, action="box", reason="fixture boxed method")
            if decision.target == scale.stable_id
            else decision
            for decision in region.decisions
        ),
    )

    assessment = CythonBackend().assess(boxed_region)

    assert assessment.status == "unsupported"
    assert set(assessment.unsupported_members) == {
        _member(region, "Worker"),
        scale,
    }
    assert any("requires concrete callable typing" in reason for reason in assessment.reasons)
    assert any("requires every class member to pass" in reason for reason in assessment.reasons)


def test_cython_lower_validates_requested_members_and_variant_id(tmp_path: Path) -> None:
    """Lowering records the prepared source hash and rejects unsupported selections."""
    regions = _regions(
        tmp_path,
        "cython_lowering",
        """
        class Worker:
            def scale(self, value: int) -> int:
                return value * 2
        """,
    )
    source_path = tmp_path / "cython_lowering.py"
    region = _region_containing(regions, "Worker", "Worker.scale")
    backend = CythonBackend()
    scale = _member(region, "Worker.scale")

    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=source_path,
            logical_module="cython_lowering",
            install_relative_dir="compiled",
            members=(scale,),
            variant_id="variant:cython",
        )
    )

    assert unit.region_id == "variant:cython"
    assert unit.backend == "cython"
    assert unit.logical_module == "cython_lowering"
    assert unit.source_paths == (source_path,)
    assert unit.members == (scale,)
    assert unit.install_relative_dir == "compiled"
    assert len(unit.source_hash) == SHA256_HEX_LENGTH

    with pytest.raises(UnsupportedBackendRegionError, match=r"cython_lowering::Worker.missing"):
        backend.lower(
            BackendLoweringRequest(
                region=region,
                source_path=source_path,
                logical_module="cython_lowering",
                members=(SymbolId("cython_lowering", "Worker.missing"),),
            )
        )

    cython_path = source_path.with_suffix(".pyx")
    cython_path.write_text(
        "# atoll scalar proof fixture\n" + source_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cython_unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=cython_path,
            logical_module="cython_lowering_specialized",
            members=(scale,),
        )
    )
    assert cython_unit.source_paths == (cython_path,)

    cython_path.write_text(
        "# atoll buffer proof fixture\n" + source_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    buffer_unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=cython_path,
            logical_module="cython_lowering_buffer",
            members=(scale,),
        )
    )
    assert buffer_unit.source_paths == (cython_path,)

    cython_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(UnsupportedBackendRegionError, match="requires Atoll proof"):
        backend.lower(
            BackendLoweringRequest(
                region=region,
                source_path=cython_path,
                logical_module="cython_lowering_unproven",
                members=(scale,),
            )
        )

    unsupported_path = source_path.with_suffix(".txt")
    unsupported_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(UnsupportedBackendRegionError, match=r"only \.py and proof-generated \.pyx"):
        backend.lower(
            BackendLoweringRequest(
                region=region,
                source_path=unsupported_path,
                logical_module="cython_lowering",
                members=(scale,),
            )
        )


def test_compiled_region_variant_validates_identity_and_binding_ownership(tmp_path: Path) -> None:
    """Backend variants require a stable ID and bindings owned by their region."""
    region = _region_containing(
        _regions(
            tmp_path,
            "cython_variant",
            """
            def score(value: int) -> int:
                return value + 1
            """,
        ),
        "score",
    )

    with pytest.raises(ValueError, match="variant ID must be non-empty"):
        CompiledRegionVariant(id=" ", region=region, backend="cython", bindings=region.bindings)
    with pytest.raises(ValueError, match="bindings must belong to the region"):
        CompiledRegionVariant(id="variant", region=region, backend="cython", bindings=())
    foreign_binding = BindingTarget(
        source=SymbolId(module="cython_variant", qualname="other"),
        compiled_name="other",
        kind="module",
        owner_class=None,
        execution_kind="sync",
    )
    with pytest.raises(ValueError, match="bindings must belong to the region"):
        CompiledRegionVariant(
            id="variant",
            region=region,
            backend="cython",
            bindings=(foreign_binding,),
        )


def test_cython_fingerprint_is_deterministic_and_strict(tmp_path: Path) -> None:
    """Source content, context options, platform, and directives feed the cache key."""
    regions = _regions(
        tmp_path,
        "cython_fingerprint",
        """
        def score(value: int) -> int:
            return value + 1
        """,
    )
    source_path = tmp_path / "cython_fingerprint.py"
    region = _region_containing(regions, "score")
    backend = CythonBackend()
    request = BackendLoweringRequest(
        region=region,
        source_path=source_path,
        logical_module="cython_fingerprint",
    )
    unit = backend.lower(request)
    context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    first = backend.fingerprint(unit, context)
    second = backend.fingerprint(unit, context)
    assert first == second

    source_path.write_text(
        dedent(
            """
            def score(value: int) -> int:
                return value + 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    changed_unit = backend.lower(request)
    assert changed_unit.source_hash != unit.source_hash
    assert backend.fingerprint(changed_unit, context) != first

    alternate_context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "alternate",
        source_roots=(tmp_path / "src",),
        backend_options=(("abi", "debug"),),
    )
    assert backend.fingerprint(changed_unit, alternate_context) != backend.fingerprint(
        changed_unit,
        context,
    )


@pytest.mark.parametrize(
    ("error", "diagnostics", "expected_code", "transient"),
    [
        (
            RuntimeError("cython failed"),
            "Error compiling Cython file:\nmodule.py:1:4: undeclared name not builtin: missing",
            "CYTHON_COMPILE_ERROR",
            False,
        ),
        (RuntimeError("compiler missing"), "", "NATIVE_BUILD_ENV_ERROR", True),
        (ImportError("module unavailable"), "", "IMPORT_PATH_ERROR", True),
        (RuntimeError("unexpected"), "", "UNKNOWN_BUILD_ERROR", True),
    ],
)
def test_cython_normalizes_backend_diagnostics(
    tmp_path: Path,
    error: BaseException,
    diagnostics: str,
    expected_code: str,
    transient: bool,
) -> None:
    """Backend exceptions become stable codes while retaining diagnostic evidence."""
    log_path = tmp_path / "cython.log"

    diagnostic = CythonBackend().normalize_diagnostic(
        error,
        diagnostics=diagnostics,
        log_path=log_path,
    )

    assert diagnostic.code == expected_code
    assert diagnostic.log_path == log_path
    assert diagnostic.transient is transient
    if diagnostics:
        assert "Captured 1 Cython error line(s)." in diagnostic.details


def test_cython_rejects_incompatible_sourceless_and_non_python_units(tmp_path: Path) -> None:
    """Compilation-unit validation fails before invoking incompatible inputs."""
    backend = CythonBackend()
    incompatible = CompilationUnit(
        region_id="fixture",
        backend="mypyc",
        logical_module="fixture",
        source_paths=(tmp_path / "fixture.py",),
        source_hash="fixture",
        members=(),
    )
    source_less = CompilationUnit(
        region_id="source-less",
        backend="cython",
        logical_module="fixture",
        source_paths=(),
        source_hash="fixture",
        members=(),
    )
    non_python = CompilationUnit(
        region_id="non-python",
        backend="cython",
        logical_module="fixture",
        source_paths=(tmp_path / "fixture.txt",),
        source_hash="fixture",
        members=(),
    )
    context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
    )

    with pytest.raises(ValueError, match="incompatible unit"):
        backend.fingerprint(incompatible, context)
    with pytest.raises(ValueError, match="source-less"):
        backend.fingerprint(source_less, context)
    with pytest.raises(ValueError, match="unsupported source"):
        backend.fingerprint(non_python, context)


def test_cython_compile_accepts_an_empty_unit_set(tmp_path: Path) -> None:
    """An empty backend batch is a successful no-op with no artifact records."""
    result = CythonBackend().compile(
        (),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            record_artifacts=False,
        ),
    )

    assert result.attempt.success is True
    assert result.attempt.stdout == "no Cython units to build"
    assert result.attempt.artifact_paths == ()
    assert result.artifacts == ()


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (SystemExit(2), "UNKNOWN_BUILD_ERROR"),
        (RuntimeError("cython crashed"), "CYTHON_COMPILE_ERROR"),
    ],
)
def test_cython_compile_normalizes_build_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_code: str,
) -> None:
    """Compiler exits and exceptions produce failed attempts with retained diagnostics."""
    source_path = tmp_path / "broken_region.py"
    source_path.write_text(
        "def score(value: int) -> int:\n    return value + 1\n", encoding="utf-8"
    )
    unit = CompilationUnit(
        region_id="broken:cython",
        backend="cython",
        logical_module="broken_region",
        source_paths=(source_path,),
        source_hash="source-hash",
        members=(),
    )

    def fail_cythonize(
        units: tuple[CompilationUnit, ...],
        build_dir: Path,
        *,
        project_root: Path,
        workers: int,
    ) -> list[object]:
        _ = (units, build_dir, project_root, workers)
        raise error

    monkeypatch.setattr(cython_backend_module, "_cythonize_extensions", fail_cythonize)
    build_dir = tmp_path / ".atoll" / "build"

    result = CythonBackend().compile(
        (unit,),
        BackendCompileContext(project_root=tmp_path, build_dir=build_dir),
    )

    assert result.attempt.success is False
    assert result.attempt.artifact_paths == ()
    assert result.artifacts == ()
    assert result.attempt.phase_timings[0].name == "cythonize"
    assert expected_code in result.attempt.stderr
    assert (build_dir / "cython.log").is_file()
