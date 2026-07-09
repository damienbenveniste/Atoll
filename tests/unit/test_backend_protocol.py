"""Tests for backend protocol conformance and mypyc capability decisions."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from textwrap import dedent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.base import CompilerBackend, UnsupportedBackendRegionError
from atoll.backends.mypyc import MYPYC_BACKEND, MypycBackend
from atoll.models import (
    ArtifactRecord,
    BackendCompileContext,
    BackendDiagnostic,
    BackendLoweringRequest,
    CompilationUnit,
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


def test_mypyc_backend_structurally_conforms_to_compiler_backend() -> None:
    """The concrete adapter and singleton satisfy the runtime protocol."""
    backend = MypycBackend()

    assert isinstance(backend, CompilerBackend)
    assert isinstance(MYPYC_BACKEND, CompilerBackend)
    assert backend.name == "mypyc"


def test_backend_diagnostics_reserve_a_cython_specific_compile_code() -> None:
    """The shared diagnostic model does not force Cython failures into mypyc labels."""
    diagnostic = BackendDiagnostic(
        code="CYTHON_COMPILE_ERROR",
        message="fixture",
        details=(),
        log_path=None,
        transient=False,
    )

    assert diagnostic.code == "CYTHON_COMPILE_ERROR"


def test_mypyc_assesses_typed_functions_and_callable_shapes(tmp_path: Path) -> None:
    """Typed functions, methods, generators, and coroutines report capabilities."""
    regions = _regions(
        tmp_path,
        "capabilities",
        """
        def typed(value: int) -> int:
            return value + 1

        def stream(limit: int) -> int:
            for value in range(limit):
                yield value

        async def fetch(value: int) -> int:
            return value

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
    backend = MypycBackend()

    function_assessments = {
        member.id.qualname: backend.assess(region)
        for region in regions
        for member in region.members
        if member.id.qualname in {"typed", "stream", "fetch"}
    }
    assert function_assessments["typed"].status == "supported"
    assert "typed_function" in function_assessments["typed"].capabilities
    assert function_assessments["stream"].status == "supported"
    assert "generator" in function_assessments["stream"].capabilities
    assert function_assessments["fetch"].status == "supported"
    assert "coroutine" in function_assessments["fetch"].capabilities

    worker_region = _region_containing(regions, "Worker", "Worker.scale")
    worker_assessment = backend.assess(worker_region)

    assert worker_region.atomic_class is True
    assert worker_assessment.status == "supported"
    assert set(worker_assessment.capabilities) >= {
        "native_class",
        "instance_method",
        "staticmethod",
        "classmethod",
    }
    assert {member.qualname for member in worker_assessment.supported_members} == {
        "Worker",
        "Worker.scale",
        "Worker.parse",
        "Worker.build",
    }


def test_async_generator_partially_rejects_without_poisoning_methods(tmp_path: Path) -> None:
    """An async generator blocks itself and the class, not supported methods."""
    regions = _regions(
        tmp_path,
        "async_shapes",
        """
        from collections.abc import AsyncIterator

        class Worker:
            def scale(self, value: int) -> int:
                return value * 2

            async def score(self, value: int) -> int:
                return value

            async def stream(self, limit: int) -> AsyncIterator[int]:
                for value in range(limit):
                    yield value
        """,
    )
    region = _region_containing(regions, "Worker", "Worker.scale", "Worker.stream")

    assessment = MypycBackend().assess(region)

    assert assessment.status == "partial"
    assert _member(region, "Worker.scale") in assessment.supported_members
    assert _member(region, "Worker.score") in assessment.supported_members
    assert _member(region, "Worker.stream") in assessment.unsupported_members
    assert _member(region, "Worker") in assessment.unsupported_members
    assert "instance_method" in assessment.capabilities
    assert "coroutine" in assessment.capabilities
    assert "async_generator" not in assessment.capabilities
    assert any("async-generator execution semantics" in reason for reason in assessment.reasons)


def test_any_and_generic_regions_are_rejected_for_mypyc_fallback(tmp_path: Path) -> None:
    """Any boxing and unresolved generic fallback remain unsupported for mypyc."""
    regions = _regions(
        tmp_path,
        "typing_fallbacks",
        """
        import typing

        def dynamic(value: typing.Any) -> typing.Any:
            return value

        def identity[T](value: T) -> T:
            return value
        """,
    )
    backend = MypycBackend()

    dynamic = backend.assess(_region_containing(regions, "dynamic"))
    identity = backend.assess(_region_containing(regions, "identity"))

    assert dynamic.status == "unsupported"
    assert [member.qualname for member in dynamic.unsupported_members] == ["dynamic"]
    assert dynamic.reasons == (
        "typing_fallbacks::dynamic: mypyc preference requires concrete typing; source uses Any",
    )
    assert identity.status == "unsupported"
    assert [member.qualname for member in identity.unsupported_members] == ["identity"]
    assert identity.reasons == (
        "typing_fallbacks::identity: member requires interpreted fallback before specialization",
    )


def test_lower_validates_requested_members_and_records_source_hash(tmp_path: Path) -> None:
    """Lowering accepts supported subsets and rejects unsupported selections."""
    regions = _regions(
        tmp_path,
        "lowering",
        """
        from collections.abc import AsyncIterator

        class Worker:
            def scale(self, value: int) -> int:
                return value * 2

            async def stream(self, limit: int) -> AsyncIterator[int]:
                for value in range(limit):
                    yield value
        """,
    )
    source_path = tmp_path / "lowering.py"
    region = _region_containing(regions, "Worker.scale", "Worker.stream")
    backend = MypycBackend()
    scale = _member(region, "Worker.scale")

    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=source_path,
            logical_module="lowering",
            members=(scale,),
        )
    )

    assert unit.region_id == region.id
    assert unit.backend == "mypyc"
    assert unit.logical_module == "lowering"
    assert unit.source_paths == (source_path,)
    assert unit.members == (scale,)
    assert len(unit.source_hash) == SHA256_HEX_LENGTH

    with pytest.raises(UnsupportedBackendRegionError, match=r"lowering::Worker\.stream"):
        backend.lower(
            BackendLoweringRequest(
                region=region,
                source_path=source_path,
                logical_module="lowering",
                members=(_member(region, "Worker.stream"),),
            )
        )


def test_fingerprint_is_deterministic_and_invalidates_on_source_change(tmp_path: Path) -> None:
    """Source content participates in stable backend cache fingerprints."""
    regions = _regions(
        tmp_path,
        "fingerprint",
        """
        def score(value: int) -> int:
            return value + 1
        """,
    )
    source_path = tmp_path / "fingerprint.py"
    region = _region_containing(regions, "score")
    backend = MypycBackend()
    request = BackendLoweringRequest(
        region=region,
        source_path=source_path,
        logical_module="fingerprint",
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

    alternate_root = tmp_path / "alternate"
    alternate_root.mkdir()
    alternate_context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "other-build",
        source_roots=(alternate_root,),
        backend_options=(("opt_level", "2"),),
    )
    assert backend.fingerprint(changed_unit, alternate_context) != backend.fingerprint(
        changed_unit,
        context,
    )


@pytest.mark.parametrize(
    ("error", "diagnostics", "expected_code", "transient"),
    [
        (
            RuntimeError("mypy type failure"),
            "module.py:1: error: fixture failure  [misc]",
            "MYPYC_TYPE_ERROR",
            False,
        ),
        (RuntimeError("compiler missing"), "", "NATIVE_BUILD_ENV_ERROR", True),
        (ImportError("module unavailable"), "", "IMPORT_PATH_ERROR", True),
        (RuntimeError("unexpected"), "", "UNKNOWN_BUILD_ERROR", True),
    ],
)
def test_mypyc_normalizes_backend_diagnostics(
    tmp_path: Path,
    error: BaseException,
    diagnostics: str,
    expected_code: str,
    transient: bool,
) -> None:
    """Backend exceptions become stable codes while retaining diagnostic evidence."""
    log_path = tmp_path / "mypyc.log"

    diagnostic = MypycBackend().normalize_diagnostic(
        error,
        diagnostics=diagnostics,
        log_path=log_path,
    )

    assert diagnostic.code == expected_code
    assert diagnostic.log_path == log_path
    assert diagnostic.transient is transient
    if diagnostics:
        assert "Captured 1 mypyc error line(s)." in diagnostic.details


def test_mypyc_rejects_incompatible_or_sourceless_units(tmp_path: Path) -> None:
    """Compilation-unit validation fails before invoking a mismatched backend."""
    backend = MypycBackend()
    incompatible = CompilationUnit(
        region_id="fixture",
        backend="cython",
        logical_module="fixture",
        source_paths=(tmp_path / "fixture.py",),
        source_hash="fixture",
        members=(),
    )
    source_less = CompilationUnit(
        region_id="source-less",
        backend="mypyc",
        logical_module="fixture",
        source_paths=(),
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


@pytest.mark.parametrize(
    "path",
    ["", "/absolute/fixture.so", "../fixture.so", r"pkg\fixture.so"],
)
def test_artifact_records_require_relative_posix_install_paths(path: str) -> None:
    """Artifact metadata cannot escape or use platform-specific payload paths."""
    with pytest.raises(ValueError, match="relative POSIX path"):
        ArtifactRecord(
            region_id="fixture",
            backend="mypyc",
            logical_module="fixture",
            role="primary",
            install_relative_path=path,
            digest="0" * SHA256_HEX_LENGTH,
            abi="abi",
            platform_tag="platform",
        )


@pytest.mark.parametrize(
    ("digest", "abi", "platform_tag", "message"),
    [
        ("invalid", "abi", "platform", "SHA-256"),
        ("0" * SHA256_HEX_LENGTH, "", "platform", "ABI"),
        ("0" * SHA256_HEX_LENGTH, "abi", "", "platform"),
    ],
)
def test_artifact_records_validate_integrity_metadata(
    digest: str,
    abi: str,
    platform_tag: str,
    message: str,
) -> None:
    """Artifact records reject malformed digests and missing compatibility tags."""
    with pytest.raises(ValueError, match=message):
        ArtifactRecord(
            region_id="fixture",
            backend="mypyc",
            logical_module="fixture",
            role="primary",
            install_relative_path="fixture.so",
            digest=digest,
            abi=abi,
            platform_tag=platform_tag,
        )
