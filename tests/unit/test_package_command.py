"""Unit tests for source-clean package artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.machinery
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll import cli as cli_module
from atoll.commands import package as package_command
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BindingTarget,
    Blocker,
    CompilationUnit,
    CompileAttempt,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    LoweringDecision,
    ModuleId,
    ModuleScan,
    RegionSpecialization,
    SymbolId,
    TypedRegion,
)
from atoll.project import DiscoveredProject, discover_project
from atoll.report import CompilationReportInput, build_compilation_report
from atoll.runtime.package_verify import PackageVerificationResult, VerificationStage
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    BenchmarkProgress,
    CommandRunEvidence,
    RuntimeMode,
)

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
TYPED_FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
EXPECTED_REBUILDS = 2
EXPECTED_PARTIAL_CACHE_BACKEND_CALLS = 3
EXPECTED_READINESS_COUNT = 2
EXPECTED_ATOMIC_SELECTION_COUNT = 2
TEST_FAILURE_RETURN_CODE = 9


@pytest.fixture(autouse=True)
def stub_native_subprocess_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package orchestration tests stop at the separately tested verification boundary."""

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)


class _Metadata(Protocol):
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


class _CacheLookup(Protocol):
    hit: bool


class _BaselinePayloadFactory(Protocol):
    def __call__(
        self,
        *,
        wheel_path: Path | None,
        build: CompileAttempt,
        baseline_install_root: Path | None = None,
        quality_project_root: Path | None = None,
    ) -> object: ...


class _QualityGateOutcomeView(Protocol):
    success: bool
    tests: tuple[CommandRunEvidence, ...]
    performance: BenchmarkGateResult
    error: str | None


class _TypedSelection(Protocol):
    backend: Backend
    variant_id: str
    region: TypedRegion
    assessment: BackendAssessment
    members: tuple[SymbolId, ...]
    specialization: RegionSpecialization | None
    conditional_on_failure_of: str | None


class _TypedGeneration(Protocol):
    backend: Backend
    region: TypedRegion
    bindings: tuple[BindingTarget, ...]


class _PreparedTypedRegion(Protocol):
    generation: _TypedGeneration
    assessment: BackendAssessment
    unit: CompilationUnit
    fallback: _PreparedTypedRegion | None
    conditional_on_failure_of: str | None


class _TypedRegionOutcome(Protocol):
    successful: tuple[_PreparedTypedRegion, ...]
    build: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]
    skipped: tuple[_TypedRegionFailure, ...]


class _TypedRegionFailure(Protocol):
    variant_id: str


class _FakeCompileBackend:
    """Backend stub used to force deterministic retry orchestration."""

    def __init__(self, result: BackendCompileResult) -> None:
        self.result = result
        self.calls: list[tuple[CompilationUnit, ...]] = []
        self.name = cast(Backend, result.attempt.command[0])

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a stable per-variant key for cache orchestration tests."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Record one invocation and return configured compiler evidence."""
        _ = context
        self.calls.append(units)
        return self.result


def _package_attr(name: str) -> object:
    return vars(package_command)[name]


_copy_atoll_artifacts = cast(
    Callable[[tuple[Path, ...], Path], None],
    _package_attr("_copy_atoll_artifacts"),
)
_copy_if_different = cast(
    Callable[[Path, Path], None],
    _package_attr("_copy_if_different"),
)
_compile_cache_key = cast(Callable[..., str], _package_attr("_compile_cache_key"))
_copy_source_roots = cast(
    Callable[[DiscoveredProject, Path], tuple[Path, ...]],
    _package_attr("_copy_source_roots"),
)
_artifact_dir = cast(
    Callable[[EnabledIslandConfig], Path],
    _package_attr("_artifact_dir"),
)
_cached_artifact_paths = cast(
    Callable[[Path, dict[str, object]], tuple[Path, ...] | None],
    _package_attr("_cached_artifact_paths"),
)
_cached_manifest_modules = cast(
    Callable[[dict[str, object]], tuple[tuple[str, ...], tuple[str, ...]] | None],
    _package_attr("_cached_manifest_modules"),
)
_compile_cache_hit = cast(Callable[..., _CacheLookup], _package_attr("_compile_cache_hit"))
_find_module = cast(
    Callable[[tuple[ModuleId, ...], str], ModuleId],
    _package_attr("_find_module"),
)
_mapping = cast(
    Callable[[object], dict[str, object]],
    _package_attr("_mapping"),
)
_project_metadata = cast(
    Callable[[Path], _Metadata],
    _package_attr("_project_metadata"),
)
_read_cache_manifest = cast(
    Callable[[Path], dict[str, object] | None],
    _package_attr("_read_cache_manifest"),
)
_relative_source_root = cast(
    Callable[[Path, Path], Path],
    _package_attr("_relative_source_root"),
)
_reset_dir = cast(Callable[[Path], None], _package_attr("_reset_dir"))
_sequence = cast(
    Callable[[object], tuple[object, ...]],
    _package_attr("_sequence"),
)
_staged_module = cast(
    Callable[[ModuleId, DiscoveredProject, tuple[Path, ...]], ModuleId],
    _package_attr("_staged_module"),
)
_store_compile_cache = cast(Callable[..., None], _package_attr("_store_compile_cache"))
_string = cast(Callable[[object], str | None], _package_attr("_string"))
_selected_scans = cast(
    Callable[[DiscoveredProject, str | None], tuple[ModuleScan, ...]],
    _package_attr("_selected_scans"),
)
_selected_typed_method_regions = cast(
    Callable[[tuple[ModuleScan, ...]], tuple[_TypedSelection, ...]],
    _package_attr("_selected_typed_method_regions"),
)
_prepare_typed_region = cast(
    Callable[..., _PreparedTypedRegion], _package_attr("_prepare_typed_region")
)
_build_typed_regions = cast(
    Callable[..., _TypedRegionOutcome], _package_attr("_build_typed_regions")
)
_TypedRegionBuildContext = cast(Callable[..., object], _package_attr("_TypedRegionBuildContext"))
_compiler_backends = cast(dict[Backend, object], _package_attr("_COMPILER_BACKENDS"))
_member_requires_source_class = cast(
    Callable[[str], bool],
    _package_attr("_member_requires_source_class"),
)
_owner_disallows_method_binding = cast(
    Callable[[str | None, str, dict[str, LoweringDecision]], bool],
    _package_attr("_owner_disallows_method_binding"),
)
_BaselineWheelPayload = cast(
    _BaselinePayloadFactory,
    _package_attr("_BaselineWheelPayload"),
)
_run_configured_quality_gate = cast(
    Callable[..., _QualityGateOutcomeView],
    _package_attr("_run_configured_quality_gate"),
)
_print_source_clean_success = cast(
    Callable[..., None],
    vars(cli_module)["_print_source_clean_success"],
)


def test_typed_region_selection_prefers_mypyc_for_safe_specializations(
    tmp_path: Path,
) -> None:
    """Subclass and closed-call specializations enter the normal automatic routing path."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    worker_selections = _selected_typed_method_regions(
        _selected_scans(project, "typed_region_project.worker")
    )
    worker_specializations = tuple(
        selection.specialization
        for selection in worker_selections
        if selection.specialization is not None
    )
    function_selections = _selected_typed_method_regions(
        _selected_scans(project, "typed_region_project.generic_functions")
    )

    assert {
        (specialization.origin, specialization.target_owner_class)
        for specialization in worker_specializations
    } == {
        ("concrete_subclass", "IntPairer"),
        ("concrete_subclass", "PayloadPairer"),
    }
    assert all(
        selection.backend == "mypyc"
        for selection in worker_selections
        if selection.specialization is not None
    )
    assert len(function_selections) == 1
    assert function_selections[0].backend == "mypyc"
    assert function_selections[0].specialization is not None
    assert function_selections[0].specialization.origin == "closed_call"


def test_atomic_class_selection_is_exclusive_and_partial_classes_split(
    tmp_path: Path,
) -> None:
    """A closed class has one class variant while mixed shapes remain per-member."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    selections = _selected_typed_method_regions(
        _selected_scans(project, "typed_region_project.worker")
    )
    scale_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("ScaleModel") for member in selection.members)
    )
    worker_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("Worker") for member in selection.members)
    )

    class_selection = next(
        selection
        for selection in scale_selections
        if selection.variant_id.endswith("@cython-class")
    )
    method_fallback = next(
        selection for selection in scale_selections if selection is not class_selection
    )
    assert len(scale_selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    assert class_selection.backend == "cython"
    assert tuple(member.qualname for member in class_selection.members) == ("ScaleModel",)
    assert method_fallback.backend == "mypyc"
    assert {member.qualname for member in method_fallback.members} == {
        "ScaleModel.apply",
        "ScaleModel.describe",
    }
    assert method_fallback.conditional_on_failure_of == class_selection.variant_id
    assert {member.qualname for selection in worker_selections for member in selection.members} == {
        "Worker.adjust",
        "Worker.exchange",
        "Worker.parse",
        "Worker.scale",
        "Worker.score",
        "Worker.values",
    }
    assert {
        selection.backend
        for selection in worker_selections
        if any(member.qualname == "Worker.exchange" for member in selection.members)
    } == {"cython"}
    assert all(
        member.qualname not in {"Worker", "Worker.__init__"}
        for selection in worker_selections
        for member in selection.members
    )


def test_method_selection_rejects_class_cell_and_private_name_semantics() -> None:
    """Top-level extraction never guesses class-cell or name-mangling behavior."""
    assert _member_requires_source_class("def value(self) -> int:\n    return super().value()\n")
    assert _member_requires_source_class("def owner(self) -> type[object]:\n    return __class__\n")
    assert _member_requires_source_class("def secret(self) -> int:\n    return self.__secret\n")
    assert not _member_requires_source_class(
        "def regular(self) -> int:\n    return len(self.__dict__)\n"
    )


def test_method_selection_preserves_registered_and_dynamic_owner_classes() -> None:
    """Method mutation is rejected when an owner may be replaced or intercept writes."""
    registered = LoweringDecision(
        target="module::Registered",
        action="fallback",
        reason="class remains interpreted because decorators may register or replace it",
    )
    eager = LoweringDecision(
        target="module::Eager",
        action="fallback",
        reason="class remains interpreted because module-time code retains its original identity",
    )

    assert _owner_disallows_method_binding(
        "Registered",
        "module",
        {registered.target: registered},
    )
    assert not _owner_disallows_method_binding(
        "Eager",
        "module",
        {eager.target: eager},
    )


def test_package_reports_build_failure_without_source_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package mode reports mypyc failures and still leaves checkout source untouched."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert args
        assert kwargs
        return CompileAttempt(
            success=False,
            command=("mypyc",),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: fixture",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "MYPYC_TYPE_ERROR: fixture"
    assert result.wheel_path is None
    assert source_path.read_text(encoding="utf-8") == original_source
    assert (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)


@pytest.mark.parametrize("failed_stage", ["payload", "wheel"])
def test_package_preserves_payload_for_subprocess_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: VerificationStage,
) -> None:
    """Routing failures keep the exact payload and rejected wheel under diagnostics."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        success = stage != failed_stage
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=success,
            exit_code=0 if success else 1,
            stdout="",
            stderr="routing failed" if not success else "",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)
    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "routing failed"
    assert result.cleanup_removed == ()
    assert result.cleanup_kept == (output_dir / "build", output_dir / "install")
    assert (output_dir / "install").exists()
    assert not tuple(output_dir.glob("*.whl"))
    assert result.verification_steps[-1].stage == failed_stage
    if failed_stage == "wheel":
        assert result.verification_steps[-1].target.parent == output_dir / "build" / "diagnostics"


def test_typed_region_build_retries_deterministic_mypyc_failure_with_cython(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mypyc type rejection uses the prepared Cython variant before fallback."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_method_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert outcome.build.stderr == ""
    assert "compiled" in outcome.build.stdout
    assert [item.generation.backend for item in outcome.successful] == ["cython"]
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_does_not_compile_speculative_cython_after_mypyc_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared Cython fallback remains dormant when the preferred backend succeeds."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_method_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert [item.generation.backend for item in outcome.successful] == ["mypyc"]
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert cython.calls == []


@pytest.mark.parametrize("class_succeeds", [True, False])
def test_atomic_class_build_conditionally_uses_method_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_succeeds: bool,
) -> None:
    """Method variants stay dormant unless the selected atomic class fails."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    selections = _selected_typed_method_regions(
        _selected_scans(project, "typed_region_project.worker")
    )
    class_selection = next(
        selection for selection in selections if selection.variant_id.endswith("@cython-class")
    )
    method_selection = next(
        selection
        for selection in selections
        if selection.conditional_on_failure_of == class_selection.variant_id
    )
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = tuple(
        _prepare_typed_region(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            selection=selection,
        )
        for selection in (class_selection, method_selection)
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=class_succeeds,
                command=("cython",),
                stdout="",
                stderr="" if class_succeeds else "CYTHON_COMPILE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=prepared,
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    if class_succeeds:
        assert [item.unit.region_id for item in outcome.successful] == [class_selection.variant_id]
        assert outcome.skipped == ()
        assert mypyc.calls == []
    else:
        assert [item.unit.region_id for item in outcome.successful] == [method_selection.variant_id]
        assert [failure.variant_id for failure in outcome.skipped] == [class_selection.variant_id]
        assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_records_real_cython_artifacts_after_mypyc_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic retry compiles a real Cython artifact owned by its variant."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_method_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    successful = outcome.successful[0]
    variant = CompiledRegionVariant(
        id=successful.unit.region_id,
        region=successful.generation.region,
        backend=successful.generation.backend,
        bindings=successful.generation.bindings,
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="compile",
            module_filter="typed_region_project.worker",
            islands=(),
            build=outcome.build,
            typed_regions=(variant.region,),
            compiled_regions=(variant.region,),
            compiled_bindings=variant.bindings,
            compiled_variants=(variant,),
            backend_assessments=(successful.assessment,),
            artifact_records=outcome.artifacts,
        )
    )

    assert outcome.build.success is True
    assert successful.generation.backend == "cython"
    assert successful.unit.region_id.endswith("@cython-mypyc-fallback")
    assert outcome.artifacts
    assert all(path.is_file() for path in outcome.build.artifact_paths)
    assert {artifact.region_id for artifact in outcome.artifacts} == {variant.id}
    assert report["compiled_regions"][0]["backend"] == "cython"
    assert report["compiled_regions"][0]["variant_id"] == variant.id
    assert report["compiled_regions"][0]["artifacts"]


def test_package_whole_project_retries_and_skips_failed_islands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode keeps buildable islands when one island fails."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "good.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "bad.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def mixed_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        if path.stem.endswith("_good"):
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            return CompileAttempt(
                success=True,
                command=("mypyc", str(path)),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.1,
            )
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: bad failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", mixed_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    good_text = (output_dir / "install" / "app" / "good.py").read_text(encoding="utf-8")
    bad_text = (output_dir / "install" / "app" / "bad.py").read_text(encoding="utf-8")
    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert tuple(island.source_module for island in result.islands) == ("app.good",)
    assert tuple(failure.island.source_module for failure in result.skipped) == ("app.bad",)
    assert "Initial batch build failed; retried islands individually" in result.build.stdout
    assert "# BEGIN ATOLL MANAGED: app.good" in good_text
    assert "# BEGIN ATOLL MANAGED" not in bad_text
    assert (output_dir / "install" / ".atoll" / "artifacts" / f"_atoll_app_good{suffix}").exists()
    assert result.wheel_path is not None


def test_package_reports_progress_for_expensive_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package builds expose phase progress to the CLI."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    messages: list[str] = []

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            progress=messages.append,
        )
    )

    assert result.success is True
    assert any(message.startswith("discovered ") for message in messages)
    assert any(message.startswith("scanned ") for message in messages)
    assert any(message.startswith("running mypyc batch") for message in messages)
    assert any(message.startswith("writing wheel") for message in messages)


def test_quality_gate_rejects_missing_source_stripped_project(tmp_path: Path) -> None:
    """Configured commands cannot silently fall back to the target checkout."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "pass"]',),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "quality-gate project is missing"
    assert outcome.performance.status == "invalid"


def test_quality_gate_rejects_missing_benchmark_baseline(tmp_path: Path) -> None:
    """Benchmarking requires a distinct unpacked baseline payload."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=tmp_path / "quality-project",
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "baseline payload is missing"


def test_quality_gate_reports_semantic_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed configured test stops before benchmarking and preserves exit evidence."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "raise SystemExit(9)"]',),
    )
    quality_root = tmp_path / "quality-project"
    quality_root.mkdir()
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=quality_root,
    )
    messages: list[str] = []

    def failing_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9,
            stdout="",
            stderr="",
            duration_seconds=0.5,
        )

    monkeypatch.setattr(package_command, "run_performance_command", failing_test)

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=messages.append,
    )

    assert outcome.success is False
    assert outcome.error == "compiled semantic test command exited 9"
    assert outcome.tests[0].returncode == TEST_FAILURE_RETURN_CODE
    assert outcome.performance.reason == "compiled semantic test command failed"
    assert messages == ["compiled semantic tests failed with exit 9 in 0.50s"]


def test_source_clean_success_summary_lists_every_fallback_kind(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI success output distinguishes build, preflight, and typed-region fallbacks."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan = _selected_scans(project, "typed_region_project.worker")[0]
    selection = _selected_typed_method_regions((scan,))[0]
    island = EnabledIslandConfig(
        source_module=scan.module.name,
        source_path=scan.module.path,
        sidecar_module="_atoll_fixture",
        sidecar_path=tmp_path / "_atoll_fixture.py",
        symbols=("passthrough",),
    )
    failed_attempt = CompileAttempt(
        success=False,
        command=("compiler",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.1,
    )
    result = package_command.PackageCommandResult(
        success=True,
        project_root=project_root,
        output_dir=tmp_path / "dist",
        install_root=tmp_path / "dist" / "install",
        wheel_path=tmp_path / "dist" / "fixture.whl",
        islands=(island,),
        build=_successful_attempt(),
        install_tree_kept=True,
        skipped=(package_command.PackageBuildFailure(island=island, build=failed_attempt),),
        preflight_skipped=(
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(Blocker(severity="hard", code="module", message="module blocker"),),
            ),
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(
                    Blocker(
                        severity="hard",
                        code="line",
                        message="line blocker",
                        lineno=7,
                    ),
                ),
            ),
        ),
        region_skipped=(
            package_command.PackageRegionBuildFailure(
                region=selection.region,
                variant_id=selection.variant_id,
                backend=selection.backend,
                assessment=selection.assessment,
                build=failed_attempt,
            ),
        ),
        performance=BenchmarkGateResult(
            status="passed",
            reason="fixture",
            minimum_speedup=1.1,
            baseline_median_seconds=1.2,
            compiled_median_seconds=1.0,
            speedup=1.2,
            warmups=(),
            samples=(),
        ),
    )

    _print_source_clean_success(
        result,
        label="source-clean compile",
        report_paths=(tmp_path / "report.json", tmp_path / "report.md"),
    )

    output = capsys.readouterr().out
    assert "Skipped 1 module(s) that mypyc could not build." in output
    assert f"- {scan.module.name}: failed" in output
    assert f"- {scan.module.name}: module: module blocker" in output
    assert f"- {scan.module.name}: line 7: line blocker" in output
    assert "Kept 1 typed region(s) as interpreted fallback." in output
    assert f"- {selection.variant_id} [{selection.backend}]: failed" in output
    assert "Install tree:" in output
    assert "Performance: 1.200x median speedup (passed)." in output


def test_package_rejects_not_profitable_wheel_after_semantic_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured benchmark below threshold removes the candidate wheel."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                'test_command = ["python", "-m", "pytest", "-q"]',
                'benchmark_command = ["python", "bench.py"]',
                "benchmark_warmups = 0",
                "benchmark_samples = 1",
                "minimum_speedup = 1.10",
                "",
            )
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    test_modes: list[RuntimeMode] = []
    target_project_root = project_root

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def passing_test_command(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        test_modes.append(mode)
        assert project_root != target_project_root
        assert not tuple((project_root / "src").rglob("*.py"))
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="passed",
            stderr="",
            duration_seconds=0.5,
        )

    def rejecting_benchmark(
        config: BenchmarkGateConfig,
        *,
        project_root: Path,
        baseline_payload_root: Path,
        compiled_payload_root: Path,
        progress: Callable[[BenchmarkProgress], None] | None = None,
    ) -> BenchmarkGateResult:
        assert config.command == ("python", "bench.py")
        assert project_root == project_root.resolve()
        assert baseline_payload_root != compiled_payload_root
        assert progress is not None
        return BenchmarkGateResult(
            status="not-profitable",
            reason="compiled median speedup 1.020 is below threshold 1.100",
            minimum_speedup=1.1,
            baseline_median_seconds=1.02,
            compiled_median_seconds=1.0,
            speedup=1.02,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)
    monkeypatch.setattr(package_command, "run_performance_command", passing_test_command)
    monkeypatch.setattr(package_command, "run_benchmark_gate", rejecting_benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.performance is not None
    assert result.performance.status == "not-profitable"
    assert result.error == result.performance.reason
    assert test_modes == ["baseline", "compiled"]
    assert not tuple(output_dir.glob("*.whl"))
    assert (output_dir / "install").exists()
    assert (output_dir / "build").exists()
    diagnostic_wheels = tuple((output_dir / "build" / "diagnostics").glob("*.whl"))
    assert diagnostic_wheels
    assert result.verification_steps[-1].target == diagnostic_wheels[0]


def test_package_reuses_compile_cache_for_unchanged_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged second source-clean package build restores cached artifacts."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        if calls > 1:
            raise AssertionError("compile cache did not skip mypyc")
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
            phase_timings=(
                CompilePhaseTiming(name="mypycify", duration_seconds=0.08),
                CompilePhaseTiming(name="build_ext", duration_seconds=0.02),
            ),
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.success is True
    assert first.build.cache_status == "miss"
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert calls == 1
    assert tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    ) == (
        "cache_lookup",
        "cache_restore",
    )
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        assert any(
            name.startswith(".atoll/artifacts/_atoll_app_ranking") for name in wheel.namelist()
        )


def test_package_reuses_partial_compile_cache_for_unchanged_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial source-clean builds cache successful artifacts and cached skips."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    ranking_source = project_root / "src" / "app" / "ranking.py"
    extra_source = project_root / "src" / "app" / "extra.py"
    extra_source.write_text(ranking_source.read_text(encoding="utf-8"), encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def partial_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        paths = cast(tuple[Path, ...], args[0])
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        assert paths
        path = paths[0]
        if "extra" in path.stem:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="extra failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", partial_build_sidecars)

    first = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )
    second = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )

    assert first.success is True
    assert first.build.cache_status == "partial"
    assert len(first.skipped) == 1
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert len(second.skipped) == 1
    assert "cached skip" in second.skipped[0].build.stderr
    assert calls == EXPECTED_PARTIAL_CACHE_BACKEND_CALLS
    assert tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    ) == (
        "cache_lookup",
        "cache_restore",
    )
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        names = set(wheel.namelist())
    assert any(name.startswith(".atoll/artifacts/_atoll_app_ranking") for name in names)
    assert "app/extra.py" in names
    with zipfile.ZipFile(second.wheel_path) as wheel:
        extra_text = wheel.read("app/extra.py").decode()
    assert "BEGIN ATOLL MANAGED" not in extra_text


def test_package_cache_invalidates_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package cache keys include the target source tree digest."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    ranking_source = project_root / "src" / "app" / "ranking.py"
    ranking_source.write_text(
        f"{ranking_source.read_text(encoding='utf-8')}\n# cache invalidation\n",
        encoding="utf-8",
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == EXPECTED_REBUILDS


def test_package_cache_invalidates_when_generator_version_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package cache keys include the sidecar generator version."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    monkeypatch.setattr(package_command, "SIDECAR_GENERATOR_VERSION", "changed")
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == EXPECTED_REBUILDS


def test_compile_cache_key_includes_selected_symbols(tmp_path: Path) -> None:
    """Source-clean compile cache keys distinguish selected island symbol sets."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    sidecar_path = project_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    source_path = project_root / "src" / "app" / "ranking.py"
    base = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=source_path,
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=sidecar_path,
        symbols=("score_user",),
    )
    expanded = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=source_path,
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=sidecar_path,
        symbols=("score_user", "rank_candidates"),
    )

    assert _compile_cache_key(
        target_project=project,
        module_name="app.ranking",
        islands=(base,),
    ) != _compile_cache_key(
        target_project=project,
        module_name="app.ranking",
        islands=(expanded,),
    )


def test_compile_cache_manifest_helpers_reject_invalid_entries(tmp_path: Path) -> None:
    """Compile cache manifests must be present, typed, and digest-matched."""
    assert _read_cache_manifest(tmp_path / "missing.json") is None

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert _read_cache_manifest(invalid_json) is None

    non_mapping_json = tmp_path / "list.json"
    non_mapping_json.write_text("[]", encoding="utf-8")
    assert _read_cache_manifest(non_mapping_json) is None

    entry_root = tmp_path / "entry"
    artifact_root = entry_root / "artifacts"
    artifact_root.mkdir(parents=True)
    artifact = artifact_root / "module.so"
    artifact.write_text("binary", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert _cached_artifact_paths(entry_root, {"artifacts": "bad"}) is None
    assert _cached_artifact_paths(entry_root, {"artifacts": [1]}) is None
    assert (
        _cached_artifact_paths(entry_root, {"artifacts": [{"name": 1, "sha256": digest}]}) is None
    )
    assert (
        _cached_artifact_paths(
            entry_root,
            {"artifacts": [{"name": "module.so", "sha256": "bad"}]},
        )
        is None
    )
    assert _cached_artifact_paths(
        entry_root,
        {"artifacts": [{"name": "module.so", "sha256": digest}]},
    ) == (artifact,)
    assert _cached_manifest_modules({}) is None
    assert _cached_manifest_modules({"successful_modules": ["app.a"]}) is None
    assert (
        _cached_manifest_modules({"successful_modules": ["app.a", 1], "skipped_modules": []})
        is None
    )
    assert (
        _cached_manifest_modules({"successful_modules": ["app.a"], "skipped_modules": ["app.a"]})
        is None
    )
    assert _cached_manifest_modules(
        {"successful_modules": ["app.a"], "skipped_modules": ["app.b"]}
    ) == (("app.a",), ("app.b",))

    stale_lookup = _compile_cache_hit(
        key="abc",
        lookup_started=0.0,
        artifact_paths=(tmp_path / "missing.so",),
        successful_modules=("app.a",),
        skipped_modules=(),
    )
    assert stale_lookup.hit is False


def test_store_compile_cache_handles_empty_and_existing_temp_dirs(tmp_path: Path) -> None:
    """Compile cache storage ignores empty artifacts and replaces stale temp dirs."""
    cache_root = tmp_path / "cache"
    _store_compile_cache(
        cache_root=cache_root,
        key="empty",
        artifact_paths=(),
        successful_modules=(),
        skipped_modules=(),
    )
    assert not cache_root.exists()

    artifact = tmp_path / "module.so"
    artifact.write_text("binary", encoding="utf-8")
    stale_temp = cache_root / "abc.tmp"
    stale_temp.mkdir(parents=True)
    (stale_temp / "stale").write_text("old", encoding="utf-8")

    _store_compile_cache(
        cache_root=cache_root,
        key="abc",
        artifact_paths=(artifact,),
        successful_modules=("app.a",),
        skipped_modules=("app.b",),
    )

    manifest = _read_cache_manifest(cache_root / "abc" / "manifest.json")
    assert manifest is not None
    assert manifest["successful_modules"] == ["app.a"]
    assert manifest["skipped_modules"] == ["app.b"]
    assert not stale_temp.exists()


def test_package_whole_project_reports_zero_successful_retries_concisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode reports one representative error when all retries fail."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "first.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "second.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr=f"MYPYC_TYPE_ERROR: {path.stem} failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("No selected islands compiled")
    assert result.error.count("MYPYC_TYPE_ERROR") == 1
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)
    assert tuple(failure.island.source_module for failure in result.skipped) == (
        "app.first",
        "app.second",
    )


def test_package_rejects_trivial_selected_module_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated candidates without repeated native work never reach mypyc."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    bad_module = project_root / "src" / "app" / "typing_features.py"
    bad_module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', infer_variance=True)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_calls: list[tuple[Path, ...]] = []

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        build_calls.append(paths)
        return CompileAttempt(
            success=False,
            command=("mypyc", *(str(path) for path in paths)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: generated sidecar failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.typing_features",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert build_calls == []
    assert result.error is not None
    assert "No performance-worthy native islands" in result.error
    assert result.preflight_skipped == ()
    assert not (output_dir / "build").exists()
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert result.cleanup_kept == ()
    assert len(result.native_readiness) == EXPECTED_READINESS_COUNT
    assert all(not readiness.eligible for readiness in result.native_readiness)


def test_package_whole_project_skips_non_native_ready_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project mode builds ready kernels and omits rejected modules."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    clean_source = package_dir / "ranking.py"
    original_clean_source = clean_source.read_text(encoding="utf-8")
    (package_dir / "blocked.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert {island.source_module for island in result.islands} == {"app.ranking"}
    assert result.islands[0].symbols == ("normalize_features",)
    assert result.preflight_skipped == ()
    assert "# BEGIN ATOLL MANAGED: app.ranking" in (
        output_dir / "install" / "app" / "ranking.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN ATOLL MANAGED: app.blocked" not in (
        output_dir / "install" / "app" / "blocked.py"
    ).read_text(encoding="utf-8")
    blocked_readiness = tuple(
        readiness
        for readiness in result.native_readiness
        if readiness.source_module == "app.blocked"
    )
    assert blocked_readiness
    assert all(not readiness.eligible for readiness in blocked_readiness)
    assert clean_source.read_text(encoding="utf-8") == original_clean_source


def test_package_helpers_handle_flat_source_roots(tmp_path: Path) -> None:
    """Flat source roots copy their contents into the build root."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    project = discover_project(tmp_path)
    build_root = tmp_path / "build"
    build_root.mkdir()

    staged_roots = _copy_source_roots(project, build_root)

    assert staged_roots == (build_root,)
    assert (build_root / "pkg" / "mod.py").exists()


def test_atoll_artifact_helpers_copy_artifacts_and_skip_same_file(tmp_path: Path) -> None:
    """Source-clean artifact helpers place compiled extensions under install `.atoll`."""
    source_root = tmp_path / "source"
    artifact_dir = source_root / ".atoll" / "artifacts"
    install_root = tmp_path / "install"
    artifact_dir.mkdir(parents=True)
    native = artifact_dir / f"_sidecar{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    native.write_text("binary", encoding="utf-8")

    _copy_atoll_artifacts((tmp_path / "missing", source_root), install_root)
    copied = install_root / ".atoll" / "artifacts" / native.name
    _copy_if_different(copied, copied)

    package_sidecar = EnabledIslandConfig(
        source_module="pkg.mod",
        source_path=source_root / "pkg" / "mod.py",
        sidecar_module="pkg._sidecar",
        sidecar_path=source_root / "pkg" / "_sidecar.py",
        symbols=("func",),
    )
    external_sidecar = EnabledIslandConfig(
        source_module="pkg.mod",
        source_path=source_root / "pkg" / "mod.py",
        sidecar_module="pkg._sidecar",
        sidecar_path=source_root / ".atoll" / "sidecars" / "_sidecar.py",
        symbols=("func",),
    )

    assert copied.read_text(encoding="utf-8") == "binary"
    assert _artifact_dir(package_sidecar) == source_root / "pkg"
    assert _artifact_dir(external_sidecar) == source_root / ".atoll" / "artifacts"


def test_project_metadata_falls_back_for_missing_or_dynamic_version(tmp_path: Path) -> None:
    """Project metadata falls back to stable Atoll values when version is dynamic."""
    fallback = _project_metadata(tmp_path)
    assert fallback.name == tmp_path.name
    assert fallback.version == "0+atoll"

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "dynamic-project"',
                'dynamic = ["version"]',
                'requires-python = ">=3.12"',
                'dependencies = ["pydantic>=2"]',
            ]
        ),
        encoding="utf-8",
    )

    metadata = _project_metadata(project_root)
    assert metadata.name == "dynamic-project"
    assert metadata.version == "0+atoll"
    assert metadata.requires_python == ">=3.12"
    assert metadata.dependencies == ("pydantic>=2",)


def test_package_small_helpers_cover_fallbacks(tmp_path: Path) -> None:
    """Small helper fallbacks stay deterministic."""
    path = tmp_path / "existing"
    path.mkdir()
    (path / "old.txt").write_text("old", encoding="utf-8")
    _reset_dir(path)
    assert path.exists()
    assert not (path / "old.txt").exists()

    assert _relative_source_root(tmp_path, tmp_path / "src") == Path("src")
    outside_root = tmp_path.parent / "not-under-root"
    assert _relative_source_root(tmp_path, outside_root) != outside_root
    assert _mapping(None) == {}
    assert _sequence(None) == ()
    assert _string(1) is None


def test_package_helpers_report_missing_modules(tmp_path: Path) -> None:
    """Module lookup helpers fail clearly for impossible paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    project = DiscoveredProject(
        config=discover_project(project_root).config,
        modules=(),
    )

    with pytest.raises(ValueError, match="module not found"):
        _find_module((), "missing")
    with pytest.raises(ValueError, match="outside configured source roots"):
        _staged_module(
            ModuleId(name="missing", path=tmp_path / "outside.py"),
            project,
            (tmp_path / "stage",),
        )


def _quality_gate_project(
    tmp_path: Path,
    compile_lines: tuple[str, ...],
) -> DiscoveredProject:
    project_root = tmp_path / "quality-gate-project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(("", "[tool.atoll.compile]", *compile_lines, "")),
        encoding="utf-8",
    )
    return discover_project(project_root)


def _successful_attempt() -> CompileAttempt:
    return CompileAttempt(
        success=True,
        command=("fixture",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.0,
    )
