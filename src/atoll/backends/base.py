"""Backend-neutral compiler contract used by source-clean typed regions.

The protocol separates capability decisions, registration of prepared source,
native compilation, cache fingerprinting, and diagnostic normalization. It does
not select a backend or generate runtime shims; command orchestration owns those
decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atoll.models import (
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BackendDiagnostic,
    BackendLoweringRequest,
    CompilationUnit,
    TypedRegion,
)


class UnsupportedBackendRegionError(ValueError):
    """Raised when lowering requests members the backend did not accept."""


@runtime_checkable
class CompilerBackend(Protocol):
    """Structural contract implemented by every native compiler backend."""

    @property
    def name(self) -> Backend:
        """Return the stable backend identifier used in reports and cache keys.

        Returns:
            Backend: Stable backend identifier used in reports and cache keys.
        """
        ...

    def assess(self, region: TypedRegion) -> BackendAssessment:
        """Classify supported members without mutating or compiling source.

        Args:
            region: Backend-neutral typed region being assessed or generated.

        Returns:
            BackendAssessment: Deterministic capability assessment for the region.
        """
        ...

    def lower(self, request: BackendLoweringRequest) -> CompilationUnit:
        """Register prepared source as a backend-specific compilation unit.

        Args:
            request: Prepared source and member selection offered to the backend.

        Returns:
            CompilationUnit: Validated backend-specific compilation unit.
        """
        ...

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Compile units and return normalized artifact and attempt evidence.

        Args:
            units: Backend compilation units submitted as one build request.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            BackendCompileResult: Structured build attempt and recorded native artifacts.
        """
        ...

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a strict toolchain-and-content cache fingerprint for a unit.

        Args:
            unit: Content-addressable backend compilation unit.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            str: Stable digest covering source, backend, options, and environment.
        """
        ...

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> BackendDiagnostic:
        """Convert backend exceptions and output into stable diagnostic fields.

        Args:
            error: Backend exception that caused compilation to fail.
            diagnostics: Captured backend diagnostic text to normalize.
            log_path: Optional path to the complete backend build log.

        Returns:
            BackendDiagnostic: Backend-independent diagnostic suitable for reports.
        """
        ...
