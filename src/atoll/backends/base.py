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
        """Return the stable backend identifier used in reports and cache keys."""
        ...

    def assess(self, region: TypedRegion) -> BackendAssessment:
        """Classify supported members without mutating or compiling source."""
        ...

    def lower(self, request: BackendLoweringRequest) -> CompilationUnit:
        """Register prepared source as a backend-specific compilation unit."""
        ...

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Compile units and return normalized artifact and attempt evidence."""
        ...

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a strict toolchain-and-content cache fingerprint for a unit."""
        ...

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> BackendDiagnostic:
        """Convert backend exceptions and output into stable diagnostic fields."""
        ...
