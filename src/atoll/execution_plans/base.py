"""Backend protocol for execution-plan lowering.

Execution-plan backends are intentionally separate from native compiler
backends because scheduler lowering assesses orchestration topology, stages
payload files, and reports diagnostics without compiling typed regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atoll.execution_plans.models import (
    ExecutionPlan,
    ExecutionPlanAssessment,
    ExecutionPlanAssessmentContext,
    ExecutionPlanDiagnostic,
    ExecutionPlanStageContext,
    StagedExecutionPlan,
)


@runtime_checkable
class ExecutionPlanBackend(Protocol):
    """Structural contract implemented by scheduler execution-plan backends."""

    @property
    def name(self) -> str:
        """Return the stable execution-plan backend identifier.

        Returns:
            str: Backend identifier used in reports, staged files, and cache keys.
        """
        ...

    def assess(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanAssessmentContext,
    ) -> ExecutionPlanAssessment:
        """Classify whether this backend can lower an execution plan.

        Args:
            plan: Scheduler-aware execution plan discovered from source.
            context: Read-only project and profile context for assessment.

        Returns:
            ExecutionPlanAssessment: Deterministic capability decision.
        """
        ...

    def stage(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> StagedExecutionPlan:
        """Stage backend payload files for an accepted execution plan.

        Args:
            plan: Scheduler-aware execution plan to stage.
            context: Filesystem boundary for payload and cache writes.

        Returns:
            StagedExecutionPlan: Staged files and guards needed by later trials.
        """
        ...

    def fingerprint(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> str:
        """Return a strict backend-and-content cache fingerprint for a plan.

        Args:
            plan: Scheduler-aware execution plan being fingerprinted.
            context: Filesystem boundary that can affect staged output.

        Returns:
            str: Stable digest covering plan content and backend semantics.
        """
        ...

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> ExecutionPlanDiagnostic:
        """Convert backend exceptions and output into stable diagnostic fields.

        Args:
            error: Backend exception that caused execution-plan staging or trial to fail.
            diagnostics: Captured backend diagnostic text to normalize.
            log_path: Optional path to a complete backend log.

        Returns:
            ExecutionPlanDiagnostic: Backend-independent diagnostic suitable for reports.
        """
        ...
