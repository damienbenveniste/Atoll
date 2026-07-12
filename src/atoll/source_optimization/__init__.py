"""Source-optimization contracts for execution-plan-derived 3x lowering."""

from atoll.source_optimization.analysis import (
    SourceOptimizationPlanningOptions,
    SourceOptimizationPlanningResult,
    build_source_optimization_plans,
)
from atoll.source_optimization.models import (
    SourceAccessKind,
    SourceAccessSite,
    SourceCallableEvidence,
    SourceEdit,
    SourceOptimizationApplicationStatus,
    SourceOptimizationAssessment,
    SourceOptimizationAssessmentStatus,
    SourceOptimizationHazard,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    SourceOptimizationTrial,
    SourceOptimizationTrialStatus,
    SourceTransformationKind,
    TransformationStep,
    stable_source_optimization_plan_id,
)

__all__ = (
    "SourceAccessKind",
    "SourceAccessSite",
    "SourceCallableEvidence",
    "SourceEdit",
    "SourceOptimizationApplicationStatus",
    "SourceOptimizationAssessment",
    "SourceOptimizationAssessmentStatus",
    "SourceOptimizationHazard",
    "SourceOptimizationIdentity",
    "SourceOptimizationPlan",
    "SourceOptimizationPlanningOptions",
    "SourceOptimizationPlanningResult",
    "SourceOptimizationTrial",
    "SourceOptimizationTrialStatus",
    "SourceTransformationKind",
    "TransformationStep",
    "build_source_optimization_plans",
    "stable_source_optimization_plan_id",
)
