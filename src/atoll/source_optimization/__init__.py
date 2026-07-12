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
from atoll.source_optimization.transforms import (
    DeclarationKind,
    GeneratedSourcePatch,
    SourceTransformationRequest,
    TransformedSourceFile,
    build_source_transformation_patch,
    materialize_transformed_files,
)

__all__ = (
    "DeclarationKind",
    "GeneratedSourcePatch",
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
    "SourceTransformationRequest",
    "TransformationStep",
    "TransformedSourceFile",
    "build_source_optimization_plans",
    "build_source_transformation_patch",
    "materialize_transformed_files",
    "stable_source_optimization_plan_id",
)
