"""Source-optimization contracts for execution-plan-derived 3x lowering."""

from atoll.source_optimization.analysis import (
    SourceOptimizationPlanningOptions,
    SourceOptimizationPlanningResult,
    build_source_optimization_plans,
)
from atoll.source_optimization.application import (
    SourcePatchApplicationResult,
    apply_source_patch_transactionally,
    validate_source_application_root,
)
from atoll.source_optimization.cache import (
    SourcePatchCacheResult,
    restore_or_build_source_patch,
)
from atoll.source_optimization.lowering import (
    SourceLoweringMode,
    SourceLoweringResult,
    SourceLoweringStatus,
    lower_batch_quiescent_plan,
    lower_state_machine_plan,
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
from atoll.source_optimization.search import (
    SourceOptimizationSearchOptions,
    SourceOptimizationSearchResult,
    run_source_optimization_search,
)
from atoll.source_optimization.transforms import (
    CallableBodyReplacement,
    DeclarationKind,
    GeneratedSourcePatch,
    SourceTransformationRequest,
    TransformedSourceFile,
    build_source_transformation_patch,
    materialize_transformed_files,
)

__all__ = (
    "CallableBodyReplacement",
    "DeclarationKind",
    "GeneratedSourcePatch",
    "SourceAccessKind",
    "SourceAccessSite",
    "SourceCallableEvidence",
    "SourceEdit",
    "SourceLoweringMode",
    "SourceLoweringResult",
    "SourceLoweringStatus",
    "SourceOptimizationApplicationStatus",
    "SourceOptimizationAssessment",
    "SourceOptimizationAssessmentStatus",
    "SourceOptimizationHazard",
    "SourceOptimizationIdentity",
    "SourceOptimizationPlan",
    "SourceOptimizationPlanningOptions",
    "SourceOptimizationPlanningResult",
    "SourceOptimizationSearchOptions",
    "SourceOptimizationSearchResult",
    "SourceOptimizationTrial",
    "SourceOptimizationTrialStatus",
    "SourcePatchApplicationResult",
    "SourcePatchCacheResult",
    "SourceTransformationKind",
    "SourceTransformationRequest",
    "TransformationStep",
    "TransformedSourceFile",
    "apply_source_patch_transactionally",
    "build_source_optimization_plans",
    "build_source_transformation_patch",
    "lower_batch_quiescent_plan",
    "lower_state_machine_plan",
    "materialize_transformed_files",
    "restore_or_build_source_patch",
    "run_source_optimization_search",
    "stable_source_optimization_plan_id",
    "validate_source_application_root",
)
