"""Execution-plan contracts and built-in scheduler dialects."""

from atoll.execution_plans.anyio_task_preserving import (
    ANYIO_TASK_PRESERVING_BACKEND,
    AnyioTaskPreservingExecutionPlanBackend,
)
from atoll.execution_plans.base import ExecutionPlanBackend
from atoll.execution_plans.callback_backed import (
    CALLBACK_BACKED_BACKEND,
    CallbackBackedExecutionPlanBackend,
)
from atoll.execution_plans.dialects import (
    AnyioOnAsyncioDialect,
    AsyncioDialect,
    SchedulerDialect,
    built_in_scheduler_dialects,
)
from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanAssessment,
    ExecutionPlanAssessmentContext,
    ExecutionPlanDiagnostic,
    ExecutionPlanIdentity,
    ExecutionPlanStageContext,
    ExecutionPlanTrial,
    PlanEdge,
    PlanGuard,
    PlanNode,
    PlanRejection,
    StagedExecutionPlan,
    stable_execution_plan_id,
)
from atoll.execution_plans.task_preserving import (
    TASK_PRESERVING_BACKEND,
    TaskPreservingExecutionPlanBackend,
)

__all__ = (
    "ANYIO_TASK_PRESERVING_BACKEND",
    "CALLBACK_BACKED_BACKEND",
    "TASK_PRESERVING_BACKEND",
    "AnyioOnAsyncioDialect",
    "AnyioTaskPreservingExecutionPlanBackend",
    "AsyncioDialect",
    "CallbackBackedExecutionPlanBackend",
    "ChangedPayloadFile",
    "ExecutionPlan",
    "ExecutionPlanAssessment",
    "ExecutionPlanAssessmentContext",
    "ExecutionPlanBackend",
    "ExecutionPlanDiagnostic",
    "ExecutionPlanIdentity",
    "ExecutionPlanStageContext",
    "ExecutionPlanTrial",
    "PlanEdge",
    "PlanGuard",
    "PlanNode",
    "PlanRejection",
    "SchedulerDialect",
    "StagedExecutionPlan",
    "TaskPreservingExecutionPlanBackend",
    "built_in_scheduler_dialects",
    "stable_execution_plan_id",
)
