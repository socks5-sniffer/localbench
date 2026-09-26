"""Explainable simultaneous-memory capacity planning for a proposed model stack."""

from localbench.stack.planner import STACK_SAFETY_HEADROOM_BYTES, plan_stack
from localbench.stack.schemas import (
    StackModelPlan,
    StackPlan,
    StackPlanError,
    StackWorkloadRequest,
)

__all__ = [
    "STACK_SAFETY_HEADROOM_BYTES",
    "StackModelPlan",
    "StackPlan",
    "StackPlanError",
    "StackWorkloadRequest",
    "plan_stack",
]
