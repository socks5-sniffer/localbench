"""Data contracts for explainable simultaneous-memory stack plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from localbench.estimation.schemas import FitClassification, MemoryComponent, ModelMemoryEstimate
from localbench.models import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class StackWorkloadRequest:
    """One named role in a proposed stack, backed by an existing single-model estimate."""

    role: str
    estimate: ModelMemoryEstimate


@dataclass(frozen=True, slots=True)
class StackModelPlan:
    """The per-model breakdown of one role's contribution to the stack, safety headroom excluded."""

    role: str
    model: str
    weights_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int
    working_memory_bytes: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StackPlan:
    collected_at: datetime
    models: tuple[StackModelPlan, ...]
    total_working_memory_bytes: int
    safety_headroom: MemoryComponent
    total_required_bytes: int
    available_memory: MemoryComponent
    headroom_bytes: int
    viability: FitClassification
    viability_basis: str
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["collected_at"] = self.collected_at.isoformat()
        return result


class StackPlanError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
