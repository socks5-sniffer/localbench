"""Data contracts for explainable model-memory estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from localbench.models import SCHEMA_VERSION


class KVCacheType(StrEnum):
    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


class FitClassification(StrEnum):
    EXCELLENT = "excellent"
    GOOD = "good"
    TIGHT = "tight"
    POOR = "poor"
    WILL_NOT_FIT = "will_not_fit"


class GPUOffloadClassification(StrEnum):
    FULL_OFFLOAD = "full_offload"
    PARTIAL_OFFLOAD = "partial_offload"
    CPU_ONLY = "cpu_only"


@dataclass(frozen=True, slots=True)
class MemoryComponent:
    bytes: int
    provenance: Literal["measured", "reported", "calculated", "inferred"]
    basis: str


@dataclass(frozen=True, slots=True)
class GPUOffloadEstimate:
    gpu_name: str
    vram: MemoryComponent
    reserved_vram: MemoryComponent
    total_layers: int
    offloadable_layers: int
    offload_fraction: float
    classification: GPUOffloadClassification
    basis: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelMemoryEstimate:
    collected_at: datetime
    model: str
    context_length: int
    model_context_limit: int | None
    kv_cache_type: KVCacheType
    weights: MemoryComponent
    kv_cache: MemoryComponent
    runtime_overhead: MemoryComponent
    safety_headroom: MemoryComponent
    working_memory_bytes: int
    estimated_total_bytes: int
    available_memory: MemoryComponent
    fit: FitClassification
    fit_basis: str
    gpu_offload: GPUOffloadEstimate | None
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["collected_at"] = self.collected_at.isoformat()
        return result


class EstimateError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
