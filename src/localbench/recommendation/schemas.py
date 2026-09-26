"""Explainable contracts for evidence-based local model recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from localbench.estimation import ModelMemoryEstimate
from localbench.models import SCHEMA_VERSION


class RecommendationUseCase(StrEnum):
    GENERAL_CHAT = "general-chat"
    CODING = "coding"
    REASONING = "reasoning"
    RAG = "rag"
    ROBOTICS = "robotics"
    AGENTS = "agents"
    LOW_LATENCY = "low-latency"
    LONG_CONTEXT = "long-context"


@dataclass(frozen=True, slots=True)
class RecommendationCandidate:
    """Comparable local evidence for one installed model."""

    model: str
    estimate: ModelMemoryEstimate
    batch_id: str
    run_count: int
    generation_tokens_per_second: float
    prompt_tokens_per_second: float
    ttft_seconds: float


@dataclass(frozen=True, slots=True)
class RecommendationWeights:
    generation: float
    prompt: float
    ttft: float
    memory: float


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    rank: int
    model: str
    batch_id: str
    run_count: int
    context_length: int
    fit: str
    estimated_total_bytes: int
    generation_tokens_per_second: float
    prompt_tokens_per_second: float
    ttft_seconds: float
    generation_score: float
    prompt_score: float
    ttft_score: float
    memory_score: float
    weighted_score: float


@dataclass(frozen=True, slots=True)
class RecommendationReport:
    use_case: RecommendationUseCase
    weights: RecommendationWeights
    recommendations: tuple[RankedRecommendation, ...]
    excluded_models: tuple[str, ...]
    caveats: tuple[str, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecommendationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
