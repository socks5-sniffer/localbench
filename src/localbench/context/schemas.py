"""Data contracts for the context-length capacity sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from localbench.estimation.schemas import KVCacheType, ModelMemoryEstimate
from localbench.models import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ContextLevelEstimate:
    """One requested context length's full explainable memory estimate."""

    context_length: int
    estimate: ModelMemoryEstimate


@dataclass(frozen=True, slots=True)
class ContextCapacityProfile:
    """How one model's estimated memory footprint grows across context lengths.

    This profiles memory growth only — the same `estimate_ollama_model` used by
    `localbench estimate`, swept across multiple context lengths sharing one available-memory
    snapshot. It does not measure prompt-processing or time-to-first-token degradation under
    load at each length; that requires live generation and is documented as separate,
    unimplemented future work rather than silently omitted.
    """

    collected_at: datetime
    model: str
    kv_cache_type: KVCacheType
    model_context_limit: int | None
    levels: tuple[ContextLevelEstimate, ...]
    skipped_context_lengths: tuple[int, ...]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["collected_at"] = self.collected_at.isoformat()
        # `asdict` recurses generically and does not run `ModelMemoryEstimate.to_dict`'s
        # own `collected_at` conversion, so each level is re-serialized explicitly here.
        result["levels"] = [
            {"context_length": level.context_length, "estimate": level.estimate.to_dict()}
            for level in self.levels
        ]
        return result


class ContextProfileError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
