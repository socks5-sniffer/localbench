"""Evidence-only model recommendations."""

from localbench.recommendation.engine import recommend_models, weights_for
from localbench.recommendation.evidence import eligible_benchmark_batches, make_candidate
from localbench.recommendation.schemas import (
    RankedRecommendation,
    RecommendationCandidate,
    RecommendationError,
    RecommendationReport,
    RecommendationUseCase,
    RecommendationWeights,
)

__all__ = [
    "RankedRecommendation",
    "RecommendationCandidate",
    "RecommendationError",
    "RecommendationReport",
    "RecommendationUseCase",
    "RecommendationWeights",
    "eligible_benchmark_batches",
    "make_candidate",
    "recommend_models",
    "weights_for",
]
