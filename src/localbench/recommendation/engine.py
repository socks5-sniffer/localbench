"""Transparent ranking of models with compatible local benchmark evidence."""

from __future__ import annotations

from collections.abc import Sequence

from localbench.estimation.schemas import FitClassification
from localbench.recommendation.schemas import (
    RankedRecommendation,
    RecommendationCandidate,
    RecommendationError,
    RecommendationReport,
    RecommendationUseCase,
    RecommendationWeights,
)


def weights_for(use_case: RecommendationUseCase) -> RecommendationWeights:
    if use_case in (RecommendationUseCase.ROBOTICS, RecommendationUseCase.LOW_LATENCY):
        return RecommendationWeights(generation=0.20, prompt=0.10, ttft=0.50, memory=0.20)
    if use_case is RecommendationUseCase.RAG:
        return RecommendationWeights(generation=0.20, prompt=0.40, ttft=0.15, memory=0.25)
    if use_case is RecommendationUseCase.LONG_CONTEXT:
        return RecommendationWeights(generation=0.15, prompt=0.25, ttft=0.15, memory=0.45)
    return RecommendationWeights(generation=0.35, prompt=0.15, ttft=0.30, memory=0.20)


def _higher_is_better(values: Sequence[float]) -> tuple[float, ...]:
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return tuple(1.0 for _value in values)
    return tuple((value - lowest) / (highest - lowest) for value in values)


def _lower_is_better(values: Sequence[float]) -> tuple[float, ...]:
    if max(values) == min(values):
        return tuple(1.0 for _value in values)
    return tuple(1.0 - score for score in _higher_is_better(values))


_MEMORY_SCORES = {
    FitClassification.EXCELLENT: 1.0,
    FitClassification.GOOD: 0.75,
    FitClassification.TIGHT: 0.50,
    FitClassification.POOR: 0.25,
    FitClassification.WILL_NOT_FIT: 0.0,
}


def recommend_models(
    candidates: Sequence[RecommendationCandidate],
    use_case: RecommendationUseCase,
    *,
    excluded_models: Sequence[str] = (),
) -> RecommendationReport:
    """Rank comparable measurements while exposing every score and policy weight."""
    if not candidates:
        raise RecommendationError(
            "no_eligible_models",
            "No installed model has a complete benchmark and usable memory estimate.",
        )
    if any(
        value <= 0
        for candidate in candidates
        for value in (
            candidate.generation_tokens_per_second,
            candidate.prompt_tokens_per_second,
            candidate.ttft_seconds,
        )
    ):
        raise RecommendationError(
            "invalid_evidence", "Recommendation metrics must all be greater than zero."
        )
    contexts = {candidate.estimate.context_length for candidate in candidates}
    if len(contexts) != 1:
        raise RecommendationError(
            "mixed_contexts", "All recommendation estimates must use the same context length."
        )

    weights = weights_for(use_case)
    generation_scores = _higher_is_better(
        [candidate.generation_tokens_per_second for candidate in candidates]
    )
    prompt_scores = _higher_is_better(
        [candidate.prompt_tokens_per_second for candidate in candidates]
    )
    ttft_scores = _lower_is_better([candidate.ttft_seconds for candidate in candidates])
    memory_scores = tuple(_MEMORY_SCORES[candidate.estimate.fit] for candidate in candidates)

    scored: list[tuple[float, RecommendationCandidate, float, float, float, float]] = []
    for candidate, generation, prompt, ttft, memory in zip(
        candidates,
        generation_scores,
        prompt_scores,
        ttft_scores,
        memory_scores,
        strict=True,
    ):
        score = (
            generation * weights.generation
            + prompt * weights.prompt
            + ttft * weights.ttft
            + memory * weights.memory
        )
        scored.append((score, candidate, generation, prompt, ttft, memory))
    scored.sort(
        key=lambda item: (
            item[1].estimate.fit is FitClassification.WILL_NOT_FIT,
            -item[0],
            -item[1].generation_tokens_per_second,
            item[1].ttft_seconds,
            item[1].model.casefold(),
        )
    )

    ranked = tuple(
        RankedRecommendation(
            rank=rank,
            model=candidate.model,
            batch_id=candidate.batch_id,
            run_count=candidate.run_count,
            context_length=candidate.estimate.context_length,
            fit=candidate.estimate.fit.value,
            estimated_total_bytes=candidate.estimate.estimated_total_bytes,
            generation_tokens_per_second=candidate.generation_tokens_per_second,
            prompt_tokens_per_second=candidate.prompt_tokens_per_second,
            ttft_seconds=candidate.ttft_seconds,
            generation_score=generation,
            prompt_score=prompt,
            ttft_score=ttft,
            memory_score=memory,
            weighted_score=score,
        )
        for rank, (score, candidate, generation, prompt, ttft, memory) in enumerate(
            scored, start=1
        )
    )
    caveats = (
        "Ranking covers measured speed and estimated memory fit only; it does not measure "
        "answer quality, correctness, or task capability.",
        "Scores are relative to the eligible models in this report, not a universal grade.",
        "Only complete runs matching the current quick workload and runtime version are eligible.",
    )
    return RecommendationReport(
        use_case=use_case,
        weights=weights,
        recommendations=ranked,
        excluded_models=tuple(excluded_models),
        caveats=caveats,
    )
