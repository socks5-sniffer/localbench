from __future__ import annotations

import dataclasses

import pytest

from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.schemas import StoredResult
from localbench.estimation import ModelMemoryEstimate
from localbench.estimation.schemas import FitClassification
from localbench.recommendation import (
    RecommendationCandidate,
    RecommendationError,
    RecommendationUseCase,
    eligible_benchmark_batches,
    make_candidate,
    recommend_models,
)


def _candidate(
    model: str,
    estimate: ModelMemoryEstimate,
    *,
    generation: float,
    prompt: float,
    ttft: float,
) -> RecommendationCandidate:
    return RecommendationCandidate(
        model=model,
        estimate=dataclasses.replace(estimate, model=model),
        batch_id=f"batch-{model}",
        run_count=3,
        generation_tokens_per_second=generation,
        prompt_tokens_per_second=prompt,
        ttft_seconds=ttft,
    )


def _stored(
    run_id: str,
    model: str,
    *,
    batch_id: str | None = None,
    batch_index: int = 1,
    batch_size: int = 1,
    runtime_version: str = "1.2.3",
    generation: float = 10.0,
    prompt: float = 30.0,
    ttft: float = 2.0,
) -> StoredResult:
    signature = quick_measurement_signature(runtime_version)
    assert signature is not None
    return StoredResult(
        run_id=run_id,
        started_at="2026-09-01T12:00:00+00:00",
        model=model,
        prompt_id=signature.prompt_id,
        generation_tokens=signature.generation_token_target,
        generation_tokens_per_second=generation,
        prompt_tokens_per_second=prompt,
        ttft_seconds=ttft,
        load_time_seconds=1.0,
        runtime="ollama",
        runtime_version=runtime_version,
        pre_run_unload_succeeded=True,
        measurement_signature=signature,
        batch_id=batch_id or run_id,
        batch_index=batch_index,
        batch_size=batch_size,
    )


def test_use_case_weights_can_change_the_recommendation(
    sample_estimate: ModelMemoryEstimate,
) -> None:
    candidates = (
        _candidate("throughput:latest", sample_estimate, generation=20, prompt=30, ttft=4),
        _candidate("responsive:latest", sample_estimate, generation=10, prompt=30, ttft=1),
    )

    balanced = recommend_models(candidates, RecommendationUseCase.GENERAL_CHAT)
    responsive = recommend_models(candidates, RecommendationUseCase.LOW_LATENCY)

    assert balanced.recommendations[0].model == "throughput:latest"
    assert responsive.recommendations[0].model == "responsive:latest"
    assert responsive.weights.ttft == 0.50


def test_long_context_prioritizes_memory_fit(sample_estimate: ModelMemoryEstimate) -> None:
    poor = dataclasses.replace(sample_estimate, fit=FitClassification.WILL_NOT_FIT)
    excellent = dataclasses.replace(sample_estimate, fit=FitClassification.EXCELLENT)
    candidates = (
        _candidate("fast:latest", poor, generation=20, prompt=40, ttft=1),
        _candidate("fits:latest", excellent, generation=18, prompt=38, ttft=1.2),
    )

    report = recommend_models(candidates, RecommendationUseCase.LONG_CONTEXT)

    assert report.recommendations[0].model == "fits:latest"
    assert report.recommendations[0].memory_score == 1.0


def test_report_exposes_components_and_quality_caveat(
    sample_estimate: ModelMemoryEstimate,
) -> None:
    report = recommend_models(
        (_candidate("example:latest", sample_estimate, generation=10, prompt=20, ttft=2),),
        RecommendationUseCase.CODING,
        excluded_models=("unbenchmarked:latest — no current benchmark",),
    )

    item = report.recommendations[0]
    assert item.weighted_score == pytest.approx(1.0)
    assert report.excluded_models == ("unbenchmarked:latest — no current benchmark",)
    assert any("does not measure answer quality" in caveat for caveat in report.caveats)
    assert report.to_dict()["schema_version"] == "1"


def test_rejects_empty_or_mixed_context_evidence(
    sample_estimate: ModelMemoryEstimate,
) -> None:
    with pytest.raises(RecommendationError, match="No installed model"):
        recommend_models((), RecommendationUseCase.GENERAL_CHAT)

    candidates = (
        _candidate("one", sample_estimate, generation=10, prompt=20, ttft=2),
        _candidate(
            "two",
            dataclasses.replace(sample_estimate, context_length=16384),
            generation=11,
            prompt=21,
            ttft=1.8,
        ),
    )
    with pytest.raises(RecommendationError) as caught:
        recommend_models(candidates, RecommendationUseCase.GENERAL_CHAT)
    assert caught.value.code == "mixed_contexts"


def test_selects_newest_complete_current_batch_per_model() -> None:
    history = (
        _stored("new-1", "example:latest", batch_id="new", batch_index=1, batch_size=2),
        _stored("new-2", "example:latest", batch_id="new", batch_index=2, batch_size=2),
        _stored("old", "example:latest", generation=99),
        _stored("stale", "other:latest", runtime_version="0.9.0"),
    )

    selected = eligible_benchmark_batches(history, "1.2.3")

    assert tuple(result.run_id for result in selected["example:latest"]) == ("new-1", "new-2")
    assert "other:latest" not in selected


def test_make_candidate_uses_batch_medians(sample_estimate: ModelMemoryEstimate) -> None:
    batch = (
        _stored("one", "example:latest", generation=8, prompt=20, ttft=3),
        _stored("two", "example:latest", generation=10, prompt=30, ttft=2),
        _stored("three", "example:latest", generation=30, prompt=40, ttft=1),
    )

    candidate = make_candidate(batch, sample_estimate)

    assert candidate.generation_tokens_per_second == 10
    assert candidate.prompt_tokens_per_second == 30
    assert candidate.ttft_seconds == 2
