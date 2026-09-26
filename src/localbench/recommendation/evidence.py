"""Select one current, complete benchmark batch per model for recommendation."""

from __future__ import annotations

from collections import defaultdict
from statistics import median

from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.schemas import StoredResult
from localbench.estimation import ModelMemoryEstimate
from localbench.recommendation.schemas import RecommendationCandidate


def eligible_benchmark_batches(
    history: tuple[StoredResult, ...], runtime_version: str | None
) -> dict[str, tuple[StoredResult, ...]]:
    """Return the newest eligible batch per model under the current quick signature."""
    expected = quick_measurement_signature(runtime_version)
    if expected is None or runtime_version is None:
        return {}
    grouped: dict[str, list[StoredResult]] = defaultdict(list)
    order: list[str] = []
    for result in history:
        batch_id = result.batch_id or result.run_id
        if batch_id not in grouped:
            order.append(batch_id)
        grouped[batch_id].append(result)

    selected: dict[str, tuple[StoredResult, ...]] = {}
    for batch_id in order:
        batch = tuple(grouped[batch_id])
        first = batch[0]
        indices = {result.batch_index for result in batch}
        eligible = (
            len(batch) == first.batch_size
            and indices == set(range(1, first.batch_size + 1))
            and all(result.model == first.model for result in batch)
            and all(result.measurement_signature == expected for result in batch)
            and all(result.pre_run_unload_succeeded is True for result in batch)
            and all(
                result.generation_tokens == expected.generation_token_target for result in batch
            )
            and all(
                result.generation_tokens_per_second is not None
                and result.generation_tokens_per_second > 0
                and result.prompt_tokens_per_second is not None
                and result.prompt_tokens_per_second > 0
                and result.ttft_seconds is not None
                and result.ttft_seconds > 0
                for result in batch
            )
        )
        key = first.model.casefold()
        if eligible and key not in selected:
            selected[key] = batch
    return selected


def make_candidate(
    batch: tuple[StoredResult, ...], estimate: ModelMemoryEstimate
) -> RecommendationCandidate:
    """Combine an already-eligible batch with a current memory estimate."""
    generation = tuple(result.generation_tokens_per_second for result in batch)
    prompt = tuple(result.prompt_tokens_per_second for result in batch)
    ttft = tuple(result.ttft_seconds for result in batch)
    assert all(value is not None for value in (*generation, *prompt, *ttft))
    return RecommendationCandidate(
        model=batch[0].model,
        estimate=estimate,
        batch_id=batch[0].batch_id or batch[0].run_id,
        run_count=len(batch),
        generation_tokens_per_second=median(value for value in generation if value is not None),
        prompt_tokens_per_second=median(value for value in prompt if value is not None),
        ttft_seconds=median(value for value in ttft if value is not None),
    )
