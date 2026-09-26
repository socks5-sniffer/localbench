"""Context-length capacity sweep: does this model still fit as context grows?

A model that fits comfortably at 2K tokens of context does not necessarily fit at 32K — KV
cache grows linearly with context length while weights and runtime overhead stay fixed. This
module answers that by calling the existing `estimate_ollama_model` once per requested context
length, all against one shared available-memory snapshot (the same "one consistent snapshot"
principle `stack.plan_stack` uses), and reporting how the fit classification changes as context
grows.

This is a memory-growth profile only. The roadmap also calls for measuring real
prompt-processing and time-to-first-token degradation at each context length, which requires
live generation at every level — a materially bigger, slower feature. That is intentionally
not built here; see `docs/context-profiling.md` for what is and is not covered.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from localbench.context.schemas import (
    ContextCapacityProfile,
    ContextLevelEstimate,
    ContextProfileError,
)
from localbench.estimation.memory import DetailsFetcher, estimate_ollama_model
from localbench.estimation.schemas import EstimateError, KVCacheType
from localbench.models import GPUProfile, RuntimeProfile
from localbench.runtimes.ollama import fetch_model_details

DEFAULT_CONTEXT_LEVELS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536)


def _memoized_fetcher(fetch_details: DetailsFetcher) -> DetailsFetcher:
    """Ollama's model metadata does not change between context lengths in one sweep.

    `estimate_ollama_model` fetches it fresh on every call; without this, a six-level sweep
    would make six identical `/api/show` round trips for the same model.
    """
    cache: dict[str, object] = {}

    def _fetch(model: str, timeout: float) -> object:
        if model not in cache:
            cache[model] = fetch_details(model, timeout)
        return cache[model]

    return _fetch


def profile_context_capacity(
    requested_model: str,
    runtime: RuntimeProfile,
    available_memory_bytes: int,
    *,
    context_levels: Sequence[int] = DEFAULT_CONTEXT_LEVELS,
    kv_cache_type: KVCacheType = KVCacheType.F16,
    gpus: tuple[GPUProfile, ...] = (),
    fetch_details: DetailsFetcher = fetch_model_details,
) -> ContextCapacityProfile:
    """Estimate one model's memory footprint at every requested context length.

    Every level is estimated against the same `available_memory_bytes` snapshot the caller
    supplies. A level beyond the model's own context ceiling is skipped (not an error) and
    listed in `skipped_context_lengths`; any other estimation failure (unavailable runtime,
    ambiguous or missing model, insufficient metadata) applies to the whole model and is
    raised immediately as the underlying `EstimateError` rather than repeated per level.
    """
    if not context_levels:
        raise ContextProfileError(
            "empty_levels", "At least one context length must be requested."
        )
    if any(level <= 0 for level in context_levels):
        raise ContextProfileError(
            "invalid_context_length", "Every requested context length must be positive."
        )
    if len(set(context_levels)) != len(context_levels):
        raise ContextProfileError(
            "duplicate_context_length", "Requested context lengths must not repeat."
        )
    sorted_levels = tuple(sorted(context_levels))
    collected_at = datetime.now(UTC)

    cached_fetch = _memoized_fetcher(fetch_details)
    levels: list[ContextLevelEstimate] = []
    skipped: list[int] = []
    model_context_limit: int | None = None
    seen_assumptions: dict[str, None] = {}
    seen_warnings: dict[str, None] = {}

    for level in sorted_levels:
        try:
            result = estimate_ollama_model(
                requested_model,
                runtime,
                available_memory_bytes,
                context_length=level,
                kv_cache_type=kv_cache_type,
                gpus=gpus,
                fetch_details=cached_fetch,
            )
        except EstimateError as error:
            if error.code == "context_exceeds_limit":
                skipped.append(level)
                continue
            raise
        model_context_limit = result.model_context_limit
        levels.append(ContextLevelEstimate(context_length=level, estimate=result))
        for assumption in result.assumptions:
            seen_assumptions.setdefault(assumption, None)
        for warning in result.warnings:
            seen_warnings.setdefault(warning, None)

    if not levels:
        raise ContextProfileError(
            "no_levels_fit",
            f"None of the requested context lengths are supported by {requested_model!r}.",
        )

    assumptions = (
        *seen_assumptions,
        "This profile estimates memory only. It does not measure prompt-processing rate or "
        "time-to-first-token degradation as context grows — that requires live generation at "
        "every level and is not implemented yet.",
    )

    return ContextCapacityProfile(
        collected_at=collected_at,
        model=levels[0].estimate.model,
        kv_cache_type=kv_cache_type,
        model_context_limit=model_context_limit,
        levels=tuple(levels),
        skipped_context_lengths=tuple(skipped),
        assumptions=assumptions,
        warnings=tuple(seen_warnings),
    )
