"""Reading back and comparing stored benchmark runs.

The governing rule here: two runs are only comparable if their canonical measurement
signatures match exactly and both runs are eligible. This is not defensive politeness —
a stale result from a superseded workload can
be numerically *better* than a correct one (see the v1 premature-stop bug, where a run
that generated 3 tokens reported a higher tokens/second than the same model's full
64-token v2 run), so silently comparing them would actively mislead.
"""

from __future__ import annotations

from dataclasses import fields
from statistics import median

from localbench.benchmark.schemas import (
    BenchmarkComparison,
    BenchmarkError,
    MetricSpread,
    StoredResult,
)


def _percent_change(baseline: float | None, candidate: float | None) -> float | None:
    """Percent change from baseline to candidate, or None if it cannot be computed."""
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100


def find_result(history: tuple[StoredResult, ...], run_reference: str) -> StoredResult:
    """Resolve a run by full id or unambiguous id prefix.

    Full run ids are UUIDs and unpleasant to type, so a prefix is accepted — but an
    ambiguous prefix is an error rather than a silent pick, matching how model selection
    already behaves elsewhere.
    """
    exact = [result for result in history if result.run_id == run_reference]
    if exact:
        return exact[0]
    matches = [result for result in history if result.run_id.startswith(run_reference)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(result.run_id for result in matches)
        raise BenchmarkError(
            "ambiguous_run",
            f"Run reference {run_reference!r} is ambiguous; choose one of: {choices}.",
        )
    raise BenchmarkError("run_not_found", f"No stored benchmark run matches {run_reference!r}.")


def results_for_batch(
    history: tuple[StoredResult, ...], selected: StoredResult
) -> tuple[StoredResult, ...]:
    batch_id = selected.batch_id or selected.run_id
    return tuple(result for result in history if (result.batch_id or result.run_id) == batch_id)


def _spread(values: tuple[float | None, ...]) -> MetricSpread:
    available = tuple(value for value in values if value is not None)
    return MetricSpread(
        count=len(available),
        median=median(available) if available else None,
        minimum=min(available) if available else None,
        maximum=max(available) if available else None,
    )


def _validate_group(group: tuple[StoredResult, ...]) -> None:
    if not group:
        raise BenchmarkError("empty_batch", "A benchmark batch cannot be empty.")
    expected = group[0].batch_size
    batch_id = group[0].batch_id or group[0].run_id
    indices = {result.batch_index for result in group}
    if (
        len(group) != expected
        or indices != set(range(1, expected + 1))
        or any(result.batch_size != expected for result in group)
        or any((result.batch_id or result.run_id) != batch_id for result in group)
        or any(result.model != group[0].model for result in group)
    ):
        raise BenchmarkError(
            "incomplete_batch",
            f"Batch {(group[0].batch_id or group[0].run_id)[:8]} has {len(group)} of "
            f"{expected} planned runs and cannot be compared yet.",
        )


def compare_result_groups(
    baseline_group: tuple[StoredResult, ...],
    candidate_group: tuple[StoredResult, ...],
) -> BenchmarkComparison:
    """Compare singleton-to-singleton or complete batch-to-batch using medians."""
    _validate_group(baseline_group)
    _validate_group(candidate_group)
    if (len(baseline_group) == 1) != (len(candidate_group) == 1):
        raise BenchmarkError(
            "incomparable_sample_sizes",
            "A singleton run cannot be compared as equivalent evidence to a repeated batch.",
        )
    baseline = baseline_group[0]
    candidate = candidate_group[0]
    if baseline.prompt_id != candidate.prompt_id:
        raise BenchmarkError(
            "incomparable_workloads",
            (
                f"Run {baseline.run_id[:8]} used workload {baseline.prompt_id} but run "
                f"{candidate.run_id[:8]} used {candidate.prompt_id}. Results from "
                "different workload versions do not measure the same thing and cannot be "
                "compared; re-run both models on the current workload instead."
            ),
        )
    baseline_signature = baseline.measurement_signature
    candidate_signature = candidate.measurement_signature
    if baseline_signature is None or candidate_signature is None:
        raise BenchmarkError(
            "unknown_measurement_signature",
            "At least one run predates a reconstructable measurement signature and cannot "
            "be compared safely; re-run it with the current LocalBench version.",
        )
    if (
        baseline_signature.runtime_version is None
        or candidate_signature.runtime_version is None
    ):
        raise BenchmarkError(
            "unknown_runtime_version",
            "Both runs need a recorded runtime version for a strict comparison.",
        )
    if baseline_signature != candidate_signature:
        differences = [
            field.name
            for field in fields(baseline_signature)
            if getattr(baseline_signature, field.name)
            != getattr(candidate_signature, field.name)
        ]
        raise BenchmarkError(
            "incomparable_measurements",
            "The runs used different measurement conditions "
            f"({', '.join(differences)}) and cannot be compared.",
        )
    if any(result.measurement_signature != baseline_signature for result in baseline_group):
        raise BenchmarkError("mixed_batch_signature", "Baseline batch signatures do not match.")
    if any(result.measurement_signature != candidate_signature for result in candidate_group):
        raise BenchmarkError("mixed_batch_signature", "Candidate batch signatures do not match.")
    if (
        any(result.pre_run_unload_succeeded is not True for result in baseline_group)
        or any(result.pre_run_unload_succeeded is not True for result in candidate_group)
    ):
        raise BenchmarkError(
            "ineligible_cold_start",
            "Both runs must have completed their requested pre-run unload for load-time and "
            "time-to-first-token comparison.",
        )
    target = baseline_signature.generation_token_target
    if any(result.generation_tokens != target for result in (*baseline_group, *candidate_group)):
        raise BenchmarkError(
            "incomplete_generation",
            f"Both runs must reach the signature's {target}-token generation target for "
            "comparison.",
        )
    baseline_generation = _spread(
        tuple(result.generation_tokens_per_second for result in baseline_group)
    )
    candidate_generation = _spread(
        tuple(result.generation_tokens_per_second for result in candidate_group)
    )
    baseline_ttft = _spread(tuple(result.ttft_seconds for result in baseline_group))
    candidate_ttft = _spread(tuple(result.ttft_seconds for result in candidate_group))
    baseline_load = _spread(tuple(result.load_time_seconds for result in baseline_group))
    candidate_load = _spread(tuple(result.load_time_seconds for result in candidate_group))
    return BenchmarkComparison(
        prompt_id=baseline.prompt_id,
        measurement_signature=baseline_signature,
        baseline=baseline,
        candidate=candidate,
        baseline_run_count=len(baseline_group),
        candidate_run_count=len(candidate_group),
        baseline_generation=baseline_generation,
        candidate_generation=candidate_generation,
        baseline_ttft=baseline_ttft,
        candidate_ttft=candidate_ttft,
        baseline_load_time=baseline_load,
        candidate_load_time=candidate_load,
        generation_rate_change_percent=_percent_change(
            baseline_generation.median, candidate_generation.median
        ),
        # Lower is better for these two, so the sign is inverted to keep the whole
        # comparison readable as "positive means the candidate improved".
        ttft_change_percent=_negated(
            _percent_change(baseline_ttft.median, candidate_ttft.median)
        ),
        load_time_change_percent=_negated(
            _percent_change(baseline_load.median, candidate_load.median)
        ),
    )


def compare_results(baseline: StoredResult, candidate: StoredResult) -> BenchmarkComparison:
    """Backward-compatible singleton comparison entry point."""
    return compare_result_groups((baseline,), (candidate,))


def _negated(value: float | None) -> float | None:
    return None if value is None else -value
