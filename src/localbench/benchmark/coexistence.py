"""Measured multi-model coexistence: does contention actually cost throughput?

`stack.plan_stack` answers whether a set of models fits in memory at once. It deliberately
does not — and must not — claim anything about how fast they run together. This module
answers the follow-up question for real: load N models and generate from all of them at the
same time, on the same Ollama instance, and measure what concurrent execution actually costs
relative to each model's own solo `bench` baseline. Nothing here is inferred or extrapolated
from single-model numbers; every figure is either measured during the concurrent run itself,
or (for the contention percentage) a direct comparison against a caller-supplied solo result
measured under matching conditions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Barrier, BrokenBarrierError
from typing import Any, Literal

from localbench.benchmark.profiles import (
    QUICK_GENERATION_TOKENS,
    quick_measurement_signature,
)
from localbench.benchmark.runner import StreamGenerator, StreamMetrics, Unloader, stream_completion
from localbench.benchmark.sampler import ResourceSample, SystemSampler
from localbench.benchmark.schemas import BenchmarkError, StoredResult
from localbench.models import SCHEMA_VERSION, RuntimeProfile
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError
from localbench.runtimes.ollama import stream_generate, unload_model

MINIMUM_CONCURRENCY = 2


@dataclass(frozen=True, slots=True)
class CoexistenceRequest:
    """One role in a proposed concurrent run.

    `baseline` is that role's own prior solo `bench` result (as read back from history via
    `StoredResult`), used only to compute the contention percentage below. It is optional:
    the concurrent measurement itself does not require one, but without it there is nothing
    to compare against and the contention fields are omitted with an explicit warning.
    """

    role: str
    model: str
    baseline: StoredResult | None = None


@dataclass(frozen=True, slots=True)
class ModelCoexistenceResult:
    role: str
    model: str
    pre_run_unload_succeeded: bool
    load_time_seconds: float
    ttft_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float | None
    generation_tokens: int
    generation_seconds: float
    generation_tokens_per_second: float | None
    total_duration_seconds: float
    done_reason: str | None
    baseline_run_id: str | None
    generation_rate_change_percent: float | None
    ttft_change_percent: float | None
    load_time_change_percent: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoexistenceResult:
    """One concurrent run of two or more models against a single Ollama instance.

    Percentage fields follow the same sign convention as `BenchmarkComparison`: positive
    always means the concurrent measurement was *better* than the solo baseline, including
    for TTFT and load time where the raw number is lower-is-better. Under real contention,
    `generation_rate_change_percent` is expected to be negative — that is the actual cost of
    running these models together, not a defect in the measurement.
    """

    started_at: datetime
    runtime: str
    runtime_version: str | None
    concurrency: int
    models: tuple[ModelCoexistenceResult, ...]
    wall_time_seconds: float
    baseline_system_ram_used_bytes: int
    peak_system_ram_used_bytes: int
    peak_system_ram_increase_bytes: int
    peak_system_cpu_percent: float
    warnings: tuple[str, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["started_at"] = self.started_at.isoformat()
        return result


class CoexistenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _select_model(runtime: RuntimeProfile, requested: str) -> str:
    """Duplicated in spirit from `runner._select_model`/`estimation.memory._select_model`.

    The project already carries two near-identical copies of this small resolver, one per
    error type it raises; this is the third, matching existing precedent rather than
    inventing a shared abstraction across modules that otherwise have no coupling.
    """
    normalized = requested.casefold()
    exact = [model.name for model in runtime.models if model.name.casefold() == normalized]
    if exact:
        return exact[0]
    if ":" not in requested:
        matches = [
            model.name
            for model in runtime.models
            if model.name.partition(":")[0].casefold() == normalized
        ]
        latest = [name for name in matches if name.casefold().endswith(":latest")]
        if latest:
            return latest[0]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(matches)
            raise CoexistenceError(
                "ambiguous_model",
                f"Model name {requested!r} is ambiguous; choose one of: {choices}.",
            )
    raise CoexistenceError(
        "model_not_found", f"Installed Ollama model {requested!r} was not found."
    )


def _contention_ineligibility_reason(
    role: str,
    model: str,
    runtime: RuntimeProfile,
    baseline: StoredResult | None,
    *,
    concurrent_unload_succeeded: bool,
    concurrent_generation_tokens: int,
) -> str | None:
    """Return why a baseline can't be used for the contention comparison, or None if it can."""
    if baseline is None:
        return None  # No baseline supplied is not itself a warning; it's simply omitted.
    if baseline.model.casefold() != model.casefold():
        return (
            f"Role {role!r}'s baseline is for {baseline.model!r}, not {model!r}; skipping "
            "the contention comparison."
        )
    if baseline.pre_run_unload_succeeded is not True:
        return (
            f"Role {role!r}'s baseline did not complete its cold unload; skipping the "
            "contention comparison."
        )
    if not concurrent_unload_succeeded:
        return (
            f"Role {role!r}'s concurrent run did not complete its cold unload; skipping the "
            "contention comparison."
        )
    if baseline.generation_tokens != QUICK_GENERATION_TOKENS:
        return (
            f"Role {role!r}'s baseline did not reach the {QUICK_GENERATION_TOKENS}-token "
            "generation target; skipping the contention comparison."
        )
    signature = baseline.measurement_signature
    if signature is None:
        return (
            f"Role {role!r}'s baseline has no known measurement signature; skipping the "
            "contention comparison."
        )
    expected_signature = quick_measurement_signature(runtime.version)
    if runtime.version is None or expected_signature is None or signature != expected_signature:
        return (
            f"Role {role!r}'s baseline used different measurement conditions (workload, "
            "context, or runtime version); skipping the contention comparison."
        )
    if concurrent_generation_tokens != QUICK_GENERATION_TOKENS:
        return (
            f"Role {role!r}'s concurrent run did not reach the "
            f"{QUICK_GENERATION_TOKENS}-token generation target; skipping the contention "
            "comparison."
        )
    if (
        baseline.generation_tokens_per_second is None
        or not baseline.generation_tokens_per_second > 0
        or baseline.ttft_seconds is None
        or not baseline.ttft_seconds > 0
        or baseline.load_time_seconds is None
        or not baseline.load_time_seconds > 0
    ):
        return (
            f"Role {role!r}'s baseline is missing one or more required metrics; skipping the "
            "contention comparison."
        )
    return None


def _percent_change(baseline: float, candidate: float, *, lower_is_better: bool) -> float:
    raw = (candidate - baseline) / baseline * 100
    return -raw if lower_is_better else raw


def _model_result(
    request: CoexistenceRequest,
    model: str,
    runtime: RuntimeProfile,
    unload_succeeded: bool,
    metrics: StreamMetrics,
) -> ModelCoexistenceResult:
    ineligible_reason = _contention_ineligibility_reason(
        request.role,
        model,
        runtime,
        request.baseline,
        concurrent_unload_succeeded=unload_succeeded,
        concurrent_generation_tokens=metrics.generation_tokens,
    )
    warnings: list[str] = []
    generation_change: float | None = None
    ttft_change: float | None = None
    load_change: float | None = None
    baseline_run_id: str | None = None
    if request.baseline is not None:
        if ineligible_reason is not None:
            warnings.append(ineligible_reason)
        else:
            baseline = request.baseline
            assert baseline.generation_tokens_per_second is not None
            assert baseline.ttft_seconds is not None
            assert baseline.load_time_seconds is not None
            baseline_run_id = baseline.run_id
            if metrics.generation_tokens_per_second is not None:
                generation_change = _percent_change(
                    baseline.generation_tokens_per_second,
                    metrics.generation_tokens_per_second,
                    lower_is_better=False,
                )
            ttft_change = _percent_change(
                baseline.ttft_seconds, metrics.ttft_seconds, lower_is_better=True
            )
            load_change = _percent_change(
                baseline.load_time_seconds, metrics.load_time_seconds, lower_is_better=True
            )
    return ModelCoexistenceResult(
        role=request.role,
        model=model,
        pre_run_unload_succeeded=unload_succeeded,
        load_time_seconds=metrics.load_time_seconds,
        ttft_seconds=metrics.ttft_seconds,
        prompt_tokens=metrics.prompt_tokens,
        prompt_tokens_per_second=metrics.prompt_tokens_per_second,
        generation_tokens=metrics.generation_tokens,
        generation_seconds=metrics.generation_seconds,
        generation_tokens_per_second=metrics.generation_tokens_per_second,
        total_duration_seconds=metrics.total_duration_seconds,
        done_reason=metrics.done_reason,
        baseline_run_id=baseline_run_id,
        generation_rate_change_percent=generation_change,
        ttft_change_percent=ttft_change,
        load_time_change_percent=load_change,
        warnings=tuple(warnings),
    )


def run_coexistence_benchmark(
    requests: Sequence[CoexistenceRequest],
    runtime: RuntimeProfile,
    *,
    stream: StreamGenerator = stream_generate,
    unload: Unloader = unload_model,
    sampler: SystemSampler | None = None,
    clock: Callable[[], float] = time.perf_counter,
    barrier_timeout_seconds: float = 60.0,
) -> CoexistenceResult:
    """Load and generate from every requested model at the same time and measure the result.

    Every model is cold-unloaded first (best effort, same as solo `bench`), then all
    requests begin streaming as close to simultaneously as a `threading.Barrier` can manage,
    so the measured window reflects real concurrent load/generate contention rather than a
    staggered sequence. One `SystemSampler` covers the whole concurrent window. If any role's
    stream fails, the whole run raises `CoexistenceError` — a partial concurrent result would
    misrepresent the contention the surviving roles experienced.
    """
    if not runtime.service_running:
        raise CoexistenceError("runtime_unavailable", "Ollama's local service must be running.")
    if len(requests) < MINIMUM_CONCURRENCY:
        raise CoexistenceError(
            "insufficient_workloads",
            f"A coexistence benchmark requires at least {MINIMUM_CONCURRENCY} concurrent roles.",
        )

    seen_roles: dict[str, str] = {}
    resolved: list[tuple[CoexistenceRequest, str]] = []
    for request in requests:
        if not request.role.strip():
            raise CoexistenceError("invalid_role", "A workload role name must not be blank.")
        role_key = request.role.casefold()
        if role_key in seen_roles:
            raise CoexistenceError(
                "duplicate_role", f"Workload role {request.role!r} was specified more than once."
            )
        seen_roles[role_key] = request.role
        resolved.append((request, _select_model(runtime, request.model)))

    unload_succeeded: dict[str, bool] = {}
    for request, model in resolved:
        try:
            unload(model, 30.0)
            unload_succeeded[request.role] = True
        except (RuntimeUnavailableError, RuntimeResponseError):
            unload_succeeded[request.role] = False

    barrier = Barrier(len(resolved))

    def _worker(model: str) -> StreamMetrics:
        try:
            barrier.wait(timeout=barrier_timeout_seconds)
        except BrokenBarrierError as error:
            raise CoexistenceError(
                "start_synchronization_failed",
                "Not all roles were ready to start generating within the barrier timeout.",
            ) from error
        started = clock()
        return stream_completion(model, stream, clock, started)

    active_sampler = sampler or SystemSampler()
    started_at = datetime.now(UTC)
    wall_started = clock()
    active_sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=len(resolved)) as executor:
            futures = {
                request.role: executor.submit(_worker, model) for request, model in resolved
            }
            outcomes: dict[str, StreamMetrics] = {}
            failures: list[str] = []
            for role, future in futures.items():
                try:
                    outcomes[role] = future.result()
                except BenchmarkError as error:
                    failures.append(f"{role!r}: {error.message}")
                except CoexistenceError as error:
                    failures.append(f"{role!r}: {error.message}")
    finally:
        resources: ResourceSample = active_sampler.stop()
    finished = clock()

    if failures:
        raise CoexistenceError(
            "concurrent_run_failed",
            "One or more roles failed during the concurrent run: " + "; ".join(failures),
        )

    models = tuple(
        _model_result(
            request, model, runtime, unload_succeeded[request.role], outcomes[request.role]
        )
        for request, model in resolved
    )
    peak_increase = max(0, resources.peak_ram_used_bytes - resources.baseline_ram_used_bytes)
    run_warnings = [
        "RAM and CPU metrics are system-wide and may include unrelated background activity.",
        "This measures real concurrent contention on this machine right now; it is not a "
        "guarantee of future performance and does not extrapolate to other hardware.",
    ]
    for request, _ in resolved:
        if not unload_succeeded[request.role]:
            run_warnings.append(
                f"Pre-run unload failed for role {request.role!r}; its load time may "
                "represent a warm model."
            )
    run_warnings.extend(
        warning for model_result in models for warning in model_result.warnings
    )

    return CoexistenceResult(
        started_at=started_at,
        runtime="ollama",
        runtime_version=runtime.version,
        concurrency=len(resolved),
        models=models,
        wall_time_seconds=finished - wall_started,
        baseline_system_ram_used_bytes=resources.baseline_ram_used_bytes,
        peak_system_ram_used_bytes=resources.peak_ram_used_bytes,
        peak_system_ram_increase_bytes=peak_increase,
        peak_system_cpu_percent=resources.peak_cpu_percent,
        warnings=tuple(run_warnings),
    )
