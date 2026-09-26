"""Ollama quick benchmark orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from localbench.benchmark.profiles import (
    QUICK_CONTEXT_LENGTH,
    QUICK_GENERATION_TOKENS,
    QUICK_PROMPT,
    QUICK_PROMPT_ID,
    QUICK_SEED,
    quick_measurement_signature,
)
from localbench.benchmark.sampler import ResourceSample, SystemSampler
from localbench.benchmark.schemas import BenchmarkError, BenchmarkResult
from localbench.models import RuntimeProfile
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError
from localbench.runtimes.ollama import stream_generate, unload_model

StreamGenerator = Callable[[Mapping[str, object], float], Iterator[Mapping[str, object]]]
Unloader = Callable[[str, float], None]


def _positive_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise BenchmarkError("invalid_metrics", f"Ollama did not report valid {key}.")
    return value


def _rate(count: int, duration_seconds: float) -> float | None:
    return count / duration_seconds if duration_seconds > 0 else None


def _select_model(runtime: RuntimeProfile, requested: str) -> str:
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
            raise BenchmarkError(
                "ambiguous_model",
                f"Model name {requested!r} is ambiguous; choose one of: {choices}.",
            )
    raise BenchmarkError("model_not_found", f"Installed Ollama model {requested!r} was not found.")


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    """Timing and token counters from one completed raw-completion stream."""

    ttft_seconds: float
    load_time_seconds: float
    prompt_tokens: int
    prompt_eval_seconds: float
    prompt_tokens_per_second: float | None
    generation_tokens: int
    generation_seconds: float
    generation_tokens_per_second: float | None
    total_duration_seconds: float
    done_reason: str | None


def stream_completion(
    model: str,
    stream: StreamGenerator,
    clock: Callable[[], float],
    started: float,
    *,
    timeout_seconds: float = 180.0,
    prompt: str = QUICK_PROMPT,
    context_length: int = QUICK_CONTEXT_LENGTH,
    generation_token_target: int = QUICK_GENERATION_TOKENS,
    seed: int = QUICK_SEED,
) -> StreamMetrics:
    """Run one raw-completion stream against `model` and return its measured metrics.

    `started` is a clock reading the caller captured immediately before this call, used as
    the time-to-first-token baseline; the caller (not this function) owns resource sampling
    and overall wall-clock timing, since those differ between the solo benchmark and the
    concurrent coexistence benchmark that both call this.
    """
    first_token_at: float | None = None
    final: Mapping[str, object] | None = None
    body: Mapping[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "raw": True,
        "keep_alive": 0,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": context_length,
            "num_predict": generation_token_target,
        },
    }
    try:
        for chunk in stream(body, timeout_seconds):
            if first_token_at is None and (chunk.get("response") or chunk.get("thinking")):
                first_token_at = clock()
            if chunk.get("done") is True:
                final = chunk
    except RuntimeUnavailableError as error:
        raise BenchmarkError("runtime_unavailable", "Ollama generation failed.") from error
    except RuntimeResponseError as error:
        raise BenchmarkError("invalid_response", "Ollama returned an invalid stream.") from error

    if final is None or first_token_at is None:
        raise BenchmarkError("incomplete_run", "Ollama did not complete the benchmark stream.")
    load_ns = _positive_integer(final, "load_duration")
    prompt_count = _positive_integer(final, "prompt_eval_count")
    prompt_ns = _positive_integer(final, "prompt_eval_duration")
    generation_count = _positive_integer(final, "eval_count")
    generation_ns = _positive_integer(final, "eval_duration")
    total_ns = _positive_integer(final, "total_duration")
    prompt_seconds = prompt_ns / 1_000_000_000
    generation_seconds = generation_ns / 1_000_000_000
    return StreamMetrics(
        ttft_seconds=first_token_at - started,
        load_time_seconds=load_ns / 1_000_000_000,
        prompt_tokens=prompt_count,
        prompt_eval_seconds=prompt_seconds,
        prompt_tokens_per_second=_rate(prompt_count, prompt_seconds),
        generation_tokens=generation_count,
        generation_seconds=generation_seconds,
        generation_tokens_per_second=_rate(generation_count, generation_seconds),
        total_duration_seconds=total_ns / 1_000_000_000,
        done_reason=str(final.get("done_reason")) if final.get("done_reason") else None,
    )


def run_quick_benchmark(
    requested_model: str,
    runtime: RuntimeProfile,
    *,
    stream: StreamGenerator = stream_generate,
    unload: Unloader = unload_model,
    sampler: SystemSampler | None = None,
    clock: Callable[[], float] = time.perf_counter,
    batch_id: str | None = None,
    batch_index: int = 1,
    batch_size: int = 1,
) -> BenchmarkResult:
    if not runtime.service_running:
        raise BenchmarkError("runtime_unavailable", "Ollama's local service must be running.")
    model = _select_model(runtime, requested_model)
    unload_succeeded = True
    try:
        unload(model, 30.0)
    except (RuntimeUnavailableError, RuntimeResponseError):
        unload_succeeded = False

    active_sampler = sampler or SystemSampler()
    started_at = datetime.now(UTC)
    started = clock()
    active_sampler.start()
    try:
        metrics = stream_completion(model, stream, clock, started)
    finally:
        resources: ResourceSample = active_sampler.stop()
    finished = clock()

    peak_increase = max(0, resources.peak_ram_used_bytes - resources.baseline_ram_used_bytes)
    warnings = [
        "RAM and CPU metrics are system-wide and may include unrelated background activity."
    ]
    if not unload_succeeded:
        warnings.append("Pre-run unload failed; load time may represent a warm model.")
    measurement_signature = quick_measurement_signature(runtime.version)
    assert measurement_signature is not None
    run_id = str(uuid4())
    return BenchmarkResult(
        run_id=run_id,
        started_at=started_at,
        model=model,
        runtime="ollama",
        runtime_version=runtime.version,
        profile="quick",
        prompt_id=QUICK_PROMPT_ID,
        context_length=QUICK_CONTEXT_LENGTH,
        generation_token_target=QUICK_GENERATION_TOKENS,
        cold_start_requested=True,
        measurement_signature=measurement_signature,
        batch_id=batch_id or run_id,
        batch_index=batch_index,
        batch_size=batch_size,
        pre_run_unload_succeeded=unload_succeeded,
        load_time_seconds=metrics.load_time_seconds,
        ttft_seconds=metrics.ttft_seconds,
        prompt_tokens=metrics.prompt_tokens,
        prompt_eval_seconds=metrics.prompt_eval_seconds,
        prompt_tokens_per_second=metrics.prompt_tokens_per_second,
        generation_tokens=metrics.generation_tokens,
        generation_seconds=metrics.generation_seconds,
        generation_tokens_per_second=metrics.generation_tokens_per_second,
        total_duration_seconds=metrics.total_duration_seconds,
        wall_time_seconds=finished - started,
        baseline_system_ram_used_bytes=resources.baseline_ram_used_bytes,
        peak_system_ram_used_bytes=resources.peak_ram_used_bytes,
        peak_system_ram_increase_bytes=peak_increase,
        peak_system_cpu_percent=resources.peak_cpu_percent,
        done_reason=metrics.done_reason,
        warnings=tuple(warnings),
    )
