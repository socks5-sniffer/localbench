"""Local-only construction of privacy-bounded community export previews.

There is intentionally no transport, endpoint, telemetry state, or filesystem write here.
The output is an allow-listed value object suitable for an exact future preview.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from localbench import __version__
from localbench.benchmark.schemas import BenchmarkResult
from localbench.community.schemas import (
    CommunityBenchmarkBatchRecord,
    CommunityEligibility,
    CommunityEnvelope,
    CommunityExportError,
    CommunityHardwareFacts,
    CommunityModelFacts,
    CommunityObservation,
)

DEFAULT_MEMORY_BUCKET_BYTES = 4 * 1024**3
DEFAULT_GPU_MEMORY_BUCKET_BYTES = 1024**3


def bucket_memory_bytes(
    value: int, *, bucket_bytes: int = DEFAULT_MEMORY_BUCKET_BYTES
) -> int:
    """Round to the nearest privacy bucket, with positive values never becoming zero."""
    if value <= 0:
        raise CommunityExportError("invalid_memory", "Memory bytes must be positive.")
    if bucket_bytes <= 0:
        raise CommunityExportError("invalid_bucket", "Memory bucket bytes must be positive.")
    rounded = ((value + bucket_bytes // 2) // bucket_bytes) * bucket_bytes
    return max(bucket_bytes, rounded)


def _hardware_reasons(hardware: CommunityHardwareFacts) -> list[str]:
    reasons: list[str] = []
    if not all(
        value.strip()
        for value in (
            hardware.os_family,
            hardware.architecture,
            hardware.cpu_model,
            hardware.execution_backend,
        )
    ):
        reasons.append("unknown_execution_or_hardware")
    if hardware.physical_cores <= 0 or hardware.logical_cores <= 0:
        reasons.append("invalid_core_count")
    if (
        hardware.installed_memory_bucket_bytes <= 0
        or hardware.installed_memory_bucket_bytes % DEFAULT_MEMORY_BUCKET_BYTES != 0
    ):
        reasons.append("invalid_memory_bucket")
    if hardware.thread_count <= 0 or hardware.gpu_offload_layers < 0:
        reasons.append("unknown_execution_placement")
    if hardware.gpu_memory_bucket_bytes is not None and (
        hardware.gpu_memory_bucket_bytes <= 0
        or hardware.gpu_memory_bucket_bytes % DEFAULT_GPU_MEMORY_BUCKET_BYTES != 0
    ):
        reasons.append("invalid_gpu_memory_bucket")
    return reasons


def _model_reasons(model: CommunityModelFacts) -> list[str]:
    reasons: list[str] = []
    if not model.digest.strip():
        reasons.append("missing_model_digest")
    if not all(
        value.strip() for value in (model.format, model.quantization, model.parameter_size)
    ):
        reasons.append("incomplete_model_identity")
    if model.weights_bytes <= 0:
        reasons.append("invalid_model_weights")
    return reasons


def _metric_values(result: BenchmarkResult) -> tuple[float | int, ...]:
    return (
        result.load_time_seconds,
        result.ttft_seconds,
        result.prompt_tokens,
        result.prompt_eval_seconds,
        result.generation_tokens,
        result.generation_seconds,
        result.total_duration_seconds,
        result.wall_time_seconds,
        result.baseline_system_ram_used_bytes,
        result.peak_system_ram_used_bytes,
        result.peak_system_ram_increase_bytes,
        result.peak_system_cpu_percent,
    )


def assess_benchmark_batch(
    results: Sequence[BenchmarkResult],
    hardware: CommunityHardwareFacts,
    model: CommunityModelFacts,
) -> CommunityEligibility:
    reasons = [*_hardware_reasons(hardware), *_model_reasons(model)]
    if not results:
        reasons.append("empty_batch")
        return CommunityEligibility(False, tuple(dict.fromkeys(reasons)))

    first = results[0]
    if not first.batch_id or not hardware.source_batch_id or not model.source_batch_id:
        reasons.append("invalid_batch_identity")
    if (
        hardware.source_batch_id != first.batch_id
        or model.source_batch_id != first.batch_id
    ):
        reasons.append("source_provenance_mismatch")
    if first.batch_size < 2:
        reasons.append("singleton_batch")
    if (
        len(results) != first.batch_size
        or {result.batch_index for result in results}
        != set(range(1, first.batch_size + 1))
        or any(result.batch_id != first.batch_id for result in results)
        or any(result.batch_size != first.batch_size for result in results)
    ):
        reasons.append("incomplete_batch")
    if any(result.measurement_signature != first.measurement_signature for result in results):
        reasons.append("mixed_measurement_signature")
    if first.measurement_signature.runtime_version is None:
        reasons.append("unknown_runtime_version")
    if any(result.model.casefold() != first.model.casefold() for result in results):
        reasons.append("mixed_model_batch")
    if any(result.pre_run_unload_succeeded is not True for result in results):
        reasons.append("failed_cold_unload")
    target = first.measurement_signature.generation_token_target
    if any(result.generation_tokens != target for result in results):
        reasons.append("incomplete_generation")
    if any(
        value < 0 or (isinstance(value, float) and not math.isfinite(value))
        for result in results
        for value in _metric_values(result)
    ):
        reasons.append("invalid_metrics")
    if any(
        result.peak_system_ram_used_bytes < result.baseline_system_ram_used_bytes
        or result.peak_system_ram_increase_bytes
        != result.peak_system_ram_used_bytes - result.baseline_system_ram_used_bytes
        for result in results
    ):
        reasons.append("inconsistent_resource_metrics")
    unique = tuple(dict.fromkeys(reasons))
    return CommunityEligibility(not unique, unique)


def _observation(result: BenchmarkResult) -> CommunityObservation:
    return CommunityObservation(
        ordinal=result.batch_index,
        load_time_seconds=result.load_time_seconds,
        ttft_seconds=result.ttft_seconds,
        prompt_tokens=result.prompt_tokens,
        prompt_eval_seconds=result.prompt_eval_seconds,
        generation_tokens=result.generation_tokens,
        generation_seconds=result.generation_seconds,
        total_duration_seconds=result.total_duration_seconds,
        wall_time_seconds=result.wall_time_seconds,
        baseline_system_ram_used_bytes=result.baseline_system_ram_used_bytes,
        peak_system_ram_used_bytes=result.peak_system_ram_used_bytes,
        peak_system_ram_increase_bytes=result.peak_system_ram_increase_bytes,
        peak_system_cpu_percent=result.peak_system_cpu_percent,
    )


def build_community_envelope(
    results: Sequence[BenchmarkResult],
    hardware: CommunityHardwareFacts,
    model: CommunityModelFacts,
    *,
    submission_id: str | None = None,
    created_month: str | None = None,
) -> CommunityEnvelope:
    eligibility = assess_benchmark_batch(results, hardware, model)
    if not eligibility.eligible:
        raise CommunityExportError(
            "ineligible_export",
            "Benchmark batch is not eligible for community export: "
            + ", ".join(eligibility.exclusion_codes),
            eligibility.exclusion_codes,
        )
    first = results[0]
    resolved_submission_id = submission_id or str(uuid4())
    try:
        UUID(resolved_submission_id)
    except ValueError as error:
        raise CommunityExportError(
            "invalid_submission_id", "Submission ID must be a UUID."
        ) from error
    resolved_month = created_month or datetime.now(UTC).strftime("%Y-%m")
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", resolved_month) is None:
        raise CommunityExportError(
            "invalid_created_month", "Created month must use YYYY-MM."
        )
    ordered = tuple(sorted(results, key=lambda result: result.batch_index))
    record = CommunityBenchmarkBatchRecord(
        record_kind="benchmark_batch",
        record_version=1,
        hardware=hardware,
        model=model,
        measurement_signature=first.measurement_signature,
        planned_runs=first.batch_size,
        completed_runs=len(results),
        observations=tuple(_observation(result) for result in ordered),
    )
    return CommunityEnvelope(
        envelope_version=1,
        submission_id=resolved_submission_id,
        created_month=resolved_month,
        localbench_version=__version__,
        records=(record,),
    )


def render_community_preview(envelope: CommunityEnvelope) -> str:
    """Return the exact deterministic JSON bytes a future confirmed upload may send."""
    return json.dumps(envelope.to_dict(), indent=2, sort_keys=True, allow_nan=False)
