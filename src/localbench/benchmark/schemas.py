"""Benchmark result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from localbench.models import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MeasurementSignature:
    """The measurement conditions that must match for a strict comparison."""

    contract_version: Literal["1"]
    benchmark_method_id: str
    profile: str
    prompt_id: str
    context_length: int
    generation_token_target: int
    raw: bool
    stream: bool
    think: bool
    temperature: float
    seed: int
    keep_alive_seconds: int
    cold_start_requested: bool
    runtime: str
    runtime_version: str | None
    resource_sampler_id: str
    resource_sample_interval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_id: str
    started_at: datetime
    model: str
    runtime: str
    runtime_version: str | None
    profile: Literal["quick"]
    prompt_id: str
    context_length: int
    generation_token_target: int
    cold_start_requested: bool
    measurement_signature: MeasurementSignature
    batch_id: str
    batch_index: int
    batch_size: int
    pre_run_unload_succeeded: bool
    load_time_seconds: float
    ttft_seconds: float
    prompt_tokens: int
    prompt_eval_seconds: float
    prompt_tokens_per_second: float | None
    generation_tokens: int
    generation_seconds: float
    generation_tokens_per_second: float | None
    total_duration_seconds: float
    wall_time_seconds: float
    baseline_system_ram_used_bytes: int
    peak_system_ram_used_bytes: int
    peak_system_ram_increase_bytes: int
    peak_system_cpu_percent: float
    done_reason: str | None
    warnings: tuple[str, ...] = ()
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["started_at"] = self.started_at.isoformat()
        return result


class BenchmarkError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StoredResult:
    """One previously stored benchmark run, as read back from local history.

    Deliberately a narrower view than `BenchmarkResult`, but it retains the canonical
    measurement signature and eligibility state needed for safe comparison.
    """

    run_id: str
    started_at: str
    model: str
    prompt_id: str
    generation_tokens: int | None
    generation_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    ttft_seconds: float | None
    load_time_seconds: float | None
    runtime: str | None = None
    runtime_version: str | None = None
    pre_run_unload_succeeded: bool | None = None
    measurement_signature: MeasurementSignature | None = None
    batch_id: str | None = None
    batch_index: int = 1
    batch_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricSpread:
    count: int
    median: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Two stored runs compared on identical workload terms.

    Construction is gated by `compare_results`, which requires identical canonical
    measurement signatures. Percentages are expressed relative to the
    baseline (the older run), positive meaning the candidate improved on it.
    """

    prompt_id: str
    measurement_signature: MeasurementSignature
    baseline: StoredResult
    candidate: StoredResult
    baseline_run_count: int
    candidate_run_count: int
    baseline_generation: MetricSpread
    candidate_generation: MetricSpread
    baseline_ttft: MetricSpread
    candidate_ttft: MetricSpread
    baseline_load_time: MetricSpread
    candidate_load_time: MetricSpread
    generation_rate_change_percent: float | None
    ttft_change_percent: float | None
    load_time_change_percent: float | None
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
