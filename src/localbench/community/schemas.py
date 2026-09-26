"""Allow-listed contracts for a future opt-in community benchmark export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from localbench.benchmark.schemas import MeasurementSignature


@dataclass(frozen=True, slots=True)
class CommunityHardwareFacts:
    source_batch_id: str
    os_family: str
    architecture: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    installed_memory_bucket_bytes: int
    gpu_model: str | None
    gpu_memory_bucket_bytes: int | None
    driver_version: str | None
    execution_backend: str
    gpu_offload_layers: int
    thread_count: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_batch_id")
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True, slots=True)
class CommunityModelFacts:
    source_batch_id: str
    digest: str
    format: str
    quantization: str
    parameter_size: str
    weights_bytes: int
    public_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_batch_id")
        if not self.public_name or not self.public_name.strip():
            result.pop("public_name")
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True, slots=True)
class CommunityObservation:
    ordinal: int
    load_time_seconds: float
    ttft_seconds: float
    prompt_tokens: int
    prompt_eval_seconds: float
    generation_tokens: int
    generation_seconds: float
    total_duration_seconds: float
    wall_time_seconds: float
    baseline_system_ram_used_bytes: int
    peak_system_ram_used_bytes: int
    peak_system_ram_increase_bytes: int
    peak_system_cpu_percent: float


@dataclass(frozen=True, slots=True)
class CommunityBenchmarkBatchRecord:
    record_kind: Literal["benchmark_batch"]
    record_version: Literal[1]
    hardware: CommunityHardwareFacts
    model: CommunityModelFacts
    measurement_signature: MeasurementSignature
    planned_runs: int
    completed_runs: int
    observations: tuple[CommunityObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "record_version": self.record_version,
            "hardware": self.hardware.to_dict(),
            "model": self.model.to_dict(),
            "measurement_signature": self.measurement_signature.to_dict(),
            "planned_runs": self.planned_runs,
            "completed_runs": self.completed_runs,
            "observations": [asdict(observation) for observation in self.observations],
        }


@dataclass(frozen=True, slots=True)
class CommunityEnvelope:
    envelope_version: Literal[1]
    submission_id: str
    created_month: str
    localbench_version: str
    records: tuple[CommunityBenchmarkBatchRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_version": self.envelope_version,
            "submission_id": self.submission_id,
            "created_month": self.created_month,
            "localbench_version": self.localbench_version,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class CommunityEligibility:
    eligible: bool
    exclusion_codes: tuple[str, ...]


class CommunityExportError(RuntimeError):
    def __init__(self, code: str, message: str, reasons: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.reasons = reasons
