"""Typed profile data and the stable JSON representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

SCHEMA_VERSION: Literal["1"] = "1"


@dataclass(frozen=True, slots=True)
class OSProfile:
    family: str
    release: str
    version: str
    architecture: str
    source: str = "python_platform"


@dataclass(frozen=True, slots=True)
class CPUProfile:
    model: str | None
    architecture: str
    physical_cores: int | None
    logical_cores: int | None
    source: str


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    utilization_percent: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_utilization_percent: float
    source: str = "psutil"


@dataclass(frozen=True, slots=True)
class GPUProfile:
    name: str
    vendor: str | None
    dedicated_memory_bytes: int | None
    shared_memory_bytes: int | None
    driver_version: str | None
    source: str


@dataclass(frozen=True, slots=True)
class SystemProfile:
    collected_at: datetime
    os: OSProfile
    cpu: CPUProfile
    memory: MemoryProfile
    gpus: tuple[GPUProfile, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, JSON-compatible public representation."""
        result = asdict(self)
        result["collected_at"] = self.collected_at.isoformat()
        return result


@dataclass(frozen=True, slots=True)
class RuntimeModelProfile:
    name: str
    size_bytes: int | None
    digest: str | None
    modified_at: str | None
    format: str | None
    family: str | None
    parameter_size: str | None
    quantization_level: str | None
    source: str


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    cli_detected: bool
    service_running: bool
    version: str | None
    models: tuple[RuntimeModelProfile, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeDiscovery:
    collected_at: datetime
    runtimes: tuple[RuntimeProfile, ...]
    schema_version: Literal["1"] = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, JSON-compatible public representation."""
        result = asdict(self)
        result["collected_at"] = self.collected_at.isoformat()
        return result
