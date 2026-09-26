from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.schemas import BenchmarkResult
from localbench.context.schemas import ContextCapacityProfile, ContextLevelEstimate
from localbench.estimation.schemas import (
    FitClassification,
    KVCacheType,
    MemoryComponent,
    ModelMemoryEstimate,
)
from localbench.models import (
    CPUProfile,
    GPUProfile,
    MemoryProfile,
    OSProfile,
    RuntimeDiscovery,
    RuntimeModelProfile,
    RuntimeProfile,
    SystemProfile,
)


@pytest.fixture
def sample_profile() -> SystemProfile:
    return SystemProfile(
        collected_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        os=OSProfile(
            family="TestOS",
            release="1",
            version="1.2.3",
            architecture="x86_64",
        ),
        cpu=CPUProfile(
            model="Test CPU",
            architecture="x86_64",
            physical_cores=4,
            logical_cores=8,
            source="test",
        ),
        memory=MemoryProfile(
            total_bytes=16 * 1024**3,
            available_bytes=10 * 1024**3,
            used_bytes=6 * 1024**3,
            utilization_percent=37.5,
            swap_total_bytes=4 * 1024**3,
            swap_used_bytes=1024**3,
            swap_utilization_percent=25.0,
        ),
        gpus=(
            GPUProfile(
                name="Test GPU",
                vendor="Test Vendor",
                dedicated_memory_bytes=8 * 1024**3,
                shared_memory_bytes=None,
                driver_version="1.0",
                source="test",
            ),
        ),
        warnings=("Example warning.",),
    )


@pytest.fixture
def sample_runtime_discovery() -> RuntimeDiscovery:
    return RuntimeDiscovery(
        collected_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        runtimes=(
            RuntimeProfile(
                name="ollama",
                cli_detected=True,
                service_running=True,
                version="1.2.3",
                models=(
                    RuntimeModelProfile(
                        name="example:latest",
                        size_bytes=4 * 1024**3,
                        digest="abc123",
                        modified_at="2026-08-30T12:00:00Z",
                        format="gguf",
                        family="example",
                        parameter_size="4B",
                        quantization_level="Q4_K_M",
                        source="ollama_local_api",
                    ),
                ),
            ),
        ),
    )


@pytest.fixture
def sample_estimate() -> ModelMemoryEstimate:
    gib = 1024**3
    return ModelMemoryEstimate(
        collected_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        model="example:latest",
        context_length=8192,
        model_context_limit=32768,
        kv_cache_type=KVCacheType.F16,
        weights=MemoryComponent(5 * gib, "reported", "test"),
        kv_cache=MemoryComponent(gib, "calculated", "test"),
        runtime_overhead=MemoryComponent(gib // 2, "inferred", "test"),
        safety_headroom=MemoryComponent(2 * gib, "inferred", "test"),
        working_memory_bytes=6 * gib + gib // 2,
        estimated_total_bytes=8 * gib + gib // 2,
        available_memory=MemoryComponent(16 * gib, "measured", "test"),
        fit=FitClassification.EXCELLENT,
        fit_basis="Test basis.",
        gpu_offload=None,
        assumptions=("Test assumption.",),
        warnings=("Test warning.",),
    )


@pytest.fixture
def sample_context_profile(sample_estimate: ModelMemoryEstimate) -> ContextCapacityProfile:
    return ContextCapacityProfile(
        collected_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        model=sample_estimate.model,
        kv_cache_type=sample_estimate.kv_cache_type,
        model_context_limit=sample_estimate.model_context_limit,
        levels=(
            ContextLevelEstimate(
                context_length=sample_estimate.context_length, estimate=sample_estimate
            ),
            ContextLevelEstimate(
                context_length=16384,
                estimate=dataclasses.replace(sample_estimate, context_length=16384),
            ),
        ),
        skipped_context_lengths=(),
        assumptions=("Test context assumption.",),
        warnings=("Test context warning.",),
    )


@pytest.fixture
def sample_benchmark() -> BenchmarkResult:
    signature = quick_measurement_signature("1.2.3", prompt_id="benchmark_generation_v1")
    assert signature is not None
    return BenchmarkResult(
        run_id="00000000-0000-0000-0000-000000000001",
        started_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        model="example:latest",
        runtime="ollama",
        runtime_version="1.2.3",
        profile="quick",
        prompt_id="benchmark_generation_v1",
        context_length=2048,
        generation_token_target=64,
        cold_start_requested=True,
        measurement_signature=signature,
        batch_id="00000000-0000-0000-0000-000000000001",
        batch_index=1,
        batch_size=1,
        pre_run_unload_succeeded=True,
        load_time_seconds=1.0,
        ttft_seconds=1.5,
        prompt_tokens=32,
        prompt_eval_seconds=0.5,
        prompt_tokens_per_second=64.0,
        generation_tokens=64,
        generation_seconds=4.0,
        generation_tokens_per_second=16.0,
        total_duration_seconds=5.5,
        wall_time_seconds=5.6,
        baseline_system_ram_used_bytes=8 * 1024**3,
        peak_system_ram_used_bytes=11 * 1024**3,
        peak_system_ram_increase_bytes=3 * 1024**3,
        peak_system_cpu_percent=88.0,
        done_reason="length",
        warnings=("System-wide metrics.",),
    )
