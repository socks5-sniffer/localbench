from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from localbench.estimation.schemas import (
    FitClassification,
    KVCacheType,
    MemoryComponent,
    ModelMemoryEstimate,
)
from localbench.stack.planner import STACK_SAFETY_HEADROOM_BYTES, plan_stack
from localbench.stack.schemas import StackPlanError, StackWorkloadRequest

WEIGHTS_BYTES = 4 * 1024**3
KV_CACHE_BYTES = 512 * 1024**2
RUNTIME_OVERHEAD_BYTES = 400 * 1024**2
WORKING_MEMORY_BYTES = WEIGHTS_BYTES + KV_CACHE_BYTES + RUNTIME_OVERHEAD_BYTES
DEFAULT_AVAILABLE_BYTES = 16 * 1024**3


def _make_estimate(
    *,
    model: str = "granite4:1b-h",
    weights_bytes: int = WEIGHTS_BYTES,
    kv_cache_bytes: int = KV_CACHE_BYTES,
    runtime_overhead_bytes: int = RUNTIME_OVERHEAD_BYTES,
    available_memory_bytes: int = DEFAULT_AVAILABLE_BYTES,
) -> ModelMemoryEstimate:
    working_memory_bytes = weights_bytes + kv_cache_bytes + runtime_overhead_bytes
    per_model_safety_headroom = 2 * 1024**3
    estimated_total_bytes = working_memory_bytes + per_model_safety_headroom
    return ModelMemoryEstimate(
        collected_at=datetime.now(UTC),
        model=model,
        context_length=8192,
        model_context_limit=131072,
        kv_cache_type=KVCacheType.F16,
        weights=MemoryComponent(bytes=weights_bytes, provenance="reported", basis="test"),
        kv_cache=MemoryComponent(bytes=kv_cache_bytes, provenance="calculated", basis="test"),
        runtime_overhead=MemoryComponent(
            bytes=runtime_overhead_bytes, provenance="inferred", basis="test"
        ),
        safety_headroom=MemoryComponent(
            bytes=per_model_safety_headroom, provenance="inferred", basis="test"
        ),
        working_memory_bytes=working_memory_bytes,
        estimated_total_bytes=estimated_total_bytes,
        available_memory=MemoryComponent(
            bytes=available_memory_bytes, provenance="measured", basis="test"
        ),
        fit=FitClassification.EXCELLENT,
        fit_basis="test",
        gpu_offload=None,
        assumptions=(),
        warnings=(),
    )


def test_plan_stack_sums_working_memory_and_applies_headroom_once() -> None:
    chat = StackWorkloadRequest(role="chat", estimate=_make_estimate(model="granite4:1b-h"))
    embed = StackWorkloadRequest(
        role="embeddings", estimate=_make_estimate(model="nomic-embed-text")
    )

    plan = plan_stack([chat, embed], DEFAULT_AVAILABLE_BYTES)

    assert plan.total_working_memory_bytes == 2 * WORKING_MEMORY_BYTES
    # Headroom applied exactly once, not once per model.
    assert plan.safety_headroom.bytes == STACK_SAFETY_HEADROOM_BYTES
    assert plan.total_required_bytes == 2 * WORKING_MEMORY_BYTES + STACK_SAFETY_HEADROOM_BYTES
    assert plan.total_required_bytes != 2 * WORKING_MEMORY_BYTES + 2 * STACK_SAFETY_HEADROOM_BYTES
    assert plan.headroom_bytes == DEFAULT_AVAILABLE_BYTES - plan.total_required_bytes


def test_plan_stack_keeps_per_model_runtime_overhead_visible() -> None:
    chat = StackWorkloadRequest(
        role="chat", estimate=_make_estimate(model="a", runtime_overhead_bytes=777)
    )
    embed = StackWorkloadRequest(
        role="embeddings", estimate=_make_estimate(model="b", runtime_overhead_bytes=333)
    )

    plan = plan_stack([chat, embed], DEFAULT_AVAILABLE_BYTES)

    by_role = {model.role: model for model in plan.models}
    assert by_role["chat"].runtime_overhead_bytes == 777
    assert by_role["embeddings"].runtime_overhead_bytes == 333
    assert by_role["chat"].weights_bytes == WEIGHTS_BYTES
    assert by_role["chat"].kv_cache_bytes == KV_CACHE_BYTES


def test_plan_stack_uses_single_shared_available_memory_snapshot() -> None:
    stale = StackWorkloadRequest(
        role="chat",
        estimate=_make_estimate(model="a", available_memory_bytes=DEFAULT_AVAILABLE_BYTES * 2),
    )
    fresh = StackWorkloadRequest(
        role="embeddings",
        estimate=_make_estimate(model="b", available_memory_bytes=DEFAULT_AVAILABLE_BYTES),
    )

    plan = plan_stack([stale, fresh], DEFAULT_AVAILABLE_BYTES)

    assert plan.available_memory.bytes == DEFAULT_AVAILABLE_BYTES
    stale_plan = next(model for model in plan.models if model.role == "chat")
    fresh_plan = next(model for model in plan.models if model.role == "embeddings")
    assert stale_plan.warnings, "mismatched snapshot should be flagged"
    assert "different available-memory snapshot" in stale_plan.warnings[0]
    assert fresh_plan.warnings == ()
    assert plan.warnings == stale_plan.warnings


def test_plan_stack_rejects_duplicate_model_across_roles() -> None:
    first = StackWorkloadRequest(role="chat", estimate=_make_estimate(model="granite4:1b-h"))
    second = StackWorkloadRequest(
        role="chat-backup", estimate=_make_estimate(model="Granite4:1B-H")
    )

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([first, second], DEFAULT_AVAILABLE_BYTES)

    assert excinfo.value.code == "duplicate_model"


def test_plan_stack_rejects_duplicate_role_name() -> None:
    first = StackWorkloadRequest(role="chat", estimate=_make_estimate(model="a"))
    second = StackWorkloadRequest(role="Chat", estimate=_make_estimate(model="b"))

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([first, second], DEFAULT_AVAILABLE_BYTES)

    assert excinfo.value.code == "duplicate_role"


def test_plan_stack_rejects_empty_workload_list() -> None:
    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([], DEFAULT_AVAILABLE_BYTES)

    assert excinfo.value.code == "empty_stack"


def test_plan_stack_rejects_non_positive_available_memory() -> None:
    request = StackWorkloadRequest(role="chat", estimate=_make_estimate())

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([request], 0)

    assert excinfo.value.code == "memory_unavailable"


def test_plan_stack_rejects_negative_available_memory() -> None:
    request = StackWorkloadRequest(role="chat", estimate=_make_estimate())

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([request], -1)

    assert excinfo.value.code == "memory_unavailable"


def test_plan_stack_rejects_blank_role() -> None:
    request = StackWorkloadRequest(role="   ", estimate=_make_estimate())

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([request], DEFAULT_AVAILABLE_BYTES)

    assert excinfo.value.code == "invalid_role"


def test_plan_stack_rejects_non_positive_working_memory() -> None:
    broken = _make_estimate(weights_bytes=0, kv_cache_bytes=0, runtime_overhead_bytes=0)
    request = StackWorkloadRequest(role="chat", estimate=broken)

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([request], DEFAULT_AVAILABLE_BYTES)

    assert excinfo.value.code == "invalid_working_memory"


def test_plan_stack_rejects_negative_memory_component() -> None:
    estimate = _make_estimate()
    broken = replace(
        estimate,
        weights=replace(estimate.weights, bytes=-1),
        working_memory_bytes=(
            -1 + estimate.kv_cache.bytes + estimate.runtime_overhead.bytes
        ),
    )

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack(
            [StackWorkloadRequest(role="chat", estimate=broken)],
            DEFAULT_AVAILABLE_BYTES,
        )

    assert excinfo.value.code == "invalid_memory_component"


def test_plan_stack_rejects_inconsistent_component_total() -> None:
    estimate = _make_estimate()
    broken = replace(estimate, working_memory_bytes=estimate.working_memory_bytes + 1)

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack(
            [StackWorkloadRequest(role="chat", estimate=broken)],
            DEFAULT_AVAILABLE_BYTES,
        )

    assert excinfo.value.code == "inconsistent_working_memory"


def test_plan_stack_rejects_negative_safety_headroom() -> None:
    request = StackWorkloadRequest(role="chat", estimate=_make_estimate())

    with pytest.raises(StackPlanError) as excinfo:
        plan_stack([request], DEFAULT_AVAILABLE_BYTES, safety_headroom_bytes=-1)

    assert excinfo.value.code == "invalid_safety_headroom"


_REQUIRED_BYTES = WORKING_MEMORY_BYTES + STACK_SAFETY_HEADROOM_BYTES


@pytest.mark.parametrize(
    ("available_bytes", "expected"),
    [
        (_REQUIRED_BYTES, FitClassification.POOR),
        (math.ceil(_REQUIRED_BYTES / 0.60), FitClassification.EXCELLENT),
        (math.ceil(_REQUIRED_BYTES / 0.80), FitClassification.GOOD),
        (math.ceil(_REQUIRED_BYTES / 0.95), FitClassification.TIGHT),
        (_REQUIRED_BYTES - 1, FitClassification.WILL_NOT_FIT),
    ],
)
def test_plan_stack_viability_thresholds(
    available_bytes: int, expected: FitClassification
) -> None:
    request = StackWorkloadRequest(role="chat", estimate=_make_estimate())

    plan = plan_stack([request], available_bytes)

    assert plan.viability == expected


def test_plan_stack_does_not_model_or_expose_concurrent_throughput() -> None:
    request = StackWorkloadRequest(role="chat", estimate=_make_estimate())

    plan = plan_stack([request], DEFAULT_AVAILABLE_BYTES)

    assert any("not measured" in assumption for assumption in plan.assumptions)
    assert not hasattr(plan, "tokens_per_second")
    assert not hasattr(plan, "throughput")
