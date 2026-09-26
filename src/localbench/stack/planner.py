"""Pure, explainable simultaneous-memory capacity planning for a proposed model stack.

This module performs no I/O and calls no runtime: it takes already-collected
`ModelMemoryEstimate` values (see `localbench.estimation`) for each named workload role and
computes whether loading all of them at once plausibly fits in a single available-memory
snapshot. It answers "does this combination fit," not "how fast will it run together" —
concurrent inference performance is not measured here and must never be extrapolated from
single-model benchmark results.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from localbench.estimation.schemas import FitClassification, MemoryComponent
from localbench.stack.schemas import StackModelPlan, StackPlan, StackPlanError, StackWorkloadRequest

# Applied exactly once per stack, not once per model — deliberately reusing the same magnitude
# as the single-model estimator's per-model headroom (see `estimation.memory`), because the
# underlying reason (OS and background-process activity) does not multiply per loaded model.
STACK_SAFETY_HEADROOM_BYTES = 2 * 1024**3


def _fit_classification(total_bytes: int, available_bytes: int) -> FitClassification:
    ratio = total_bytes / available_bytes
    if ratio <= 0.60:
        return FitClassification.EXCELLENT
    if ratio <= 0.80:
        return FitClassification.GOOD
    if ratio <= 0.95:
        return FitClassification.TIGHT
    if ratio <= 1.0:
        return FitClassification.POOR
    return FitClassification.WILL_NOT_FIT


def _validate_workload(request: StackWorkloadRequest) -> None:
    if not request.role.strip():
        raise StackPlanError("invalid_role", "A workload role name must not be blank.")
    if request.estimate.working_memory_bytes <= 0:
        raise StackPlanError(
            "invalid_working_memory",
            f"Role {request.role!r} has a non-positive working-memory estimate "
            f"({request.estimate.working_memory_bytes} bytes).",
        )
    components = (
        request.estimate.weights.bytes,
        request.estimate.kv_cache.bytes,
        request.estimate.runtime_overhead.bytes,
    )
    if any(component < 0 for component in components):
        raise StackPlanError(
            "invalid_memory_component",
            f"Role {request.role!r} has a negative memory component.",
        )
    if sum(components) != request.estimate.working_memory_bytes:
        raise StackPlanError(
            "inconsistent_working_memory",
            f"Role {request.role!r}'s weights, KV cache, and runtime overhead do not add "
            "up to its working-memory total.",
        )


def plan_stack(
    workloads: Sequence[StackWorkloadRequest],
    available_memory_bytes: int,
    *,
    safety_headroom_bytes: int = STACK_SAFETY_HEADROOM_BYTES,
) -> StackPlan:
    """Combine per-model memory estimates into one explainable simultaneous-load plan.

    Raises `StackPlanError` for empty input, invalid roles, non-positive memory figures,
    or a model requested under more than one role — concurrent-load memory sharing between
    two instances of the same model is unmeasured and is not guessed at.
    """
    if not workloads:
        raise StackPlanError("empty_stack", "A stack plan requires at least one workload role.")
    if available_memory_bytes <= 0:
        raise StackPlanError(
            "memory_unavailable", "Available system memory could not be measured."
        )
    if safety_headroom_bytes < 0:
        raise StackPlanError(
            "invalid_safety_headroom", "Safety headroom cannot be negative."
        )

    for request in workloads:
        _validate_workload(request)

    seen_roles: dict[str, str] = {}
    seen_models: dict[str, str] = {}
    for request in workloads:
        role_key = request.role.casefold()
        if role_key in seen_roles:
            raise StackPlanError(
                "duplicate_role", f"Workload role {request.role!r} was specified more than once."
            )
        seen_roles[role_key] = request.role

        model_key = request.estimate.model.casefold()
        if model_key in seen_models:
            raise StackPlanError(
                "duplicate_model",
                f"Model {request.estimate.model!r} is requested by both role "
                f"{seen_models[model_key]!r} and role {request.role!r}. Memory sharing between "
                "two simultaneously loaded instances of the same model is not modeled, so "
                "duplicate model entries are refused rather than silently deduplicated or "
                "double-counted.",
            )
        seen_models[model_key] = request.role

    models: list[StackModelPlan] = []
    for request in workloads:
        estimate = request.estimate
        warnings: list[str] = []
        if estimate.available_memory.bytes != available_memory_bytes:
            warnings.append(
                f"Role {request.role!r}'s estimate was computed against a different "
                f"available-memory snapshot ({estimate.available_memory.bytes} bytes) than "
                f"this plan's snapshot ({available_memory_bytes} bytes); it may be stale."
            )
        models.append(
            StackModelPlan(
                role=request.role,
                model=estimate.model,
                weights_bytes=estimate.weights.bytes,
                kv_cache_bytes=estimate.kv_cache.bytes,
                runtime_overhead_bytes=estimate.runtime_overhead.bytes,
                working_memory_bytes=estimate.working_memory_bytes,
                warnings=tuple(warnings),
            )
        )

    total_working_memory_bytes = sum(model.working_memory_bytes for model in models)
    total_required_bytes = total_working_memory_bytes + safety_headroom_bytes
    headroom_bytes = available_memory_bytes - total_required_bytes
    viability = _fit_classification(total_required_bytes, available_memory_bytes)

    plan_warnings = [warning for model in models for warning in model.warnings]

    assumptions = (
        "Safety headroom is applied once for the whole stack, not once per model; each "
        "role's working-memory figure already excludes it.",
        "Per-model runtime overhead is preserved and reported individually per role, "
        "not folded into a single stack-wide figure.",
        "All roles are evaluated against one shared available-memory snapshot supplied to "
        "this plan, not each estimate's own possibly-stale snapshot.",
        "This plan answers whether the combination fits in memory. Concurrent inference "
        "performance (throughput, latency, contention) is not measured here and must not be "
        "extrapolated from single-model benchmark results.",
        "One active sequence per model is assumed, matching the underlying single-model "
        "estimates; parallel requests within a role multiply that role's KV-cache demand.",
    )

    return StackPlan(
        collected_at=datetime.now(UTC),
        models=tuple(models),
        total_working_memory_bytes=total_working_memory_bytes,
        safety_headroom=MemoryComponent(
            bytes=safety_headroom_bytes,
            provenance="inferred",
            basis="LocalBench stack-level safety policy, applied once for the whole stack",
        ),
        total_required_bytes=total_required_bytes,
        available_memory=MemoryComponent(
            bytes=available_memory_bytes,
            provenance="measured",
            basis="Caller-supplied shared available-memory snapshot",
        ),
        headroom_bytes=headroom_bytes,
        viability=viability,
        viability_basis=(
            "Total required bytes (sum of per-model working memory plus one stack-level "
            "safety headroom) divided by the shared available-memory snapshot, using the "
            "same fit thresholds as single-model estimation"
        ),
        assumptions=assumptions,
        warnings=tuple(plan_warnings),
    )
