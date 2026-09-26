"""Ollama model-memory calculation with explicit assumptions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from localbench.estimation.schemas import (
    EstimateError,
    FitClassification,
    GPUOffloadClassification,
    GPUOffloadEstimate,
    KVCacheType,
    MemoryComponent,
    ModelMemoryEstimate,
)
from localbench.models import GPUProfile, RuntimeModelProfile, RuntimeProfile
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError
from localbench.runtimes.ollama import API_TIMEOUT_SECONDS, fetch_model_details

DEFAULT_CONTEXT_LENGTH = 8192
RUNTIME_OVERHEAD_FLOOR_BYTES = 512 * 1024**2
SAFETY_HEADROOM_BYTES = 2 * 1024**3
RUNTIME_OVERHEAD_FRACTION = 0.10
GPU_RESERVED_VRAM_FLOOR_BYTES = 256 * 1024**2
GPU_RESERVED_VRAM_FRACTION = 0.10

DetailsFetcher = Callable[[str, float], object]


@dataclass(frozen=True, slots=True)
class _CacheGeometry:
    total_layers: int
    attention_layers: int
    kv_heads: int
    key_length: int
    value_length: int
    recurrent_state_bytes: int
    basis: str


def _find_integer(metadata: Mapping[object, object], suffix: str) -> int | None:
    for key, value in metadata.items():
        if str(key).endswith(suffix) and isinstance(value, int) and value > 0:
            return value
    return None


def _select_model(runtime: RuntimeProfile, requested: str) -> RuntimeModelProfile:
    normalized = requested.casefold()
    exact = [model for model in runtime.models if model.name.casefold() == normalized]
    if exact:
        return exact[0]

    if ":" not in requested:
        family_matches = [
            model
            for model in runtime.models
            if model.name.partition(":")[0].casefold() == normalized
        ]
        latest = [model for model in family_matches if model.name.casefold().endswith(":latest")]
        if latest:
            return latest[0]
        if len(family_matches) == 1:
            return family_matches[0]
        if len(family_matches) > 1:
            choices = ", ".join(model.name for model in family_matches)
            raise EstimateError(
                "ambiguous_model",
                f"Model name {requested!r} is ambiguous; choose one of: {choices}.",
            )

    raise EstimateError("model_not_found", f"Installed Ollama model {requested!r} was not found.")


def _granite_hybrid_geometry(metadata: Mapping[object, object]) -> _CacheGeometry | None:
    """Recognize the published Granite 4 H Tiny/1B architecture exactly.

    Ollama's JSON `/api/show` representation turns the per-layer GGUF KV-head array into
    `null`. The remaining metadata identifies this one architecture signature without
    treating every future `granitehybrid` model as equivalent.
    """
    if metadata.get("general.architecture") != "granitehybrid":
        return None
    expected = {
        ".block_count": 40,
        ".embedding_length": 1536,
        ".attention.head_count": 12,
        ".rope.dimension_count": 128,
        ".ssm.conv_kernel": 4,
        ".ssm.group_count": 1,
        ".ssm.inner_size": 3072,
        ".ssm.state_size": 128,
    }
    if any(_find_integer(metadata, suffix) != value for suffix, value in expected.items()):
        return None

    attention_layers = 4
    recurrent_layers = expected[".block_count"] - attention_layers
    conv_elements = (expected[".ssm.conv_kernel"] - 1) * (
        expected[".ssm.inner_size"]
        + 2 * expected[".ssm.group_count"] * expected[".ssm.state_size"]
    )
    state_elements = expected[".ssm.state_size"] * expected[".ssm.inner_size"]
    recurrent_state_bytes = recurrent_layers * (conv_elements + state_elements) * 4
    return _CacheGeometry(
        total_layers=expected[".block_count"],
        attention_layers=attention_layers,
        kv_heads=4,
        key_length=128,
        value_length=128,
        recurrent_state_bytes=recurrent_state_bytes,
        basis=(
            "Exact Granite 4 H signature: 4 attention layers with 4 KV heads and 36 "
            "Mamba-2 layers; recurrent state calculated from reported SSM dimensions at f32"
        ),
    )


def _cache_geometry(metadata: Mapping[object, object]) -> _CacheGeometry:
    granite_hybrid = _granite_hybrid_geometry(metadata)
    if granite_hybrid is not None:
        return granite_hybrid

    blocks = _find_integer(metadata, ".block_count")
    kv_heads = _find_integer(metadata, ".attention.head_count_kv")
    key_length = _find_integer(metadata, ".attention.key_length")
    value_length = _find_integer(metadata, ".attention.value_length")
    basis = "Ollama model_info block, KV-head, key-length, and value-length fields"

    if key_length is None or value_length is None:
        embedding = _find_integer(metadata, ".embedding_length")
        attention_heads = _find_integer(metadata, ".attention.head_count")
        if embedding and attention_heads and embedding % attention_heads == 0:
            head_dimension = embedding // attention_heads
            key_length = key_length or head_dimension
            value_length = value_length or head_dimension
            basis = "Head dimension derived from embedding length and attention-head count"

    missing = [
        name
        for name, value in (
            ("block count", blocks),
            ("KV-head count", kv_heads),
            ("key length", key_length),
            ("value length", value_length),
        )
        if value is None
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise EstimateError(
            "insufficient_metadata",
            f"Ollama metadata cannot support a KV-cache estimate; missing {missing_text}.",
        )
    assert blocks is not None
    assert kv_heads is not None
    assert key_length is not None
    assert value_length is not None
    return _CacheGeometry(
        total_layers=blocks,
        attention_layers=blocks,
        kv_heads=kv_heads,
        key_length=key_length,
        value_length=value_length,
        recurrent_state_bytes=0,
        basis=basis,
    )


def _estimate_gpu_offload(
    gpus: tuple[GPUProfile, ...],
    total_layers: int,
    weights_bytes: int,
    kv_bytes: int,
) -> GPUOffloadEstimate | None:
    candidates = [gpu for gpu in gpus if (gpu.dedicated_memory_bytes or 0) > 0]
    if not candidates:
        return None
    gpu = max(candidates, key=lambda candidate: candidate.dedicated_memory_bytes or 0)
    vram_bytes = gpu.dedicated_memory_bytes
    assert vram_bytes is not None

    reserved_bytes = max(
        GPU_RESERVED_VRAM_FLOOR_BYTES, math.ceil(vram_bytes * GPU_RESERVED_VRAM_FRACTION)
    )
    usable_vram_bytes = max(0, vram_bytes - reserved_bytes)
    per_layer_bytes = math.ceil((weights_bytes + kv_bytes) / total_layers)
    offloadable_layers = (
        min(total_layers, usable_vram_bytes // per_layer_bytes) if per_layer_bytes > 0 else 0
    )
    offload_fraction = offloadable_layers / total_layers if total_layers else 0.0

    if offloadable_layers >= total_layers:
        classification = GPUOffloadClassification.FULL_OFFLOAD
    elif offloadable_layers > 0:
        classification = GPUOffloadClassification.PARTIAL_OFFLOAD
    else:
        classification = GPUOffloadClassification.CPU_ONLY

    return GPUOffloadEstimate(
        gpu_name=gpu.name,
        vram=MemoryComponent(
            bytes=vram_bytes,
            provenance="measured",
            basis=f"{gpu.source} dedicated-memory report",
        ),
        reserved_vram=MemoryComponent(
            bytes=reserved_bytes,
            provenance="inferred",
            basis="10% of VRAM reserved for the driver and OS compositor, 256 MiB floor",
        ),
        total_layers=total_layers,
        offloadable_layers=offloadable_layers,
        offload_fraction=offload_fraction,
        classification=classification,
        basis=(
            "Weights and KV cache are assumed evenly distributed across model layers; "
            "layers whose share fits in usable VRAM are counted as offloadable"
        ),
        assumptions=(
            "Assumes one GPU is used and layers are uniformly sized, which undercounts "
            "capacity for models with a large embedding or output layer.",
            "Does not model Ollama's actual layer-placement algorithm, multi-GPU splitting, "
            "or other processes already holding VRAM.",
        ),
    )


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


def estimate_ollama_model(
    requested_model: str,
    runtime: RuntimeProfile,
    available_memory_bytes: int,
    *,
    context_length: int | None = None,
    kv_cache_type: KVCacheType = KVCacheType.F16,
    gpus: tuple[GPUProfile, ...] = (),
    fetch_details: DetailsFetcher = fetch_model_details,
) -> ModelMemoryEstimate:
    if runtime.name != "ollama":
        raise EstimateError("unsupported_runtime", "The initial estimator supports only Ollama.")
    if not runtime.service_running:
        raise EstimateError(
            "runtime_unavailable",
            "Ollama's local service must be running to inspect installed model metadata.",
        )
    if available_memory_bytes <= 0:
        raise EstimateError("memory_unavailable", "Available system memory could not be measured.")

    model = _select_model(runtime, requested_model)
    if model.size_bytes is None or model.size_bytes <= 0:
        raise EstimateError("weight_size_unavailable", "Ollama did not report the model's size.")

    try:
        detail_payload = fetch_details(model.name, API_TIMEOUT_SECONDS)
    except RuntimeUnavailableError as error:
        raise EstimateError("runtime_unavailable", "Ollama stopped responding.") from error
    except RuntimeResponseError as error:
        message = "Ollama returned invalid model metadata."
        raise EstimateError("invalid_metadata", message) from error
    if not isinstance(detail_payload, Mapping):
        raise EstimateError("invalid_metadata", "Ollama model metadata was not an object.")
    raw_model_info = detail_payload.get("model_info")
    if not isinstance(raw_model_info, Mapping):
        raise EstimateError("insufficient_metadata", "Ollama did not return model_info metadata.")

    model_context_limit = _find_integer(raw_model_info, ".context_length")
    if context_length is None:
        context_ceiling = model_context_limit or DEFAULT_CONTEXT_LENGTH
        selected_context = min(DEFAULT_CONTEXT_LENGTH, context_ceiling)
    else:
        selected_context = context_length
    if selected_context <= 0:
        raise EstimateError("invalid_context", "Context length must be greater than zero.")
    if model_context_limit is not None and selected_context > model_context_limit:
        raise EstimateError(
            "context_exceeds_limit",
            f"Requested context {selected_context} exceeds the model limit {model_context_limit}.",
        )

    cache_geometry = _cache_geometry(raw_model_info)
    bytes_per_element = {
        KVCacheType.F16: 2.0,
        KVCacheType.Q8_0: 1.0,
        KVCacheType.Q4_0: 0.5,
    }[kv_cache_type]
    kv_bytes = math.ceil(
        cache_geometry.attention_layers
        * cache_geometry.kv_heads
        * (cache_geometry.key_length + cache_geometry.value_length)
        * selected_context
        * bytes_per_element
    ) + cache_geometry.recurrent_state_bytes
    runtime_overhead_bytes = max(
        RUNTIME_OVERHEAD_FLOOR_BYTES,
        math.ceil(model.size_bytes * RUNTIME_OVERHEAD_FRACTION),
    )
    working_memory_bytes = model.size_bytes + kv_bytes + runtime_overhead_bytes
    estimated_total_bytes = working_memory_bytes + SAFETY_HEADROOM_BYTES
    gpu_offload = _estimate_gpu_offload(
        gpus, cache_geometry.total_layers, model.size_bytes, kv_bytes
    )

    warnings: tuple[str, ...] = (
        "System-memory fit assumes no GPU offload; a partial or full offload reduces "
        "actual RAM pressure below this estimate.",
    )
    if gpu_offload is not None:
        warnings += (
            "GPU offload is a coarse per-layer approximation, not Ollama's real placement "
            "algorithm, and does not model unified-memory (e.g. Apple Silicon) behavior.",
        )
    assumptions: tuple[str, ...] = (
        "One active sequence is assumed; parallel requests multiply KV-cache demand.",
        f"KV cache uses {kv_cache_type.value} at {bytes_per_element:g} bytes per element.",
        "Runtime overhead is inferred as 10% of model size with a 512 MiB floor.",
        "Safety headroom is a fixed 2 GiB for the OS and background activity.",
    )
    if cache_geometry.recurrent_state_bytes:
        assumptions += (
            "Granite 4 H Mamba-2 recurrent state is modeled at f32; unlike attention KV, "
            "this fixed per-sequence state does not grow with context length.",
        )
    return ModelMemoryEstimate(
        collected_at=datetime.now(UTC),
        model=model.name,
        context_length=selected_context,
        model_context_limit=model_context_limit,
        kv_cache_type=kv_cache_type,
        weights=MemoryComponent(
            bytes=model.size_bytes,
            provenance="reported",
            basis="Ollama /api/tags model size",
        ),
        kv_cache=MemoryComponent(
            bytes=kv_bytes, provenance="calculated", basis=cache_geometry.basis
        ),
        runtime_overhead=MemoryComponent(
            bytes=runtime_overhead_bytes,
            provenance="inferred",
            basis="10% of model size with a 512 MiB floor",
        ),
        safety_headroom=MemoryComponent(
            bytes=SAFETY_HEADROOM_BYTES,
            provenance="inferred",
            basis="LocalBench initial safety policy",
        ),
        working_memory_bytes=working_memory_bytes,
        estimated_total_bytes=estimated_total_bytes,
        available_memory=MemoryComponent(
            bytes=available_memory_bytes,
            provenance="measured",
            basis="psutil available-memory snapshot",
        ),
        fit=_fit_classification(estimated_total_bytes, available_memory_bytes),
        fit_basis="Estimated total divided by currently available system memory",
        gpu_offload=gpu_offload,
        assumptions=assumptions,
        warnings=warnings,
    )
