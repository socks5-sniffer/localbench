from __future__ import annotations

import pytest

from localbench.estimation import (
    EstimateError,
    GPUOffloadClassification,
    KVCacheType,
    estimate_ollama_model,
)
from localbench.models import GPUProfile, RuntimeModelProfile, RuntimeProfile

GIB = 1024**3


def _gpu(
    dedicated_memory_bytes: int | None,
    name: str = "Test GPU",
) -> GPUProfile:
    return GPUProfile(
        name=name,
        vendor="Test",
        dedicated_memory_bytes=dedicated_memory_bytes,
        shared_memory_bytes=None,
        driver_version="1.0",
        source="test",
    )


def _model(name: str = "example:latest", size_bytes: int | None = 5 * GIB) -> RuntimeModelProfile:
    return RuntimeModelProfile(
        name=name,
        size_bytes=size_bytes,
        digest="abc",
        modified_at=None,
        format="gguf",
        family="example",
        parameter_size="4B",
        quantization_level="Q4_K_M",
        source="test",
    )


def _runtime(*models: RuntimeModelProfile, running: bool = True) -> RuntimeProfile:
    return RuntimeProfile(
        name="ollama",
        cli_detected=True,
        service_running=running,
        version="1.0.0",
        models=models or (_model(),),
    )


def _details(context_limit: int = 32768) -> dict[str, object]:
    return {
        "model_info": {
            "example.block_count": 32,
            "example.attention.head_count_kv": 8,
            "example.attention.key_length": 128,
            "example.attention.value_length": 128,
            "example.context_length": context_limit,
        }
    }


def _granite_hybrid_details() -> dict[str, object]:
    return {
        "model_info": {
            "general.architecture": "granitehybrid",
            "granitehybrid.attention.head_count": 12,
            "granitehybrid.attention.head_count_kv": None,
            "granitehybrid.block_count": 40,
            "granitehybrid.context_length": 1048576,
            "granitehybrid.embedding_length": 1536,
            "granitehybrid.rope.dimension_count": 128,
            "granitehybrid.ssm.conv_kernel": 4,
            "granitehybrid.ssm.group_count": 1,
            "granitehybrid.ssm.inner_size": 3072,
            "granitehybrid.ssm.state_size": 128,
        }
    }


def test_estimate_calculates_explainable_components() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert estimate.model == "example:latest"
    assert estimate.weights.bytes == 5 * GIB
    assert estimate.weights.provenance == "reported"
    assert estimate.kv_cache.bytes == GIB
    assert estimate.kv_cache.provenance == "calculated"
    assert estimate.runtime_overhead.bytes == GIB // 2
    assert estimate.estimated_total_bytes == 8 * GIB + GIB // 2
    assert estimate.fit.value == "excellent"
    assert estimate.gpu_offload is None


def test_quantized_kv_cache_changes_only_cache_component() -> None:
    f16 = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        fetch_details=lambda _model_name, _timeout: _details(),
    )
    q8 = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        kv_cache_type=KVCacheType.Q8_0,
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert q8.kv_cache.bytes == f16.kv_cache.bytes // 2
    assert q8.weights == f16.weights


def test_default_context_is_capped_to_model_limit() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        fetch_details=lambda _model_name, _timeout: _details(context_limit=4096),
    )

    assert estimate.context_length == 4096


def test_head_dimensions_can_be_derived() -> None:
    details = {
        "model_info": {
            "example.block_count": 32,
            "example.attention.head_count_kv": 8,
            "example.attention.head_count": 32,
            "example.embedding_length": 4096,
            "example.context_length": 8192,
        }
    }

    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        fetch_details=lambda _model_name, _timeout: details,
    )

    assert estimate.kv_cache.bytes == GIB
    assert "derived" in estimate.kv_cache.basis.lower()


def test_granite_hybrid_uses_exact_attention_and_recurrent_geometry() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        fetch_details=lambda _model_name, _timeout: _granite_hybrid_details(),
    )

    attention_kv_bytes = 4 * 4 * (128 + 128) * 8192 * 2
    recurrent_state_bytes = 36 * (((4 - 1) * (3072 + 2 * 1 * 128)) + 128 * 3072) * 4
    assert estimate.kv_cache.bytes == attention_kv_bytes + recurrent_state_bytes
    assert "4 attention layers" in estimate.kv_cache.basis
    assert any("recurrent state" in assumption for assumption in estimate.assumptions)


def test_granite_hybrid_kv_quantization_does_not_quantize_recurrent_state() -> None:
    f16 = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        fetch_details=lambda _model_name, _timeout: _granite_hybrid_details(),
    )
    q8 = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        kv_cache_type=KVCacheType.Q8_0,
        fetch_details=lambda _model_name, _timeout: _granite_hybrid_details(),
    )

    recurrent_state_bytes = 36 * (((4 - 1) * (3072 + 2 * 1 * 128)) + 128 * 3072) * 4
    assert q8.kv_cache.bytes - recurrent_state_bytes == (
        f16.kv_cache.bytes - recurrent_state_bytes
    ) // 2


def test_unknown_granite_hybrid_signature_still_fails_closed() -> None:
    details = _granite_hybrid_details()
    model_info = details["model_info"]
    assert isinstance(model_info, dict)
    model_info["granitehybrid.embedding_length"] = 2048

    with pytest.raises(EstimateError) as caught:
        estimate_ollama_model(
            "example",
            _runtime(),
            16 * GIB,
            fetch_details=lambda _model_name, _timeout: details,
        )

    assert caught.value.code == "insufficient_metadata"


@pytest.mark.parametrize(
    ("runtime", "requested", "details", "code"),
    [
        (_runtime(running=False), "example", _details(), "runtime_unavailable"),
        (_runtime(), "missing", _details(), "model_not_found"),
        (_runtime(_model(size_bytes=None)), "example", _details(), "weight_size_unavailable"),
        (_runtime(), "example", {"model_info": {}}, "insufficient_metadata"),
    ],
)
def test_estimate_rejects_unsupported_inputs(
    runtime: RuntimeProfile,
    requested: str,
    details: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(EstimateError) as caught:
        estimate_ollama_model(
            requested,
            runtime,
            16 * GIB,
            fetch_details=lambda _model_name, _timeout: details,
        )

    assert caught.value.code == code


def test_estimate_rejects_context_above_model_limit() -> None:
    with pytest.raises(EstimateError) as caught:
        estimate_ollama_model(
            "example",
            _runtime(),
            16 * GIB,
            context_length=65536,
            fetch_details=lambda _model_name, _timeout: _details(),
        )

    assert caught.value.code == "context_exceeds_limit"


def test_untagged_ambiguous_model_requires_exact_name() -> None:
    runtime = _runtime(_model("example:q4"), _model("example:q8"))

    with pytest.raises(EstimateError) as caught:
        estimate_ollama_model(
            "example",
            runtime,
            16 * GIB,
            fetch_details=lambda _model_name, _timeout: _details(),
        )

    assert caught.value.code == "ambiguous_model"


def test_gpu_with_no_dedicated_memory_is_ignored() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        gpus=(_gpu(dedicated_memory_bytes=None),),
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert estimate.gpu_offload is None


def test_small_vram_yields_cpu_only_classification() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        gpus=(_gpu(dedicated_memory_bytes=128 * 1024**2),),
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert estimate.gpu_offload is not None
    assert estimate.gpu_offload.classification == GPUOffloadClassification.CPU_ONLY
    assert estimate.gpu_offload.offloadable_layers == 0


def test_moderate_vram_yields_partial_offload() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        gpus=(_gpu(dedicated_memory_bytes=4 * GIB),),
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert estimate.gpu_offload is not None
    assert estimate.gpu_offload.classification == GPUOffloadClassification.PARTIAL_OFFLOAD
    assert 0 < estimate.gpu_offload.offloadable_layers < estimate.gpu_offload.total_layers
    assert estimate.gpu_offload.total_layers == 32


def test_ample_vram_yields_full_offload() -> None:
    estimate = estimate_ollama_model(
        "example",
        _runtime(),
        16 * GIB,
        context_length=8192,
        gpus=(
            _gpu(dedicated_memory_bytes=16 * GIB),
            _gpu(dedicated_memory_bytes=2 * GIB, name="Weaker GPU"),
        ),
        fetch_details=lambda _model_name, _timeout: _details(),
    )

    assert estimate.gpu_offload is not None
    assert estimate.gpu_offload.gpu_name == "Test GPU"
    assert estimate.gpu_offload.classification == GPUOffloadClassification.FULL_OFFLOAD
    assert estimate.gpu_offload.offloadable_layers == estimate.gpu_offload.total_layers
    assert estimate.gpu_offload.offload_fraction == 1.0
