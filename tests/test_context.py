from __future__ import annotations

import json

import pytest

from localbench.context import (
    DEFAULT_CONTEXT_LEVELS,
    ContextProfileError,
    profile_context_capacity,
)
from localbench.estimation.schemas import EstimateError
from localbench.models import RuntimeModelProfile, RuntimeProfile

GIB = 1024**3


def _model(name: str = "example:latest", size_bytes: int = 5 * GIB) -> RuntimeModelProfile:
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


def test_profiles_memory_growth_across_default_levels() -> None:
    calls: list[str] = []

    def _fetch(model: str, _timeout: float) -> object:
        calls.append(model)
        return _details(context_limit=32768)

    profile = profile_context_capacity(
        "example", _runtime(), 16 * GIB, fetch_details=_fetch
    )

    assert [level.context_length for level in profile.levels] == [
        2048,
        4096,
        8192,
        16384,
        32768,
    ]
    assert profile.skipped_context_lengths == (65536,)
    assert profile.model_context_limit == 32768
    assert profile.model == "example:latest"
    # KV cache grows linearly with context: doubling context doubles KV bytes.
    kv_by_level = {level.context_length: level.estimate.kv_cache.bytes for level in profile.levels}
    assert kv_by_level[4096] == kv_by_level[2048] * 2
    assert kv_by_level[8192] == kv_by_level[2048] * 4
    # Weights and runtime overhead are fixed regardless of context length.
    assert profile.levels[0].estimate.weights.bytes == profile.levels[-1].estimate.weights.bytes


def test_fetches_model_details_only_once_across_the_whole_sweep() -> None:
    calls: list[str] = []

    def _fetch(model: str, _timeout: float) -> object:
        calls.append(model)
        return _details()

    profile_context_capacity(
        "example",
        _runtime(),
        16 * GIB,
        context_levels=(2048, 4096, 8192),
        fetch_details=_fetch,
    )

    assert calls == ["example:latest"]


def test_custom_levels_are_evaluated_in_sorted_order() -> None:
    profile = profile_context_capacity(
        "example",
        _runtime(),
        16 * GIB,
        context_levels=(8192, 2048, 4096),
        fetch_details=lambda _model, _timeout: _details(),
    )

    assert [level.context_length for level in profile.levels] == [2048, 4096, 8192]


def test_rejects_empty_levels() -> None:
    with pytest.raises(ContextProfileError) as caught:
        profile_context_capacity(
            "example",
            _runtime(),
            16 * GIB,
            context_levels=(),
            fetch_details=lambda _model, _timeout: _details(),
        )

    assert caught.value.code == "empty_levels"


def test_rejects_non_positive_context_length() -> None:
    with pytest.raises(ContextProfileError) as caught:
        profile_context_capacity(
            "example",
            _runtime(),
            16 * GIB,
            context_levels=(2048, 0),
            fetch_details=lambda _model, _timeout: _details(),
        )

    assert caught.value.code == "invalid_context_length"


def test_rejects_duplicate_context_length() -> None:
    with pytest.raises(ContextProfileError) as caught:
        profile_context_capacity(
            "example",
            _runtime(),
            16 * GIB,
            context_levels=(2048, 4096, 2048),
            fetch_details=lambda _model, _timeout: _details(),
        )

    assert caught.value.code == "duplicate_context_length"


def test_all_levels_exceeding_model_limit_raises_no_levels_fit() -> None:
    with pytest.raises(ContextProfileError) as caught:
        profile_context_capacity(
            "example",
            _runtime(),
            16 * GIB,
            context_levels=DEFAULT_CONTEXT_LEVELS,
            fetch_details=lambda _model, _timeout: _details(context_limit=1024),
        )

    assert caught.value.code == "no_levels_fit"


def test_propagates_model_resolution_errors_immediately() -> None:
    models = (
        _model(name="example:3b"),
        _model(name="example:8b"),
    )

    with pytest.raises(EstimateError) as caught:
        profile_context_capacity(
            "example",
            _runtime(*models),
            16 * GIB,
            fetch_details=lambda _model, _timeout: _details(),
        )

    assert caught.value.code == "ambiguous_model"


def test_partial_skip_still_reports_supported_levels() -> None:
    profile = profile_context_capacity(
        "example",
        _runtime(),
        16 * GIB,
        context_levels=(2048, 8192, 65536),
        fetch_details=lambda _model, _timeout: _details(context_limit=8192),
    )

    assert [level.context_length for level in profile.levels] == [2048, 8192]
    assert profile.skipped_context_lengths == (65536,)


def test_assumptions_disclose_what_is_not_measured() -> None:
    profile = profile_context_capacity(
        "example",
        _runtime(),
        16 * GIB,
        context_levels=(2048,),
        fetch_details=lambda _model, _timeout: _details(),
    )

    assert any("does not measure" in assumption for assumption in profile.assumptions)


def test_to_dict_serializes_collected_at_and_schema_version() -> None:
    profile = profile_context_capacity(
        "example",
        _runtime(),
        16 * GIB,
        context_levels=(2048, 4096),
        fetch_details=lambda _model, _timeout: _details(),
    )

    payload = profile.to_dict()
    assert isinstance(payload["collected_at"], str)
    assert payload["schema_version"] == "1"
    assert len(payload["levels"]) == 2
    assert isinstance(payload["levels"][0]["estimate"]["collected_at"], str)
    json.dumps(payload)
