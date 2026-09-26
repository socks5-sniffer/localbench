from __future__ import annotations

from collections.abc import Mapping

import pytest

from localbench.benchmark.coexistence import (
    CoexistenceError,
    CoexistenceRequest,
    run_coexistence_benchmark,
)
from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.sampler import ResourceSample
from localbench.benchmark.schemas import StoredResult
from localbench.models import RuntimeModelProfile, RuntimeProfile


class FakeSampler:
    def start(self) -> None:
        pass

    def stop(self) -> ResourceSample:
        return ResourceSample(8_000, 12_000, 87.5)


def _runtime(*models: str, running: bool = True) -> RuntimeProfile:
    profiles = tuple(
        RuntimeModelProfile(
            name=name,
            size_bytes=1000,
            digest=None,
            modified_at=None,
            format="gguf",
            family="example",
            parameter_size="1B",
            quantization_level="Q4",
            source="test",
        )
        for name in models
    )
    return RuntimeProfile("ollama", True, running, "1.2.3", profiles)


def _final_chunk(
    *, generation_count: int = 64, generation_ns: int = 4_000_000_000
) -> Mapping[str, object]:
    return {
        "response": "",
        "done": True,
        "done_reason": "length" if generation_count >= 64 else "stop",
        "load_duration": 1_000_000_000,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 500_000_000,
        "eval_count": generation_count,
        "eval_duration": generation_ns,
        "total_duration": generation_ns + 1_500_000_000,
    }


def _chunks(**final_kwargs: object) -> list[Mapping[str, object]]:
    return [{"response": "First", "done": False}, _final_chunk(**final_kwargs)]  # type: ignore[arg-type]


def _stream_for(per_model: dict[str, list[Mapping[str, object]]]):  # noqa: ANN201
    def _stream(body: object, _timeout: float) -> object:
        assert isinstance(body, dict)
        model = body["model"]
        assert isinstance(model, str)
        return iter(per_model[model])

    return _stream


def _stored_baseline(
    *,
    model: str = "chat:latest",
    generation_rate: float | None = 10.0,
    ttft: float | None = 2.0,
    load_time: float | None = 1.0,
    generation_tokens: int | None = 64,
    runtime_version: str | None = "1.2.3",
) -> StoredResult:
    signature = quick_measurement_signature(runtime_version)
    return StoredResult(
        run_id="baseline-run",
        started_at="2026-08-31T02:00:00+00:00",
        model=model,
        prompt_id="benchmark_generation_v2",
        generation_tokens=generation_tokens,
        generation_tokens_per_second=generation_rate,
        prompt_tokens_per_second=30.0,
        ttft_seconds=ttft,
        load_time_seconds=load_time,
        runtime="ollama",
        runtime_version=runtime_version,
        pre_run_unload_succeeded=True,
        measurement_signature=signature,
        batch_id="baseline-run",
        batch_index=1,
        batch_size=1,
    )


def test_requires_running_service() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(requests, _runtime("chat:latest", "embed:latest", running=False))

    assert caught.value.code == "runtime_unavailable"


def test_requires_at_least_two_workloads() -> None:
    requests = [CoexistenceRequest(role="chat", model="chat:latest")]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(requests, _runtime("chat:latest"))

    assert caught.value.code == "insufficient_workloads"


def test_rejects_blank_role() -> None:
    requests = [
        CoexistenceRequest(role="  ", model="chat:latest"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(requests, _runtime("chat:latest", "embed:latest"))

    assert caught.value.code == "invalid_role"


def test_rejects_duplicate_role() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="Chat", model="embed:latest"),
    ]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(requests, _runtime("chat:latest", "embed:latest"))

    assert caught.value.code == "duplicate_role"


def test_rejects_unknown_model() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="embed", model="missing:latest"),
    ]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(requests, _runtime("chat:latest"))

    assert caught.value.code == "model_not_found"


def test_resolves_latest_suffix_and_single_family_match() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat"),  # resolves to chat:latest
        CoexistenceRequest(role="embed", model="embed"),  # resolves to the one embed:* match
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:v1": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:v1"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    by_role = {m.role: m for m in result.models}
    assert by_role["chat"].model == "chat:latest"
    assert by_role["embed"].model == "embed:v1"


def test_rejects_ambiguous_model() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(
            requests, _runtime("chat:3b", "chat:8b", "embed:latest")
        )

    assert caught.value.code == "ambiguous_model"


def test_runs_two_models_concurrently_and_reports_metrics() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for(
        {
            "chat:latest": _chunks(),
            "embed:latest": _chunks(generation_count=64, generation_ns=2_000_000_000),
        }
    )
    unloaded: list[str] = []

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda model, _timeout: unloaded.append(model),
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    assert result.concurrency == 2
    assert sorted(unloaded) == ["chat:latest", "embed:latest"]
    by_role = {m.role: m for m in result.models}
    assert by_role["chat"].model == "chat:latest"
    assert by_role["chat"].generation_tokens_per_second == pytest.approx(16.0)
    assert by_role["embed"].generation_tokens_per_second == pytest.approx(32.0)
    assert by_role["chat"].baseline_run_id is None
    assert by_role["chat"].generation_rate_change_percent is None
    assert result.peak_system_ram_increase_bytes == 4000
    assert result.peak_system_cpu_percent == 87.5
    assert any("does not extrapolate" in warning for warning in result.warnings)


def test_computes_contention_percentage_against_matching_baseline() -> None:
    baseline = _stored_baseline(model="chat:latest", generation_rate=32.0, load_time=2.0)
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for(
        {
            # Half the solo rate: 64 tokens / 4s = 16 tok/s vs. baseline's 32 tok/s.
            "chat:latest": _chunks(generation_count=64, generation_ns=4_000_000_000),
            "embed:latest": _chunks(),
        }
    )

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.baseline_run_id == "baseline-run"
    assert chat.generation_rate_change_percent == pytest.approx(-50.0)
    # The fake chunk's fixed 1s load_duration beats the 2s baseline: positive (better),
    # since load time is lower-is-better and the sign is normalized accordingly.
    assert chat.load_time_change_percent == pytest.approx(50.0)
    assert chat.ttft_change_percent is not None
    assert chat.warnings == ()


def test_baseline_model_mismatch_skips_contention_with_warning() -> None:
    baseline = _stored_baseline(model="different:latest")
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert chat.warnings
    assert "different:latest" in chat.warnings[0]
    assert chat.warnings[0] in result.warnings


def test_baseline_missing_signature_skips_contention_with_warning() -> None:
    import dataclasses

    baseline = dataclasses.replace(_stored_baseline(), measurement_signature=None)
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "no known measurement signature" in chat.warnings[0]


def test_baseline_different_runtime_version_skips_contention_with_warning() -> None:
    baseline = _stored_baseline(runtime_version="0.9.0")
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),  # runtime.version == "1.2.3"
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "different measurement conditions" in chat.warnings[0]


def test_baseline_any_signature_field_mismatch_skips_contention() -> None:
    import dataclasses

    baseline = _stored_baseline()
    assert baseline.measurement_signature is not None
    baseline = dataclasses.replace(
        baseline,
        measurement_signature=dataclasses.replace(
            baseline.measurement_signature, seed=999
        ),
    )
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "different measurement conditions" in chat.warnings[0]


def test_failed_baseline_cold_unload_skips_contention() -> None:
    import dataclasses

    baseline = dataclasses.replace(
        _stored_baseline(), pre_run_unload_succeeded=False
    )
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "baseline did not complete its cold unload" in chat.warnings[0]


def test_short_concurrent_generation_skips_contention() -> None:
    requests = [
        CoexistenceRequest(
            role="chat", model="chat:latest", baseline=_stored_baseline()
        ),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for(
        {
            "chat:latest": _chunks(generation_count=3),
            "embed:latest": _chunks(),
        }
    )

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "concurrent run did not reach" in chat.warnings[0]


def test_baseline_missing_metric_skips_contention_with_warning() -> None:
    baseline = _stored_baseline(ttft=None)
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "missing one or more required metrics" in chat.warnings[0]


def test_baseline_short_generation_skips_contention_with_warning() -> None:
    baseline = _stored_baseline(generation_tokens=15)
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest", baseline=baseline),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    assert chat.generation_rate_change_percent is None
    assert "generation target" in chat.warnings[0]


def test_one_role_failing_raises_coexistence_error() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": []})

    with pytest.raises(CoexistenceError) as caught:
        run_coexistence_benchmark(
            requests,
            _runtime("chat:latest", "embed:latest"),
            stream=stream,  # type: ignore[arg-type]
            unload=lambda _model, _timeout: None,
            sampler=FakeSampler(),  # type: ignore[arg-type]
        )

    assert caught.value.code == "concurrent_run_failed"
    assert "embed" in caught.value.message


def test_pre_run_unload_failure_produces_warning() -> None:
    from localbench.runtimes.base import RuntimeUnavailableError

    requests = [
        CoexistenceRequest(
            role="chat", model="chat:latest", baseline=_stored_baseline()
        ),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    def _unload(model: str, _timeout: float) -> None:
        if model == "chat:latest":
            raise RuntimeUnavailableError("boom")

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=_unload,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    chat = next(m for m in result.models if m.role == "chat")
    embed = next(m for m in result.models if m.role == "embed")
    assert chat.pre_run_unload_succeeded is False
    assert chat.generation_rate_change_percent is None
    assert any(
        "concurrent run did not complete its cold unload" in warning
        for warning in chat.warnings
    )
    assert embed.pre_run_unload_succeeded is True
    assert any("chat" in warning and "unload failed" in warning for warning in result.warnings)


def test_to_dict_serializes_started_at_and_schema_version() -> None:
    requests = [
        CoexistenceRequest(role="chat", model="chat:latest"),
        CoexistenceRequest(role="embed", model="embed:latest"),
    ]
    stream = _stream_for({"chat:latest": _chunks(), "embed:latest": _chunks()})

    result = run_coexistence_benchmark(
        requests,
        _runtime("chat:latest", "embed:latest"),
        stream=stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
    )

    payload = result.to_dict()
    assert isinstance(payload["started_at"], str)
    assert payload["schema_version"] == "1"
    assert len(payload["models"]) == 2
