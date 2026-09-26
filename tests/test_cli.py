from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from localbench import __version__, cli
from localbench.benchmark import (
    BenchmarkError,
    BenchmarkResult,
    CoexistenceResult,
    ModelCoexistenceResult,
    StoredResult,
)
from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.storage import BenchmarkStorageError
from localbench.context import ContextCapacityProfile
from localbench.estimation import EstimateError, ModelMemoryEstimate
from localbench.models import RuntimeDiscovery, SystemProfile

runner = CliRunner()


def test_profile_json_is_valid_and_contains_no_human_output(
    monkeypatch: object, sample_profile: SystemProfile
) -> None:
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["profile", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["schema_version"] == "1"
    assert decoded["memory"]["total_bytes"] == 16 * 1024**3
    # Guards JSON purity: the human banner must never leak into piped output. Pinned to
    # __version__ so it keeps testing something after a version bump.
    assert f"LocalBench {__version__}" not in result.stdout


def test_profile_human_output(
    monkeypatch: object, sample_profile: SystemProfile
) -> None:
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["profile"])

    assert result.exit_code == 0
    # Asserted against __version__ rather than a hardcoded literal: this test broke on
    # the 0.2 and 0.3 bumps for exactly that reason.
    assert f"LocalBench {__version__}" in result.stdout
    assert "Test CPU" in result.stdout
    assert "Test GPU" in result.stdout
    assert "Example warning" in result.stdout


def test_root_without_command_displays_help() -> None:
    result = runner.invoke(cli.app, [])

    assert result.exit_code == 2
    assert "profile" in result.stdout
    assert "runtimes" in result.stdout


def test_runtimes_json_output(
    monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["runtimes", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["schema_version"] == "1"
    assert decoded["runtimes"][0]["models"][0]["name"] == "example:latest"


def test_runtimes_human_output(
    monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["runtimes"])

    assert result.exit_code == 0
    assert "Running" in result.stdout
    assert "example:latest" in result.stdout


def test_estimate_json_output(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
    sample_estimate: ModelMemoryEstimate,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "estimate_ollama_model", lambda *_args, **_kwargs: sample_estimate
    )

    result = runner.invoke(cli.app, ["estimate", "example", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["fit"] == "excellent"
    assert decoded["weights"]["provenance"] == "reported"
    assert decoded["available_memory"]["provenance"] == "measured"


def test_estimate_human_output(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
    sample_estimate: ModelMemoryEstimate,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "estimate_ollama_model", lambda *_args, **_kwargs: sample_estimate
    )

    result = runner.invoke(cli.app, ["estimate", "example"])

    assert result.exit_code == 0
    assert "EXCELLENT" in result.stdout
    assert "Test assumption" in result.stdout


def test_estimate_json_error_is_structured(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]

    def fail(*_args: object, **_kwargs: object) -> None:
        raise EstimateError("model_not_found", "Missing model.")

    monkeypatch.setattr(cli, "estimate_ollama_model", fail)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["estimate", "missing", "--json"])

    assert result.exit_code == 2
    decoded = json.loads(result.stdout)
    assert decoded["error"]["code"] == "model_not_found"


def test_context_json_output(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
    sample_context_profile: ContextCapacityProfile,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "profile_context_capacity", lambda *_args, **_kwargs: sample_context_profile
    )

    result = runner.invoke(cli.app, ["context", "example", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["model"] == "example:latest"
    assert len(decoded["levels"]) == 2
    assert decoded["levels"][0]["estimate"]["fit"] == "excellent"


def test_context_human_output(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
    sample_context_profile: ContextCapacityProfile,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "profile_context_capacity", lambda *_args, **_kwargs: sample_context_profile
    )

    result = runner.invoke(cli.app, ["context", "example"])

    assert result.exit_code == 0
    assert "EXCELLENT" in result.stdout
    assert "Test context assumption" in result.stdout


def test_context_custom_levels_are_parsed(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_profile: SystemProfile,
    sample_context_profile: ContextCapacityProfile,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    captured: dict[str, object] = {}

    def profile_context_capacity(*_args: object, **kwargs: object) -> ContextCapacityProfile:
        captured.update(kwargs)
        return sample_context_profile

    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "profile_context_capacity", profile_context_capacity
    )

    result = runner.invoke(cli.app, ["context", "example", "--levels", "4096, 8192"])

    assert result.exit_code == 0
    assert captured["context_levels"] == (4096, 8192)


def test_context_invalid_levels_is_structured_error(
    monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery, sample_profile: SystemProfile
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["context", "example", "--levels", "not-a-number", "--json"])

    assert result.exit_code == 2
    decoded = json.loads(result.stdout)
    assert decoded["error"]["code"] == "invalid_levels"


def test_stack_json_uses_one_memory_snapshot_for_all_models(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_estimate: ModelMemoryEstimate,
) -> None:
    available = 20 * 1024**3
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "collect_memory", lambda: SimpleNamespace(available_bytes=available)
    )
    calls: list[tuple[str, int]] = []

    def estimate(model: str, _runtime: object, memory: int, **_kwargs: object) -> object:
        calls.append((model, memory))
        return dataclasses.replace(
            sample_estimate,
            model=f"{model}:latest",
            available_memory=dataclasses.replace(
                sample_estimate.available_memory, bytes=memory
            ),
        )

    monkeypatch.setattr(cli, "estimate_ollama_model", estimate)  # type: ignore[attr-defined]

    result = runner.invoke(
        cli.app,
        [
            "stack",
            "--model",
            "reasoning=example",
            "--model",
            "embeddings=nomic-embed-text",
            "--json",
        ],
    )

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert [model["role"] for model in decoded["models"]] == ["reasoning", "embeddings"]
    assert decoded["available_memory"]["bytes"] == available
    assert calls == [("example", available), ("nomic-embed-text", available)]

    human = runner.invoke(
        cli.app,
        ["stack", "--model", "reasoning=example", "--model", "embeddings=other"],
    )

    assert human.exit_code == 0
    assert "Local AI stack plan" in human.stdout
    assert "Memory feasibility only" in human.stdout


def test_stack_invalid_workload_is_structured_and_skips_discovery(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "discover_runtimes",
        lambda: pytest.fail("invalid input should fail before runtime discovery"),
    )

    result = runner.invoke(cli.app, ["stack", "--model", "missing-separator", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_workload_spec"


def test_stack_requires_at_least_one_model() -> None:
    result = runner.invoke(cli.app, ["stack", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "empty_stack"


def test_bench_json_output(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_benchmark: BenchmarkResult,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "run_quick_benchmark", lambda *_args, **_kwargs: sample_benchmark
    )
    saved: list[str] = []
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "save_benchmark_result", lambda result: saved.append(result.run_id)
    )

    result = runner.invoke(cli.app, ["bench", "example", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["generation_tokens_per_second"] == 16.0
    assert saved == [sample_benchmark.run_id]


def test_bench_no_save(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_benchmark: BenchmarkResult,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "run_quick_benchmark", lambda *_args, **_kwargs: sample_benchmark
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "save_benchmark_result",
        lambda _result: pytest.fail("save should not be called"),
    )

    result = runner.invoke(cli.app, ["bench", "example", "--no-save"])

    assert result.exit_code == 0
    assert "16.00 tok/s" in result.stdout


def test_bench_repeated_batch_outputs_median_and_saves_raw_runs(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_benchmark: BenchmarkResult,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    rates = iter((8.0, 10.0, 30.0))

    def run(*_args: object, **kwargs: object) -> BenchmarkResult:
        index = int(kwargs["batch_index"])
        rate = next(rates)
        return dataclasses.replace(
            sample_benchmark,
            run_id=f"run-{index}",
            batch_id=str(kwargs["batch_id"]),
            batch_index=index,
            batch_size=int(kwargs["batch_size"]),
            generation_tokens_per_second=rate,
        )

    monkeypatch.setattr(cli, "run_quick_benchmark", run)  # type: ignore[attr-defined]
    saved: list[BenchmarkResult] = []
    monkeypatch.setattr(cli, "save_benchmark_result", saved.append)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["bench", "example", "--runs", "3", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["completed_runs"] == 3
    assert decoded["generation_tokens_per_second"]["median"] == 10.0
    assert [item.batch_index for item in saved] == [1, 2, 3]
    assert len({item.batch_id for item in saved}) == 1


def test_bench_storage_failure_still_shows_result(
    monkeypatch: object,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_benchmark: BenchmarkResult,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "run_quick_benchmark", lambda *_args, **_kwargs: sample_benchmark
    )

    def fail_to_save(_result: object) -> None:
        raise BenchmarkStorageError("disk full")

    monkeypatch.setattr(cli, "save_benchmark_result", fail_to_save)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["bench", "example", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["generation_tokens_per_second"] == 16.0
    assert any("could not be saved" in warning for warning in decoded["warnings"])


def _stored(run_id: str, prompt_id: str = "benchmark_generation_v2") -> StoredResult:
    signature = quick_measurement_signature("1.2.3", prompt_id=prompt_id)
    return StoredResult(
        run_id=run_id,
        started_at="2026-08-31T02:00:00+00:00",
        model="example:latest",
        prompt_id=prompt_id,
        generation_tokens=64,
        generation_tokens_per_second=10.0,
        prompt_tokens_per_second=30.0,
        ttft_seconds=2.0,
        load_time_seconds=1.0,
        runtime="ollama",
        runtime_version="1.2.3",
        pre_run_unload_succeeded=True,
        measurement_signature=signature,
    )


def test_recommend_json_ranks_only_current_benchmarked_installed_models(
    monkeypatch: object,
    sample_profile: SystemProfile,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_estimate: ModelMemoryEstimate,
) -> None:
    runtime = sample_runtime_discovery.runtimes[0]
    second_model = dataclasses.replace(runtime.models[0], name="other:latest")
    discovery = dataclasses.replace(
        sample_runtime_discovery,
        runtimes=(dataclasses.replace(runtime, models=(*runtime.models, second_model)),),
    )
    monkeypatch.setattr(cli, "discover_runtimes", lambda: discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "load_benchmark_history",
        lambda: (
            dataclasses.replace(
                _stored("example-run"),
                generation_tokens_per_second=20.0,
                prompt_tokens_per_second=30.0,
                ttft_seconds=3.0,
            ),
            dataclasses.replace(
                _stored("other-run"),
                model="other:latest",
                generation_tokens_per_second=10.0,
                prompt_tokens_per_second=30.0,
                ttft_seconds=1.0,
            ),
        ),
    )
    estimate_calls: list[tuple[str, int]] = []

    def estimate(model: str, _runtime: object, _memory: int, **kwargs: object) -> object:
        estimate_calls.append((model, int(kwargs["context_length"])))
        return dataclasses.replace(sample_estimate, model=model, context_length=16384)

    monkeypatch.setattr(cli, "estimate_ollama_model", estimate)  # type: ignore[attr-defined]

    result = runner.invoke(
        cli.app,
        ["recommend", "--use-case", "low-latency", "--context", "16384", "--json"],
    )

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["use_case"] == "low-latency"
    assert decoded["weights"]["ttft"] == 0.5
    assert decoded["recommendations"][0]["model"] == "other:latest"
    assert estimate_calls == [("example:latest", 16384), ("other:latest", 16384)]


def test_recommend_human_output_is_explicit_about_quality_limits(
    monkeypatch: object,
    sample_profile: SystemProfile,
    sample_runtime_discovery: RuntimeDiscovery,
    sample_estimate: ModelMemoryEstimate,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "load_benchmark_history", lambda: (_stored("run"),))  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "estimate_ollama_model", lambda *_args, **_kwargs: sample_estimate
    )

    result = runner.invoke(cli.app, ["recommend", "--use-case", "coding"])

    assert result.exit_code == 0
    assert "Model recommendation" in result.stdout
    assert "Scoring components" in result.stdout
    assert "answer quality" in result.stdout


def test_recommend_without_current_benchmarks_is_structured(
    monkeypatch: object,
    sample_profile: SystemProfile,
    sample_runtime_discovery: RuntimeDiscovery,
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "collect_system_profile", lambda: sample_profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "load_benchmark_history", lambda: ())  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["recommend", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "no_eligible_models"


def test_history_json_output(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "load_benchmark_history", lambda **_kwargs: (_stored("aaaa1111"),)
    )

    result = runner.invoke(cli.app, ["history", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["runs"][0]["run_id"] == "aaaa1111"
    assert decoded["runs"][0]["prompt_id"] == "benchmark_generation_v2"


def test_history_empty_is_not_an_error(monkeypatch: object) -> None:
    monkeypatch.setattr(cli, "load_benchmark_history", lambda **_kwargs: ())  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["history"])

    assert result.exit_code == 0
    assert "No stored benchmark runs yet" in result.stdout


def test_history_warns_when_workload_versions_are_mixed(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "load_benchmark_history",
        lambda **_kwargs: (
            _stored("aaaa1111", "benchmark_generation_v1"),
            _stored("bbbb2222", "benchmark_generation_v2"),
        ),
    )

    result = runner.invoke(cli.app, ["history"])

    assert result.exit_code == 0
    assert "multiple workload versions" in result.stdout


def test_compare_json_output(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "load_benchmark_history",
        lambda **_kwargs: (_stored("aaaa1111"), _stored("bbbb2222")),
    )

    result = runner.invoke(cli.app, ["compare", "aaaa", "bbbb", "--json"])

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["baseline"]["run_id"] == "aaaa1111"
    assert decoded["candidate"]["run_id"] == "bbbb2222"
    assert decoded["prompt_id"] == "benchmark_generation_v2"


def test_compare_rejects_mixed_workload_versions(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "load_benchmark_history",
        lambda **_kwargs: (
            _stored("aaaa1111", "benchmark_generation_v1"),
            _stored("bbbb2222", "benchmark_generation_v2"),
        ),
    )

    result = runner.invoke(cli.app, ["compare", "aaaa", "bbbb", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "incomparable_workloads"


def _coexistence_result() -> CoexistenceResult:
    model = ModelCoexistenceResult(
        role="reasoning",
        model="example:latest",
        pre_run_unload_succeeded=True,
        load_time_seconds=1.0,
        ttft_seconds=2.0,
        prompt_tokens=20,
        prompt_tokens_per_second=40.0,
        generation_tokens=64,
        generation_seconds=4.0,
        generation_tokens_per_second=16.0,
        total_duration_seconds=5.5,
        done_reason="length",
        baseline_run_id="aaaa1111",
        generation_rate_change_percent=-20.0,
        ttft_change_percent=-10.0,
        load_time_change_percent=5.0,
        warnings=(),
    )
    return CoexistenceResult(
        started_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        runtime="ollama",
        runtime_version="1.2.3",
        concurrency=2,
        models=(model, dataclasses.replace(model, role="secondary", model="other:latest")),
        wall_time_seconds=8.0,
        baseline_system_ram_used_bytes=8 * 1024**3,
        peak_system_ram_used_bytes=12 * 1024**3,
        peak_system_ram_increase_bytes=4 * 1024**3,
        peak_system_cpu_percent=95.0,
        warnings=("System-wide metrics.",),
    )


def test_coexist_json_resolves_optional_solo_baseline(
    monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "load_benchmark_history", lambda: (_stored("aaaa1111"),)
    )
    captured: list[object] = []

    def run(requests: object, _runtime: object) -> CoexistenceResult:
        captured.append(requests)
        return _coexistence_result()

    monkeypatch.setattr(cli, "run_coexistence_benchmark", run)  # type: ignore[attr-defined]

    result = runner.invoke(
        cli.app,
        [
            "coexist",
            "--model",
            "reasoning=example",
            "--model",
            "secondary=other",
            "--baseline",
            "reasoning=aaaa",
            "--json",
        ],
    )

    assert result.exit_code == 0
    decoded = json.loads(result.stdout)
    assert decoded["concurrency"] == 2
    requests = captured[0]
    assert isinstance(requests, tuple)
    assert requests[0].baseline is not None
    assert requests[0].baseline.run_id == "aaaa1111"
    assert requests[1].baseline is None


def test_coexist_human_output(
    monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery
) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "run_coexistence_benchmark", lambda *_args: _coexistence_result()
    )

    result = runner.invoke(
        cli.app,
        ["coexist", "--model", "reasoning=example", "--model", "secondary=other"],
    )

    assert result.exit_code == 0
    assert "Measured coexistence" in result.stdout
    assert "-20.0%" in result.stdout


def test_coexist_requires_two_models_before_discovery(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "discover_runtimes",
        lambda: pytest.fail("invalid input should fail before discovery"),
    )

    result = runner.invoke(
        cli.app, ["coexist", "--model", "reasoning=example", "--json"]
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "insufficient_workloads"


def test_coexist_rejects_baseline_for_unknown_role() -> None:
    result = runner.invoke(
        cli.app,
        [
            "coexist",
            "--model",
            "reasoning=example",
            "--model",
            "secondary=other",
            "--baseline",
            "missing=aaaa",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "unknown_baseline_role"


def test_bench_json_error(monkeypatch: object, sample_runtime_discovery: RuntimeDiscovery) -> None:
    monkeypatch.setattr(cli, "discover_runtimes", lambda: sample_runtime_discovery)  # type: ignore[attr-defined]

    def fail(*_args: object, **_kwargs: object) -> None:
        raise BenchmarkError("incomplete_run", "Incomplete.")

    monkeypatch.setattr(cli, "run_quick_benchmark", fail)  # type: ignore[attr-defined]

    result = runner.invoke(cli.app, ["bench", "example", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "incomplete_run"
