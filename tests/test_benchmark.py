from __future__ import annotations

import gc
import json
import threading
import warnings
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from localbench.benchmark import sampler as sampler_module
from localbench.benchmark.history import compare_results, find_result
from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.runner import run_quick_benchmark
from localbench.benchmark.sampler import ResourceSample, SystemSampler
from localbench.benchmark.schemas import BenchmarkError, StoredResult
from localbench.benchmark.storage import (
    BenchmarkStorageError,
    load_benchmark_history,
    save_benchmark_result,
)
from localbench.models import RuntimeModelProfile, RuntimeProfile


class FakeSampler:
    def start(self) -> None:
        pass

    def stop(self) -> ResourceSample:
        return ResourceSample(8_000, 12_000, 87.5)


def test_system_sampler_tracks_peaks_and_stops_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_calls = 0
    background_sampled = threading.Event()

    def virtual_memory() -> SimpleNamespace:
        nonlocal memory_calls
        memory_calls += 1
        if memory_calls >= 2:
            background_sampled.set()
        return SimpleNamespace(used=memory_calls * 1_000)

    cpu_calls = 0

    def cpu_percent(*, interval: None) -> float:
        nonlocal cpu_calls
        cpu_calls += 1
        return 0.0 if cpu_calls == 1 else 72.5

    monkeypatch.setattr(sampler_module.psutil, "virtual_memory", virtual_memory)
    monkeypatch.setattr(sampler_module.psutil, "cpu_percent", cpu_percent)
    sampler = SystemSampler(interval_seconds=0.001)

    sampler.start()
    assert background_sampled.wait(timeout=1.0)
    sample = sampler.stop()

    assert sample.baseline_ram_used_bytes == 1_000
    assert sample.peak_ram_used_bytes >= 2_000
    assert sample.peak_cpu_percent == 72.5
    assert sampler._thread is not None
    assert not sampler._thread.is_alive()


def _runtime(running: bool = True) -> RuntimeProfile:
    model = RuntimeModelProfile(
        name="example:latest",
        size_bytes=1000,
        digest=None,
        modified_at=None,
        format="gguf",
        family="example",
        parameter_size="1B",
        quantization_level="Q4",
        source="test",
    )
    return RuntimeProfile("ollama", True, running, "1.2.3", (model,))


def test_quick_benchmark_calculates_metrics() -> None:
    chunks = iter(
        [
            {"response": "First", "done": False},
            {
                "response": "",
                "done": True,
                "done_reason": "length",
                "load_duration": 1_000_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 500_000_000,
                "eval_count": 64,
                "eval_duration": 4_000_000_000,
                "total_duration": 5_500_000_000,
            },
        ]
    )
    times = iter([10.0, 11.5, 15.6])
    unloaded: list[str] = []

    result = run_quick_benchmark(
        "example",
        _runtime(),
        stream=lambda _body, _timeout: chunks,
        unload=lambda model, _timeout: unloaded.append(model),
        sampler=FakeSampler(),  # type: ignore[arg-type]
        clock=lambda: next(times),
    )

    assert unloaded == ["example:latest"]
    assert result.pre_run_unload_succeeded is True
    assert result.ttft_seconds == 1.5
    assert result.prompt_tokens_per_second == 40.0
    assert result.generation_tokens_per_second == 16.0
    assert result.peak_system_ram_increase_bytes == 4000
    assert result.peak_system_cpu_percent == 87.5


def test_quick_benchmark_requests_raw_completion_mode() -> None:
    """v2 sends the prompt in Ollama's raw mode so instruction-tuned models don't treat
    the continuation prompt as a single chat turn and stop after one item (the v1 bug)."""
    captured: dict[str, object] = {}

    def capture_stream(body: object, _timeout: float) -> object:
        assert isinstance(body, dict)
        captured.update(body)
        return iter(
            [
                {"response": "X", "done": False},
                {
                    "response": "",
                    "done": True,
                    "done_reason": "length",
                    "load_duration": 1_000_000_000,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 64,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 2_100_000_000,
                },
            ]
        )

    times = iter([10.0, 10.5, 12.0])
    run_quick_benchmark(
        "example",
        _runtime(),
        stream=capture_stream,  # type: ignore[arg-type]
        unload=lambda _model, _timeout: None,
        sampler=FakeSampler(),  # type: ignore[arg-type]
        clock=lambda: next(times),
    )

    assert captured["raw"] is True


def test_quick_benchmark_reports_ambiguous_model_choices() -> None:
    models = (
        RuntimeModelProfile(
            name="example:3b",
            size_bytes=1000,
            digest=None,
            modified_at=None,
            format="gguf",
            family="example",
            parameter_size="3B",
            quantization_level="Q4",
            source="test",
        ),
        RuntimeModelProfile(
            name="example:8b",
            size_bytes=2000,
            digest=None,
            modified_at=None,
            format="gguf",
            family="example",
            parameter_size="8B",
            quantization_level="Q4",
            source="test",
        ),
    )
    runtime = RuntimeProfile("ollama", True, True, "1.2.3", models)

    with pytest.raises(BenchmarkError) as caught:
        run_quick_benchmark("example", runtime)

    assert caught.value.code == "ambiguous_model"
    assert "example:3b" in caught.value.message
    assert "example:8b" in caught.value.message


def test_quick_benchmark_requires_running_service() -> None:
    with pytest.raises(BenchmarkError) as caught:
        run_quick_benchmark("example", _runtime(running=False))

    assert caught.value.code == "runtime_unavailable"


def test_quick_benchmark_rejects_incomplete_stream() -> None:
    with pytest.raises(BenchmarkError) as caught:
        run_quick_benchmark(
            "example",
            _runtime(),
            stream=lambda _body, _timeout: iter([]),
            unload=lambda _model, _timeout: None,
            sampler=FakeSampler(),  # type: ignore[arg-type]
            clock=lambda: 1.0,
        )

    assert caught.value.code == "incomplete_run"


def _stored(
    run_id: str,
    *,
    prompt_id: str = "benchmark_generation_v2",
    model: str = "example:latest",
    generation_rate: float | None = 10.0,
    ttft: float | None = 2.0,
    load_time: float | None = 1.0,
    batch_id: str | None = None,
    batch_index: int = 1,
    batch_size: int = 1,
) -> StoredResult:
    signature = quick_measurement_signature("1.2.3", prompt_id=prompt_id)
    return StoredResult(
        run_id=run_id,
        started_at="2026-08-31T02:00:00+00:00",
        model=model,
        prompt_id=prompt_id,
        generation_tokens=64,
        generation_tokens_per_second=generation_rate,
        prompt_tokens_per_second=30.0,
        ttft_seconds=ttft,
        load_time_seconds=load_time,
        runtime="ollama",
        runtime_version="1.2.3",
        pre_run_unload_succeeded=True,
        measurement_signature=signature,
        batch_id=batch_id or run_id,
        batch_index=batch_index,
        batch_size=batch_size,
    )


def test_compare_refuses_mixed_workload_versions() -> None:
    """The v1 premature-stop bug produced results that look better than correct v2 runs,
    so comparing across workload versions must fail loudly rather than mislead."""
    v1 = _stored("aaaa1111", prompt_id="benchmark_generation_v1", generation_rate=14.06)
    v2 = _stored("bbbb2222", prompt_id="benchmark_generation_v2", generation_rate=8.67)

    with pytest.raises(BenchmarkError) as caught:
        compare_results(v1, v2)

    assert caught.value.code == "incomparable_workloads"
    assert "benchmark_generation_v1" in caught.value.message
    assert "benchmark_generation_v2" in caught.value.message


def test_compare_reports_direction_correctly() -> None:
    baseline = _stored("aaaa1111", generation_rate=10.0, ttft=4.0, load_time=2.0)
    candidate = _stored("bbbb2222", generation_rate=15.0, ttft=2.0, load_time=3.0)

    comparison = compare_results(baseline, candidate)

    # Faster generation is an improvement; lower TTFT is an improvement (sign inverted);
    # a longer load time is a regression.
    assert comparison.generation_rate_change_percent == pytest.approx(50.0)
    assert comparison.ttft_change_percent == pytest.approx(50.0)
    assert comparison.load_time_change_percent == pytest.approx(-50.0)


def test_compare_refuses_runtime_version_change() -> None:
    import dataclasses

    baseline = _stored("aaaa1111")
    candidate = _stored("bbbb2222")
    assert candidate.measurement_signature is not None
    candidate = dataclasses.replace(
        candidate,
        runtime_version="2.0.0",
        measurement_signature=dataclasses.replace(
            candidate.measurement_signature, runtime_version="2.0.0"
        ),
    )

    with pytest.raises(BenchmarkError) as caught:
        compare_results(baseline, candidate)

    assert caught.value.code == "incomparable_measurements"
    assert "runtime_version" in caught.value.message


def test_compare_refuses_unknown_signature() -> None:
    import dataclasses

    baseline = _stored("aaaa1111")
    candidate = dataclasses.replace(_stored("bbbb2222"), measurement_signature=None)

    with pytest.raises(BenchmarkError) as caught:
        compare_results(baseline, candidate)

    assert caught.value.code == "unknown_measurement_signature"


def test_compare_refuses_failed_cold_start() -> None:
    import dataclasses

    baseline = _stored("aaaa1111")
    candidate = dataclasses.replace(
        _stored("bbbb2222"), pre_run_unload_succeeded=False
    )

    with pytest.raises(BenchmarkError) as caught:
        compare_results(baseline, candidate)

    assert caught.value.code == "ineligible_cold_start"


def test_compare_refuses_incomplete_generation() -> None:
    import dataclasses

    baseline = _stored("aaaa1111")
    candidate = dataclasses.replace(_stored("bbbb2222"), generation_tokens=15)

    with pytest.raises(BenchmarkError) as caught:
        compare_results(baseline, candidate)

    assert caught.value.code == "incomplete_generation"


def test_compare_batches_uses_medians_and_retains_spread() -> None:
    from localbench.benchmark.history import compare_result_groups

    baseline = tuple(
        _stored(
            f"aaaa111{i}",
            generation_rate=rate,
            batch_id="batch-a",
            batch_index=i,
            batch_size=3,
        )
        for i, rate in enumerate((8.0, 10.0, 30.0), start=1)
    )
    candidate = tuple(
        _stored(
            f"bbbb222{i}",
            generation_rate=rate,
            batch_id="batch-b",
            batch_index=i,
            batch_size=3,
        )
        for i, rate in enumerate((12.0, 15.0, 18.0), start=1)
    )

    comparison = compare_result_groups(baseline, candidate)

    assert comparison.baseline_generation.median == 10.0
    assert comparison.baseline_generation.maximum == 30.0
    assert comparison.candidate_generation.median == 15.0
    assert comparison.generation_rate_change_percent == pytest.approx(50.0)


def test_compare_refuses_incomplete_batch() -> None:
    from localbench.benchmark.history import compare_result_groups

    incomplete = (
        _stored("aaaa1111", batch_id="batch-a", batch_index=1, batch_size=3),
        _stored("aaaa2222", batch_id="batch-a", batch_index=2, batch_size=3),
    )

    with pytest.raises(BenchmarkError) as caught:
        compare_result_groups(incomplete, (_stored("bbbb2222"),))

    assert caught.value.code == "incomplete_batch"


def test_compare_refuses_singleton_against_batch() -> None:
    from localbench.benchmark.history import compare_result_groups

    batch = tuple(
        _stored(
            f"bbbb222{i}", batch_id="batch-b", batch_index=i, batch_size=2
        )
        for i in (1, 2)
    )

    with pytest.raises(BenchmarkError) as caught:
        compare_result_groups((_stored("aaaa1111"),), batch)

    assert caught.value.code == "incomparable_sample_sizes"


def test_compare_handles_missing_metrics() -> None:
    baseline = _stored("aaaa1111", generation_rate=None)
    candidate = _stored("bbbb2222", generation_rate=12.0)

    comparison = compare_results(baseline, candidate)

    assert comparison.generation_rate_change_percent is None


def test_find_result_accepts_unique_prefix() -> None:
    history = (_stored("aaaa1111"), _stored("bbbb2222"))

    assert find_result(history, "aaaa").run_id == "aaaa1111"


def test_find_result_rejects_ambiguous_prefix() -> None:
    history = (_stored("aaaa1111"), _stored("aaaa2222"))

    with pytest.raises(BenchmarkError) as caught:
        find_result(history, "aaaa")

    assert caught.value.code == "ambiguous_run"


def test_find_result_reports_missing_run() -> None:
    with pytest.raises(BenchmarkError) as caught:
        find_result((_stored("aaaa1111"),), "zzzz")

    assert caught.value.code == "run_not_found"


def test_load_history_returns_empty_when_database_absent(tmp_path: Path) -> None:
    """Nothing benchmarked yet is a normal state, not a storage error."""
    assert load_benchmark_history(tmp_path / "missing.db") == ()


def test_save_then_load_round_trip(sample_benchmark: object, tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]

    stored = load_benchmark_history(database)

    assert len(stored) == 1
    assert stored[0].run_id == sample_benchmark.run_id  # type: ignore[attr-defined]
    assert stored[0].prompt_id == "benchmark_generation_v1"
    assert stored[0].generation_tokens_per_second == 16.0
    assert stored[0].batch_id == sample_benchmark.batch_id  # type: ignore[attr-defined]


def test_repeated_batch_round_trip_preserves_every_raw_run(
    sample_benchmark: object, tmp_path: Path
) -> None:
    import dataclasses

    database = tmp_path / "history.db"
    for index, rate in enumerate((8.0, 10.0, 30.0), start=1):
        result = dataclasses.replace(  # type: ignore[type-var]
            sample_benchmark,
            run_id=f"run-{index}",
            batch_id="batch-a",
            batch_index=index,
            batch_size=3,
            generation_tokens_per_second=rate,
        )
        save_benchmark_result(result, database)

    stored = load_benchmark_history(database)

    assert len(stored) == 3
    assert {result.batch_id for result in stored} == {"batch-a"}
    assert {result.batch_index for result in stored} == {1, 2, 3}
    assert sorted(result.generation_tokens_per_second for result in stored) == [
        8.0,
        10.0,
        30.0,
    ]


def test_load_history_filters_by_model_and_limit(
    sample_benchmark: object, tmp_path: Path
) -> None:
    import dataclasses

    database = tmp_path / "history.db"
    save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]
    other = dataclasses.replace(
        sample_benchmark,  # type: ignore[type-var]
        run_id="00000000-0000-0000-0000-000000000002",
        model="other:latest",
    )
    save_benchmark_result(other, database)

    assert len(load_benchmark_history(database)) == 2
    assert len(load_benchmark_history(database, model="other:latest")) == 1
    assert len(load_benchmark_history(database, limit=1)) == 1


def test_load_history_skips_unreadable_rows(sample_benchmark: object, tmp_path: Path) -> None:
    """A corrupt or pre-schema row should be skipped, not crash the whole listing."""
    import sqlite3

    database = tmp_path / "history.db"
    save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """INSERT INTO benchmark_results
            (run_id, started_at, model, profile, result_json)
            VALUES (?, ?, ?, ?, ?)""",
            ("bad", "2026-08-31T00:00:00+00:00", "x", "quick", "not valid json"),
        )

    stored = load_benchmark_history(database)

    assert len(stored) == 1
    assert stored[0].run_id == sample_benchmark.run_id  # type: ignore[attr-defined]


def test_save_benchmark_result_does_not_leak_connection(
    sample_benchmark: object, tmp_path: Path
) -> None:
    """Regression guard for the sqlite3 context-manager gotcha: `with sqlite3.connect()`
    commits the transaction but does not close the connection."""
    database = tmp_path / "history.db"

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]
        load_benchmark_history(database)
        gc.collect()


def test_save_benchmark_result(sample_benchmark: object, tmp_path: Path) -> None:
    database = tmp_path / "history.db"

    save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]

    import sqlite3

    with closing(sqlite3.connect(database)) as connection, connection:
        stored = connection.execute("SELECT result_json FROM benchmark_results").fetchone()[0]
    assert json.loads(stored)["run_id"] == sample_benchmark.run_id  # type: ignore[attr-defined]


def test_new_database_records_explicit_schema_version(
    sample_benchmark: object, tmp_path: Path
) -> None:
    import sqlite3

    database = tmp_path / "history.db"
    save_benchmark_result(sample_benchmark, database)  # type: ignore[arg-type]

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(benchmark_results)")
        )

    assert version == 3
    assert "measurement_signature_json" in columns
    assert columns[-3:] == ("batch_id", "batch_index", "batch_size")


def test_save_migrates_legacy_database_without_rewriting_results(
    sample_benchmark: object, tmp_path: Path
) -> None:
    import dataclasses
    import sqlite3

    database = tmp_path / "history.db"
    original_payload = sample_benchmark.to_dict()  # type: ignore[attr-defined]
    original_payload.pop("measurement_signature")
    original_json = json.dumps(original_payload, sort_keys=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE benchmark_results (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                model TEXT NOT NULL,
                profile TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?)",
            (
                sample_benchmark.run_id,  # type: ignore[attr-defined]
                sample_benchmark.started_at.isoformat(),  # type: ignore[attr-defined]
                sample_benchmark.model,  # type: ignore[attr-defined]
                sample_benchmark.profile,  # type: ignore[attr-defined]
                original_json,
            ),
        )
    second = dataclasses.replace(  # type: ignore[type-var]
        sample_benchmark,
        run_id="00000000-0000-0000-0000-000000000002",
    )

    save_benchmark_result(second, database)

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        migrated_json, signature_json = connection.execute(
            """SELECT result_json, measurement_signature_json
            FROM benchmark_results WHERE run_id = ?""",
            (sample_benchmark.run_id,),  # type: ignore[attr-defined]
        ).fetchone()
    assert version == 3
    assert migrated_json == original_json
    assert json.loads(signature_json)["prompt_id"] == "benchmark_generation_v1"
    assert len(load_benchmark_history(database)) == 2


def test_load_refuses_newer_database_schema(tmp_path: Path) -> None:
    import sqlite3

    database = tmp_path / "future.db"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(BenchmarkStorageError, match="newer than"):
        load_benchmark_history(database)
