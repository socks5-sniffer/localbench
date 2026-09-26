"""Local SQLite persistence for benchmark history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, suppress
from pathlib import Path

from platformdirs import user_data_path

from localbench.benchmark.profiles import quick_measurement_signature
from localbench.benchmark.schemas import (
    BenchmarkResult,
    MeasurementSignature,
    StoredResult,
)

CURRENT_DATABASE_SCHEMA_VERSION = 3
_LEGACY_COLUMNS = ("run_id", "started_at", "model", "profile", "result_json")
_LEGACY_SHAPE = (
    ("run_id", "TEXT", False, 1),
    ("started_at", "TEXT", True, 0),
    ("model", "TEXT", True, 0),
    ("profile", "TEXT", True, 0),
    ("result_json", "TEXT", True, 0),
)
_V2_SHAPE = (*_LEGACY_SHAPE, ("measurement_signature_json", "TEXT", False, 0))
_CURRENT_SHAPE = (
    *_V2_SHAPE,
    ("batch_id", "TEXT", False, 0),
    ("batch_index", "INTEGER", False, 0),
    ("batch_size", "INTEGER", False, 0),
)


class BenchmarkStorageError(RuntimeError):
    """Benchmark history could not be read or written."""


def default_database_path() -> Path:
    return user_data_path("LocalBench", appauthor=False) / "localbench.db"


def _measurement_signature(payload: object) -> MeasurementSignature | None:
    if not isinstance(payload, dict):
        return None
    string_fields = (
        "benchmark_method_id",
        "profile",
        "prompt_id",
        "runtime",
        "resource_sampler_id",
    )
    integer_fields = ("context_length", "generation_token_target", "seed", "keep_alive_seconds")
    boolean_fields = ("raw", "stream", "think", "cold_start_requested")
    if payload.get("contract_version") != "1":
        return None
    if not all(isinstance(payload.get(field), str) for field in string_fields):
        return None
    if not all(
        isinstance(payload.get(field), int) and not isinstance(payload.get(field), bool)
        for field in integer_fields
    ):
        return None
    if not all(isinstance(payload.get(field), bool) for field in boolean_fields):
        return None
    temperature = payload.get("temperature")
    sample_interval = payload.get("resource_sample_interval_seconds")
    runtime_version = payload.get("runtime_version")
    if (
        not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or not isinstance(sample_interval, int | float)
        or isinstance(sample_interval, bool)
        or not (runtime_version is None or isinstance(runtime_version, str))
    ):
        return None
    return MeasurementSignature(
        contract_version="1",
        benchmark_method_id=payload["benchmark_method_id"],
        profile=payload["profile"],
        prompt_id=payload["prompt_id"],
        context_length=payload["context_length"],
        generation_token_target=payload["generation_token_target"],
        raw=payload["raw"],
        stream=payload["stream"],
        think=payload["think"],
        temperature=float(temperature),
        seed=payload["seed"],
        keep_alive_seconds=payload["keep_alive_seconds"],
        cold_start_requested=payload["cold_start_requested"],
        runtime=payload["runtime"],
        runtime_version=runtime_version,
        resource_sampler_id=payload["resource_sampler_id"],
        resource_sample_interval_seconds=float(sample_interval),
    )


def _stored_result(
    row: str,
    signature_json: str | None = None,
    batch_id: str | None = None,
    batch_index: int = 1,
    batch_size: int = 1,
) -> StoredResult | None:
    """Decode one stored row, skipping anything unreadable rather than failing the query."""
    try:
        payload = json.loads(row)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    model = payload.get("model")
    started_at = payload.get("started_at")
    prompt_id = payload.get("prompt_id")
    if not (
        isinstance(run_id, str)
        and isinstance(model, str)
        and isinstance(started_at, str)
        and isinstance(prompt_id, str)
    ):
        return None
    generation_rate = payload.get("generation_tokens_per_second")
    prompt_rate = payload.get("prompt_tokens_per_second")
    ttft = payload.get("ttft_seconds")
    load_time = payload.get("load_time_seconds")
    generation_tokens = payload.get("generation_tokens")
    persisted_signature: MeasurementSignature | None = None
    if signature_json is not None:
        with suppress(TypeError, ValueError):
            persisted_signature = _measurement_signature(json.loads(signature_json))
    signature = _measurement_signature(payload.get("measurement_signature"))
    runtime = payload.get("runtime")
    runtime_version = payload.get("runtime_version")
    unload_succeeded = payload.get("pre_run_unload_succeeded")
    if signature is None:
        signature = quick_measurement_signature(
            runtime_version if isinstance(runtime_version, str) else None,
            prompt_id=prompt_id,
        )
    if persisted_signature is not None and persisted_signature != signature:
        signature = None
    profile = payload.get("profile")
    context_length = payload.get("context_length")
    generation_target = payload.get("generation_token_target")
    cold_start = payload.get("cold_start_requested")
    if signature is not None and (
        signature.prompt_id != prompt_id
        or signature.runtime != runtime
        or signature.runtime_version != runtime_version
        or signature.profile != profile
        or signature.context_length != context_length
        or signature.generation_token_target != generation_target
        or signature.cold_start_requested != cold_start
    ):
        signature = None
    return StoredResult(
        run_id=run_id,
        started_at=started_at,
        model=model,
        prompt_id=prompt_id,
        generation_tokens=generation_tokens if isinstance(generation_tokens, int) else None,
        generation_tokens_per_second=(
            float(generation_rate) if isinstance(generation_rate, int | float) else None
        ),
        prompt_tokens_per_second=(
            float(prompt_rate) if isinstance(prompt_rate, int | float) else None
        ),
        ttft_seconds=float(ttft) if isinstance(ttft, int | float) else None,
        load_time_seconds=float(load_time) if isinstance(load_time, int | float) else None,
        runtime=runtime if isinstance(runtime, str) else None,
        runtime_version=runtime_version if isinstance(runtime_version, str) else None,
        pre_run_unload_succeeded=(
            unload_succeeded if isinstance(unload_succeeded, bool) else None
        ),
        measurement_signature=signature,
        batch_id=batch_id or run_id,
        batch_index=batch_index,
        batch_size=batch_size,
    )


def _table_shape(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, bool, int], ...]:
    rows = connection.execute("PRAGMA table_info(benchmark_results)").fetchall()
    return tuple((str(row[1]), str(row[2]).upper(), bool(row[3]), int(row[5])) for row in rows)


def _database_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _validate_database(connection: sqlite3.Connection) -> tuple[str, ...]:
    version = _database_version(connection)
    if version > CURRENT_DATABASE_SCHEMA_VERSION:
        raise BenchmarkStorageError(
            f"Benchmark database schema {version} is newer than this LocalBench supports"
        )
    shape = _table_shape(connection)
    columns = tuple(item[0] for item in shape)
    if shape and shape not in (_LEGACY_SHAPE, _V2_SHAPE, _CURRENT_SHAPE):
        raise BenchmarkStorageError("Benchmark database has an unrecognized schema")
    if not shape and version != 0:
        raise BenchmarkStorageError("Benchmark database version has no matching table")
    if version == 0 and shape and shape != _LEGACY_SHAPE:
        raise BenchmarkStorageError("Unversioned benchmark database has an unknown schema")
    if version == 1 and shape != _LEGACY_SHAPE:
        raise BenchmarkStorageError("Benchmark database v1 has an unexpected schema")
    if version == 2 and shape != _V2_SHAPE:
        raise BenchmarkStorageError("Benchmark database v2 has an unexpected schema")
    if version == 3 and shape != _CURRENT_SHAPE:
        raise BenchmarkStorageError("Benchmark database v3 has an unexpected schema")
    return columns


def _initialize_or_migrate(connection: sqlite3.Connection) -> None:
    columns = _validate_database(connection)
    version = _database_version(connection)
    if not columns:
        connection.execute(
            """
            CREATE TABLE benchmark_results (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                model TEXT NOT NULL,
                profile TEXT NOT NULL,
                result_json TEXT NOT NULL,
                measurement_signature_json TEXT,
                batch_id TEXT,
                batch_index INTEGER,
                batch_size INTEGER
            )
            """
        )
        connection.execute(f"PRAGMA user_version = {CURRENT_DATABASE_SCHEMA_VERSION}")
        return
    if version == 0:
        if columns != _LEGACY_COLUMNS:
            raise BenchmarkStorageError("Unversioned benchmark database has an unknown schema")
        connection.execute("PRAGMA user_version = 1")
        version = 1
    if version == 1:
        connection.execute(
            "ALTER TABLE benchmark_results ADD COLUMN measurement_signature_json TEXT"
        )
        rows = connection.execute(
            "SELECT run_id, result_json FROM benchmark_results"
        ).fetchall()
        for run_id, result_json in rows:
            stored = _stored_result(result_json)
            signature_json = (
                json.dumps(stored.measurement_signature.to_dict(), sort_keys=True)
                if stored is not None and stored.measurement_signature is not None
                else None
            )
            connection.execute(
                "UPDATE benchmark_results SET measurement_signature_json = ? WHERE run_id = ?",
                (signature_json, run_id),
            )
        connection.execute("PRAGMA user_version = 2")
        version = 2
    if version == 2:
        connection.execute("ALTER TABLE benchmark_results ADD COLUMN batch_id TEXT")
        connection.execute("ALTER TABLE benchmark_results ADD COLUMN batch_index INTEGER")
        connection.execute("ALTER TABLE benchmark_results ADD COLUMN batch_size INTEGER")
        connection.execute(
            """UPDATE benchmark_results
            SET batch_id = run_id, batch_index = 1, batch_size = 1"""
        )
        connection.execute("PRAGMA user_version = 3")


def load_benchmark_history(
    database_path: Path | None = None,
    *,
    model: str | None = None,
    limit: int | None = None,
) -> tuple[StoredResult, ...]:
    """Return stored results, newest first, optionally filtered to one model.

    A missing database is an empty history, not an error — nothing has been benchmarked
    yet is a normal state, distinct from a history that cannot be read.
    """
    path = database_path or default_database_path()
    if not path.exists():
        return ()
    try:
        with closing(sqlite3.connect(path)) as connection:
            columns = _validate_database(connection)
            if not columns:
                return ()
            signature_column = (
                "measurement_signature_json" if "measurement_signature_json" in columns else "NULL"
            )
            batch_columns = (
                "batch_id, batch_index, batch_size"
                if "batch_id" in columns
                else "NULL, 1, 1"
            )
            query = (
                f"SELECT result_json, {signature_column}, {batch_columns} "
                "FROM benchmark_results"
            )
            parameters: list[object] = []
            if model is not None:
                query += " WHERE model = ?"
                parameters.append(model)
            query += " ORDER BY started_at DESC"
            if limit is not None:
                query += " LIMIT ?"
                parameters.append(limit)
            rows = connection.execute(query, parameters).fetchall()
    except (OSError, sqlite3.Error) as error:
        raise BenchmarkStorageError("Benchmark history could not be read") from error
    decoded = (_stored_result(row[0], row[1], row[2], row[3], row[4]) for row in rows)
    return tuple(result for result in decoded if result is not None)


def save_benchmark_result(
    result: BenchmarkResult, database_path: Path | None = None
) -> Path:
    path = database_path or default_database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # `closing` is required as well as `with`: sqlite3's own context manager commits
        # or rolls back the transaction but never closes the connection.
        with closing(sqlite3.connect(path)) as connection, connection:
            _initialize_or_migrate(connection)
            connection.execute(
                """
                INSERT INTO benchmark_results (
                    run_id, started_at, model, profile, result_json,
                    measurement_signature_json, batch_id, batch_index, batch_size
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.started_at.isoformat(),
                    result.model,
                    result.profile,
                    json.dumps(result.to_dict(), sort_keys=True),
                    json.dumps(result.measurement_signature.to_dict(), sort_keys=True),
                    result.batch_id,
                    result.batch_index,
                    result.batch_size,
                ),
            )
    except (OSError, sqlite3.Error) as error:
        raise BenchmarkStorageError("Benchmark history could not be written") from error
    return path
