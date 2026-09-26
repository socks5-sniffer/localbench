"""LocalBench command-line interface."""

from __future__ import annotations

import dataclasses
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console

from localbench.benchmark import (
    BenchmarkError,
    CoexistenceError,
    CoexistenceRequest,
    compare_result_groups,
    find_result,
    load_benchmark_history,
    results_for_batch,
    run_coexistence_benchmark,
    run_quick_benchmark,
)
from localbench.benchmark.storage import BenchmarkStorageError, save_benchmark_result
from localbench.context import (
    DEFAULT_CONTEXT_LEVELS,
    ContextProfileError,
    profile_context_capacity,
)
from localbench.estimation import EstimateError, KVCacheType, estimate_ollama_model
from localbench.hardware import collect_system_profile
from localbench.hardware.memory import collect_memory
from localbench.recommendation import (
    RecommendationError,
    RecommendationUseCase,
    eligible_benchmark_batches,
    make_candidate,
    recommend_models,
)
from localbench.render import (
    render_benchmark,
    render_benchmark_batch,
    render_benchmark_batch_json,
    render_benchmark_error_json,
    render_benchmark_json,
    render_coexistence,
    render_coexistence_error_json,
    render_coexistence_json,
    render_comparison,
    render_comparison_json,
    render_context,
    render_context_error_json,
    render_context_json,
    render_estimate,
    render_estimate_error_json,
    render_estimate_json,
    render_history,
    render_history_json,
    render_human,
    render_json,
    render_recommendation,
    render_recommendation_error_json,
    render_recommendation_json,
    render_runtime_json,
    render_runtimes,
    render_stack,
    render_stack_error_json,
    render_stack_json,
)
from localbench.runtimes import discover_runtimes
from localbench.stack import StackPlanError, StackWorkloadRequest, plan_stack

app = typer.Typer(help="Profile this computer for local AI workloads.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Profile this computer for local AI workloads."""


@app.command()
def profile(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the versioned machine-readable JSON profile."),
    ] = False,
) -> None:
    """Collect and display a snapshot of this computer's hardware resources."""
    system_profile = collect_system_profile()
    if json_output:
        typer.echo(render_json(system_profile))
    else:
        render_human(system_profile, Console())


@app.command()
def runtimes(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit versioned machine-readable runtime discovery."),
    ] = False,
) -> None:
    """Detect supported local inference runtimes and installed models."""
    discovery = discover_runtimes()
    if json_output:
        typer.echo(render_runtime_json(discovery))
    else:
        render_runtimes(discovery, Console())


@app.command()
def estimate(
    model: Annotated[str, typer.Argument(help="Installed Ollama model name or tag.")],
    context: Annotated[
        int | None,
        typer.Option("--context", min=1, help="Context length in tokens; defaults up to 8K."),
    ] = None,
    kv_cache_type: Annotated[
        KVCacheType,
        typer.Option("--kv-cache-type", help="Assumed Ollama KV-cache precision."),
    ] = KVCacheType.F16,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable estimate."),
    ] = False,
) -> None:
    """Estimate system-memory requirements for an installed Ollama model."""
    discovery = discover_runtimes()
    ollama = next((runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None)
    try:
        if ollama is None:
            raise EstimateError("runtime_unavailable", "Ollama discovery is unavailable.")
        system_profile = collect_system_profile()
        estimate_result = estimate_ollama_model(
            model,
            ollama,
            system_profile.memory.available_bytes,
            context_length=context,
            kv_cache_type=kv_cache_type,
            gpus=system_profile.gpus,
        )
    except EstimateError as error:
        if json_output:
            typer.echo(render_estimate_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Estimate unavailable:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_estimate_json(estimate_result))
    else:
        render_estimate(estimate_result, Console())


def _parse_context_levels(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",")]
    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError as error:
            raise ContextProfileError(
                "invalid_levels",
                f"Invalid --levels value {raw!r}; expected comma-separated integers.",
            ) from error
    return tuple(parsed)


@app.command()
def context(
    model: Annotated[str, typer.Argument(help="Installed Ollama model name or tag.")],
    levels: Annotated[
        str | None,
        typer.Option(
            "--levels",
            help="Comma-separated context lengths in tokens; defaults to "
            + ",".join(str(level) for level in DEFAULT_CONTEXT_LEVELS) + ".",
        ),
    ] = None,
    kv_cache_type: Annotated[
        KVCacheType,
        typer.Option("--kv-cache-type", help="Assumed Ollama KV-cache precision."),
    ] = KVCacheType.F16,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable context-capacity profile."),
    ] = False,
) -> None:
    """Profile how an installed Ollama model's memory footprint grows across context lengths."""
    discovery = discover_runtimes()
    ollama = next((runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None)
    try:
        if ollama is None:
            raise EstimateError("runtime_unavailable", "Ollama discovery is unavailable.")
        context_levels = _parse_context_levels(levels) if levels is not None else None
        system_profile = collect_system_profile()
        profile_result = profile_context_capacity(
            model,
            ollama,
            system_profile.memory.available_bytes,
            context_levels=context_levels or DEFAULT_CONTEXT_LEVELS,
            kv_cache_type=kv_cache_type,
            gpus=system_profile.gpus,
        )
    except (EstimateError, ContextProfileError) as error:
        if json_output:
            typer.echo(render_context_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Context profile unavailable:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_context_json(profile_result))
    else:
        render_context(profile_result, Console())


def _parse_stack_models(models: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not models:
        raise StackPlanError(
            "empty_stack",
            "Supply at least one --model ROLE=MODEL workload.",
        )
    parsed: list[tuple[str, str]] = []
    for value in models:
        role, separator, model = value.partition("=")
        role = role.strip()
        model = model.strip()
        if separator != "=" or not role or not model:
            raise StackPlanError(
                "invalid_workload_spec",
                f"Invalid workload {value!r}; expected --model ROLE=MODEL.",
            )
        parsed.append((role, model))
    return tuple(parsed)


def _parse_baselines(baselines: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in baselines or []:
        role, separator, run_reference = value.partition("=")
        role = role.strip()
        run_reference = run_reference.strip()
        if separator != "=" or not role or not run_reference:
            raise CoexistenceError(
                "invalid_baseline_spec",
                f"Invalid baseline {value!r}; expected --baseline ROLE=RUN_ID.",
            )
        role_key = role.casefold()
        if role_key in parsed:
            raise CoexistenceError(
                "duplicate_baseline", f"Role {role!r} has more than one baseline."
            )
        parsed[role_key] = run_reference
    return parsed


@app.command()
def stack(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Workload as ROLE=MODEL; repeat for each stack role."),
    ] = None,
    context: Annotated[
        int | None,
        typer.Option("--context", min=1, help="Context length applied to every model."),
    ] = None,
    kv_cache_type: Annotated[
        KVCacheType,
        typer.Option("--kv-cache-type", help="KV-cache precision applied to every model."),
    ] = KVCacheType.F16,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable stack plan."),
    ] = False,
) -> None:
    """Plan simultaneous memory for multiple installed Ollama models."""
    try:
        requested = _parse_stack_models(models)
        discovery = discover_runtimes()
        ollama = next(
            (runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None
        )
        if ollama is None:
            raise EstimateError("runtime_unavailable", "Ollama discovery is unavailable.")
        available_memory_bytes = collect_memory().available_bytes
        workloads = tuple(
            StackWorkloadRequest(
                role=role,
                estimate=estimate_ollama_model(
                    model,
                    ollama,
                    available_memory_bytes,
                    context_length=context,
                    kv_cache_type=kv_cache_type,
                ),
            )
            for role, model in requested
        )
        stack_plan = plan_stack(workloads, available_memory_bytes)
    except (EstimateError, StackPlanError) as error:
        if json_output:
            typer.echo(render_stack_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Stack plan unavailable:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_stack_json(stack_plan))
    else:
        render_stack(stack_plan, Console())


@app.command()
def coexist(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Concurrent workload as ROLE=MODEL; repeat per role."),
    ] = None,
    baselines: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            help="Optional solo baseline as ROLE=RUN_ID; repeat per matching role.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable coexistence result."),
    ] = False,
) -> None:
    """Measure two or more Ollama models generating concurrently."""
    try:
        requested = _parse_stack_models(models)
        if len(requested) < 2:
            raise CoexistenceError(
                "insufficient_workloads",
                "A coexistence benchmark requires at least two --model roles.",
            )
        baseline_references = _parse_baselines(baselines)
        requested_roles = {role.casefold() for role, _model in requested}
        unknown_roles = set(baseline_references) - requested_roles
        if unknown_roles:
            raise CoexistenceError(
                "unknown_baseline_role",
                "Baseline roles must also appear in --model: "
                + ", ".join(sorted(unknown_roles)),
            )
        history = load_benchmark_history() if baseline_references else ()
        coexistence_requests = tuple(
            CoexistenceRequest(
                role=role,
                model=model,
                baseline=(
                    find_result(history, baseline_references[role.casefold()])
                    if role.casefold() in baseline_references
                    else None
                ),
            )
            for role, model in requested
        )
        discovery = discover_runtimes()
        ollama = next(
            (runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None
        )
        if ollama is None:
            raise CoexistenceError(
                "runtime_unavailable", "Ollama discovery is unavailable."
            )
        coexistence_result = run_coexistence_benchmark(coexistence_requests, ollama)
    except BenchmarkStorageError as error:
        coexistence_error = CoexistenceError("storage_error", str(error))
        if json_output:
            typer.echo(render_coexistence_error_json(coexistence_error))
        else:
            Console(stderr=True).print(
                f"[red]Coexistence benchmark unavailable:[/red] {coexistence_error.message}"
            )
        raise typer.Exit(code=2) from error
    except (BenchmarkError, CoexistenceError, StackPlanError) as error:
        if json_output:
            typer.echo(render_coexistence_error_json(error))
        else:
            Console(stderr=True).print(
                f"[red]Coexistence benchmark unavailable:[/red] {error.message}"
            )
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_coexistence_json(coexistence_result))
    else:
        render_coexistence(coexistence_result, Console())


@app.command()
def bench(
    model: Annotated[str, typer.Argument(help="Installed Ollama model name or tag.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable benchmark result."),
    ] = False,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Persist the result in local benchmark history."),
    ] = True,
    runs: Annotated[
        int,
        typer.Option("--runs", min=1, max=50, help="Repeat the workload in one batch."),
    ] = 1,
) -> None:
    """Run the deterministic quick benchmark against an installed Ollama model."""
    discovery = discover_runtimes()
    ollama = next((runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None)
    results = []
    batch_id = str(uuid4())
    try:
        if ollama is None:
            raise BenchmarkError("runtime_unavailable", "Ollama discovery is unavailable.")
        for index in range(1, runs + 1):
            result = run_quick_benchmark(
                model,
                ollama,
                batch_id=batch_id,
                batch_index=index,
                batch_size=runs,
            )
            if save:
                try:
                    save_benchmark_result(result)
                except BenchmarkStorageError:
                    result = dataclasses.replace(
                        result,
                        warnings=(
                            *result.warnings,
                            "The benchmark completed, but its result could not be saved.",
                        ),
                    )
            results.append(result)
    except BenchmarkError as error:
        if json_output:
            typer.echo(render_benchmark_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Benchmark failed:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    completed = tuple(results)
    if json_output and runs > 1:
        typer.echo(render_benchmark_batch_json(completed))
    elif json_output:
        typer.echo(render_benchmark_json(completed[0]))
    elif runs > 1:
        render_benchmark_batch(completed, Console())
    else:
        render_benchmark(completed[0], Console())


@app.command()
def recommend(
    use_case: Annotated[
        RecommendationUseCase,
        typer.Option("--use-case", help="Performance policy for the intended workload."),
    ] = RecommendationUseCase.GENERAL_CHAT,
    context: Annotated[
        int,
        typer.Option("--context", min=1, help="Context length used for every memory estimate."),
    ] = 8192,
    kv_cache_type: Annotated[
        KVCacheType,
        typer.Option("--kv-cache-type", help="KV-cache precision used for every estimate."),
    ] = KVCacheType.F16,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable recommendation."),
    ] = False,
) -> None:
    """Rank benchmarked installed models using visible performance and memory weights."""
    try:
        discovery = discover_runtimes()
        ollama = next(
            (runtime for runtime in discovery.runtimes if runtime.name == "ollama"), None
        )
        if ollama is None or not ollama.service_running:
            raise RecommendationError(
                "runtime_unavailable", "Ollama's local service must be running."
            )
        if ollama.version is None:
            raise RecommendationError(
                "unknown_runtime_version",
                "Ollama's version is required to select comparable benchmark evidence.",
            )
        history_results = load_benchmark_history()
        batches = eligible_benchmark_batches(history_results, ollama.version)
        system_profile = collect_system_profile()
        candidates = []
        excluded: list[str] = []
        for installed in ollama.models:
            batch = batches.get(installed.name.casefold())
            if batch is None:
                excluded.append(f"{installed.name} — no complete current benchmark")
                continue
            try:
                estimate_result = estimate_ollama_model(
                    installed.name,
                    ollama,
                    system_profile.memory.available_bytes,
                    context_length=context,
                    kv_cache_type=kv_cache_type,
                    gpus=system_profile.gpus,
                )
            except EstimateError as error:
                excluded.append(f"{installed.name} — estimate unavailable ({error.code})")
                continue
            candidates.append(make_candidate(batch, estimate_result))
        report = recommend_models(
            candidates,
            use_case,
            excluded_models=excluded,
        )
    except BenchmarkStorageError as error:
        recommendation_error = RecommendationError("storage_error", str(error))
        if json_output:
            typer.echo(render_recommendation_error_json(recommendation_error))
        else:
            Console(stderr=True).print(
                f"[red]Recommendation unavailable:[/red] {recommendation_error.message}"
            )
        raise typer.Exit(code=2) from error
    except RecommendationError as error:
        if json_output:
            typer.echo(render_recommendation_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Recommendation unavailable:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_recommendation_json(report))
    else:
        render_recommendation(report, Console())


@app.command()
def history(
    model: Annotated[
        str | None,
        typer.Option("--model", help="Show only runs for this model name."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Show at most this many runs."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit versioned machine-readable benchmark history."),
    ] = False,
) -> None:
    """List previously saved benchmark runs, newest first."""
    try:
        stored = load_benchmark_history(model=model, limit=limit)
    except BenchmarkStorageError as error:
        benchmark_error = BenchmarkError("storage_error", str(error))
        if json_output:
            typer.echo(render_benchmark_error_json(benchmark_error))
        else:
            Console(stderr=True).print(f"[red]History unavailable:[/red] {benchmark_error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_history_json(stored))
    else:
        render_history(stored, Console())


@app.command()
def compare(
    baseline_run: Annotated[str, typer.Argument(help="Baseline run ID or unique prefix.")],
    candidate_run: Annotated[str, typer.Argument(help="Run ID or unique prefix to compare.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a versioned machine-readable comparison."),
    ] = False,
) -> None:
    """Compare two saved benchmark runs that used the same workload version."""
    try:
        stored = load_benchmark_history()
        baseline = find_result(stored, baseline_run)
        candidate = find_result(stored, candidate_run)
        comparison = compare_result_groups(
            results_for_batch(stored, baseline), results_for_batch(stored, candidate)
        )
    except BenchmarkStorageError as error:
        benchmark_error = BenchmarkError("storage_error", str(error))
        if json_output:
            typer.echo(render_benchmark_error_json(benchmark_error))
        else:
            Console(stderr=True).print(
                f"[red]Comparison unavailable:[/red] {benchmark_error.message}"
            )
        raise typer.Exit(code=2) from error
    except BenchmarkError as error:
        if json_output:
            typer.echo(render_benchmark_error_json(error))
        else:
            Console(stderr=True).print(f"[red]Comparison unavailable:[/red] {error.message}")
        raise typer.Exit(code=2) from error

    if json_output:
        typer.echo(render_comparison_json(comparison))
    else:
        render_comparison(comparison, Console())


if __name__ == "__main__":
    app()
