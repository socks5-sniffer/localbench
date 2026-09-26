"""Human and machine-readable profile presentation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from statistics import median
from typing import cast

from rich.console import Console
from rich.table import Table

from localbench import __version__
from localbench.benchmark.coexistence import CoexistenceError, CoexistenceResult
from localbench.benchmark.schemas import (
    BenchmarkComparison,
    BenchmarkError,
    BenchmarkResult,
    MetricSpread,
    StoredResult,
)
from localbench.context.schemas import ContextCapacityProfile, ContextProfileError
from localbench.estimation.schemas import EstimateError, ModelMemoryEstimate
from localbench.models import RuntimeDiscovery, RuntimeProfile, SystemProfile
from localbench.recommendation import RecommendationError, RecommendationReport
from localbench.stack import StackPlan, StackPlanError


def render_json(profile: SystemProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True)


def render_runtime_json(discovery: RuntimeDiscovery) -> str:
    return json.dumps(discovery.to_dict(), indent=2, sort_keys=True)


def render_estimate_json(estimate: ModelMemoryEstimate) -> str:
    return json.dumps(estimate.to_dict(), indent=2, sort_keys=True)


def render_estimate_error_json(error: EstimateError) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def render_benchmark_json(result: BenchmarkResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


def render_benchmark_error_json(error: BenchmarkError) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    gibibytes = value / (1024**3)
    return f"{gibibytes:.1f} GiB"


def render_human(profile: SystemProfile, console: Console) -> None:
    console.print(f"[bold cyan]LocalBench {__version__}[/bold cyan]")

    system = Table(title="System", show_header=False, box=None, pad_edge=False)
    system.add_column(style="bold")
    system.add_column()
    system.add_row("OS", f"{profile.os.family} {profile.os.release}")
    system.add_row("Architecture", profile.os.architecture)
    system.add_row("CPU", profile.cpu.model or "Unknown")
    core_text = f"{profile.cpu.physical_cores or 'Unknown'} physical / "
    core_text += f"{profile.cpu.logical_cores or 'Unknown'} logical"
    system.add_row("Cores", core_text)
    console.print(system)

    memory = Table(title="Memory", show_header=False, box=None, pad_edge=False)
    memory.add_column(style="bold")
    memory.add_column(justify="right")
    memory.add_row("Installed", _format_bytes(profile.memory.total_bytes))
    memory.add_row("Available", _format_bytes(profile.memory.available_bytes))
    memory.add_row("Used", _format_bytes(profile.memory.used_bytes))
    memory.add_row("Swap total", _format_bytes(profile.memory.swap_total_bytes))
    memory.add_row("Swap used", _format_bytes(profile.memory.swap_used_bytes))
    console.print(memory)

    gpu = Table(title="GPU", show_header=True, box=None, pad_edge=False)
    gpu.add_column("Adapter", style="bold")
    gpu.add_column("Vendor")
    gpu.add_column("Dedicated memory", justify="right")
    gpu.add_column("Driver")
    for adapter in profile.gpus:
        gpu.add_row(
            adapter.name,
            adapter.vendor or "Unknown",
            _format_bytes(adapter.dedicated_memory_bytes),
            adapter.driver_version or "Unknown",
        )
    if not profile.gpus:
        gpu.add_row("Unavailable", "—", "—", "—")
    console.print(gpu)

    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _runtime_status(runtime: RuntimeProfile) -> str:
    if runtime.service_running:
        return "Running"
    if runtime.cli_detected:
        return "Installed; service stopped"
    return "Not detected"


def render_runtimes(discovery: RuntimeDiscovery, console: Console) -> None:
    console.print("[bold cyan]LocalBench runtimes[/bold cyan]")
    for runtime in discovery.runtimes:
        display_name = runtime.name.replace("_", " ").title()
        summary = Table(title=display_name, show_header=False, box=None, pad_edge=False)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Status", _runtime_status(runtime))
        summary.add_row("CLI", "Detected" if runtime.cli_detected else "Not detected")
        summary.add_row("Version", runtime.version or "Unknown")
        summary.add_row("Installed models", str(len(runtime.models)))
        console.print(summary)

        if runtime.models:
            models = Table(show_header=True, box=None, pad_edge=False)
            models.add_column("Model", style="bold")
            models.add_column("Parameters")
            models.add_column("Quantization")
            models.add_column("Size", justify="right")
            for model in runtime.models:
                models.add_row(
                    model.name,
                    model.parameter_size or "Unknown",
                    model.quantization_level or "Unknown",
                    _format_bytes(model.size_bytes),
                )
            console.print(models)

        for warning in runtime.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")


def render_estimate(estimate: ModelMemoryEstimate, console: Console) -> None:
    console.print(f"[bold cyan]Model memory estimate — {estimate.model}[/bold cyan]")
    console.print(
        f"Context: {estimate.context_length:,} tokens | KV cache: {estimate.kv_cache_type.value}"
    )

    components = Table(show_header=True, box=None, pad_edge=False)
    components.add_column("Component", style="bold")
    components.add_column("Amount", justify="right")
    components.add_column("Evidence")
    components.add_row(
        "Weights", _format_bytes(estimate.weights.bytes), estimate.weights.provenance
    )
    components.add_row(
        "KV cache", _format_bytes(estimate.kv_cache.bytes), estimate.kv_cache.provenance
    )
    components.add_row(
        "Runtime overhead",
        _format_bytes(estimate.runtime_overhead.bytes),
        estimate.runtime_overhead.provenance,
    )
    components.add_row(
        "Safety headroom",
        _format_bytes(estimate.safety_headroom.bytes),
        estimate.safety_headroom.provenance,
    )
    components.add_row("Estimated total", _format_bytes(estimate.estimated_total_bytes), "mixed")
    components.add_row(
        "Currently available",
        _format_bytes(estimate.available_memory.bytes),
        estimate.available_memory.provenance,
    )
    console.print(components)
    console.print(f"Fit: [bold]{estimate.fit.value.upper().replace('_', ' ')}[/bold]")

    if estimate.gpu_offload is None:
        console.print("GPU offload: [dim]not estimated (no dedicated-VRAM GPU detected)[/dim]")
    else:
        offload = estimate.gpu_offload
        offload_table = Table(show_header=True, box=None, pad_edge=False)
        offload_table.add_column("GPU", style="bold")
        offload_table.add_column("VRAM", justify="right")
        offload_table.add_column("Reserved", justify="right")
        offload_table.add_column("Layers offloadable", justify="right")
        offload_table.add_row(
            offload.gpu_name,
            _format_bytes(offload.vram.bytes),
            _format_bytes(offload.reserved_vram.bytes),
            f"{offload.offloadable_layers}/{offload.total_layers} "
            f"({offload.offload_fraction:.0%})",
        )
        console.print(offload_table)
        offload_label = offload.classification.value.upper().replace("_", " ")
        console.print(f"GPU offload: [bold]{offload_label}[/bold]")

    console.print("[bold]Assumptions[/bold]")
    for assumption in estimate.assumptions:
        console.print(f"• {assumption}")
    for warning in estimate.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def render_context_json(profile: ContextCapacityProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True)


def render_context_error_json(error: ContextProfileError | EstimateError) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def render_context(profile: ContextCapacityProfile, console: Console) -> None:
    console.print(f"[bold cyan]Context capacity — {profile.model}[/bold cyan]")
    limit_text = (
        f"{profile.model_context_limit:,} tokens"
        if profile.model_context_limit is not None
        else "Unknown"
    )
    console.print(f"KV cache: {profile.kv_cache_type.value} | Model context limit: {limit_text}")

    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Context", justify="right", style="bold")
    table.add_column("KV cache", justify="right")
    table.add_column("Estimated total", justify="right")
    table.add_column("Fit")
    table.add_column("GPU offload")
    for level in profile.levels:
        estimate = level.estimate
        offload = "—"
        if estimate.gpu_offload is not None:
            offload_layers = estimate.gpu_offload
            offload = (
                f"{offload_layers.offloadable_layers}/{offload_layers.total_layers} layers "
                f"({offload_layers.classification.value.replace('_', ' ')})"
            )
        table.add_row(
            f"{level.context_length:,}",
            _format_bytes(estimate.kv_cache.bytes),
            _format_bytes(estimate.estimated_total_bytes),
            estimate.fit.value.upper().replace("_", " "),
            offload,
        )
    console.print(table)

    if profile.skipped_context_lengths:
        skipped = ", ".join(f"{length:,}" for length in profile.skipped_context_lengths)
        console.print(f"[dim]Skipped (exceeds model context limit): {skipped}[/dim]")

    console.print("[bold]Assumptions[/bold]")
    for assumption in profile.assumptions:
        console.print(f"• {assumption}")
    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def render_stack_json(plan: StackPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)


def render_stack_error_json(error: StackPlanError | EstimateError) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def render_stack(plan: StackPlan, console: Console) -> None:
    console.print("[bold cyan]Local AI stack plan[/bold cyan]")
    models = Table(show_header=True, box=None, pad_edge=False)
    models.add_column("Role", style="bold")
    models.add_column("Model")
    models.add_column("Weights", justify="right")
    models.add_column("KV cache", justify="right")
    models.add_column("Runtime", justify="right")
    models.add_column("Working", justify="right")
    for model in plan.models:
        models.add_row(
            model.role,
            model.model,
            _format_bytes(model.weights_bytes),
            _format_bytes(model.kv_cache_bytes),
            _format_bytes(model.runtime_overhead_bytes),
            _format_bytes(model.working_memory_bytes),
        )
    console.print(models)

    totals = Table(show_header=False, box=None, pad_edge=False)
    totals.add_column(style="bold")
    totals.add_column(justify="right")
    totals.add_row("Models", _format_bytes(plan.total_working_memory_bytes))
    totals.add_row("Safety headroom", _format_bytes(plan.safety_headroom.bytes))
    totals.add_row("Required", _format_bytes(plan.total_required_bytes))
    totals.add_row("Available", _format_bytes(plan.available_memory.bytes))
    totals.add_row("Headroom", _format_bytes(plan.headroom_bytes))
    totals.add_row("Classification", plan.viability.value.upper().replace("_", " "))
    console.print(totals)
    for warning in plan.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(
        "[dim]Memory feasibility only; concurrent throughput and contention are not measured.[/dim]"
    )


def render_coexistence_json(result: CoexistenceResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


def render_coexistence_error_json(
    error: CoexistenceError | BenchmarkError | StackPlanError,
) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def render_coexistence(result: CoexistenceResult, console: Console) -> None:
    console.print(
        f"[bold cyan]Measured coexistence — {result.concurrency} concurrent roles[/bold cyan]"
    )
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Role", style="bold")
    table.add_column("Model")
    table.add_column("Generation", justify="right")
    table.add_column("vs solo", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Load", justify="right")
    for model in result.models:
        table.add_row(
            model.role,
            model.model,
            _format_rate(model.generation_tokens_per_second),
            _format_change(model.generation_rate_change_percent),
            _format_seconds(model.ttft_seconds),
            _format_seconds(model.load_time_seconds),
        )
    console.print(table)
    console.print(
        f"Peak system RAM: {_format_bytes(result.peak_system_ram_used_bytes)} | "
        f"Increase: {_format_bytes(result.peak_system_ram_increase_bytes)} | "
        f"Peak CPU: {result.peak_system_cpu_percent:.1f}% | "
        f"Wall time: {result.wall_time_seconds:.3f} s"
    )
    for warning in dict.fromkeys(result.warnings):
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _format_rate(value: float | None) -> str:
    return f"{value:.2f} tok/s" if value is not None else "Unavailable"


def render_benchmark(result: BenchmarkResult, console: Console) -> None:
    console.print(f"[bold cyan]Quick benchmark — {result.model}[/bold cyan]")
    metrics = Table(show_header=False, box=None, pad_edge=False)
    metrics.add_column(style="bold")
    metrics.add_column(justify="right")
    metrics.add_row("Load time", f"{result.load_time_seconds:.3f} s")
    metrics.add_row("Time to first token", f"{result.ttft_seconds:.3f} s")
    metrics.add_row("Prompt processing", _format_rate(result.prompt_tokens_per_second))
    metrics.add_row("Generation", _format_rate(result.generation_tokens_per_second))
    metrics.add_row("Generated tokens", str(result.generation_tokens))
    metrics.add_row("Wall time", f"{result.wall_time_seconds:.3f} s")
    metrics.add_row("Peak system RAM", _format_bytes(result.peak_system_ram_used_bytes))
    metrics.add_row("RAM increase", _format_bytes(result.peak_system_ram_increase_bytes))
    metrics.add_row("Peak system CPU", f"{result.peak_system_cpu_percent:.1f}%")
    metrics.add_row("Run ID", result.run_id)
    console.print(metrics)
    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _batch_summary(results: tuple[BenchmarkResult, ...]) -> dict[str, object]:
    def summary(values: tuple[float | None, ...]) -> dict[str, float | int | None]:
        available = tuple(value for value in values if value is not None)
        return {
            "count": len(available),
            "median": median(available) if available else None,
            "minimum": min(available) if available else None,
            "maximum": max(available) if available else None,
        }

    return {
        "schema_version": "1",
        "batch_id": results[0].batch_id,
        "planned_runs": results[0].batch_size,
        "completed_runs": len(results),
        "generation_tokens_per_second": summary(
            tuple(result.generation_tokens_per_second for result in results)
        ),
        "ttft_seconds": summary(tuple(result.ttft_seconds for result in results)),
        "load_time_seconds": summary(tuple(result.load_time_seconds for result in results)),
        "runs": [result.to_dict() for result in results],
    }


def render_benchmark_batch_json(results: tuple[BenchmarkResult, ...]) -> str:
    return json.dumps(_batch_summary(results), indent=2, sort_keys=True)


def render_benchmark_batch(results: tuple[BenchmarkResult, ...], console: Console) -> None:
    summary = _batch_summary(results)
    console.print(
        f"[bold cyan]Quick benchmark batch — {results[0].model} "
        f"({len(results)} runs)[/bold cyan]"
    )
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Metric", style="bold")
    table.add_column("Median", justify="right")
    table.add_column("Range", justify="right")
    metrics: tuple[tuple[str, str, Callable[[float], str]], ...] = (
        ("Generation", "generation_tokens_per_second", lambda value: f"{value:.2f} tok/s"),
        ("Time to first token", "ttft_seconds", lambda value: f"{value:.3f} s"),
        ("Load time", "load_time_seconds", lambda value: f"{value:.3f} s"),
    )
    for label, key, formatter in metrics:
        values = cast(dict[str, float | int | None], summary[key])
        median_value = values["median"]
        minimum = values["minimum"]
        maximum = values["maximum"]
        if (
            not isinstance(median_value, int | float)
            or not isinstance(minimum, int | float)
            or not isinstance(maximum, int | float)
        ):
            table.add_row(label, "Unavailable", "Unavailable")
            continue
        table.add_row(
            label,
            formatter(float(median_value)),
            f"{formatter(float(minimum))}–{formatter(float(maximum))}",
        )
    console.print(table)
    console.print(f"Batch ID: {results[0].batch_id}")
    for warning in dict.fromkeys(
        warning for result in results for warning in result.warnings
    ):
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def render_history_json(history: tuple[StoredResult, ...]) -> str:
    return json.dumps(
        {"schema_version": "1", "runs": [result.to_dict() for result in history]},
        indent=2,
        sort_keys=True,
    )


def render_comparison_json(comparison: BenchmarkComparison) -> str:
    return json.dumps(comparison.to_dict(), indent=2, sort_keys=True)


def render_recommendation_json(report: RecommendationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_recommendation_error_json(error: RecommendationError) -> str:
    return json.dumps(
        {"schema_version": "1", "error": {"code": error.code, "message": error.message}},
        indent=2,
        sort_keys=True,
    )


def render_recommendation(report: RecommendationReport, console: Console) -> None:
    console.print(
        f"[bold cyan]Model recommendation — {report.use_case.value.replace('-', ' ')}[/bold cyan]"
    )
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Generation", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Fit")
    table.add_column("Score", justify="right")
    for item in report.recommendations:
        table.add_row(
            str(item.rank),
            item.model,
            _format_rate(item.generation_tokens_per_second),
            _format_rate(item.prompt_tokens_per_second),
            _format_seconds(item.ttft_seconds),
            _format_bytes(item.estimated_total_bytes),
            item.fit.upper().replace("_", " "),
            f"{item.weighted_score * 100:.1f}",
        )
    console.print(table)

    weights = report.weights
    console.print(
        "[bold]Scoring components (relative 0–100)[/bold] "
        f"generation {weights.generation:.0%}, prompt {weights.prompt:.0%}, "
        f"TTFT {weights.ttft:.0%}, memory {weights.memory:.0%}"
    )
    components = Table(show_header=True, box=None, pad_edge=False)
    components.add_column("Model", style="bold")
    components.add_column("Generation", justify="right")
    components.add_column("Prompt", justify="right")
    components.add_column("TTFT", justify="right")
    components.add_column("Memory", justify="right")
    components.add_column("Evidence")
    for item in report.recommendations:
        components.add_row(
            item.model,
            f"{item.generation_score * 100:.0f}",
            f"{item.prompt_score * 100:.0f}",
            f"{item.ttft_score * 100:.0f}",
            f"{item.memory_score * 100:.0f}",
            f"{item.batch_id[:8]} (n={item.run_count})",
        )
    console.print(components)
    for excluded in report.excluded_models:
        console.print(f"[yellow]Excluded:[/yellow] {excluded}")
    for caveat in report.caveats:
        console.print(f"[dim]{caveat}[/dim]")


def _format_seconds(value: float | None) -> str:
    return f"{value:.3f} s" if value is not None else "Unavailable"


def _short_workload(prompt_id: str) -> str:
    """Reduce a workload id to its trailing version token for narrow-terminal display.

    `benchmark_generation_v2` becomes `v2`. The full identifier is always printed in a
    legend beneath the table, so nothing is lost — this only keeps the column from
    wrapping and burying the one field that determines comparability.
    """
    tail = prompt_id.rsplit("_", 1)[-1]
    return tail if re.fullmatch(r"v\d+", tail) else prompt_id


def render_history(history: tuple[StoredResult, ...], console: Console) -> None:
    console.print("[bold cyan]Benchmark history[/bold cyan]")
    if not history:
        console.print("No stored benchmark runs yet. Run [bold]localbench bench MODEL[/bold].")
        return

    # The workload column is the reason history is trustworthy, so it must never be the
    # column that gets truncated away on a narrow terminal: `no_wrap=False` lets it fold
    # instead, and the shared "benchmark_" prefix is dropped to buy back width.
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Run", style="bold", no_wrap=True)
    table.add_column("Batch", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Started", no_wrap=True)
    table.add_column("Generation", justify="right", no_wrap=True)
    table.add_column("TTFT", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Workload", overflow="fold")
    for result in history:
        table.add_row(
            result.run_id[:8],
            (
                f"{(result.batch_id or result.run_id)[:8]} "
                f"{result.batch_index}/{result.batch_size}"
            ),
            result.model,
            # Year omitted: history is read for relative recency, and the width is worth
            # more to the model name. Full timestamps remain in --json.
            result.started_at[5:16].replace("T", " "),
            _format_rate(result.generation_tokens_per_second),
            _format_seconds(result.ttft_seconds),
            str(result.generation_tokens) if result.generation_tokens is not None else "?",
            _short_workload(result.prompt_id),
        )
    console.print(table)

    # Surfacing this is the whole point of showing the workload column: results from
    # different workload versions are not comparable, and the user needs to know before
    # they read the table as a ranking.
    workloads = {result.prompt_id for result in history}
    console.print(f"[dim]Workload: {', '.join(sorted(workloads))}[/dim]")
    runtimes = {
        f"{result.runtime or 'unknown'} {result.runtime_version or 'unknown version'}"
        for result in history
    }
    console.print(f"[dim]Runtime: {', '.join(sorted(runtimes))}[/dim]")
    if len(workloads) > 1:
        console.print(
            "[yellow]Warning:[/yellow] this history spans multiple workload versions. "
            "Only runs sharing a workload version can be compared."
        )
    signatures = {result.measurement_signature for result in history}
    if None in signatures or len(signatures) > 1:
        console.print(
            "[yellow]Warning:[/yellow] these runs do not all share one canonical "
            "measurement signature. Strict comparison may be refused."
        )


def _format_change(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"[green]+{value:.1f}%[/green]" if value >= 0 else f"[red]{value:.1f}%[/red]"


def _format_spread(
    spread: MetricSpread, formatter: Callable[[float | None], str]
) -> str:
    median_value = spread.median
    minimum = spread.minimum
    maximum = spread.maximum
    if median_value is None or minimum is None or maximum is None:
        return "Unavailable"
    rendered = formatter(median_value)
    if spread.count == 1:
        return rendered
    return f"{rendered} [{formatter(minimum)}–{formatter(maximum)}]"


def render_comparison(comparison: BenchmarkComparison, console: Console) -> None:
    baseline, candidate = comparison.baseline, comparison.candidate
    console.print(
        f"[bold cyan]Comparison — {baseline.model} → {candidate.model}[/bold cyan]"
    )
    console.print(f"Workload: {comparison.prompt_id} (identical for both runs)")
    signature = comparison.measurement_signature
    console.print(
        f"Runtime: {signature.runtime} {signature.runtime_version} "
        f"(measurement signature v{signature.contract_version} matched)"
    )

    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Metric", style="bold")
    table.add_column(
        f"{(baseline.batch_id or baseline.run_id)[:8]} (n={comparison.baseline_run_count})",
        justify="right",
    )
    table.add_column(
        f"{(candidate.batch_id or candidate.run_id)[:8]} (n={comparison.candidate_run_count})",
        justify="right",
    )
    table.add_column("Change", justify="right")
    table.add_row(
        "Generation",
        _format_spread(comparison.baseline_generation, _format_rate),
        _format_spread(comparison.candidate_generation, _format_rate),
        _format_change(comparison.generation_rate_change_percent),
    )
    table.add_row(
        "Time to first token",
        _format_spread(comparison.baseline_ttft, _format_seconds),
        _format_spread(comparison.candidate_ttft, _format_seconds),
        _format_change(comparison.ttft_change_percent),
    )
    table.add_row(
        "Load time",
        _format_spread(comparison.baseline_load_time, _format_seconds),
        _format_spread(comparison.candidate_load_time, _format_seconds),
        _format_change(comparison.load_time_change_percent),
    )
    console.print(table)
    console.print(
        "[dim]Positive change favours the second run. Single runs carry real "
        "variance; treat small differences as noise.[/dim]"
    )
