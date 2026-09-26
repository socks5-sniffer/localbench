# LocalBench

LocalBench is a local-first tool for understanding what AI workloads a computer can
realistically support. Version 1.0 provides system profiling, Ollama and llama.cpp runtime
discovery, an explainable memory estimator with GPU/VRAM offload estimation, and a
context-length capacity profiler. It also includes a reproducible quick benchmark for installed
Ollama models, retained repeated-run batches, strict comparison, simultaneous-memory stack
planning, and measured multi-model coexistence benchmarking. Its evidence-based recommender
turns compatible local measurements into transparent, workload-weighted operational advice.

## Requirements

- Python 3.12 or newer
- Windows 10/11 or Linux for the initial supported release
- macOS is best-effort in 1.0

## Install as a terminal command

With `uv` installed, run this once from the repository folder:

```bash
uv tool install .
```

Then run LocalBench from any folder:

```bash
localbench --help
localbench profile
localbench profile --json
```

If your shell cannot find `localbench`, run `uv tool update-shell` and open a new
terminal. This installs a separate copy in your user tools directory; the repository
and its external drive are not needed to run the installed command. The examples
below also work with `localbench` in place of `uv run localbench`.

After updating the source, reinstall from the repository folder with
`uv tool install --reinstall .`. To uninstall, run `uv tool uninstall localbench`.

## Install for development

```bash
uv sync --extra dev
```

Run the human-readable profile:

```bash
uv run localbench profile
```

Produce machine-readable JSON:

```bash
uv run localbench profile --json
```

Discover Ollama and llama.cpp (`llama-server`):

```bash
uv run localbench runtimes
uv run localbench runtimes --json
```

Ollama discovery reports its installed model registry. llama.cpp has no equivalent registry,
so LocalBench reports only models currently loaded by a server on its fixed default loopback
port. Estimation and benchmarking remain Ollama-only in 1.0.

Estimate memory for an installed Ollama model while its local service is running:

```bash
uv run localbench estimate gemma3:4b
uv run localbench estimate gemma3:4b --context 16384 --kv-cache-type q8_0
uv run localbench estimate gemma3:4b --json
```

The estimator reports weights, KV cache, runtime overhead, and safety headroom separately and
labels each component as reported, calculated, or inferred. When a GPU with dedicated VRAM is
detected, it also reports a coarse layer-offload estimate (full, partial, or CPU-only) — a
uniform-layer approximation, not Ollama's actual placement algorithm, and it does not yet model
unified-memory (e.g. Apple Silicon) behavior.

See how a model's memory footprint grows as context length grows:

```bash
uv run localbench context gemma3:4b
uv run localbench context gemma3:4b --levels 4096,8192,16384 --kv-cache-type q8_0
uv run localbench context gemma3:4b --json
```

Each requested context length gets a full `estimate`-style breakdown against one shared
available-memory snapshot, including GPU offload if a GPU with dedicated VRAM is detected — KV
cache grows with context, so offload capacity typically shrinks at longer lengths. A length
beyond the model's own context ceiling is skipped rather than failing the whole sweep. This
profiles memory only; it does not measure prompt-processing rate or time-to-first-token
degradation under real context load.

Plan whether several installed models plausibly fit in memory at the same time:

```bash
uv run localbench stack --model reasoning=phi4-mini:latest \
  --model embeddings=nomic-embed-text:latest
uv run localbench stack --model reasoning=phi4-mini:latest \
  --model vision=moondream:latest --context 4096 --json
```

Each `--model` value is `ROLE=MODEL`. The planner estimates every role against one current
available-memory snapshot, adds OS/background safety headroom once for the whole stack, and
shows weights, KV cache, and runtime overhead separately. It predicts memory feasibility only;
it does not infer concurrent throughput or contention from single-model benchmarks. Models
already resident in memory can make the current available-memory snapshot conservative.

Measure real throughput while two or more models generate concurrently:

```bash
uv run localbench coexist --model reasoning=phi4-mini:latest \
  --model secondary=granite4.2:3b
uv run localbench coexist --model reasoning=phi4-mini:latest \
  --model secondary=granite4.2:3b --baseline reasoning=RUN_ID --json
```

Each optional `--baseline` is `ROLE=RUN_ID` and compares that role against a compatible saved
solo benchmark. The command actively unloads, loads, and generates from every requested model;
coexistence results are currently rendered but not saved to benchmark history.

Run the deterministic quick benchmark and save the result locally:

```bash
uv run localbench bench phi4-mini:latest
uv run localbench bench phi4-mini:latest --runs 5
uv run localbench bench phi4-mini:latest --json
uv run localbench bench phi4-mini:latest --no-save
```

Use `--runs N` to retain a repeated batch and report median plus range instead of relying on a
single sample. Benchmark history is stored in the operating system's LocalBench
application-data directory.
Prompts and generated responses are never stored; only the versioned prompt identifier,
configuration, timing counters, and system-wide resource measurements are persisted.

Review and compare saved runs:

```bash
uv run localbench history
uv run localbench history --model granite4:1b-h --limit 5
uv run localbench compare RUN_ID RUN_ID
```

Run IDs may be abbreviated to any unique prefix. Selecting a run from a repeated batch selects
the complete batch and compares medians plus ranges. Comparisons are refused when canonical
measurement conditions differ, when a batch is incomplete, or when one side is a singleton and
the other is a repeated batch — see [benchmark methodology](docs/benchmark-methodology.md).

Rank benchmarked installed models using an explicit performance policy:

```bash
uv run localbench recommend
uv run localbench recommend --use-case low-latency
uv run localbench recommend --use-case rag --context 16384 --kv-cache-type q8_0
uv run localbench recommend --use-case coding --json
```

Only complete benchmark batches matching the current quick workload and Ollama version are
eligible. The report exposes raw measurements, relative component scores, policy weights,
memory fit, and excluded models. It ranks operational speed and capacity only; it does not
pretend to measure answer quality, coding correctness, or reasoning ability. See the
[recommendation methodology](docs/recommendations.md) for the exact eligibility and math.

All memory values in JSON are integer bytes. Unavailable observations are represented by
`null`; collection failures are reported in `warnings`. JSON is written alone to standard
output so it can safely be piped to another program.

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

LocalBench performs no telemetry or external network requests. Runtime discovery makes
read-only requests to fixed loopback APIs for Ollama (`127.0.0.1:11434`) and llama.cpp
(`127.0.0.1:8080`); it never follows an environment variable to a remote server. llama.cpp
model IDs that contain paths are reduced to basename-only before output. A profile does not
collect usernames, hostnames, IP addresses, MAC addresses, serial numbers, prompts, or
filesystem paths.

See [the architecture](docs/architecture.md) for the initial module boundaries and JSON
contract, [runtime discovery](docs/runtimes.md) for the Ollama and llama.cpp adapter boundary,
and [memory-estimation methodology](docs/memory-estimation.md) for formulas and limitations.
[Benchmark methodology](docs/benchmark-methodology.md) defines the workload and metrics.
The [recommendation methodology](docs/recommendations.md) defines evidence eligibility,
performance policies, scoring, and quality limitations.
The [stack-planning methodology](docs/stack-planning.md) defines simultaneous-memory arithmetic
and its current limitations.
The [coexistence-benchmarking methodology](docs/coexistence-benchmarking.md) defines how
concurrent generation and contention against solo baselines are measured.
The [community export contract](docs/community-export.md) defines the opt-in privacy and
scientific-validity boundary before any uploader or service is built.
