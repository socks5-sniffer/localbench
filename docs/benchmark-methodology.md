# Benchmark methodology

Milestone 0.4 implements one intentionally narrow, versioned workload:

```bash
localbench bench MODEL
localbench bench MODEL --runs 5
```

## Quick profile

| Setting | Value |
|---|---|
| Prompt identifier | `benchmark_generation_v2` |
| Context allocation | 2,048 tokens |
| Generation target | 64 tokens |
| Temperature | 0 |
| Seed | 42 |
| Streaming | enabled |
| Delivery mode | raw (bypasses chat templating) |
| Ollama keep-alive | 0 (immediate unload afterward) |

The fixed prompt asks the model to continue a numbered sequence with short factual sentences.
The prompt text lives in `src/localbench/benchmark/profiles.py`; benchmark results store only
its identifier, never prompt or response content.

**v1 → v2:** v1 sent this prompt through Ollama's default chat template. Several
instruction-tuned models treated "continue this list" as a single chat turn, emitted their
end-of-turn token after one item, and stopped after 3-15 tokens instead of the 64-token
target — producing a throughput figure measured over a handful of tokens, not a meaningful
sample. v2 sends the identical prompt text with `"raw": true`, the correct delivery mode for
a text-continuation prompt like this one; generation then runs to the full target. v1 results
are not comparable to v2 results — they carry different prompt identifiers for exactly this
reason.

Before the run, LocalBench asks Ollama to unload the selected model. Whether that request
succeeded is recorded. A failed pre-run unload does not abort the workload, but the result warns
that load time may describe a warm model.

## Timing

- **Load time:** Ollama's `load_duration`, converted from nanoseconds.
- **Time to first token:** client monotonic time from request start until the first non-empty
  streamed response or thinking fragment.
- **Prompt throughput:** Ollama `prompt_eval_count / prompt_eval_duration`.
- **Generation throughput:** Ollama `eval_count / eval_duration`.
- **Total duration:** Ollama's server-reported total duration.
- **Wall time:** client monotonic duration for the complete streamed request.

Ollama reports its duration fields in nanoseconds. Streaming uses newline-delimited JSON with
fixed loopback access, redirect rejection, a three-minute timeout, and response-size limits.

## Resource sampling

LocalBench samples system-wide used RAM and CPU utilization every 100 ms during generation. It
reports baseline RAM, peak RAM, peak increase, and peak CPU. These metrics are labeled
system-wide because background applications can affect them; they are not attributed solely to
the Ollama process.

GPU utilization, VRAM sampling, temperatures, and sustained multi-minute performance remain
future benchmark profiles.

## Repeated batches

`--runs N` repeats the exact workload N times. Each observation is retained as its own raw row
with a shared `batch_id`, a one-based `batch_index`, and the planned `batch_size`; no aggregate
replaces the source measurements. The command reports the median and min–max spread for
generation rate, time-to-first-token, and load time. An interrupted batch remains honest partial
history and is refused by comparison until all planned indices exist.

Selecting any run ID from a complete batch in `compare` selects the whole batch. Batch
comparisons use metric medians and display their ranges. Singleton-to-singleton comparison is
still supported for old and default runs, but singleton-to-batch comparison is refused because
those are not equivalent levels of evidence. Every raw row in both batches must share the same
measurement signature, have a successful cold unload, and reach the generation target.

## History and comparison

```bash
localbench history
localbench history --model granite4:1b-h --limit 5
localbench compare RUN_ID RUN_ID
```

`history` lists saved runs newest first; run IDs may be abbreviated to any unique prefix
when passed to `compare`. Both commands support `--json`.

**Comparisons require identical canonical measurement signatures.** The signature contract
is versioned independently of the public result schema. It includes the benchmark method,
profile, workload identifier, context and generation target, request options (`raw`, streaming,
thinking, temperature, seed, and keep-alive), cold-start intent, runtime and runtime version,
and resource-sampler method and interval. Models, timestamps, run IDs, and measured values are
not part of the signature because those are observations or comparison subjects.

Different workload identifiers still raise `incomparable_workloads`. Any other signature
difference raises `incomparable_measurements`; unknown signatures or runtime versions are not
treated as equal. Runtime version is strict by design: an upgrade can change execution kernels,
scheduling, and metric behavior, so silently mixing versions would confound the comparison.
A future explicitly labeled runtime-regression mode may relax that rule, but the default does
not. A failed pre-run unload also makes a run ineligible for the current comparison, because
load time and time-to-first-token would no longer share the requested cold-start condition.
Both runs must also reach the signature's generation target, so a prematurely stopped sample
cannot silently influence a comparison or a future batch aggregate.

This is a correctness requirement, not a formality: a result from a superseded workload can
be numerically better than a correct one. A real example from
this project's own history — under v1, `granite4:1b-h` reported 14.06 tok/s measured over
3 tokens (the premature-stop bug), while the same model under v2 reports 8.67 tok/s over
the full 64. A comparison that ignored workload version would have ranked the broken
measurement first. `history` likewise warns whenever the runs it lists span more than one
workload version. Human and JSON history output also retain runtime-version and signature data.

Comparison percentages are expressed relative to the first (baseline) run, with the sign
normalized so that positive always means the second run performed better — including for
time-to-first-token and load time, where a lower raw number is the improvement.

## Persistence and privacy

Results are stored by default in a local SQLite database in the operating system's LocalBench
application-data directory. `--no-save` disables persistence. Stored records contain the result
schema, model name, runtime version, workload identifier, metrics, and warnings. They do not
contain prompt text, generated responses, usernames, hostnames, IP addresses, or filesystem
paths.

The SQLite schema has its own integer version in `PRAGMA user_version`; it is separate from the
public JSON `schema_version`. New databases are created at the latest database version. The
write path recognizes the original exact five-column table as database v1 and migrates it
transactionally through v2 (canonical signatures) and v3 (batch identity/index/size). Legacy
rows become honest singleton batches whose batch ID equals their run ID. Known signatures are
backfilled, while original result JSON is preserved byte-for-byte. Unknown table shapes and
databases newer than the installed LocalBench supports are refused rather than guessed at.
