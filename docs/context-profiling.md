# Context-length capacity profiling

A model that fits comfortably at 2K tokens of context does not necessarily fit at 32K. KV cache
grows linearly with context length while weights and runtime overhead stay fixed, so the same
model can move from "excellent" to "will not fit" purely by asking for a longer context window.
`localbench.context.profile_context_capacity` answers that by sweeping
[`estimate_ollama_model`](memory-estimation.md) across a set of context lengths.

```bash
uv run localbench context gemma3:4b
uv run localbench context gemma3:4b --levels 4096,8192,16384 --kv-cache-type q8_0
uv run localbench context gemma3:4b --json
```

`--levels` is a comma-separated list of context lengths in tokens; it defaults to
`DEFAULT_CONTEXT_LEVELS` (below). Each level's row includes GPU offload if a GPU with measured
dedicated VRAM is detected — see [memory-estimation.md](memory-estimation.md#gpu-offload-estimate);
because KV cache grows with context length, offload capacity typically shrinks at longer levels.

## What it does

```text
for each requested context length, ascending:
    call estimate_ollama_model at that length, against one shared available-memory snapshot
    if the length exceeds the model's own context ceiling: skip it, don't fail the sweep
    otherwise: record its full explainable ModelMemoryEstimate (weights, KV cache, fit, ...)
```

The default sweep is `2048, 4096, 8192, 16384, 32768, 65536` tokens
(`DEFAULT_CONTEXT_LEVELS`), matching the shape from the original roadmap:

```text
2K       Excellent
4K       Excellent
8K       Good
16K      Tight
32K      Swap expected
64K      Not recommended
```

Every level's `fit` classification reuses the exact same `FitClassification` enum and
thresholds as `localbench estimate`, so a context sweep and a single-context estimate read on
one consistent scale.

## Design notes

- **One shared available-memory snapshot for the whole sweep.** `available_memory_bytes` is a
  single argument, not measured per level — the same principle `stack.plan_stack` uses, so
  every level in the sweep is judged against the same system state.
- **Model metadata is fetched once, not once per level.** Ollama's `/api/show` response (block
  count, KV-head count, key/value lengths, context ceiling) does not change between context
  lengths within one sweep, so the details fetcher is wrapped in a small memoizing cache — a
  six-level sweep makes one metadata round trip, not six.
- **A too-large context length is skipped, not a failure.** If a requested length exceeds the
  model's own context ceiling, it moves to `skipped_context_lengths` and the sweep continues.
  If *every* requested length exceeds the ceiling, the whole call raises
  `ContextProfileError("no_levels_fit", ...)` rather than returning an empty, useless profile.
- **Other estimation failures propagate immediately as the underlying `EstimateError`.**
  Ambiguous or missing models, a stopped runtime, or insufficient KV-cache metadata are
  properties of the model as a whole, not of one context length — repeating the same failure
  once per requested level would add noise, not information. `ContextProfileError` is reserved
  for problems specific to the sweep itself (empty, invalid, or duplicate requested lengths).

## What this does not do

The roadmap describes context profiling as covering four things: memory growth,
prompt-processing degradation, TTFT degradation, and KV-cache consumption. This increment
covers memory growth and KV-cache consumption only (both are inherent to
`estimate_ollama_model`, which already reports the KV-cache component explicitly). It does
**not** measure real prompt-processing rate or time-to-first-token degradation as context
grows — that requires live generation at every requested context length, which is a
materially bigger and slower feature (real wall-clock model runs times the number of context
levels), and is intentionally left as separate future work rather than silently promised by
the module name. `ContextCapacityProfile.assumptions` states this explicitly so a caller
reading the typed result sees the same disclosure documented here.
