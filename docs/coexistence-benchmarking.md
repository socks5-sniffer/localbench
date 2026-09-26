# Coexistence benchmarking methodology

`localbench stack` (see [`stack-planning.md`](stack-planning.md)) answers a memory-feasibility
question only: does this combination of models fit in available memory? It is explicit that it
never claims anything about speed. This document covers the follow-up layer,
`localbench.benchmark.coexistence`, which answers the speed question for real by actually
running the models together and measuring what happens. The CLI exposes it as:

```bash
localbench coexist --model chat=phi4-mini:latest --model helper=granite4.2:3b
localbench coexist --model chat=phi4-mini:latest --model helper=granite4.2:3b \
  --baseline chat=RUN_ID --json
```

Each repeated `--model` value is `ROLE=MODEL`. Each optional `--baseline` value is
`ROLE=RUN_ID`, where the run ID may be any unique saved-history prefix and its role must also
appear in `--model`.

## What it measures

`run_coexistence_benchmark` takes two or more named `CoexistenceRequest` roles (each a role
name plus an installed Ollama model, optionally with that model's own prior solo `bench`
result as a baseline) and:

1. Cold-unloads every requested model (best effort, same as solo `bench`).
2. Starts every role's generation stream at the same time, synchronized with a
   `threading.Barrier` so the measured window reflects real concurrent load/generate
   contention rather than a staggered sequence.
3. Samples system-wide RAM/CPU with one `SystemSampler` across the whole concurrent window.
4. If a solo baseline was supplied for a role, computes how much the concurrent run differed
   from that baseline — the actual, measured cost of contention, not a projection.

If any role's stream fails, the whole run raises `CoexistenceError`: a partial concurrent
result would misrepresent the contention the surviving roles actually experienced, the same
fail-fast principle the solo benchmark already uses.

## Reused, not duplicated, measurement code

Both the solo benchmark (`run_quick_benchmark`) and this module call the same
`stream_completion` helper in `benchmark/runner.py` to run one request and derive its timing
and token counters. Only orchestration differs: solo owns unload timing, sampling, and a
single stream; coexistence adds barrier-synchronized concurrent scheduling across N streams.
This guarantees a concurrent measurement and its solo baseline were produced by identical
request semantics — the same prompt, `raw` mode, temperature, seed, and context length — so a
contention comparison is measuring contention, not a difference in how the two runs asked
Ollama to generate.

## The contention percentage

For a role with a baseline, `ModelCoexistenceResult` reports `generation_rate_change_percent`,
`ttft_change_percent`, and `load_time_change_percent`. These follow the same sign convention
as `BenchmarkComparison` in [`benchmark-methodology.md`](benchmark-methodology.md): **positive
always means the concurrent run measured as better than the solo baseline**, including for
TTFT and load time where the raw number is lower-is-better. Under genuine contention,
`generation_rate_change_percent` is expected to come back negative — that is the real
throughput cost of running these models together, not a defect in the measurement.

### When the comparison is skipped

A baseline is only used if it is a solid apples-to-apples match: same model, both its saved
solo run and the current concurrent run completed their cold unloads, both reached the full
generation target, and the baseline's complete canonical measurement signature exactly
matches the current quick workload and runtime version. Any mismatch skips the comparison
for that role — the fields come back `None` — with an explicit warning explaining why, rather
than silently comparing runs that were not measured the same way. A role with no baseline
supplied at all is not a warning; it simply has nothing to compare against.

## What this does not do

- **No throughput extrapolation.** Nothing here infers concurrent performance from
  single-model numbers; every figure is either measured live during the concurrent run or a
  direct comparison against an already-measured solo baseline.
- **No persistence yet.** `localbench coexist` renders the typed result in human-readable or
  versioned JSON form, but does not add it to solo benchmark history. Repeated coexistence
  batches and retained coexistence history remain future work.
- **No claim about other hardware.** A coexistence measurement reflects real contention on
  this machine right now; it is not a guarantee of future performance and does not
  generalize to different hardware, model versions, or Ollama versions.

## Real-machine validation

[`coexistence-validation.md`](coexistence-validation.md) preserves explicitly labeled local
validation runs until first-class coexistence persistence and repeated batches are available.
