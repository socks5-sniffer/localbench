# Stack-planning methodology (Phase 7 foundation)

This is the first increment of Phase 7 ("can I run more than one model at once?"). A pure,
explainable capacity-planning core underpins a thin CLI and renderer. There is no YAML manifest
parser or new dependency yet. The CLI accepts repeated flags:

```bash
localbench stack --model reasoning=phi4-mini:latest \
  --model embeddings=nomic-embed-text:latest
```

Each value uses `ROLE=MODEL`. `--context`, `--kv-cache-type`, and `--json` are also available;
context and KV-cache precision currently apply uniformly to every role. A YAML/TOML manifest
remains future work.

`localbench.stack.plan_stack` answers one narrow question: **given a set of named workload
roles, each backed by an existing single-model
[`ModelMemoryEstimate`](memory-estimation.md), does loading all of them simultaneously fit in
one available-memory snapshot?** It does not answer how fast the stack would run — concurrent
inference performance is unmeasured and this module never extrapolates it from single-model
benchmark numbers.

The shared snapshot is current available memory, not installed RAM. A model already resident in
memory may therefore make the plan conservative because its observed footprint is already
absent from available memory while the estimate also includes loading it. LocalBench does not
currently infer residency or subtract shared mappings.

## Shape

```text
for each role:  weights + KV cache + runtime overhead  = per-role working memory
sum of all roles' working memory                        = total working memory
                                                         + one stack-level safety headroom
                                                         = total required memory
total required memory vs. one shared available-memory snapshot = viability
```

## Design rules

### Safety headroom is applied once per stack, not once per model

A single-model [`ModelMemoryEstimate`](memory-estimation.md) already carries its own 2 GiB
safety headroom for the OS and background activity. That reasoning does not multiply just
because more models are loaded at once — the OS and background processes are still the same
system. `plan_stack` therefore uses each estimate's `working_memory_bytes` (weights + KV cache
+ runtime overhead, headroom excluded) and adds exactly one stack-level headroom
(`STACK_SAFETY_HEADROOM_BYTES`, same 2 GiB magnitude as the single-model default, but applied
once) to the sum.

### Per-model runtime overhead stays visible

Each role's contribution — weights, KV cache, and runtime overhead — is reported individually
in `StackPlan.models`, not collapsed into an opaque stack total. A user should be able to see
which role is expensive and why.

### All estimates are evaluated against one shared snapshot

`plan_stack` takes `available_memory_bytes` as an explicit argument rather than trusting each
estimate's own embedded `available_memory` field, because those estimates may have been
collected at different times and could disagree. If a role's own estimate was computed against
a different snapshot than the one passed to the plan, that role carries an explicit warning
noting it may be stale; the plan itself always reports figures against the one shared snapshot
it was given.

### Duplicate models are refused, not deduplicated

Whether two simultaneously loaded instances of the same model would share memory (e.g. via
mmap'd weights) is runtime- and OS-dependent and is not modeled here. Rather than silently
counting a duplicated model once (undercounting) or twice (possibly overcounting), `plan_stack`
raises `StackPlanError` with code `duplicate_model` whenever the same model name (compared
case-insensitively) appears under more than one role. Duplicate role names are refused the same
way, under code `duplicate_role`.

### Concurrent performance is out of scope

`StackPlan` has no throughput, latency, or contention field. The design assumption is stated
explicitly in `StackPlan.assumptions`: this plan establishes memory feasibility only. Measuring
real multi-model contention is later Phase 7 work, once the fit question above is trustworthy.

## Structured errors

`plan_stack` raises `StackPlanError(code, message)` for:

| Code | Cause |
|---|---|
| `empty_stack` | No workload roles were supplied. |
| `memory_unavailable` | `available_memory_bytes` is not positive. |
| `invalid_safety_headroom` | A negative `safety_headroom_bytes` override was supplied. |
| `invalid_role` | A role name is blank. |
| `invalid_working_memory` | An estimate's `working_memory_bytes` is not positive. |
| `invalid_memory_component` | Weights, KV cache, or runtime overhead is negative. |
| `inconsistent_working_memory` | The displayed components do not sum to the working-memory total. |
| `duplicate_role` | The same role name appears more than once. |
| `duplicate_model` | The same model name appears under more than one role. |

## Viability classification

`StackPlan.viability` reuses the same `FitClassification` enum and the same ratio thresholds
(excellent ≤ 60%, good ≤ 80%, tight ≤ 95%, poor ≤ 100%, otherwise will-not-fit) as single-model
estimation, applied to `total_required_bytes / available_memory_bytes`, so a user reads one
consistent scale across `estimate` and the eventual `stack` command.
