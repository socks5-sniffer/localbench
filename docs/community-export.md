# Community export contract (design prerequisite)

LocalBench may eventually let people contribute benchmark data, but local operation must remain
complete without an account, network connection, or telemetry. This document defines the data
boundary before an uploader or service exists. It is intentionally a contract, not an
implementation plan.

The first local-only implementation now lives in `localbench.community`. It provides typed
allow-listed envelope values, coarse memory bucketing, eligibility assessment, and deterministic
JSON preview rendering. It has no CLI command, persistent consent flag, filesystem write,
endpoint, uploader, or network code.

## Consent and transport rules

- Collection and upload are off by default. Installing, profiling, estimating, benchmarking,
  or enabling another feature must never imply consent.
- The first implementation should create a local export file and a byte-for-byte preview. A
  later upload action must send exactly the previewed envelope, with a separate explicit
  confirmation. It must not enrich the payload after confirmation.
- Consent is per submission. A future persistent opt-in may reduce prompts, but it must remain
  visible, reversible, and narrower than blanket application telemetry.
- There is no stable installation, machine, account, or advertising identifier. Each envelope
  receives a random one-use submission ID solely for idempotent retry.
- A server necessarily observes the connection's IP address. The client and privacy notice must
  not promise otherwise. The service must not place it in the benchmark record and should
  minimize or disable infrastructure-log retention where operationally possible.
- A successful upload should return a random deletion receipt. Keeping that receipt locally
  permits deletion without creating a user account.

## Envelope structure

The transport envelope and each record kind have independent integer schema versions. Unknown
major versions are refused rather than guessed at.

```json
{
  "envelope_version": 1,
  "submission_id": "one-use random UUID",
  "created_month": "2026-08",
  "localbench_version": "0.6.0",
  "records": [
    {
      "record_kind": "benchmark_batch",
      "record_version": 1,
      "hardware": {},
      "model": {},
      "measurement_signature": {},
      "batch": {}
    }
  ]
}
```

Exact timestamps are reduced to a month. Local run IDs, batch IDs, database keys, and paths are
never copied into the envelope; record-local random IDs may connect raw observations within one
submission only.

### Hardware facts

The standard privacy tier contains only facts needed to stratify results:

- OS family and architecture, but not build number, hostname, locale, or installation ID.
- Normalized CPU marketing model, physical/logical core counts, and architecture.
- Installed RAM rounded to a documented bucket (initially 4 GiB).
- Normalized GPU model, dedicated-memory bucket, and whether memory is unified/shared/dedicated
  when known. Driver version is included only when it materially defines the execution stratum.
- Execution backend facts actually observed for the run: CPU/GPU offload placement, thread
  count, and accelerator backend. Unknown stays `null`; it is never inferred from GPU presence.

The combination can still be identifying, especially for unusual hardware. The preview must
label these as quasi-identifiers. The service should suppress public groups below a documented
minimum cohort (initial recommendation: 10 submissions) and coarsen rare hardware buckets.

### Model identity

Model tags are mutable, while private/custom names may be sensitive. An aggregation-eligible
record therefore needs:

- content digest;
- format, quantization, reported parameter size, and weights bytes;
- normalized public registry name only when it can be resolved safely.

The digest is itself a model fingerprint and must be visible in the preview. A private or
unresolved model name is omitted, not uploaded as typed. Records without a digest may be
exported locally for the user but are not eligible for public model rankings.

### Measurement and batch data

The complete canonical `MeasurementSignature` is included; community aggregation must use
exact signature equality, including runtime version. Hardware and model identity are additional
stratification keys, not signature fields.

The batch object contains planned/completed counts and raw observations with only measurement
values: generation/prompt token counts and durations, TTFT, load/total/wall duration, and
system-wide baseline/peak RAM and peak CPU. It contains no prompt, response, arbitrary warning
text, local run ID, or exact timestamp. Derived medians and spread may be included for
convenience, but raw observations remain authoritative.

Only complete repeated batches whose rows share one signature, reached the generation target,
and completed the requested cold unload are aggregation-eligible. Singletons and incomplete or
invalid batches may be exported locally for inspection but must be labeled ineligible rather
than mixed into community distributions. Future stack estimates and measured contention runs
need separate `record_kind` contracts; estimates must never be pooled with measurements.

## Fields never placed in the envelope

- username, hostname, email, account identifiers, IP/MAC addresses, and serial numbers;
- filesystem paths, environment variables, shell history, or application/process lists;
- prompts, generated text, embeddings, tool calls, or model responses;
- local database paths, run IDs, batch IDs, or exact collection timestamps;
- arbitrary warning/error strings, which may contain unexpected local data;
- a persistent installation or machine fingerprint.

Structured, allow-listed quality codes may replace warning text when scientifically necessary.

## Eligibility and preview

Envelope construction is allow-list based: serialize a dedicated export type rather than remove
fields from `SystemProfile` or `BenchmarkResult`. Validation produces structured reasons such
as `missing_model_digest`, `unknown_execution_placement`, `incomplete_batch`, or
`mixed_measurement_signature`.

Existing benchmark rows do not contain a benchmark-time hardware snapshot, model digest, or
resolved execution placement. A current profile must not be silently attached to an old run.
Those rows remain valid local history but are not retroactively community-eligible. Capture the
required facts atomically with future eligible batches. The local envelope builder enforces this
boundary by requiring hardware and model facts to carry the source batch ID; it validates that
binding, then deliberately removes the local ID from serialized output.

A future `localbench telemetry preview` should show:

1. the exact formatted JSON that would be sent;
2. every quasi-identifier and why it is needed;
3. eligibility or exclusion reasons;
4. endpoint, retention policy, cohort-suppression rule, and deletion mechanism;
5. a statement that core LocalBench behavior is unchanged if the user declines.

## Aggregation rules

The service must partition results by measurement signature, model digest/quantization, and
material execution/hardware buckets before calculating statistics. Report cohort size, median,
P10/P90, and the exclusion policy. Never rank tiny cohorts, silently merge runtime versions, or
treat multiple batches from one submission as independent machines.

Public downloads should use the same documented schema with submission IDs removed or rotated,
rare cohorts suppressed, and a data dictionary explaining measured versus reported, calculated,
and inferred fields. Schema changes, retention changes, or new fields require renewed visible
review; consent to envelope v1 is not consent to an expanded v2.
