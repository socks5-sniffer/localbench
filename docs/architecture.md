# Milestone 0.1 architecture

LocalBench separates observation, structured data, and presentation:

```text
OS / CPU / memory / GPU collectors
                 |
            SystemProfile
                 |
        Rich renderer / JSON encoder
                 |
              Typer CLI
```

Collectors return typed dataclasses and never print. A failed optional observation produces a
warning and an empty or null value instead of failing the entire profile. Presentation code
does not inspect the host directly.

## Support boundary

Windows and Linux are the supported 0.1 targets. macOS uses standard system tools on a
best-effort basis. Other platforms still return OS, CPU, and memory observations when Python
and psutil support them; GPU detection reports a warning.

GPU discovery is intentionally conservative. It reports adapters visible to standard OS
interfaces but does not infer that a CUDA, ROCm, DirectML, or other inference runtime is
installed. `no adapters found`, `unsupported platform`, and `collector failed` are distinct
outcomes through the GPU list and warnings. VRAM and driver-version enrichment sources are
documented separately in [`docs/gpu-detection.md`](gpu-detection.md).

## JSON contract

The root object contains:

- `schema_version`: currently `1`
- `collected_at`: UTC ISO 8601 timestamp
- `os`, `cpu`, and `memory`: required profile objects
- `gpus`: zero or more adapter objects
- `warnings`: collection warnings safe to display to the user

Memory quantities use integer bytes. Missing observations are JSON `null`. Every profile
section includes a `source` describing how the operating system reported the value. Breaking
schema changes require a new `schema_version`.

Hardware values are observations, not performance claims. Model-fit estimates, runtime
detection, and AI-readiness scoring are outside Milestone 0.1.

