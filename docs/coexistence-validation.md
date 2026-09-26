# Local coexistence validation

This note preserves real-machine validation results that are not currently retained by
`localbench coexist`. It is evidence from one machine, not a performance guarantee for other
hardware.

## Granite 1.5B hybrid plus Granite 3.7B dense

Environment:

- Intel Core i5-8265U (4 cores / 8 threads), Intel UHD 620 integrated graphics, approximately
  24 GiB RAM, CPU-only inference
- Ollama 0.33.2
- `granite4:1b-h` Q8_0 as role `hybrid`
- `granite4.2:3b` Q4_K_M as role `dense`
- Four concurrent trials; every stream completed the 64-token target
- Peak system CPU was 100% in every trial

The command used retained, comparable `benchmark_generation_v2` solo baselines.
The IDs below are placeholders; substitute your own saved run IDs:

- `granite4:1b-h`: `HYBRID_RUN_ID`
- `granite4.2:3b`: `DENSE_RUN_ID`

```powershell
uv run localbench coexist `
  --model hybrid=granite4:1b-h `
  --model dense=granite4.2:3b `
  --baseline hybrid=HYBRID_RUN_ID `
  --baseline dense=DENSE_RUN_ID `
  --json
```

### Raw trial summary

Load time and TTFT are seconds; generation is tokens per second. Peak RAM is the increase in
system-wide used RAM from the pre-run sample.

| Trial | Hybrid load | Hybrid TTFT | Hybrid generation | Dense load | Dense TTFT | Dense generation | Wall time | Peak RAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.69 | 27.42 | 10.05 | 15.48 | 17.19 | 6.69 | 33.79 | 3.75 GiB |
| 2 | 1.84 | 3.24 | 5.38 | 7.27 | 10.89 | 5.94 | 21.67 | 3.94 GiB |
| 3 | 1.83 | 3.27 | 5.75 | 6.97 | 9.91 | 5.88 | 20.69 | 3.98 GiB |
| 4 | 1.91 | 3.34 | 5.71 | 7.07 | 10.30 | 5.87 | 21.20 | 3.93 GiB |
| Median | 1.87 | 3.30 | 5.73 | 7.17 | 10.59 | 5.91 | 21.44 | 3.94 GiB |

Solo baseline generation was 8.67 tok/s for the hybrid model and 6.41 tok/s for the dense
model. Median concurrent generation was therefore approximately 34% slower for
`granite4:1b-h` and 8% slower for `granite4.2:3b`. Each retained solo baseline is a single
observation, so the percentages are useful directional evidence rather than confidence
intervals.

### Interpretation and limits

The pair fits in memory and completed reliably, but CPU contention is real after loading:
steady generation did not merely inherit the startup delay. The dense model remained fairly
close to its solo generation rate, while the hybrid model lost roughly one-third at the
four-trial median.

Trial 1 was a startup-timing outlier, especially for the hybrid role. Its cause was not
isolated, so it remains in the recorded range and median rather than being discarded.

These are repeated cold-start trials, not warm-resident trials. The current coexistence
contract deliberately unloads every model before each run so results remain comparable to the
solo cold-start baselines. Generation rate is measured only after a model has loaded, which
separates the steady generation phase from load time and TTFT, but a future explicitly labeled
warm-resident mode would be required to measure already-loaded service behavior directly.
