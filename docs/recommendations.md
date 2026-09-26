# Evidence-based model recommendations

`localbench recommend` answers a deliberately bounded question:

> Of the installed models benchmarked under the current LocalBench workload and Ollama
> version, which has the best measured speed and estimated memory fit for this performance
> policy?

It does not claim to measure answer quality, coding correctness, reasoning ability, factuality,
or vision support. A fast model can still be wrong. The command makes that limitation part of
both human and JSON output rather than burying it in documentation.

## Usage

```bash
localbench recommend
localbench recommend --use-case low-latency
localbench recommend --use-case rag --context 16384 --kv-cache-type q8_0
localbench recommend --use-case coding --json
```

Supported policies are `general-chat`, `coding`, `reasoning`, `rag`, `robotics`, `agents`,
`low-latency`, and `long-context`. Semantic names such as `coding` and `reasoning` do not imply
that LocalBench has evaluated task quality; in 1.0 they select the balanced operational policy.

## Evidence eligibility

A model is ranked only when all of the following are true:

- it is currently installed in Ollama;
- its newest eligible saved batch uses the current quick workload's complete canonical
  measurement signature, including the current Ollama version;
- every planned batch run is present and reached the generation-token target;
- every run completed its requested cold unload and has positive generation rate,
  prompt-processing rate, and TTFT measurements; and
- LocalBench can produce a current memory estimate at the requested context length.

Anything else is listed as excluded. This prevents an old runtime, superseded prompt, partial
batch, or unmeasured model from silently entering a ranking.

Repeated batches use the median of generation rate, prompt-processing rate, and TTFT. The
memory estimate uses one system snapshot shared by every candidate. The speed benchmark remains
the versioned quick workload at 2K context; `--context` controls the current memory estimate,
not the saved benchmark workload.

## Transparent scoring

Generation and prompt rates use min-max normalization where higher is better. TTFT uses the
same normalization inverted because lower is better. If all candidates have the same value,
they all receive 100 for that component. Memory fit maps to a fixed score:

| Fit | Score |
|---|---:|
| Excellent | 100 |
| Good | 75 |
| Tight | 50 |
| Poor | 25 |
| Will not fit | 0 |

The weighted score is the sum of those four visible components:

| Policy | Generation | Prompt | TTFT | Memory |
|---|---:|---:|---:|---:|
| General chat, coding, reasoning, agents | 35% | 15% | 30% | 20% |
| Robotics, low latency | 20% | 10% | 50% | 20% |
| RAG | 20% | 40% | 15% | 25% |
| Long context | 15% | 25% | 15% | 45% |

A model classified `will_not_fit` always sorts below models that fit, regardless of its speed
score. Scores are relative to the eligible candidates in one report and must not be compared
across different reports as if they were universal grades.
