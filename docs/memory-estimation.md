# Memory-estimation methodology

Milestone 0.3 answers a narrow question: how much system memory should one installed Ollama
model require at a specified context length?

```text
reported model size
+ calculated KV cache
+ inferred runtime overhead
= working memory
+ inferred safety headroom
= estimated total
```

The command is:

```bash
localbench estimate MODEL [--context TOKENS] [--kv-cache-type f16|q8_0|q4_0]
```

Ollama must be running because LocalBench reads installed models from `/api/tags` and detailed
architecture metadata from `/api/show`. It does not inspect model files or download metadata.

## Components

### Weights — reported

The model byte size reported by Ollama's `/api/tags` response is used as the initial weight
footprint. LocalBench displays this as reported rather than measured runtime memory.

### KV cache — calculated

For one active sequence:

```text
layers × KV heads × (key length + value length) × context tokens × bytes per element
```

Key and value lengths come directly from Ollama `model_info` when present. For conventional
attention architectures that omit those fields, LocalBench derives head dimension as embedding
length divided by attention-head count and explicitly reports that basis. If the required
metadata is unavailable, estimation stops rather than silently substituting a generic number.

Granite 4 H hybrid models require a different calculation because only 4 of their 40 layers
use attention; the other 36 are Mamba-2 recurrent layers. Ollama exposes their per-layer
KV-head array as `null` in JSON. LocalBench recognizes only the exact published Tiny/1B
signature (40 total layers, 4 attention layers, 4 KV heads, 128-wide heads, and matching SSM
dimensions). It calculates:

```text
attention cache = 4 × 4 × (128 + 128) × context tokens × configured KV bytes
recurrent state = 36 × (convolution state + SSM state) × 4-byte f32
```

The recurrent state is fixed per active sequence rather than growing with context. A different
or future `granitehybrid` signature still fails with `insufficient_metadata`; LocalBench does
not generalize these constants to an architecture it has not identified exactly.

The signature and layer layout come from IBM's published
[Granite 4.0 H Tiny model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny).
The recurrent-state formulas mirror llama.cpp's
[`llama_hparams::n_embd_r()` and `n_embd_s()`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-hparams.cpp).

Cache precision assumptions:

| Type | Bytes per element |
|---|---:|
| `f16` | 2 |
| `q8_0` | 1 |
| `q4_0` | 0.5 |

Ollama defaults to `f16`; selecting a quantized type tells LocalBench what configuration to
model. It does not alter Ollama's configuration. Quantized KV-cache support depends on the model
and Ollama runtime configuration.

### Runtime overhead — inferred

The initial policy uses 10% of reported model size with a 512 MiB floor. This is a deliberately
visible heuristic and should eventually be replaced by runtime- and backend-specific measured
data.

### Safety headroom — inferred

A fixed 2 GiB reserve is included for the operating system and background activity. This policy
will become configurable after real benchmark data supports better defaults.

## Fit classification

The estimated total is divided by currently available system memory:

| Share of available memory | Classification |
|---|---|
| up to 60% | Excellent |
| over 60%, up to 80% | Good |
| over 80%, up to 95% | Tight |
| over 95%, up to 100% | Poor |
| over 100% | Will not fit |

Available memory is a measured snapshot and can change immediately as other applications start
or stop.

## GPU offload estimate

When a GPU with measured dedicated VRAM is detected, the estimator also reports a coarse
layer-offload projection:

```text
usable VRAM = dedicated VRAM − max(256 MiB, 10% of dedicated VRAM)
per-layer bytes = (weights + KV cache) ÷ model layer count
offloadable layers = min(layer count, usable VRAM ÷ per-layer bytes)
```

Layers are assumed uniform in size — a simplification that undercounts capacity for
architectures with an outsized embedding or output layer. The offloadable-layer count is
classified as:

| Offloadable layers | Classification |
|---|---|
| all layers | Full offload |
| some, not all | Partial offload |
| none | CPU only |

The highest-VRAM detected GPU is used when more than one is present. This does not reproduce
Ollama's actual layer-placement algorithm, does not model multi-GPU splitting or other
processes already holding VRAM, and does not model unified-memory (e.g. Apple Silicon)
behavior, where GPU and system memory are the same pool. When no GPU with measured dedicated
VRAM is detected, `gpu_offload` is omitted (`null` in JSON) rather than guessed at.

## Current limitations

- One active sequence; concurrent requests multiply KV-cache demand.
- GPU offload is a coarse per-layer approximation, not Ollama's real placement algorithm, and
  unified-memory behavior is not modeled separately.
- Runtime overhead and safety headroom are policy estimates, not measurements.
- Multimodal projector memory is not modeled. Recurrent state is modeled only for the exact
  supported Granite 4 H signature described above; other recurrent architectures fail closed.
