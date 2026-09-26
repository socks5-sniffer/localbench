# Runtime discovery

Milestone 0.2 introduced a runtime adapter protocol and the first adapter for Ollama. A second
adapter for llama.cpp's `llama-server` was added later to prove the `RuntimeDetector` protocol
generalizes, not merely to add coverage — see below.

```text
RuntimeDetector protocol
      /            \
OllamaDetector   LlamaCppDetector
  /       \          /        \
CLI     fixed      CLI       fixed
presence loopback  presence  loopback
        API                  API
          \                  /
           RuntimeDiscovery
                    |
        Rich renderer / JSON encoder
```

`localbench runtimes` distinguishes three observations per adapter:

- whether its CLI is on `PATH`;
- whether its local service responds correctly on its fixed loopback port;
- which models that service reports.

Executable paths are deliberately not included in output because filesystem paths can carry
personal information. API and subprocess operations are read-only and timeout-bounded. Every
adapter talks only to a fixed `127.0.0.1` port and ignores environment variables that could
redirect it to a remote host, so discovery cannot unexpectedly contact a remote service.

The local API version is preferred when both the CLI and service report versions because it
describes the process serving the model inventory. Failure to reach a runtime is normal when it
is not installed; partial and malformed responses become warnings without terminating discovery.

## Ollama

Talks to the fixed loopback API at `http://127.0.0.1:11434` and lists installed models through
`GET /api/tags`. Model metadata includes the reported name, byte size, digest, modification
time, format, family, parameter size, and quantization level. LocalBench does not inspect model
files or expose their paths — Ollama's model names are clean tags (`gemma3:4b`), not filesystem
paths.

## llama.cpp

Talks to `llama-server`'s fixed loopback API at `http://127.0.0.1:8080` (its documented
default port) via its OpenAI-compatible `GET /v1/models`, and reads build version from
`GET /props`'s `build_info` field, falling back to `llama-server --version` when the service
isn't reachable but the CLI is.

**A real difference from Ollama's model, not papered over:** llama.cpp has no installed-model
registry. A `llama-server` process is started against exactly one GGUF file (or a small preset
list), so `/v1/models` describes only what that process currently has loaded — not everything
on disk. LocalBench reports that as the runtime's "installed models" list rather than
pretending to enumerate a catalog that does not exist.

**Privacy note, unlike Ollama:** unless the server was started with an `--alias`, the `id`
field `/v1/models` returns is typically the model's local filesystem path. LocalBench reduces
values containing either POSIX or Windows path separators to basename-only before they enter
the runtime profile, and emits a generic warning that sanitization occurred. The original path
never appears in human or JSON output.

Model metadata is limited to what `/v1/models`' `meta` object reports: byte size and a
parameter-count-derived size string (e.g. `4.3B`). Digest, modification time, family, and
quantization level are not available through this API and are reported as `null`.
