from __future__ import annotations

from localbench.runtimes import discover_runtimes


def test_default_detectors_cover_ollama_and_llama_cpp() -> None:
    """No `detectors` override: the real composition root must run both adapters.

    Uses real detection (no mocking) deliberately — this is the one test that exercises the
    actual default tuple `discover_runtimes()` falls back to, so it must call the real thing
    rather than an injected fake. Detection results vary by machine; only the runtime names
    (not their status) are asserted.
    """
    discovery = discover_runtimes()

    assert [runtime.name for runtime in discovery.runtimes] == ["ollama", "llama_cpp"]
