"""Composition root for inference-runtime discovery."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from localbench.models import RuntimeDiscovery
from localbench.runtimes.base import RuntimeDetector
from localbench.runtimes.llama_cpp import LlamaCppDetector
from localbench.runtimes.ollama import OllamaDetector


def discover_runtimes(detectors: Iterable[RuntimeDetector] | None = None) -> RuntimeDiscovery:
    active_detectors = (
        tuple(detectors) if detectors is not None else (OllamaDetector(), LlamaCppDetector())
    )
    return RuntimeDiscovery(
        collected_at=datetime.now(UTC),
        runtimes=tuple(detector.detect() for detector in active_detectors),
    )

