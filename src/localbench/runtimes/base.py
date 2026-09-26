"""Interfaces and errors shared by runtime adapters."""

from __future__ import annotations

from typing import Protocol

from localbench.models import RuntimeProfile


class RuntimeUnavailableError(RuntimeError):
    """The local runtime service could not be reached."""


class RuntimeResponseError(RuntimeError):
    """The runtime returned an invalid or unsupported response."""


class RuntimeDetector(Protocol):
    name: str

    def detect(self) -> RuntimeProfile:
        """Return one runtime observation without raising for expected absence."""
        ...

