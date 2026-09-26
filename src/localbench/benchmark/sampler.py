"""Low-overhead system-wide resource sampling during a benchmark."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class ResourceSample:
    baseline_ram_used_bytes: int
    peak_ram_used_bytes: int
    peak_cpu_percent: float


class SystemSampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak_ram = 0
        self._peak_cpu = 0.0

    def start(self) -> None:
        self._baseline = psutil.virtual_memory().used
        self._peak_ram = self._baseline
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak_ram = max(self._peak_ram, psutil.virtual_memory().used)
            self._peak_cpu = max(self._peak_cpu, psutil.cpu_percent(interval=None))

    def stop(self) -> ResourceSample:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 5))
        self._peak_ram = max(self._peak_ram, psutil.virtual_memory().used)
        return ResourceSample(self._baseline, self._peak_ram, self._peak_cpu)

