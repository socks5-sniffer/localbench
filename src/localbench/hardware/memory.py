"""Physical and swap memory observations."""

from __future__ import annotations

import psutil

from localbench.models import MemoryProfile


def collect_memory() -> MemoryProfile:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryProfile(
        total_bytes=virtual.total,
        available_bytes=virtual.available,
        used_bytes=virtual.used,
        utilization_percent=virtual.percent,
        swap_total_bytes=swap.total,
        swap_used_bytes=swap.used,
        swap_utilization_percent=swap.percent,
    )

