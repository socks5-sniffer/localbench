"""Composition root for a complete system profile."""

from __future__ import annotations

from datetime import UTC, datetime

from localbench.hardware.cpu import collect_cpu
from localbench.hardware.gpu import collect_gpus
from localbench.hardware.memory import collect_memory
from localbench.hardware.system import collect_os
from localbench.models import SystemProfile


def collect_system_profile() -> SystemProfile:
    os_profile = collect_os()
    gpus, warnings = collect_gpus(os_profile.family)
    return SystemProfile(
        collected_at=datetime.now(UTC),
        os=os_profile,
        cpu=collect_cpu(os_profile.family),
        memory=collect_memory(),
        gpus=gpus,
        warnings=warnings,
    )

