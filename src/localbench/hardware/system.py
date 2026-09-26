"""Operating-system observations."""

from __future__ import annotations

import platform

from localbench.models import OSProfile


def collect_os() -> OSProfile:
    return OSProfile(
        family=platform.system() or "Unknown",
        release=platform.release() or "Unknown",
        version=platform.version() or "Unknown",
        architecture=platform.machine() or "Unknown",
    )

