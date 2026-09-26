"""Cross-platform CPU observations."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from localbench.models import CPUProfile


def _windows_cpu_model() -> str | None:
    try:
        winreg: Any = importlib.import_module("winreg")
        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            value: Any = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except (ImportError, OSError):
        return None
    return str(value).strip() or None


def _linux_cpu_model(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str | None:
    try:
        candidates: dict[str, str] = {}
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            normalized_key = key.strip().lower()
            if (
                separator
                and normalized_key in {"model name", "hardware", "processor"}
                and value.strip()
                and not value.strip().isdigit()
            ):
                candidates.setdefault(normalized_key, value.strip())
        for preferred_key in ("model name", "hardware", "processor"):
            if preferred_key in candidates:
                return candidates[preferred_key]
    except OSError:
        return None
    return None


def _macos_cpu_model() -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_cpu(system: str | None = None) -> CPUProfile:
    family = system or platform.system()
    source = "python_platform"
    model = platform.processor().strip() or None

    if family == "Windows":
        registry_model = _windows_cpu_model()
        model = registry_model or os.environ.get("PROCESSOR_IDENTIFIER", "").strip() or model
        source = "windows_registry" if registry_model else "windows_environment"
    elif family == "Linux":
        linux_model = _linux_cpu_model()
        model = linux_model or model
        source = "proc_cpuinfo" if linux_model else "python_platform"
    elif family == "Darwin":
        macos_model = _macos_cpu_model()
        model = macos_model or model
        source = "sysctl" if macos_model else "python_platform"

    return CPUProfile(
        model=model,
        architecture=platform.machine() or "Unknown",
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        source=source,
    )
