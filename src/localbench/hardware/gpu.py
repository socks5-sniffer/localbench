"""Conservative, best-effort GPU discovery using operating-system tools.

Adapter presence always comes from a primary, OS-native source (`Get-CimInstance` on
Windows, `lspci` on Linux, `system_profiler` on macOS). VRAM and driver-version fields
are then enriched, best-effort, from additional sources that are cheap, safe (argument
arrays, no `shell=True`, bounded timeouts), and never treated as required: if an
enrichment source is missing or fails, the adapter is still reported with `null` for
whatever field it could not fill in. A missing enrichment source is never presented as
a collection failure, and it is never used to infer that any inference runtime or
compute backend is installed.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from localbench.models import GPUProfile

COMMAND_TIMEOUT_SECONDS = 5

_LSPCI_PATTERN = re.compile(
    r"^(?P<address>[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s+"
    r"(?:VGA compatible controller|3D controller|Display controller):\s*(?P<name>.+)$"
)
_CARD_DIR_PATTERN = re.compile(r"^card\d+$")
_PCI_SLOT_PATTERN = re.compile(
    r"^PCI_SLOT_NAME=(?:[0-9a-fA-F]{4}:)?([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s*$"
)
_MACOS_MEMORY_STRING_PATTERN = re.compile(r"^\s*([\d.]+)\s*(GB|MB|KB)\s*$", re.IGNORECASE)
_MACOS_MEMORY_UNIT_MULTIPLIERS = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
_MACOS_VRAM_KEYS = ("sppci_vram", "spdisplays_vram")


def _vendor(name: str) -> str | None:
    lowered = name.lower()
    for needle, vendor in (
        ("nvidia", "NVIDIA"),
        ("advanced micro devices", "AMD"),
        ("amd", "AMD"),
        ("radeon", "AMD"),
        ("intel", "Intel"),
        ("apple", "Apple"),
    ):
        if needle in lowered:
            return vendor
    return None


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def _normalize_pci_address(raw: str) -> str | None:
    """Reduce a PCI address with an optional domain prefix to lspci's `BB:DD.F` form."""
    parts = raw.strip().split(":")
    if len(parts) < 2:
        return None
    bus, device_function = parts[-2], parts[-1]
    if not re.fullmatch(r"[0-9a-fA-F]{2}", bus):
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{2}\.[0-9a-fA-F]", device_function):
        return None
    return f"{bus.lower()}:{device_function.lower()}"


def _windows_gpus() -> tuple[GPUProfile, ...]:
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    result = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    if result.returncode != 0:
        return _windows_registry_gpus()
    if not result.stdout.strip():
        return ()
    decoded: Any = json.loads(result.stdout)
    records = decoded if isinstance(decoded, list) else [decoded]
    profiles: list[GPUProfile] = []
    for record in records:
        if not isinstance(record, dict) or not record.get("Name"):
            continue
        memory = record.get("AdapterRAM")
        profiles.append(
            GPUProfile(
                name=str(record["Name"]),
                vendor=_vendor(str(record["Name"])),
                dedicated_memory_bytes=int(memory) if isinstance(memory, int) else None,
                shared_memory_bytes=None,
                driver_version=str(record["DriverVersion"])
                if record.get("DriverVersion")
                else None,
                source="windows_cim",
            )
        )
    return tuple(profiles)


def _windows_registry_gpus() -> tuple[GPUProfile, ...]:
    """Fall back to the display-driver registry when Windows CIM is restricted."""
    try:
        winreg: Any = importlib.import_module("winreg")
    except ImportError:
        return ()

    root_path = r"SYSTEM\CurrentControlSet\Control\Video"
    profiles: list[GPUProfile] = []
    seen: set[tuple[str, str | None]] = set()
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path) as root:
            index = 0
            while True:
                try:
                    adapter_key = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                path = f"{root_path}\\{adapter_key}\\0000"
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                        name_value: Any = winreg.QueryValueEx(key, "DriverDesc")[0]
                        name = str(name_value).strip()
                        if not name:
                            continue
                        try:
                            driver_value: Any = winreg.QueryValueEx(key, "DriverVersion")[0]
                            driver = str(driver_value).strip() or None
                        except OSError:
                            driver = None
                        try:
                            memory_value: Any = winreg.QueryValueEx(
                                key, "HardwareInformation.MemorySize"
                            )[0]
                            memory = int(memory_value) if isinstance(memory_value, int) else None
                        except OSError:
                            memory = None
                except OSError:
                    continue
                identity = (name, driver)
                if identity in seen:
                    continue
                seen.add(identity)
                profiles.append(
                    GPUProfile(
                        name=name,
                        vendor=_vendor(name),
                        dedicated_memory_bytes=memory,
                        shared_memory_bytes=None,
                        driver_version=driver,
                        source="windows_registry",
                    )
                )
    except OSError:
        return ()
    return tuple(profiles)


def _linux_sysfs_vram_by_pci_address(drm_root: Path = Path("/sys/class/drm")) -> dict[str, int]:
    """Best-effort VRAM sizes keyed by short PCI address (e.g. ``01:00.0``).

    Reads the kernel driver's own ``mem_info_vram_total`` sysfs file (exposed by
    amdgpu, nouveau, and the proprietary nvidia driver for cards it owns) and matches
    it to a PCI address via the adjacent ``uevent`` file's ``PCI_SLOT_NAME`` line.
    Returns an empty mapping wherever sysfs is unavailable or a card exposes neither
    file — this is best-effort enrichment, not a required discovery path.
    """
    mapping: dict[str, int] = {}
    try:
        entries = sorted(drm_root.iterdir())
    except OSError:
        return mapping
    for entry in entries:
        if not _CARD_DIR_PATTERN.match(entry.name):
            continue
        device_dir = entry / "device"
        try:
            vram_text = (device_dir / "mem_info_vram_total").read_text(encoding="utf-8")
            vram_bytes = int(vram_text.strip())
        except (OSError, ValueError):
            continue
        try:
            uevent_text = (device_dir / "uevent").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in uevent_text.splitlines():
            match = _PCI_SLOT_PATTERN.match(line.strip())
            if match:
                mapping[match.group(1).lower()] = vram_bytes
                break
    return mapping


def _nvidia_smi_info() -> dict[str, tuple[int | None, str | None]]:
    """Best-effort VRAM (bytes) and driver version per PCI address, from ``nvidia-smi``.

    Silently returns an empty mapping when the tool is missing, times out, or fails —
    its absence never means NVIDIA hardware is absent, only that this enrichment
    source could not run.
    """
    try:
        result = _run(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    info: dict[str, tuple[int | None, str | None]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        bus_id, memory_mib, driver = parts
        address = _normalize_pci_address(bus_id)
        if address is None:
            continue
        try:
            memory_bytes: int | None = int(memory_mib) * 1024 * 1024
        except ValueError:
            memory_bytes = None
        info[address] = (memory_bytes, driver or None)
    return info


def _linux_gpus() -> tuple[GPUProfile, ...]:
    result = _run(["lspci"])
    if result.returncode != 0:
        raise RuntimeError("lspci query failed")

    vram_by_address = _linux_sysfs_vram_by_pci_address()
    nvidia_by_address = _nvidia_smi_info()

    profiles: list[GPUProfile] = []
    for line in result.stdout.splitlines():
        match = _LSPCI_PATTERN.search(line)
        if not match:
            continue
        address = match.group("address").lower()
        name = match.group("name").strip()

        dedicated_memory_bytes = vram_by_address.get(address)
        extra_sources: list[str] = ["sysfs"] if dedicated_memory_bytes is not None else []

        driver_version: str | None = None
        nvidia_entry = nvidia_by_address.get(address)
        if nvidia_entry is not None:
            nvidia_memory, nvidia_driver = nvidia_entry
            if dedicated_memory_bytes is None and nvidia_memory is not None:
                dedicated_memory_bytes = nvidia_memory
                extra_sources.append("nvidia_smi")
            if nvidia_driver is not None:
                driver_version = nvidia_driver
                if "nvidia_smi" not in extra_sources:
                    extra_sources.append("nvidia_smi")

        profiles.append(
            GPUProfile(
                name=name,
                vendor=_vendor(name),
                dedicated_memory_bytes=dedicated_memory_bytes,
                shared_memory_bytes=None,
                driver_version=driver_version,
                source="+".join(["linux_lspci", *extra_sources]),
            )
        )
    return tuple(profiles)


def _parse_macos_memory_string(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = _MACOS_MEMORY_STRING_PATTERN.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    multiplier = _MACOS_MEMORY_UNIT_MULTIPLIERS[match.group(2).upper()]
    return int(amount * multiplier)


def _macos_gpus() -> tuple[GPUProfile, ...]:
    result = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if result.returncode != 0:
        raise RuntimeError("system_profiler display query failed")
    decoded: Any = json.loads(result.stdout)
    records = decoded.get("SPDisplaysDataType", []) if isinstance(decoded, dict) else []
    profiles: list[GPUProfile] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("sppci_model") or record.get("_name")
        if not name:
            continue
        name_text = str(name)

        dedicated_memory_bytes: int | None = None
        for key in _MACOS_VRAM_KEYS:
            dedicated_memory_bytes = _parse_macos_memory_string(record.get(key))
            if dedicated_memory_bytes is not None:
                break

        profiles.append(
            GPUProfile(
                name=name_text,
                vendor=_vendor(name_text),
                dedicated_memory_bytes=dedicated_memory_bytes,
                # Apple Silicon's unified memory has no separate "VRAM" byte count to
                # report, and system_profiler exposes no driver-version concept at
                # all on macOS; both stay null by design, not by omission.
                shared_memory_bytes=None,
                driver_version=None,
                source="macos_system_profiler",
            )
        )
    return tuple(profiles)


def collect_gpus(system: str) -> tuple[tuple[GPUProfile, ...], tuple[str, ...]]:
    collectors = {"Windows": _windows_gpus, "Linux": _linux_gpus, "Darwin": _macos_gpus}
    collector = collectors.get(system)
    if collector is None:
        return (), (f"GPU detection is unsupported on {system or 'this platform'}.",)
    try:
        profiles = collector()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return (), (f"GPU detection failed on {system}; GPU information is unavailable.",)
    if not profiles:
        return (), ("No GPU adapters were reported by the operating system.",)
    return profiles, ()
