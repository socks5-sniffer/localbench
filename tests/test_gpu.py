from __future__ import annotations

import json
import subprocess
from pathlib import Path

from localbench.hardware import gpu


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _no_enrichment(monkeypatch: object) -> None:
    """Force both Linux enrichment sources empty so tests are deterministic even on
    a real Linux CI runner, which may have its own /sys/class/drm entries."""
    monkeypatch.setattr(gpu, "_linux_sysfs_vram_by_pci_address", lambda: {})  # type: ignore[attr-defined]
    monkeypatch.setattr(gpu, "_nvidia_smi_info", lambda: {})  # type: ignore[attr-defined]


def test_linux_gpu_parser(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)
    output = (
        "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n"
        "01:00.0 3D controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q]\n"
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Linux")

    assert warnings == ()
    assert [profile.vendor for profile in profiles] == ["Intel", "NVIDIA"]
    assert all(profile.source == "linux_lspci" for profile in profiles)


def test_linux_gpu_parser_handles_display_controller_label(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)
    output = "00:02.0 Display controller: Intel Corporation TigerLake-LP GT2 [Iris Xe]\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Linux")

    assert warnings == ()
    assert profiles[0].name == "Intel Corporation TigerLake-LP GT2 [Iris Xe]"


def test_linux_gpu_no_matching_lines_reports_no_adapters(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)
    output = "00:1f.3 Audio device: Intel Corporation Sunrise Point-LP HD Audio\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Linux")

    assert profiles == ()
    assert "No GPU adapters" in warnings[0]


def test_linux_gpu_reports_null_when_no_enrichment_source_available(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)
    output = "01:00.0 VGA compatible controller: NVIDIA Corporation AD104 [GeForce RTX 4070]\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Linux")

    assert profiles[0].dedicated_memory_bytes is None
    assert profiles[0].driver_version is None
    assert profiles[0].source == "linux_lspci"


def test_linux_gpu_enriches_vram_from_sysfs(monkeypatch: object) -> None:
    output = "01:00.0 VGA compatible controller: AMD Radeon RX 6700 XT\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu, "_linux_sysfs_vram_by_pci_address", lambda: {"01:00.0": 12 * 1024**3}
    )
    monkeypatch.setattr(gpu, "_nvidia_smi_info", lambda: {})  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Linux")

    assert profiles[0].dedicated_memory_bytes == 12 * 1024**3
    assert profiles[0].driver_version is None
    assert profiles[0].source == "linux_lspci+sysfs"


def test_linux_gpu_enriches_vram_and_driver_from_nvidia_smi(monkeypatch: object) -> None:
    output = "01:00.0 3D controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q]\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]
    monkeypatch.setattr(gpu, "_linux_sysfs_vram_by_pci_address", lambda: {})  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu, "_nvidia_smi_info", lambda: {"01:00.0": (8 * 1024**3, "550.54.14")}
    )

    profiles, _warnings = gpu.collect_gpus("Linux")

    assert profiles[0].dedicated_memory_bytes == 8 * 1024**3
    assert profiles[0].driver_version == "550.54.14"
    assert profiles[0].source == "linux_lspci+nvidia_smi"


def test_linux_gpu_prefers_sysfs_vram_but_still_takes_nvidia_driver(
    monkeypatch: object,
) -> None:
    output = "01:00.0 3D controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q]\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu, "_linux_sysfs_vram_by_pci_address", lambda: {"01:00.0": 8 * 1024**3}
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu, "_nvidia_smi_info", lambda: {"01:00.0": (8100 * 1024**2, "550.54.14")}
    )

    profiles, _warnings = gpu.collect_gpus("Linux")

    assert profiles[0].dedicated_memory_bytes == 8 * 1024**3
    assert profiles[0].driver_version == "550.54.14"
    assert profiles[0].source == "linux_lspci+sysfs+nvidia_smi"


def test_linux_sysfs_vram_reads_matching_card(tmp_path: Path) -> None:
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_info_vram_total").write_text("17179869184\n", encoding="utf-8")
    (device_dir / "uevent").write_text(
        "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:01:00.0\nMODALIAS=x\n", encoding="utf-8"
    )

    mapping = gpu._linux_sysfs_vram_by_pci_address(tmp_path)

    assert mapping == {"01:00.0": 17179869184}


def test_linux_sysfs_vram_skips_card_without_vram_file(tmp_path: Path) -> None:
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "uevent").write_text("PCI_SLOT_NAME=0000:01:00.0\n", encoding="utf-8")

    assert gpu._linux_sysfs_vram_by_pci_address(tmp_path) == {}


def test_linux_sysfs_vram_skips_card_without_uevent(tmp_path: Path) -> None:
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_info_vram_total").write_text("17179869184\n", encoding="utf-8")

    assert gpu._linux_sysfs_vram_by_pci_address(tmp_path) == {}


def test_linux_sysfs_vram_ignores_non_card_entries(tmp_path: Path) -> None:
    device_dir = tmp_path / "card0-DP-1" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_info_vram_total").write_text("123\n", encoding="utf-8")
    (device_dir / "uevent").write_text("PCI_SLOT_NAME=0000:01:00.0\n", encoding="utf-8")

    assert gpu._linux_sysfs_vram_by_pci_address(tmp_path) == {}


def test_linux_sysfs_vram_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert gpu._linux_sysfs_vram_by_pci_address(tmp_path / "does-not-exist") == {}


def test_normalize_pci_address_strips_domain_prefix() -> None:
    assert gpu._normalize_pci_address("0000:01:00.0") == "01:00.0"


def test_normalize_pci_address_accepts_bare_address() -> None:
    assert gpu._normalize_pci_address("01:00.0") == "01:00.0"


def test_normalize_pci_address_rejects_malformed_input() -> None:
    assert gpu._normalize_pci_address("not-an-address") is None


def test_nvidia_smi_info_parses_csv_output(monkeypatch: object) -> None:
    output = "00000000:01:00.0, 8188, 550.54.14\n"
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    info = gpu._nvidia_smi_info()

    assert info == {"01:00.0": (8188 * 1024 * 1024, "550.54.14")}


def test_nvidia_smi_info_returns_empty_when_binary_missing(monkeypatch: object) -> None:
    def _raise(_command: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(gpu, "_run", _raise)  # type: ignore[attr-defined]

    assert gpu._nvidia_smi_info() == {}


def test_nvidia_smi_info_returns_empty_on_nonzero_exit(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed("", 1))  # type: ignore[attr-defined]

    assert gpu._nvidia_smi_info() == {}


def test_macos_gpu_parser_prefers_sppci_model(monkeypatch: object) -> None:
    payload = json.dumps(
        {
            "SPDisplaysDataType": [
                {"_name": "Apple M2 Pro", "sppci_model": "Apple M2 Pro"},
                {"_name": "Intel UHD Graphics 630"},
            ]
        }
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Darwin")

    assert warnings == ()
    assert [profile.name for profile in profiles] == ["Apple M2 Pro", "Intel UHD Graphics 630"]
    assert [profile.vendor for profile in profiles] == ["Apple", "Intel"]
    assert all(profile.source == "macos_system_profiler" for profile in profiles)


def test_macos_gpu_parses_dedicated_vram(monkeypatch: object) -> None:
    payload = json.dumps(
        {"SPDisplaysDataType": [{"_name": "AMD Radeon Pro 5500M", "sppci_vram": "8 GB"}]}
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Darwin")

    assert profiles[0].dedicated_memory_bytes == 8 * 1024**3


def test_macos_gpu_parses_alternate_vram_key(monkeypatch: object) -> None:
    payload = json.dumps(
        {"SPDisplaysDataType": [{"_name": "Intel Iris Plus", "spdisplays_vram": "1536 MB"}]}
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Darwin")

    assert profiles[0].dedicated_memory_bytes == 1536 * 1024**2


def test_macos_gpu_driver_version_is_always_none_by_design(monkeypatch: object) -> None:
    """macOS exposes no driver-version concept via system_profiler; this is a documented
    platform limitation, not a parsing gap."""
    payload = json.dumps(
        {"SPDisplaysDataType": [{"_name": "AMD Radeon Pro 5500M", "sppci_vram": "8 GB"}]}
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Darwin")

    assert profiles[0].driver_version is None


def test_macos_gpu_apple_silicon_has_no_vram_field_by_design(monkeypatch: object) -> None:
    """Apple Silicon's unified memory has no separate VRAM byte count to report; leaving
    it null is honest, not a gap to fix."""
    payload = json.dumps({"SPDisplaysDataType": [{"_name": "Apple M2 Pro"}]})
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, _warnings = gpu.collect_gpus("Darwin")

    assert profiles[0].dedicated_memory_bytes is None
    assert profiles[0].driver_version is None


def test_macos_gpu_skips_records_without_a_name(monkeypatch: object) -> None:
    payload = json.dumps({"SPDisplaysDataType": [{"spdisplays_vendor": "sppci_vendor_id"}]})
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Darwin")

    assert profiles == ()
    assert "No GPU adapters" in warnings[0]


def test_macos_gpu_empty_display_data_reports_no_adapters(monkeypatch: object) -> None:
    payload = json.dumps({"SPDisplaysDataType": []})
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(payload))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Darwin")

    assert profiles == ()
    assert "No GPU adapters" in warnings[0]


def test_macos_gpu_malformed_json_becomes_warning(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed("not json"))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Darwin")

    assert profiles == ()
    assert "failed" in warnings[0]


def test_macos_gpu_command_failure_becomes_warning(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed("", 1))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Darwin")

    assert profiles == ()
    assert "failed" in warnings[0]


def test_parse_macos_memory_string_handles_units() -> None:
    assert gpu._parse_macos_memory_string("8 GB") == 8 * 1024**3
    assert gpu._parse_macos_memory_string("1536 MB") == 1536 * 1024**2
    assert gpu._parse_macos_memory_string("512KB") == 512 * 1024


def test_parse_macos_memory_string_rejects_invalid_input() -> None:
    assert gpu._parse_macos_memory_string("yes") is None
    assert gpu._parse_macos_memory_string(None) is None
    assert gpu._parse_macos_memory_string(1536) is None


def test_windows_gpu_parser_handles_single_record(monkeypatch: object) -> None:
    output = '{"Name":"Intel UHD Graphics","AdapterRAM":1073741824,"DriverVersion":"31.0"}'
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed(output))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Windows")

    assert warnings == ()
    assert profiles[0].name == "Intel UHD Graphics"
    assert profiles[0].dedicated_memory_bytes == 1073741824


def test_windows_gpu_falls_back_when_cim_is_restricted(monkeypatch: object) -> None:
    fallback = (
        gpu.GPUProfile(
            name="Registry GPU",
            vendor=None,
            dedicated_memory_bytes=None,
            shared_memory_bytes=None,
            driver_version=None,
            source="windows_registry",
        ),
    )
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed("", 1))  # type: ignore[attr-defined]
    monkeypatch.setattr(gpu, "_windows_registry_gpus", lambda: fallback)  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Windows")

    assert profiles == fallback
    assert warnings == ()


def test_unsupported_platform_is_distinct() -> None:
    profiles, warnings = gpu.collect_gpus("Plan9")

    assert profiles == ()
    assert "unsupported" in warnings[0]


def test_gpu_command_failure_becomes_warning(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)
    monkeypatch.setattr(gpu, "_run", lambda _command: _completed("", 1))  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Linux")

    assert profiles == ()
    assert "failed" in warnings[0]


def test_gpu_command_timeout_becomes_warning(monkeypatch: object) -> None:
    _no_enrichment(monkeypatch)

    def _raise(_command: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="lspci", timeout=gpu.COMMAND_TIMEOUT_SECONDS)

    monkeypatch.setattr(gpu, "_run", _raise)  # type: ignore[attr-defined]

    profiles, warnings = gpu.collect_gpus("Linux")

    assert profiles == ()
    assert "failed" in warnings[0]


def test_vendor_recognizes_full_amd_vendor_string() -> None:
    assert gpu._vendor("Advanced Micro Devices, Inc. [AMD/ATI] Navi 22") == "AMD"


def test_vendor_returns_none_for_unknown_vendor() -> None:
    assert gpu._vendor("Matrox Electronics Systems Ltd. G200") is None
