from __future__ import annotations

import subprocess
from pathlib import Path

from localbench.hardware import cpu
from localbench.hardware.cpu import _linux_cpu_model


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_linux_cpu_model_reads_first_named_processor() -> None:
    assert _linux_cpu_model(Path("tests/fixtures/cpuinfo-x86.txt")) == "Example CPU 9000"


def test_linux_cpu_model_falls_back_to_hardware() -> None:
    assert _linux_cpu_model(Path("tests/fixtures/cpuinfo-arm.txt")) == "Example ARM Board"


def test_linux_cpu_model_prefers_model_name_over_hardware() -> None:
    assert _linux_cpu_model(Path("tests/fixtures/cpuinfo-mixed.txt")) == "Priority CPU"


def test_linux_cpu_model_skips_digit_only_processor_field() -> None:
    assert _linux_cpu_model(Path("tests/fixtures/cpuinfo-digits-only.txt")) is None


def test_linux_cpu_model_returns_none_when_file_missing() -> None:
    assert _linux_cpu_model(Path("tests/fixtures/does-not-exist.txt")) is None


def test_macos_cpu_model_reads_sysctl_output(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cpu.subprocess, "run", lambda *_a, **_k: _completed("Apple M2 Pro\n")
    )

    assert cpu._macos_cpu_model() == "Apple M2 Pro"


def test_macos_cpu_model_returns_none_on_nonzero_exit(monkeypatch: object) -> None:
    monkeypatch.setattr(cpu.subprocess, "run", lambda *_a, **_k: _completed("", 1))  # type: ignore[attr-defined]

    assert cpu._macos_cpu_model() is None


def test_macos_cpu_model_returns_none_on_timeout(monkeypatch: object) -> None:
    def _raise(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="sysctl", timeout=3)

    monkeypatch.setattr(cpu.subprocess, "run", _raise)  # type: ignore[attr-defined]

    assert cpu._macos_cpu_model() is None


def test_collect_cpu_linux_reports_proc_cpuinfo_source(monkeypatch: object) -> None:
    monkeypatch.setattr(cpu, "_linux_cpu_model", lambda: "Linux CPU")  # type: ignore[attr-defined]

    profile = cpu.collect_cpu("Linux")

    assert profile.model == "Linux CPU"
    assert profile.source == "proc_cpuinfo"


def test_collect_cpu_linux_falls_back_when_cpuinfo_unavailable(monkeypatch: object) -> None:
    monkeypatch.setattr(cpu, "_linux_cpu_model", lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(cpu.platform, "processor", lambda: "x86_64")  # type: ignore[attr-defined]

    profile = cpu.collect_cpu("Linux")

    assert profile.model == "x86_64"
    assert profile.source == "python_platform"


def test_collect_cpu_macos_reports_sysctl_source(monkeypatch: object) -> None:
    monkeypatch.setattr(cpu, "_macos_cpu_model", lambda: "Apple M2")  # type: ignore[attr-defined]

    profile = cpu.collect_cpu("Darwin")

    assert profile.model == "Apple M2"
    assert profile.source == "sysctl"


def test_collect_cpu_macos_falls_back_when_sysctl_unavailable(monkeypatch: object) -> None:
    monkeypatch.setattr(cpu, "_macos_cpu_model", lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(cpu.platform, "processor", lambda: "")  # type: ignore[attr-defined]

    profile = cpu.collect_cpu("Darwin")

    assert profile.model is None
    assert profile.source == "python_platform"
