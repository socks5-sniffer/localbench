"""Release packaging smoke test.

Builds the real wheel, installs it into a throwaway virtual environment, and runs the
installed `localbench` console script — the same artifact a user would get from `pip
install localbench`. This is the only test that exercises hatchling's package discovery
(catches a subpackage silently missing from the wheel) and the `project.scripts` entry
point, neither of which `uv run` from the source tree would catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_TIMEOUT_SECONDS = 180


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # type: ignore[call-overload]
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        **kwargs,
    )


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_entry_point(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "localbench.exe"
    return venv_dir / "bin" / "localbench"


@pytest.fixture(scope="module")
def installed_entry_point(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel and install it into an isolated venv; return the CLI script."""
    workspace = tmp_path_factory.mktemp("localbench-packaging")
    dist_dir = workspace / "dist"
    venv_dir = workspace / "venv"

    build_result = _run(["uv", "build", "--wheel", "-o", str(dist_dir), str(REPO_ROOT)])
    assert build_result.returncode == 0, f"uv build failed:\n{build_result.stderr}"

    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"

    venv_result = _run(["uv", "venv", "--python", sys.executable, str(venv_dir)])
    assert venv_result.returncode == 0, f"uv venv failed:\n{venv_result.stderr}"

    install_result = _run(
        ["uv", "pip", "install", "--python", str(_venv_python(venv_dir)), str(wheels[0])]
    )
    assert install_result.returncode == 0, f"uv pip install failed:\n{install_result.stderr}"

    entry_point = _venv_entry_point(venv_dir)
    assert entry_point.exists(), f"console script not found at {entry_point}"
    return entry_point


def test_installed_cli_shows_help(installed_entry_point: Path) -> None:
    result = _run([str(installed_entry_point), "--help"])

    assert result.returncode == 0
    assert "profile" in result.stdout
    assert "runtimes" in result.stdout
    assert "estimate" in result.stdout
    assert "context" in result.stdout
    assert "recommend" in result.stdout
    assert "history" in result.stdout
    assert "compare" in result.stdout
    assert "bench" in result.stdout
    assert "stack" in result.stdout
    assert "coexist" in result.stdout


def test_installed_cli_profile_json_matches_schema(installed_entry_point: Path) -> None:
    result = _run([str(installed_entry_point), "profile", "--json"])

    assert result.returncode == 0
    decoded = json.loads(result.stdout)
    assert decoded["schema_version"] == "1"
    for key in ("collected_at", "os", "cpu", "memory", "gpus", "warnings"):
        assert key in decoded


def test_installed_cli_runtimes_json_matches_schema(installed_entry_point: Path) -> None:
    """Regression guard: the `runtimes` subpackage must actually ship in the wheel."""
    result = _run([str(installed_entry_point), "runtimes", "--json"])

    assert result.returncode == 0
    decoded = json.loads(result.stdout)
    assert decoded["schema_version"] == "1"
    assert "runtimes" in decoded
    assert {runtime["name"] for runtime in decoded["runtimes"]} == {"ollama", "llama_cpp"}
