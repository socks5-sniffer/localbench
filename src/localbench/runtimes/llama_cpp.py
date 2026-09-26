"""Local-only llama.cpp (`llama-server`) runtime discovery.

llama.cpp has no persistent model registry like Ollama's `/api/tags`: a `llama-server`
process is started against exactly one GGUF file (or a small preset list) and exposes an
OpenAI-compatible `/v1/models` endpoint describing only what is currently loaded, not
everything installed on disk. This detector reports that currently loaded model as the
runtime's "installed models" list rather than pretending to enumerate a catalog that does
not exist — a real difference from Ollama's model, called out explicitly rather than
papered over.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from localbench import __version__
from localbench.models import RuntimeModelProfile, RuntimeProfile
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError

LLAMA_CPP_API_BASE = "http://127.0.0.1:8080"
COMMAND_TIMEOUT_SECONDS = 3.0
API_TIMEOUT_SECONDS = 2.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024

CommandResult = subprocess.CompletedProcess[str]
WhichCommand = Callable[[str], str | None]
CommandRunner = Callable[[Sequence[str], float], CommandResult]
JsonFetcher = Callable[[str, float], object]


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        return None


def _run_command(command: Sequence[str], timeout: float) -> CommandResult:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def _request_json(path: str, timeout: float) -> object:
    headers = {"Accept": "application/json", "User-Agent": f"LocalBench/{__version__}"}
    request = Request(f"{LLAMA_CPP_API_BASE}{path}", headers=headers, method="GET")
    try:
        with build_opener(_NoRedirectHandler()).open(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    response_size = int(content_length)
                except ValueError as error:
                    message = "llama.cpp returned an invalid content length"
                    raise RuntimeResponseError(message) from error
                if response_size > MAX_RESPONSE_BYTES:
                    raise RuntimeResponseError("llama.cpp response exceeded the size limit")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeUnavailableError("llama.cpp local API is unavailable") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeResponseError("llama.cpp response exceeded the size limit")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeResponseError("llama.cpp returned invalid JSON") from error


def _fetch_json(path: str, timeout: float) -> object:
    return _request_json(path, timeout)


def _parse_cli_version(output: str) -> str | None:
    match = re.search(r"\bversion:\s*(\d+\s*\([0-9a-fA-F]+\))", output)
    return match.group(1) if match else None


def _optional_text(value: object) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _build_info_version(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    return _optional_text(payload.get("build_info"))


def _safe_model_name(value: object) -> tuple[str | None, bool]:
    raw = _optional_text(value)
    if raw is None:
        return None, False
    # llama-server commonly exposes the GGUF path as its model id. Normalize both Windows
    # and POSIX separators without resolving or otherwise retaining the personal path.
    basename = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return (basename or None), basename != raw


def _parse_models(
    payload: object, *, warnings: list[str] | None = None
) -> tuple[RuntimeModelProfile, ...]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise RuntimeResponseError("llama.cpp model response did not contain a data list")

    profiles: list[RuntimeModelProfile] = []
    for record in payload["data"]:
        if not isinstance(record, Mapping):
            continue
        name, path_sanitized = _safe_model_name(record.get("id"))
        if name is None:
            continue
        if path_sanitized and warnings is not None and not any(
            "basename-only" in warning for warning in warnings
        ):
            warnings.append(
                "llama.cpp reported a model path; LocalBench displayed its basename-only "
                "value to avoid exposing personal filesystem information."
            )
        raw_meta: Any = record.get("meta")
        meta: Mapping[object, object] = raw_meta if isinstance(raw_meta, Mapping) else {}
        raw_size = meta.get("size")
        size = raw_size if isinstance(raw_size, int) and raw_size >= 0 else None
        raw_params = meta.get("n_params")
        parameter_size = (
            f"{raw_params / 1_000_000_000:.1f}B"
            if isinstance(raw_params, int) and raw_params > 0
            else None
        )
        profiles.append(
            RuntimeModelProfile(
                name=name,
                size_bytes=size,
                digest=None,
                modified_at=None,
                format="gguf",
                family=None,
                parameter_size=parameter_size,
                quantization_level=None,
                source="llama_cpp_server_api",
            )
        )
    return tuple(profiles)


class LlamaCppDetector:
    name = "llama_cpp"

    def __init__(
        self,
        *,
        which: WhichCommand = shutil.which,
        run_command: CommandRunner = _run_command,
        fetch_json: JsonFetcher = _fetch_json,
    ) -> None:
        self._which = which
        self._run_command = run_command
        self._fetch_json = fetch_json

    def _cli_version(self, executable: str, warnings: list[str]) -> str | None:
        try:
            result = self._run_command([executable, "--version"], COMMAND_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            warnings.append("llama.cpp CLI version detection failed or timed out.")
            return None
        if result.returncode != 0:
            warnings.append("llama.cpp CLI was found but did not report its version.")
            return None
        version = _parse_cli_version(f"{result.stdout}\n{result.stderr}")
        if version is None:
            warnings.append("llama.cpp CLI returned an unrecognized version string.")
        return version

    def detect(self) -> RuntimeProfile:
        executable = self._which("llama-server")
        warnings: list[str] = []
        cli_version = self._cli_version(executable, warnings) if executable else None

        service_running = False
        api_version: str | None = None
        models: tuple[RuntimeModelProfile, ...] = ()
        try:
            models = _parse_models(
                self._fetch_json("/v1/models", API_TIMEOUT_SECONDS), warnings=warnings
            )
            service_running = True
        except RuntimeUnavailableError:
            if executable:
                warnings.append(
                    "llama.cpp CLI is installed, but no llama-server is reachable on "
                    f"{LLAMA_CPP_API_BASE}."
                )
        except RuntimeResponseError:
            warnings.append("A service responded on llama.cpp's port with invalid model data.")

        if service_running:
            try:
                api_version = _build_info_version(
                    self._fetch_json("/props", API_TIMEOUT_SECONDS)
                )
            except (RuntimeUnavailableError, RuntimeResponseError):
                warnings.append("llama.cpp is running, but its build version is unavailable.")

        return RuntimeProfile(
            name=self.name,
            cli_detected=executable is not None,
            service_running=service_running,
            version=api_version or cli_version,
            models=models,
            warnings=tuple(warnings),
        )
