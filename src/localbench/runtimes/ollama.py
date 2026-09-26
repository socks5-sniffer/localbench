"""Local-only Ollama runtime discovery."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from localbench.models import RuntimeModelProfile, RuntimeProfile
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError

OLLAMA_API_BASE = "http://127.0.0.1:11434"
COMMAND_TIMEOUT_SECONDS = 3.0
API_TIMEOUT_SECONDS = 2.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_STREAM_LINE_BYTES = 1024 * 1024

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


def _request_json(
    path: str,
    timeout: float,
    body: Mapping[str, object] | None = None,
) -> object:
    encoded_body = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json", "User-Agent": "LocalBench/0.3"}
    if encoded_body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{OLLAMA_API_BASE}{path}",
        data=encoded_body,
        headers=headers,
        method="POST" if encoded_body is not None else "GET",
    )
    try:
        with build_opener(_NoRedirectHandler()).open(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    response_size = int(content_length)
                except ValueError as error:
                    message = "Ollama returned an invalid content length"
                    raise RuntimeResponseError(message) from error
                if response_size > MAX_RESPONSE_BYTES:
                    raise RuntimeResponseError("Ollama response exceeded the size limit")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeUnavailableError("Ollama local API is unavailable") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeResponseError("Ollama response exceeded the size limit")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeResponseError("Ollama returned invalid JSON") from error


def _fetch_json(path: str, timeout: float) -> object:
    return _request_json(path, timeout)


def fetch_model_details(model: str, timeout: float = API_TIMEOUT_SECONDS) -> object:
    """Fetch metadata for one installed model from Ollama's fixed local API."""
    return _request_json("/api/show", timeout, {"model": model, "verbose": False})


def stream_generate(
    body: Mapping[str, object], timeout: float = 180.0
) -> Iterator[Mapping[str, object]]:
    """Yield validated NDJSON objects from Ollama's fixed local generate endpoint."""
    encoded_body = json.dumps(body).encode("utf-8")
    request = Request(
        f"{OLLAMA_API_BASE}/api/generate",
        data=encoded_body,
        headers={
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
            "User-Agent": "LocalBench/0.4",
        },
        method="POST",
    )
    total_bytes = 0
    try:
        with build_opener(_NoRedirectHandler()).open(request, timeout=timeout) as response:
            for raw_line in response:
                total_bytes += len(raw_line)
                if len(raw_line) > MAX_STREAM_LINE_BYTES or total_bytes > MAX_RESPONSE_BYTES:
                    raise RuntimeResponseError("Ollama stream exceeded the size limit")
                if not raw_line.strip():
                    continue
                try:
                    decoded: Any = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeResponseError("Ollama returned invalid streaming JSON") from error
                if not isinstance(decoded, Mapping):
                    raise RuntimeResponseError("Ollama stream item was not an object")
                yield decoded
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeUnavailableError("Ollama generation request failed") from error


def unload_model(model: str, timeout: float = 30.0) -> None:
    """Request immediate unload without storing or generating user content."""
    _request_json(
        "/api/generate", timeout, {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    )


def _parse_cli_version(output: str) -> str | None:
    match = re.search(r"\bv?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b", output)
    return match.group(1) if match else None


def _optional_text(value: object) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _parse_models(payload: object) -> tuple[RuntimeModelProfile, ...]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), list):
        raise RuntimeResponseError("Ollama model response did not contain a models list")

    profiles: list[RuntimeModelProfile] = []
    for record in payload["models"]:
        if not isinstance(record, Mapping):
            continue
        name = _optional_text(record.get("name") or record.get("model"))
        if name is None:
            continue
        raw_size = record.get("size")
        size = raw_size if isinstance(raw_size, int) and raw_size >= 0 else None
        raw_details: Any = record.get("details")
        details: Mapping[object, object] = raw_details if isinstance(raw_details, Mapping) else {}
        profiles.append(
            RuntimeModelProfile(
                name=name,
                size_bytes=size,
                digest=_optional_text(record.get("digest")),
                modified_at=_optional_text(record.get("modified_at")),
                format=_optional_text(details.get("format")),
                family=_optional_text(details.get("family")),
                parameter_size=_optional_text(details.get("parameter_size")),
                quantization_level=_optional_text(details.get("quantization_level")),
                source="ollama_local_api",
            )
        )
    return tuple(profiles)


class OllamaDetector:
    name = "ollama"

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
            result = self._run_command(
                [executable, "--version"],
                COMMAND_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            warnings.append("Ollama CLI version detection failed or timed out.")
            return None
        if result.returncode != 0:
            warnings.append("Ollama CLI was found but did not report its version.")
            return None
        version = _parse_cli_version(f"{result.stdout}\n{result.stderr}")
        if version is None:
            warnings.append("Ollama CLI returned an unrecognized version string.")
        return version

    def detect(self) -> RuntimeProfile:
        executable = self._which("ollama")
        warnings: list[str] = []
        cli_version = self._cli_version(executable, warnings) if executable else None

        service_running = False
        api_version: str | None = None
        models: tuple[RuntimeModelProfile, ...] = ()
        try:
            version_payload = self._fetch_json("/api/version", API_TIMEOUT_SECONDS)
            if not isinstance(version_payload, Mapping):
                raise RuntimeResponseError("Ollama version response was not an object")
            api_version = _optional_text(version_payload.get("version"))
            if api_version is None:
                raise RuntimeResponseError("Ollama version response did not contain a version")
            service_running = True
            try:
                models = _parse_models(self._fetch_json("/api/tags", API_TIMEOUT_SECONDS))
            except RuntimeUnavailableError:
                warnings.append("Ollama stopped responding while installed models were queried.")
            except RuntimeResponseError:
                warnings.append("Ollama returned invalid installed-model data.")
        except RuntimeUnavailableError:
            if executable:
                warnings.append("Ollama CLI is installed, but its local service is not reachable.")
        except RuntimeResponseError:
            warnings.append("A service responded on Ollama's port with invalid version data.")

        return RuntimeProfile(
            name=self.name,
            cli_detected=executable is not None,
            service_running=service_running,
            version=api_version or cli_version,
            models=models,
            warnings=tuple(warnings),
        )
