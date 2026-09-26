from __future__ import annotations

import io
import subprocess
from email.message import Message
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from localbench.runtimes import ollama
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError
from localbench.runtimes.ollama import (
    MAX_RESPONSE_BYTES,
    MAX_STREAM_LINE_BYTES,
    OllamaDetector,
    _NoRedirectHandler,
    _parse_cli_version,
    _parse_models,
)


class _FakeResponse:
    def __init__(self, payload: bytes, *, content_length: str | None = None) -> None:
        self._payload = payload
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload[:size]

    def __iter__(self) -> Any:
        return iter(io.BytesIO(self._payload))


class _FakeOpener:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response

    def open(self, _request: Request, *, timeout: float) -> _FakeResponse:
        del timeout
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _install_response(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse | Exception
) -> None:
    monkeypatch.setattr(ollama, "build_opener", lambda *_handlers: _FakeOpener(response))


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ollama", "--version"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_parse_cli_version() -> None:
    assert _parse_cli_version("ollama version is 0.12.6") == "0.12.6"
    assert _parse_cli_version("ollama version 1.0.0-rc.1") == "1.0.0-rc.1"
    assert _parse_cli_version("unknown") is None


def test_parse_models_extracts_supported_metadata() -> None:
    models = _parse_models(
        {
            "models": [
                {
                    "name": "gemma3:latest",
                    "size": 3338801804,
                    "digest": "abc",
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "format": "gguf",
                        "family": "gemma3",
                        "parameter_size": "4.3B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        }
    )

    assert models[0].name == "gemma3:latest"
    assert models[0].size_bytes == 3338801804
    assert models[0].quantization_level == "Q4_K_M"


def test_parse_models_rejects_invalid_root() -> None:
    with pytest.raises(RuntimeResponseError):
        _parse_models({"unexpected": []})


def test_request_json_rejects_declared_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_response(
        monkeypatch,
        _FakeResponse(b"{}", content_length=str(MAX_RESPONSE_BYTES + 1)),
    )

    with pytest.raises(RuntimeResponseError, match="size limit"):
        ollama._request_json("/api/version", 1.0)


@pytest.mark.parametrize("content_length", [None, "1"])
def test_request_json_enforces_actual_size_when_length_is_missing_or_false(
    monkeypatch: pytest.MonkeyPatch, content_length: str | None
) -> None:
    _install_response(
        monkeypatch,
        _FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1), content_length=content_length),
    )

    with pytest.raises(RuntimeResponseError, match="size limit"):
        ollama._request_json("/api/version", 1.0)


def test_request_json_rejects_invalid_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_response(monkeypatch, _FakeResponse(b"{}", content_length="many"))

    with pytest.raises(RuntimeResponseError, match="invalid content length"):
        ollama._request_json("/api/version", 1.0)


def test_request_json_translates_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_response(monkeypatch, URLError("offline"))

    with pytest.raises(RuntimeUnavailableError, match="local API is unavailable"):
        ollama._request_json("/api/version", 1.0)


def test_stream_generate_rejects_oversized_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_response(monkeypatch, _FakeResponse(b"x" * (MAX_STREAM_LINE_BYTES + 1) + b"\n"))

    with pytest.raises(RuntimeResponseError, match="size limit"):
        tuple(ollama.stream_generate({"model": "example"}))


def test_stream_generate_enforces_total_size(monkeypatch: pytest.MonkeyPatch) -> None:
    line = b'{"response":"' + (b"x" * (MAX_STREAM_LINE_BYTES - 30)) + b'"}\n'
    line_count = (MAX_RESPONSE_BYTES // len(line)) + 1
    _install_response(monkeypatch, _FakeResponse(line * line_count))

    with pytest.raises(RuntimeResponseError, match="size limit"):
        tuple(ollama.stream_generate({"model": "example"}))


def test_stream_generate_rejects_invalid_items(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_response(monkeypatch, _FakeResponse(b"[]\n"))

    with pytest.raises(RuntimeResponseError, match="not an object"):
        tuple(ollama.stream_generate({"model": "example"}))


def test_stream_generate_translates_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request_url = f"{ollama.OLLAMA_API_BASE}/api/generate"
    _install_response(monkeypatch, HTTPError(request_url, 302, "redirect", Message(), None))

    with pytest.raises(RuntimeUnavailableError, match="generation request failed"):
        tuple(ollama.stream_generate({"model": "example"}))


def test_redirect_handler_refuses_redirects() -> None:
    request = Request(f"{ollama.OLLAMA_API_BASE}/api/version")

    redirected = _NoRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        Message(),
        "http://example.com/not-local",
    )

    assert redirected is None


def test_detect_running_ollama_prefers_api_version() -> None:
    responses = {
        "/api/version": {"version": "0.12.6"},
        "/api/tags": {"models": [{"model": "gemma3", "size": 100}]},
    }
    detector = OllamaDetector(
        which=lambda _name: "C:/Ollama/ollama.exe",
        run_command=lambda _command, _timeout: _completed("ollama version is 0.11.0"),
        fetch_json=lambda path, _timeout: responses[path],
    )

    profile = detector.detect()

    assert profile.cli_detected is True
    assert profile.service_running is True
    assert profile.version == "0.12.6"
    assert profile.models[0].name == "gemma3"
    assert profile.warnings == ()


def test_detect_absent_ollama_is_not_an_error() -> None:
    def unavailable(_path: str, _timeout: float) -> object:
        raise RuntimeUnavailableError

    profile = OllamaDetector(which=lambda _name: None, fetch_json=unavailable).detect()

    assert profile.cli_detected is False
    assert profile.service_running is False
    assert profile.version is None
    assert profile.warnings == ()


def test_detect_installed_cli_with_stopped_service() -> None:
    def unavailable(_path: str, _timeout: float) -> object:
        raise RuntimeUnavailableError

    detector = OllamaDetector(
        which=lambda _name: "ollama",
        run_command=lambda _command, _timeout: _completed("ollama version is 0.12.6"),
        fetch_json=unavailable,
    )

    profile = detector.detect()

    assert profile.service_running is False
    assert profile.version == "0.12.6"
    assert "not reachable" in profile.warnings[-1]


def test_detect_malformed_service_response() -> None:
    profile = OllamaDetector(
        which=lambda _name: None,
        fetch_json=lambda _path, _timeout: [],
    ).detect()

    assert profile.service_running is False
    assert "invalid version data" in profile.warnings[0]


def test_model_query_failure_preserves_running_status() -> None:
    def fetch(path: str, _timeout: float) -> object:
        if path == "/api/version":
            return {"version": "0.12.6"}
        raise RuntimeUnavailableError

    profile = OllamaDetector(which=lambda _name: None, fetch_json=fetch).detect()

    assert profile.service_running is True
    assert profile.models == ()
    assert "stopped responding" in profile.warnings[0]
