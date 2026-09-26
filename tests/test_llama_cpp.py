from __future__ import annotations

import io
import subprocess
from email.message import Message
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from localbench.runtimes import llama_cpp
from localbench.runtimes.base import RuntimeResponseError, RuntimeUnavailableError
from localbench.runtimes.llama_cpp import (
    MAX_RESPONSE_BYTES,
    LlamaCppDetector,
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
    monkeypatch.setattr(llama_cpp, "build_opener", lambda *_handlers: _FakeOpener(response))


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["llama-server", "--version"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_parse_cli_version() -> None:
    assert _parse_cli_version("version: 8931 (9725a31)") == "8931 (9725a31)"
    assert _parse_cli_version("version: 8931 (9725a31) built with GNU 15.2.0") == "8931 (9725a31)"
    assert _parse_cli_version("unknown") is None


def test_parse_models_extracts_supported_metadata() -> None:
    models = _parse_models(
        {
            "object": "list",
            "data": [
                {
                    "id": "../models/gemma3.gguf",
                    "object": "model",
                    "owned_by": "llamacpp",
                    "meta": {
                        "n_params": 4_300_000_000,
                        "size": 3338801804,
                        "n_ctx_train": 8192,
                    },
                }
            ],
        }
    )

    assert models[0].name == "gemma3.gguf"
    assert models[0].size_bytes == 3338801804
    assert models[0].parameter_size == "4.3B"
    assert models[0].format == "gguf"
    assert models[0].source == "llama_cpp_server_api"


def test_parse_models_strips_posix_and_windows_paths() -> None:
    warnings: list[str] = []
    models = _parse_models(
        {
            "data": [
                {"id": "/home/alice/private/gemma.gguf"},
                {"id": "C:\\Users\\alice\\models\\phi.gguf"},
            ]
        },
        warnings=warnings,
    )

    assert [model.name for model in models] == ["gemma.gguf", "phi.gguf"]
    assert len(warnings) == 1
    assert "basename-only" in warnings[0]
    assert "alice" not in warnings[0]


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
        llama_cpp._request_json("/v1/models", 1.0)


def test_request_json_translates_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_response(monkeypatch, URLError("offline"))

    with pytest.raises(RuntimeUnavailableError, match="local API is unavailable"):
        llama_cpp._request_json("/v1/models", 1.0)


def test_redirect_handler_refuses_redirects() -> None:
    request = Request(f"{llama_cpp.LLAMA_CPP_API_BASE}/v1/models")

    redirected = _NoRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        Message(),
        "http://example.com/not-local",
    )

    assert redirected is None


def test_detect_running_server_prefers_build_info_version() -> None:
    responses = {
        "/v1/models": {"data": [{"id": "gemma3.gguf", "meta": {"size": 100}}]},
        "/props": {"build_info": "b8931-9725a31"},
    }
    detector = LlamaCppDetector(
        which=lambda _name: "C:/llama.cpp/llama-server.exe",
        run_command=lambda _command, _timeout: _completed("version: 8900 (aaaaaaa)"),
        fetch_json=lambda path, _timeout: responses[path],
    )

    profile = detector.detect()

    assert profile.cli_detected is True
    assert profile.service_running is True
    assert profile.version == "b8931-9725a31"
    assert profile.models[0].name == "gemma3.gguf"
    assert profile.warnings == ()


def test_detect_absent_llama_cpp_is_not_an_error() -> None:
    def unavailable(_path: str, _timeout: float) -> object:
        raise RuntimeUnavailableError

    profile = LlamaCppDetector(which=lambda _name: None, fetch_json=unavailable).detect()

    assert profile.cli_detected is False
    assert profile.service_running is False
    assert profile.version is None
    assert profile.models == ()
    assert profile.warnings == ()


def test_detect_installed_cli_with_stopped_server() -> None:
    def unavailable(_path: str, _timeout: float) -> object:
        raise RuntimeUnavailableError

    detector = LlamaCppDetector(
        which=lambda _name: "llama-server",
        run_command=lambda _command, _timeout: _completed("version: 8900 (aaaaaaa)"),
        fetch_json=unavailable,
    )

    profile = detector.detect()

    assert profile.service_running is False
    assert profile.version == "8900 (aaaaaaa)"
    assert "no llama-server is reachable" in profile.warnings[-1]


def test_detect_malformed_model_response() -> None:
    profile = LlamaCppDetector(
        which=lambda _name: None,
        fetch_json=lambda _path, _timeout: {"unexpected": []},
    ).detect()

    assert profile.service_running is False
    assert "invalid model data" in profile.warnings[0]


def test_build_info_failure_falls_back_to_cli_version() -> None:
    def fetch(path: str, _timeout: float) -> object:
        if path == "/v1/models":
            return {"data": []}
        raise RuntimeUnavailableError

    detector = LlamaCppDetector(
        which=lambda _name: "llama-server",
        run_command=lambda _command, _timeout: _completed("version: 8900 (aaaaaaa)"),
        fetch_json=fetch,
    )

    profile = detector.detect()

    assert profile.service_running is True
    assert profile.version == "8900 (aaaaaaa)"
    assert "build version is unavailable" in profile.warnings[-1]
