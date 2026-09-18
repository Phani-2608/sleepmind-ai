"""Tests for the dashboard's API client.

These reproduce the failure that made the live dashboard show
"The upload didn't go through (429)": a free-tier instance that is
asleep, answers the first calls at the edge with 429 or 502, and only
then starts serving. The client must ride that out instead of
surfacing the status code to the user.

A real HTTP server is used rather than a mocked transport, so the
multipart upload path is exercised end to end. Sleeps are stubbed, so
the suite stays fast.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from sleepmind_ai.dashboard.api_client import (
    ApiClient,
    ApiUnavailable,
    friendly_error,
    retry_after_seconds,
)


class _Script:
    """Status codes each endpoint returns, consumed one call at a time."""

    def __init__(self, health: list[int], upload: list[int]) -> None:
        self.health = list(health)
        self.upload = list(upload)
        self.calls: list[tuple[str, str]] = []
        self.upload_bodies: list[int] = []

    def next_status(self, path: str) -> int:
        queue = self.health if path == "/health" else self.upload
        return queue.pop(0) if queue else 200


def _make_server(script: _Script) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence the default stderr logging
            pass

        def _respond(self, path: str, body_len: int = 0) -> None:
            script.calls.append((self.command, path))
            status = script.next_status(path)
            payload = json.dumps(
                {"status": "ok"} if status == 200 else {"detail": "not ready"}
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if status == 429:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            self._respond(self.path)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            script.upload_bodies.append(len(body))
            self._respond(self.path)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def slept():
    """Collect sleep durations instead of actually waiting."""
    return []


def _client(base_url: str, slept: list[float], **kwargs) -> ApiClient:
    return ApiClient(base_url, sleep=slept.append, **kwargs)


def test_upload_survives_a_429_while_the_instance_wakes(slept):
    """The exact live failure: edge answers 429, then the app comes up."""
    script = _Script(health=[503, 200, 200], upload=[429, 200])
    server, url = _make_server(script)
    try:
        client = _client(url, slept)
        response = client.post("/upload", files={"file": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")})
    finally:
        server.shutdown()

    assert response.status_code == 200
    # It woke the backend with a cheap GET before sending the file.
    assert script.calls[0] == ("GET", "/health")
    assert ("POST", "/upload") in script.calls
    # The file really was transmitted, twice (first attempt plus retry).
    assert len(script.upload_bodies) == 2
    assert all(size > 0 for size in script.upload_bodies)


def test_wake_polls_health_until_it_answers(slept):
    script = _Script(health=[502, 503, 429, 200], upload=[])
    server, url = _make_server(script)
    try:
        client = _client(url, slept)
        assert client.wake() is True
    finally:
        server.shutdown()

    assert client.awake is True
    assert len([c for c in script.calls if c == ("GET", "/health")]) == 4


def test_wake_gives_up_and_request_raises_api_unavailable(slept):
    script = _Script(health=[503] * 50, upload=[])
    server, url = _make_server(script)
    try:
        # A tiny budget: monotonic advances by each stubbed sleep.
        clock = [0.0]

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += seconds

        client = ApiClient(url, wake_timeout_sec=10, sleep=fake_sleep, monotonic=lambda: clock[0])
        with pytest.raises(ApiUnavailable):
            client.post("/upload", files={"file": ("a.pdf", b"x", "application/pdf")})
    finally:
        server.shutdown()


def test_non_transient_status_is_returned_not_retried(slept):
    """A 400 is the user's problem, not the platform's: report it at once."""
    script = _Script(health=[200], upload=[400])
    server, url = _make_server(script)
    try:
        client = _client(url, slept)
        response = client.post("/upload", files={"file": ("a.txt", b"x", "application/pdf")})
    finally:
        server.shutdown()

    assert response.status_code == 400
    assert len([c for c in script.calls if c[0] == "POST"]) == 1


def test_persistent_429_eventually_returns_the_response(slept):
    script = _Script(health=[200] * 20, upload=[429] * 20)
    server, url = _make_server(script)
    try:
        client = _client(url, slept, max_attempts=3)
        response = client.post("/upload", files={"file": ("a.pdf", b"x", "application/pdf")})
    finally:
        server.shutdown()

    assert response.status_code == 429
    assert len([c for c in script.calls if c[0] == "POST"]) == 3
    assert "busy waking up" in friendly_error(response)


def test_retry_after_header_is_honoured_and_capped():
    response = httpx.Response(429, headers={"Retry-After": "7"})
    assert retry_after_seconds(response, fallback=3.0) == 7.0

    response = httpx.Response(429, headers={"Retry-After": "9999"})
    assert retry_after_seconds(response, fallback=3.0) == 30.0

    response = httpx.Response(429, headers={"Retry-After": "soon"})
    assert retry_after_seconds(response, fallback=3.0) == 3.0

    assert retry_after_seconds(httpx.Response(429), fallback=3.0) == 3.0


def test_friendly_error_never_leaks_a_status_code():
    cases = [
        httpx.Response(429),
        httpx.Response(502),
        httpx.Response(400),
        httpx.Response(500, json={"detail": "No OpenAI API key configured."}),
        httpx.Response(500, json={"detail": "insufficient_quota"}),
    ]
    for response in cases:
        message = friendly_error(response)
        assert str(response.status_code) not in message
        assert message and message[0].isupper()


def test_api_key_and_quota_problems_are_named_plainly():
    assert "AI key" in friendly_error(httpx.Response(503, json={"detail": "No OpenAI API key configured."}))
    assert "out of credit" in friendly_error(httpx.Response(500, json={"detail": "insufficient_quota"}))
