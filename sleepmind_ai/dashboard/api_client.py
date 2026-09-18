"""HTTP client for the SleepMind API, built for a backend that sleeps.

The API runs on a free hosting tier that spins the instance down after
about 15 minutes of inactivity. The first request after that is answered
by the platform edge, not the application, and can come back as 429, 502
or 503 while the instance is starting. Firing a multi megabyte upload at
a sleeping instance and treating any non-200 as fatal is what made the
dashboard look broken.

This module keeps that handling in one place, free of any Streamlit
import, so it can be tested directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

# Statuses that mean "not ready yet" rather than "this request was bad".
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

DEFAULT_WAKE_TIMEOUT_SEC = 150
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_REQUEST_TIMEOUT_SEC = 300

Progress = Callable[[str], None] | None


class ApiUnavailable(Exception):
    """The backend could not be reached or woken within the time budget."""


def retry_after_seconds(response: httpx.Response, fallback: float, cap: float = 30.0) -> float:
    """Seconds to wait before retrying, honouring Retry-After when present."""
    raw = response.headers.get("retry-after")
    if not raw:
        return fallback
    try:
        return max(0.0, min(float(raw), cap))
    except ValueError:
        return fallback


class ApiClient:
    """Calls the SleepMind API, waking it first and retrying transient failures."""

    def __init__(
        self,
        base_url: str,
        *,
        wake_timeout_sec: int = DEFAULT_WAKE_TIMEOUT_SEC,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        request_timeout_sec: int = DEFAULT_REQUEST_TIMEOUT_SEC,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.wake_timeout_sec = wake_timeout_sec
        self.max_attempts = max_attempts
        self.request_timeout_sec = request_timeout_sec
        self._sleep = sleep
        self._monotonic = monotonic
        self.awake = False

    # -- waking ------------------------------------------------------------
    def wake(self, progress: Progress = None) -> bool:
        """Poll /health until the backend answers, or the budget runs out.

        A cheap GET is a much better thing to absorb a cold start with than
        the user's file upload.
        """
        if self.awake:
            return True

        deadline = self._monotonic() + self.wake_timeout_sec
        delay = 2.0
        attempt = 0

        with httpx.Client(timeout=30) as client:
            while self._monotonic() < deadline:
                attempt += 1
                try:
                    response = client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        self.awake = True
                        return True
                    wait = retry_after_seconds(response, delay)
                except httpx.RequestError:
                    wait = delay

                if progress is not None:
                    progress(
                        "Waking the server (free hosting sleeps when idle), "
                        f"attempt {attempt}..."
                    )

                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    break
                self._sleep(min(wait, remaining))
                delay = min(delay * 1.6, 12.0)

        return False

    # -- requests ----------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        progress: Progress = None,
        **kwargs,
    ) -> httpx.Response:
        """Send a request, waking the backend first and retrying transients.

        Returns the response for anything that is not a transient failure,
        so the caller can report a real error. Raises ApiUnavailable if the
        backend never became reachable at all.
        """
        if not self.wake(progress=progress):
            raise ApiUnavailable(
                "The server is not responding. Free hosting spins down when "
                "idle and can take a couple of minutes to come back. Please "
                "try again shortly."
            )

        delay = 3.0
        last_response: httpx.Response | None = None

        for attempt in range(1, self.max_attempts + 1):
            last_attempt = attempt == self.max_attempts
            try:
                with httpx.Client(timeout=self.request_timeout_sec) as client:
                    response = client.request(method, f"{self.base_url}{path}", **kwargs)
            except (httpx.TimeoutException, httpx.RequestError):
                if last_attempt:
                    raise
                self.awake = False
                if progress is not None:
                    progress(f"Connection problem. Retrying ({attempt} of {self.max_attempts})...")
                self._sleep(delay)
                delay = min(delay * 1.8, 20.0)
                continue

            if response.status_code not in RETRYABLE_STATUSES:
                return response

            last_response = response
            # The instance most likely went back to sleep between the health
            # check and this call, so re-wake before trying again.
            self.awake = False
            if last_attempt:
                break

            wait = retry_after_seconds(response, delay)
            if progress is not None:
                progress(
                    f"The server is busy waking up. Retrying "
                    f"({attempt} of {self.max_attempts})..."
                )
            self._sleep(wait)
            delay = min(delay * 1.8, 20.0)
            self.wake(progress=progress)

        assert last_response is not None
        return last_response

    def post(self, path: str, *, progress: Progress = None, **kwargs) -> httpx.Response:
        return self.request("POST", path, progress=progress, **kwargs)

    def get(self, path: str, *, progress: Progress = None, **kwargs) -> httpx.Response:
        return self.request("GET", path, progress=progress, **kwargs)


def friendly_error(response: httpx.Response) -> str:
    """Turn a failed response into something a non-technical reader can act on."""
    try:
        detail = str(response.json().get("detail", ""))
    except Exception:
        detail = ""

    lowered = detail.lower()
    if "api key" in lowered:
        return "The server isn't set up with an AI key yet."
    if "quota" in lowered or "insufficient" in lowered:
        return "The AI account behind this demo is out of credit."

    if response.status_code == 429:
        return "The free server is still busy waking up. Give it a minute and try once more."
    if response.status_code in (502, 503, 504):
        return "The server is restarting. Please try again in a minute."
    if response.status_code == 400:
        return "That file couldn't be read. Try a text-based PDF rather than a scan."
    return "Please try again."
