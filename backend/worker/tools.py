"""HTTP clients for the service desk backend, behind a provider abstraction.

The concrete system (Jira, ServiceNow, a generic REST API, ...) is chosen via
SERVICEDESK_PROVIDER in backend/worker/integrations/__init__.py. This module keeps the
same function surface the agent's tools have always used, so brain.py does not
care which backend is wired in.

One shared provider — creating a client per call would add a TCP handshake to
every tool call, which the latency budget cannot afford.
"""

import asyncio
import os
import time

from backend.core import integrations

# Telemetry only: the debug UI feed. Failures here are swallowed — telemetry
# must never break a live call.
DEBUG_API_URL = os.getenv("DEBUG_API_URL", "http://localhost:8002")

_provider = None


def _get_provider():
    global _provider
    if _provider is None:
        _provider = integrations.get_provider()
    return _provider


async def aclose() -> None:
    global _provider, _client
    if _provider is not None:
        await _provider.aclose()
        _provider = None
    if _client is not None:
        await _client.aclose()
        _client = None


_client: "httpx.AsyncClient | None" = None
# Tasks are parked here because asyncio holds no strong reference to a running
# task, so a fire-and-forget one can be garbage collected mid-flight.
_pending: set = set()

# Ceiling on in-flight telemetry posts. 64 is far more than a healthy endpoint
# ever has outstanding, and small enough that a dead one cannot grow without
# bound during a long call.
MAX_PENDING_EVENTS = int(os.getenv("MAX_PENDING_EVENTS", "64"))


def _telemetry_client():
    """One shared client. Building one per event paid a TCP handshake each time."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(2.0, connect=1.0),
            limits=httpx.Limits(max_keepalive_connections=4),
        )
    return _client


async def _post_event(kind: str, detail: dict) -> None:
    try:
        await _telemetry_client().post(
            f"{DEBUG_API_URL}/_debug/log", json={"kind": kind, "detail": detail}
        )
    except Exception:
        # Telemetry must never affect a live call, including by raising into a
        # task nobody awaits.
        pass


def log_event(kind: str, **detail) -> None:
    """Fire-and-forget event for operational telemetry.

    Deliberately not a coroutine to await. It used to be, and `timed` awaited it
    in a `finally`, so every single tool call paid an HTTP round trip before its
    result reached the model — and if DEBUG_API_URL was unroutable rather than
    refused, that was a multi-second stall on the call path for a log line.
    """
    # If the telemetry endpoint is unreachable, every event sits in flight until
    # its timeout. Without a ceiling a long call against a dead endpoint would
    # accumulate a task per tool call. Dropping log lines is the correct
    # sacrifice here; the alternative is memory growth on a live call.
    if len(_pending) >= MAX_PENDING_EVENTS:
        return
    try:
        task = asyncio.create_task(_post_event(kind, detail))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    except RuntimeError:
        # No running loop (a synchronous caller). Telemetry is not worth one.
        pass


async def timed(kind: str, coro):
    """Run coro, report how long it took, and return its result.

    The report is scheduled, not awaited: the caller is a voice agent mid-turn
    and the caller is a person waiting for an answer.
    """
    started = time.perf_counter()
    try:
        return await coro
    finally:
        log_event(kind, ms=round((time.perf_counter() - started) * 1000, 1))


async def get_employee(employee_id: str) -> dict | None:
    return await _get_provider().get_employee(employee_id)


async def list_tickets(employee_id: str, status: str = "open") -> list[dict]:
    return await _get_provider().list_tickets(employee_id, status)


async def create_ticket(employee_id: str, title: str, description: str = "") -> dict | None:
    return await _get_provider().create_ticket(employee_id, title, description)


async def reset_password(employee_id: str, idempotency_key: str) -> dict | None:
    return await _get_provider().reset_password(employee_id, idempotency_key)
