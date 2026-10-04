"""Event bus for API diagnostics and operational telemetry.

The agent-facing middleware and several /_debug endpoints publish events here;
/_debug/logs polls them and /_debug/events streams them over SSE.
"""

import asyncio
from datetime import datetime, timezone

_subscribers: set[asyncio.Queue] = set()
_recent: list[dict] = []
_seq = 0

# How long one /_debug/events connection lives before it hangs up and lets the
# client reconnect.
MAX_STREAM_SECONDS = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def publish(event: dict) -> None:
    global _seq
    _seq += 1
    event.setdefault("at", _now())
    # Sequence number so a reconnecting feed can ask for what it missed instead
    # of replaying the whole buffer and showing every event twice.
    event.setdefault("seq", _seq)
    _recent.append(event)
    del _recent[:-200]
    for q in _subscribers:
        q.put_nowait(event)


def recent(since: int = 0) -> list[dict]:
    return [e for e in _recent if e.get("seq", 0) > since]


def last_seq() -> int:
    return _recent[-1].get("seq", 0) if _recent else 0