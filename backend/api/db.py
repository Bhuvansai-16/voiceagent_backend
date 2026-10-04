"""Database access for the API. Was common.py, backed by SQLite.

Two things changed beyond the dialect.

`sqlite3.connect()` was called inside `async def` handlers. Against a local file
that costs microseconds and nobody notices; against a network database it blocks
the event loop for a full round trip on every request, stalling every other
in-flight request. Everything here is async and goes through the shared pool.

Placeholders are `%s`, not `?`, and booleans are real booleans rather than
integer 0/1.
"""

from datetime import datetime, timezone

from backend.store.pool import get_pool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mask(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[0]}***@{domain}" if local else email


def _iso(value) -> str:
    """Timestamps go out as ISO strings, the shape the agent and UI already read."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    async with get_pool().connection() as conn:
        return await (await conn.execute(sql, params)).fetchall()


async def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    async with get_pool().connection() as conn:
        return await (await conn.execute(sql, params)).fetchone()


async def execute(sql: str, params: tuple = ()) -> int:
    async with get_pool().connection() as conn:
        cur = await conn.execute(sql, params)
        return cur.rowcount
