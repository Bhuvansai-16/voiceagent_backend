"""Test fixtures.

The database is Neon now, not a throwaway SQLite file, so the `client` fixture
needs a connection. Tests that do not use `client` still run with no network and
no credentials — that is roughly 120 of them, and keeping it that way is
deliberate.

Fixture rows are inserted and removed per session. They are test fixtures, not
seed data: the application starts empty and these exist only so the assertions
have something to assert against.
"""

import os

from backend.eventloop import use_selector_loop_on_windows

use_selector_loop_on_windows()

import pytest  # noqa: E402

EMPLOYEES = [
    ("E1042", "Priya N", "priya.n@corp.com", False),
    ("E1043", "Marcus Lee", "marcus.lee@corp.com", False),
    ("E1044", "Aisha Khan", "aisha.khan@corp.com", False),
    ("E1046", "Sofia Berg", "sofia.berg@corp.com", False),
    ("E1045", "Tom Rivera", "tom.rivera@corp.com", True),
]

TICKETS = [
    ("INC0041", "E1042", "Outlook keeps asking for password", "open", "2026-07-20T09:14:00Z"),
    ("INC0042", "E1042", "VPN drops every 10 minutes", "open", "2026-07-22T14:02:00Z"),
    ("INC0043", "E1042", "Requested second monitor", "closed", "2026-07-11T11:30:00Z"),
    ("INC0048", "E1046", "Printer on 4th floor offline", "open", "2026-07-25T13:40:00Z"),
]


def _database_configured() -> bool:
    return bool(os.getenv("NEON_CONNECTION_URI", "").strip())


@pytest.fixture(scope="session")
def client():
    """A TestClient with fixture rows present.

    Skips rather than fails when there is no database, so a contributor without
    credentials still gets a useful run out of the other ~120 tests.
    """
    from dotenv import load_dotenv

    load_dotenv(".env")
    if not _database_configured():
        pytest.skip("NEON_CONNECTION_URI not set")

    import psycopg
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.store.pool import direct_dsn

    with psycopg.connect(direct_dsn(), connect_timeout=30, autocommit=True) as conn:
        for row in EMPLOYEES:
            conn.execute(
                "INSERT INTO employees (id, name, email, locked) VALUES (%s,%s,%s,%s)"
                " ON CONFLICT (id) DO NOTHING", row,
            )
        for tid, eid, title, status, updated in TICKETS:
            conn.execute(
                "INSERT INTO tickets (id, employee_id, title, description, status, updated)"
                " VALUES (%s,%s,%s,'',%s,%s::timestamptz) ON CONFLICT (id) DO NOTHING",
                (tid, eid, title, status, updated),
            )

    with TestClient(app) as c:   # entering the context runs lifespan
        yield c

    with psycopg.connect(direct_dsn(), connect_timeout=30, autocommit=True) as conn:
        conn.execute("DELETE FROM password_resets WHERE employee_id = ANY(%s)",
                     ([e[0] for e in EMPLOYEES],))
        conn.execute("DELETE FROM tickets WHERE employee_id = ANY(%s)",
                     ([e[0] for e in EMPLOYEES],))
        conn.execute("DELETE FROM employees WHERE id = ANY(%s)",
                     ([e[0] for e in EMPLOYEES],))


@pytest.fixture
def store():
    """An empty in-process store, for tests that only need agent configuration."""
    from backend.store import MemoryStore, set_store

    impl = MemoryStore()
    set_store(impl)
    yield impl
    set_store(None)
