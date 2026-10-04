"""One suite, run against both store implementations.

MemoryStore only earns its place if it behaves like PostgresStore. A double that
replaces where the real one merges lets a test pass while production silently
drops fields — which is exactly the class of bug that hit this codebase before,
when CONFIGURABLE lagged behind agents.json and months of speech settings
vanished at load with a single log line.

The Postgres half is skipped unless NEON_CONNECTION_URI is set, so the default
`uv run pytest` needs no network. Run it against the real database with:

    uv run pytest backend/tests/test_store_contract.py -v
"""

import os
import uuid

import pytest

from backend.store.memory import MemoryStore

pytestmark = pytest.mark.asyncio


def _postgres_available() -> bool:
    return bool(os.getenv("NEON_CONNECTION_URI", "").strip())


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    """Each test runs twice: once in memory, once against Neon."""
    if request.param == "memory":
        yield MemoryStore()
        return

    if not _postgres_available():
        pytest.skip("NEON_CONNECTION_URI not set")

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from backend.store.pool import dsn
    from backend.store.postgres import PostgresStore

    # A pool of this test's own, on this test's event loop. Reusing the shared
    # one fails with "attached to a different loop" once any other fixture has
    # opened it on a different loop.
    pool = AsyncConnectionPool(dsn(), min_size=1, max_size=2,
                               kwargs={"row_factory": dict_row}, open=False)
    await pool.open(wait=True, timeout=30)

    impl = PostgresStore(connection=lambda: pool.connection())
    # Namespaced ids so a run cannot collide with real rows or a parallel run.
    impl._prefix = f"ct-{uuid.uuid4().hex[:8]}-"
    try:
        yield impl
    finally:
        async with pool.connection() as conn:
            for table in ("agents", "flows", "connectors"):
                await conn.execute(
                    f"DELETE FROM {table} WHERE id LIKE %s", (impl._prefix + "%",)
                )
        await pool.close()


def _id(store, name: str) -> str:
    return getattr(store, "_prefix", "ct-mem-") + name


# --- agents -----------------------------------------------------------------

async def test_unknown_agent_is_absent(store):
    assert _id(store, "nope") not in await store.agent_overrides()


async def test_upsert_then_read_back(store):
    aid = _id(store, "a1")
    await store.upsert_agent(aid, {"model": "gemini-3.5-flash-lite"})
    assert (await store.agent_overrides())[aid]["model"] == "gemini-3.5-flash-lite"


async def test_upsert_merges_rather_than_replaces(store):
    """The bug this prevents: a PUT carrying only {id, provider} used to wipe
    that agent's label, purpose, skills and tools."""
    aid = _id(store, "a2")
    await store.upsert_agent(aid, {"label": "Support", "tools": ["end_call"]})
    await store.upsert_agent(aid, {"provider": "nvidia"})

    saved = (await store.agent_overrides())[aid]
    assert saved["provider"] == "nvidia"
    assert saved["label"] == "Support", "merge dropped a field it was not asked to change"
    assert saved["tools"] == ["end_call"]


async def test_delete_reports_whether_anything_went(store):
    aid = _id(store, "a3")
    await store.upsert_agent(aid, {"label": "X"})
    assert await store.delete_agent(aid) is True
    assert await store.delete_agent(aid) is False


# --- flows ------------------------------------------------------------------

async def test_flow_round_trips_its_graph(store):
    fid = _id(store, "f1")
    await store.upsert_flow({"id": fid, "name": "Speechy", "nodes": [{"type": "agent_brain"}]})
    got = next(f for f in await store.flows() if f["id"] == fid)
    assert got["name"] == "Speechy"
    assert got["nodes"] == [{"type": "agent_brain"}]


async def test_flow_upsert_merges(store):
    fid = _id(store, "f2")
    await store.upsert_flow({"id": fid, "name": "One", "nodes": [1]})
    await store.upsert_flow({"id": fid, "name": "Two"})
    got = next(f for f in await store.flows() if f["id"] == fid)
    assert got["name"] == "Two"
    assert got["nodes"] == [1]


async def test_deleting_a_flow(store):
    fid = _id(store, "f3")
    await store.upsert_flow({"id": fid, "name": "Doomed"})
    assert await store.delete_flow(fid) is True
    assert fid not in {f["id"] for f in await store.flows()}


# --- connectors -------------------------------------------------------------

async def test_connector_round_trips(store):
    cid = _id(store, "c1")
    await store.upsert_connector({"id": cid, "name": "Slack", "slug": "slack"})
    got = next(c for c in await store.connectors() if c["id"] == cid)
    assert got["name"] == "Slack"
    assert got["connected"] is False


async def test_upsert_does_not_change_connection_state(store):
    """Editing a connector's definition must not silently connect or disconnect
    it. set_connector_connected is the only thing that flips that flag."""
    cid = _id(store, "c2")
    await store.upsert_connector({"id": cid, "name": "Gmail"})
    await store.set_connector_connected(cid, True)
    await store.upsert_connector({"id": cid, "name": "Gmail", "connected": False})

    got = next(c for c in await store.connectors() if c["id"] == cid)
    assert got["connected"] is True, "an upsert changed the connection state"


async def test_set_connected_reports_whether_the_row_existed(store):
    cid = _id(store, "c3")
    assert await store.set_connector_connected(cid, True) is False
    await store.upsert_connector({"id": cid, "name": "Linear"})
    assert await store.set_connector_connected(cid, True) is True


async def test_deleting_a_connector(store):
    cid = _id(store, "c4")
    await store.upsert_connector({"id": cid, "name": "Gone"})
    assert await store.delete_connector(cid) is True
    assert await store.delete_connector(cid) is False
