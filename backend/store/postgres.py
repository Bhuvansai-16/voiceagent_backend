"""The real store: Neon Postgres via psycopg3.

Every write is a single statement. The read-modify-write dance the JSON files
forced — load the whole file, mutate a dict, write it back, hope nobody else
did the same — is what `jsonb ||` and ON CONFLICT replace. Five separate code
paths used to do that against config/agents.json with no locking; concurrent
saves silently lost each other's changes.
"""

from typing import Any, Callable

from psycopg.types.json import Jsonb

from backend.store.pool import get_pool


class PostgresStore:
    """`connection` defaults to the shared pool.

    It is injectable because a psycopg pool is bound to the event loop that
    opened it, and pytest-asyncio gives each test its own loop. Sharing one
    pool across them fails with "attached to a different loop", which reads as
    a psycopg bug and is not one.
    """

    def __init__(self, connection: Callable[[], Any] | None = None) -> None:
        self._connection = connection or (lambda: get_pool().connection())

    # --- agents -------------------------------------------------------------

    async def agent_overrides(self) -> dict[str, dict]:
        async with self._connection() as conn:
            rows = await (await conn.execute("SELECT id, fields FROM agents")).fetchall()
        return {r["id"]: (r["fields"] or {}) for r in rows}

    async def upsert_agent(self, agent_id: str, fields: dict) -> dict:
        # `fields || excluded.fields` merges at the database rather than in
        # Python, so two concurrent saves of different keys both survive.
        async with self._connection() as conn:
            row = await (await conn.execute(
                """
                INSERT INTO agents (id, fields) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE
                   SET fields  = agents.fields || EXCLUDED.fields,
                       updated = now()
                RETURNING id, fields
                """,
                (agent_id, Jsonb(fields)),
            )).fetchone()
        return row["fields"] or {}

    async def delete_agent(self, agent_id: str) -> bool:
        async with self._connection() as conn:
            cur = await conn.execute("DELETE FROM agents WHERE id = %s", (agent_id,))
        return cur.rowcount > 0

    # --- flows --------------------------------------------------------------

    async def flows(self) -> list[dict]:
        async with self._connection() as conn:
            rows = await (await conn.execute(
                "SELECT id, graph, deployed_agent_id FROM flows ORDER BY updated DESC"
            )).fetchall()
        out = []
        for r in rows:
            flow = dict(r["graph"] or {})
            flow["id"] = r["id"]
            if r["deployed_agent_id"]:
                flow["deployed_agent_id"] = r["deployed_agent_id"]
            out.append(flow)
        return out

    async def upsert_flow(self, flow: dict) -> dict:
        flow_id = flow["id"]
        graph = {k: v for k, v in flow.items() if k != "id"}
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO flows (id, graph, deployed_agent_id) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                   SET graph             = flows.graph || EXCLUDED.graph,
                       deployed_agent_id = COALESCE(EXCLUDED.deployed_agent_id,
                                                    flows.deployed_agent_id),
                       updated           = now()
                """,
                (flow_id, Jsonb(graph), flow.get("deployed_agent_id")),
            )
        return flow

    async def delete_flow(self, flow_id: str) -> bool:
        async with self._connection() as conn:
            cur = await conn.execute("DELETE FROM flows WHERE id = %s", (flow_id,))
        return cur.rowcount > 0

    # --- connectors ---------------------------------------------------------

    async def connectors(self) -> list[dict]:
        async with self._connection() as conn:
            rows = await (await conn.execute(
                "SELECT id, definition, connected FROM connectors ORDER BY id"
            )).fetchall()
        return [{**(r["definition"] or {}), "id": r["id"], "connected": r["connected"]}
                for r in rows]

    async def upsert_connector(self, connector: dict) -> dict:
        cid = connector["id"]
        definition = {k: v for k, v in connector.items() if k not in ("id", "connected")}
        async with self._connection() as conn:
            row = await (await conn.execute(
                """
                INSERT INTO connectors (id, definition, connected) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                   SET definition = connectors.definition || EXCLUDED.definition,
                       updated    = now()
                RETURNING id, definition, connected
                """,
                (cid, Jsonb(definition), bool(connector.get("connected", False))),
            )).fetchone()
        return {**(row["definition"] or {}), "id": row["id"], "connected": row["connected"]}

    async def set_connector_connected(self, connector_id: str, connected: bool) -> bool:
        async with self._connection() as conn:
            cur = await conn.execute(
                "UPDATE connectors SET connected = %s, updated = now() WHERE id = %s",
                (connected, connector_id),
            )
        return cur.rowcount > 0

    async def delete_connector(self, connector_id: str) -> bool:
        async with self._connection() as conn:
            cur = await conn.execute("DELETE FROM connectors WHERE id = %s", (connector_id,))
        return cur.rowcount > 0
