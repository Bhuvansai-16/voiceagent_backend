"""Call traces in Postgres.

The worker wrote JSONL to its own disk and the API globbed TRACE_DIR and
JSON-parsed every file on every request. Two processes, one assumed filesystem —
so in production the analytics panels went blank and nothing errored — and an
O(all files) scan per poll even when it worked.

Aggregation that Postgres does well happens in SQL. `reply_latencies` stays in
Python: it is not a percentile, it pairs each user turn with the first audio
after it and *skips* turns where a second turn ended in between, because
ownership is genuinely ambiguous there. Guessing once reported a twelve-second
wait for a turn that answered in 806ms.
"""

from typing import Any, Callable

from psycopg.types.json import Jsonb


class TraceStore:
    def __init__(self, connection: Callable[[], Any] | None = None) -> None:
        if connection is None:
            from backend.store.pool import get_pool

            connection = lambda: get_pool().connection()  # noqa: E731
        self._connection = connection

    # --- writing ------------------------------------------------------------

    async def begin_call(self, call_id: str, **meta) -> None:
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO calls (id, agent, model, provider, voice, room)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                   SET agent = COALESCE(EXCLUDED.agent, calls.agent),
                       model = COALESCE(EXCLUDED.model, calls.model),
                       provider = COALESCE(EXCLUDED.provider, calls.provider),
                       voice = COALESCE(EXCLUDED.voice, calls.voice),
                       room  = COALESCE(EXCLUDED.room, calls.room)
                """,
                (call_id, meta.get("agent"), meta.get("model"),
                 meta.get("provider"), meta.get("voice"), meta.get("room")),
            )

    async def add_events(self, call_id: str, rows: list[dict]) -> int:
        """Append span and quality rows. `stage` and `ms` are columns; the rest
        is payload, because the useful fields differ per stage and a column per
        field would need a migration every time a new one appears."""
        if not rows:
            return 0
        async with self._connection() as conn:
            # The call row may not exist if meta was never written (a crash
            # before the session started). The FK would reject the events, so
            # make sure there is a parent first.
            await conn.execute(
                "INSERT INTO calls (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                (call_id,),
            )
            for row in rows:
                payload = {k: v for k, v in row.items() if k not in ("stage", "ms", "at")}
                await conn.execute(
                    "INSERT INTO call_events (call_id, stage, ms, payload)"
                    " VALUES (%s, %s, %s, %s)",
                    (call_id, row.get("stage", "unknown"), row.get("ms"), Jsonb(payload)),
                )
        return len(rows)

    async def end_call(self, call_id: str) -> None:
        async with self._connection() as conn:
            await conn.execute(
                "UPDATE calls SET ended = now() WHERE id = %s AND ended IS NULL",
                (call_id,),
            )

    # --- reading ------------------------------------------------------------

    async def per_model_stats(self) -> list[dict]:
        """p50/p95 per model per stage, computed by Postgres.

        Replaces reading and parsing every trace file in Python on each request.
        """
        async with self._connection() as conn:
            return await (await conn.execute(
                """
                SELECT COALESCE(c.model, 'unattributed') AS model,
                       MIN(c.provider)                   AS provider,
                       e.stage                           AS stage,
                       count(*)                          AS n,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY e.ms) AS p50,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY e.ms) AS p95
                  FROM call_events e
                  JOIN calls c ON c.id = e.call_id
                 WHERE e.ms IS NOT NULL
                 GROUP BY 1, 3
                """
            )).fetchall()

    async def per_model_calls(self) -> list[dict]:
        """How many calls each model produced, and which agents used it.

        Separate from per_model_stats because that one groups by stage, so
        counting calls there would count each call once per stage.
        """
        async with self._connection() as conn:
            return await (await conn.execute(
                """
                SELECT COALESCE(model, 'unattributed') AS model,
                       MIN(provider)                   AS provider,
                       count(*)                        AS calls,
                       array_remove(array_agg(DISTINCT agent), NULL) AS agents
                  FROM calls
                 GROUP BY 1
                """
            )).fetchall()

    async def calls_summary(self) -> list[dict]:
        async with self._connection() as conn:
            return await (await conn.execute(
                """
                SELECT c.id, c.agent, c.model, c.provider, c.started,
                       count(*) FILTER (WHERE e.stage = 'turn.total')      AS turns,
                       count(*) FILTER (WHERE e.stage LIKE 'tool.%')       AS tools,
                       array_agg(DISTINCT c.agent)                          AS agents,
                       avg(e.ms) FILTER (WHERE e.stage = 'llm.first_token') AS avg_first_token
                  FROM calls c LEFT JOIN call_events e ON e.call_id = c.id
                 GROUP BY c.id
                 ORDER BY c.started DESC
                """
            )).fetchall()

    async def rows_for(self, call_id: str) -> list[dict]:
        """Every event for one call, flattened back to the row shape the pure
        analytics functions already understand."""
        async with self._connection() as conn:
            rows = await (await conn.execute(
                "SELECT stage, ms, payload FROM call_events"
                " WHERE call_id = %s ORDER BY seq",
                (call_id,),
            )).fetchall()
        return [{"stage": r["stage"], "ms": r["ms"], **(r["payload"] or {})} for r in rows]

    async def call_ids(self, limit: int = 200) -> list[str]:
        async with self._connection() as conn:
            rows = await (await conn.execute(
                "SELECT id FROM calls ORDER BY started DESC LIMIT %s", (limit,)
            )).fetchall()
        return [r["id"] for r in rows]

    async def total_calls(self) -> int:
        async with self._connection() as conn:
            row = await (await conn.execute("SELECT count(*) AS n FROM calls")).fetchone()
        return row["n"]
