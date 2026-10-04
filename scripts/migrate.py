"""Apply the schema and move the old file-based state into Neon.

    uv run python scripts/migrate.py            # schema + import, idempotent
    uv run python scripts/migrate.py --schema   # schema only

Runs over the DIRECT endpoint. Neon requires it for DDL, and PgBouncer's
transaction pooling cannot serve schema changes reliably.

Idempotent and re-runnable: every statement is IF NOT EXISTS or ON CONFLICT, so
running it twice imports nothing twice. Reads config/*.json and
data/servicedesk.db if they are still present and says so if they are not.
"""

import json
import sqlite3
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from backend.store.pool import direct_dsn  # noqa: E402


def apply_schema(conn) -> None:
    conn.execute((ROOT / "backend" / "store" / "schema.sql").read_text(encoding="utf-8"))
    print("  schema applied")


def migrate_agents(conn) -> int:
    path = ROOT / "config" / "agents.json"
    if not path.is_file():
        print("  agents.json: absent, nothing to import")
        return 0
    entries = json.loads(path.read_text(encoding="utf-8")).get("agents", [])
    n = 0
    for entry in entries:
        agent_id = (entry.get("id") or "").strip()
        if not agent_id:
            continue
        fields = {k: v for k, v in entry.items() if k != "id"}
        conn.execute(
            "INSERT INTO agents (id, fields) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET fields = agents.fields || EXCLUDED.fields",
            (agent_id, Jsonb(fields)),
        )
        n += 1
    print(f"  agents: {n} imported")
    return n


def migrate_flows(conn) -> int:
    path = ROOT / "config" / "flows.json"
    if not path.is_file():
        print("  flows.json: absent, nothing to import")
        return 0
    flows = json.loads(path.read_text(encoding="utf-8")).get("flows", [])
    n = 0
    for flow in flows:
        flow_id = (flow.get("id") or "").strip()
        if not flow_id:
            continue
        graph = {k: v for k, v in flow.items() if k != "id"}
        conn.execute(
            "INSERT INTO flows (id, graph, deployed_agent_id) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET graph = flows.graph || EXCLUDED.graph",
            (flow_id, Jsonb(graph), flow.get("deployed_agent_id")),
        )
        n += 1
    print(f"  flows: {n} imported")
    return n


def migrate_connectors(conn) -> int:
    """Only the state. The catalog is code now, in core/connector_catalog.py."""
    path = ROOT / "config" / "connectors.json"
    if not path.is_file():
        print("  connectors.json: absent, nothing to import")
        return 0
    items = json.loads(path.read_text(encoding="utf-8")).get("connectors", [])
    n = 0
    for c in items:
        cid = (c.get("id") or "").strip().lower()
        if not cid or not c.get("connected"):
            continue  # a disconnected catalog entry needs no row
        definition = {k: v for k, v in c.items() if k not in ("id", "connected")}
        conn.execute(
            "INSERT INTO connectors (id, definition, connected) VALUES (%s, %s, true) "
            "ON CONFLICT (id) DO UPDATE SET definition = connectors.definition || EXCLUDED.definition",
            (cid, Jsonb(definition)),
        )
        n += 1
    print(f"  connectors: {n} connected row(s) imported")
    return n


def migrate_servicedesk(conn) -> int:
    path = ROOT / "data" / "servicedesk.db"
    if not path.is_file():
        print("  servicedesk.db: absent, nothing to import")
        return 0

    src = sqlite3.connect(path)
    src.row_factory = sqlite3.Row
    total = 0

    def table_exists(name: str) -> bool:
        return src.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    if table_exists("employees"):
        for r in src.execute("SELECT id, name, email, locked FROM employees"):
            conn.execute(
                "INSERT INTO employees (id, name, email, locked) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (id) DO NOTHING",
                (r["id"], r["name"], r["email"], bool(r["locked"])),
            )
            total += 1

    if table_exists("tickets"):
        for r in src.execute(
            "SELECT id, employee_id, title, description, status, updated FROM tickets"
        ):
            conn.execute(
                "INSERT INTO tickets (id, employee_id, title, description, status, updated)"
                " VALUES (%s,%s,%s,%s,%s,COALESCE(%s::timestamptz, now()))"
                " ON CONFLICT (id) DO NOTHING",
                (r["id"], r["employee_id"], r["title"], r["description"],
                 r["status"], r["updated"]),
            )
            total += 1

    if table_exists("password_resets"):
        for r in src.execute(
            "SELECT idempotency_key, reset_id, employee_id, status, sent_to, created"
            " FROM password_resets"
        ):
            conn.execute(
                "INSERT INTO password_resets"
                " (idempotency_key, reset_id, employee_id, status, sent_to, created)"
                " VALUES (%s,%s,%s,%s,%s,COALESCE(%s::timestamptz, now()))"
                " ON CONFLICT (idempotency_key) DO NOTHING",
                (r["idempotency_key"], r["reset_id"], r["employee_id"],
                 r["status"], r["sent_to"], r["created"]),
            )
            total += 1

    src.close()
    print(f"  servicedesk.db: {total} row(s) imported")
    return total


def main(argv: list[str]) -> int:
    schema_only = "--schema" in argv
    print(f"migrating into {direct_dsn().split('@')[-1].split('/')[0]} (direct endpoint)")

    with psycopg.connect(direct_dsn(), connect_timeout=30, autocommit=True) as conn:
        apply_schema(conn)
        if schema_only:
            return 0
        migrate_agents(conn)
        migrate_flows(conn)
        migrate_connectors(conn)
        migrate_servicedesk(conn)

        counts = {
            t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("agents", "flows", "connectors", "employees", "tickets",
                      "password_resets", "doc_chunks")
        }
    print("\nrow counts now:")
    for table, n in counts.items():
        print(f"  {table:16} {n}")
    print("\nconfig/ and data/ can now be deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
