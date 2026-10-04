"""The Postgres connection pool, created once at module scope.

Neon documents that creating a pool inside a request handler is the common
mistake, so there is exactly one here and everything shares it.

Two connection strings, for two different jobs:

  NEON_CONNECTION_URI   the pooled endpoint (-pooler in the hostname). PgBouncer
                        in transaction mode, good for up to 10k clients. This is
                        what the application uses.

  direct_uri()          the same string with "-pooler" removed. Neon requires the
                        direct endpoint for schema migrations, and PgBouncer's
                        transaction pooling does not support the session state
                        that DDL and prepared statements rely on.

psycopg3 rather than asyncpg for the same reason: asyncpg caches prepared
statements, which transaction-mode PgBouncer breaks. Neon's own Python guidance
uses psycopg_pool.
"""

import os

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None


def dsn() -> str:
    uri = os.getenv("NEON_CONNECTION_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "NEON_CONNECTION_URI is not set. Copy the pooled connection string "
            "from the Neon console (the hostname contains '-pooler')."
        )
    return uri


def direct_dsn() -> str:
    """The non-pooler endpoint, for DDL only."""
    return dsn().replace("-pooler", "", 1)


def get_pool() -> AsyncConnectionPool:
    """The shared pool. Opened lazily so importing this module needs no network."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            dsn(),
            min_size=1,
            # Neon's pooled endpoint fronts PgBouncer, so a large local pool buys
            # nothing and just holds server-side slots open.
            max_size=int(os.getenv("DB_POOL_MAX", "8")),
            kwargs={"row_factory": dict_row},
            open=False,
        )
    return _pool


async def open_pool() -> None:
    await get_pool().open(wait=True, timeout=30)


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def configured() -> bool:
    return bool(os.getenv("NEON_CONNECTION_URI", "").strip())
