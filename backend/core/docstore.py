"""Document chunks, shared between the API and the worker.

This replaces `rag._LOCAL_DOC_STORE`, a module-level Python list. The API
appended to it on upload and the worker read its own separate copy during
`search_documents`, so uploading a document and then asking the agent about it
found nothing at all unless Pinecone happened to be configured. Two processes,
one list that was never shared. The tests passed because they indexed and
queried inside a single process.

A table is what makes the two processes agree.

Kept out of rag.py so core/rag.py stays about extraction and embedding, and so
the fallback path has one obvious home. Takes a connection-providing callable so
core does not import backend.store.
"""

import logging
import re
from typing import Any, Callable

logger = logging.getLogger("service-desk.docstore")

# Set once at startup by the API and the worker. core/ must not import
# backend.store, so the connection source is injected rather than imported.
_connection: Callable[[], Any] | None = None


def bind(connection_factory: Callable[[], Any]) -> None:
    global _connection
    _connection = connection_factory


def bound() -> bool:
    return _connection is not None


async def add_chunks(rows: list[dict]) -> int:
    """Store chunk rows. Returns how many were written."""
    if not rows or _connection is None:
        return 0
    async with _connection() as conn:
        for row in rows:
            await conn.execute(
                """
                INSERT INTO doc_chunks (id, filename, caller_id, chunk_index, text, object_key)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text
                """,
                (row["id"], row["filename"], row["caller_id"],
                 row["chunk_index"], row["text"], row.get("object_key")),
            )
    return len(rows)


async def search(query_text: str, caller_id: str = "default", top_k: int = 3) -> list[dict]:
    """Word-overlap search over this caller's chunks.

    Deliberately not a vector search: this is the fallback for when Pinecone is
    absent or returns nothing, and Postgres full-text or pgvector would be a
    second embedding pipeline to keep in sync. Overlap is enough to answer
    "did my upload land", which is what this path is for.
    """
    if _connection is None:
        return []
    words = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2]
    if not words:
        return []

    async with _connection() as conn:
        rows = await (await conn.execute(
            """
            SELECT filename, text,
                   (SELECT count(*) FROM unnest(%s::text[]) w
                     WHERE position(w in lower(doc_chunks.text)) > 0) AS hits
              FROM doc_chunks
             WHERE caller_id = %s
             ORDER BY hits DESC, chunk_index ASC
             LIMIT %s
            """,
            (words, caller_id, top_k),
        )).fetchall()
    return [r for r in rows if (r["hits"] or 0) > 0]


async def list_chunks(limit: int = 200) -> list[dict]:
    if _connection is None:
        return []
    async with _connection() as conn:
        return await (await conn.execute(
            "SELECT id, filename, caller_id, chunk_index, text, object_key, created"
            " FROM doc_chunks ORDER BY created DESC LIMIT %s",
            (limit,),
        )).fetchall()


async def delete_chunk(doc_id: str) -> int:
    if _connection is None:
        return 0
    async with _connection() as conn:
        cur = await conn.execute("DELETE FROM doc_chunks WHERE id = %s", (doc_id,))
        return cur.rowcount


async def clear() -> int:
    if _connection is None:
        return 0
    async with _connection() as conn:
        cur = await conn.execute("DELETE FROM doc_chunks")
        return cur.rowcount
