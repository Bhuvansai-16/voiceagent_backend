"""Per-caller memory, backed by supermemory.

Run it locally: `npx supermemory local`, then `supermemory-server`. It listens on
6767 and embeds locally with bge-base-en-v1.5, so nothing leaves the machine and
nothing is billed.

Two rules shape this module.

**Recall must never delay a turn.** It runs on every user turn, so a slow or
absent memory server has to cost nothing. Every call is wrapped in a timeout and
every failure returns empty rather than raising. A caller waiting on a memory
lookup is worse off than a caller whose agent forgot something.

**Memory is scoped per caller and never shared.** `container_tag` carries the
identity. Leaking one caller's memory into another's session is the worst thing
this system could do, and it would be a one-line mistake, so the tag is
mandatory rather than defaulted.
"""

import asyncio
import logging
import os

from supermemory import AsyncSupermemory

logger = logging.getLogger("service-desk.memory")

# Roadmap budget: recall runs inside the turn, so it gets 200ms and no more.
RECALL_TIMEOUT_S = float(os.getenv("MEMORY_RECALL_TIMEOUT", "0.2"))
# Writing happens after the reply, off the critical path, so it can be patient.
WRITE_TIMEOUT_S = float(os.getenv("MEMORY_WRITE_TIMEOUT", "5"))
RECALL_LIMIT = int(os.getenv("MEMORY_RECALL_LIMIT", "3"))

_client: AsyncSupermemory | None = None


def configured() -> bool:
    return bool(os.getenv("SUPERMEMORY_API_KEY", "").strip())


def _get() -> AsyncSupermemory:
    global _client
    if _client is None:
        # base_url is deliberately not defaulted here. The SDK already reads
        # SUPERMEMORY_BASE_URL and otherwise points at api.supermemory.ai, so
        # hardcoding localhost:6767 as the fallback sent a cloud `sm_` key to a
        # local server that was not running. Every call then failed and returned
        # empty, which is exactly what this module does on purpose, so the agent
        # simply never remembered anything and said nothing about why.
        # Running locally still works: set SUPERMEMORY_BASE_URL=http://localhost:6767.
        _client = AsyncSupermemory(api_key=os.environ["SUPERMEMORY_API_KEY"])
    return _client


def tag_for(caller_id: str) -> str:
    """The per-caller scope key. Explicit function so the shape is in one place."""
    return f"caller:{caller_id}"


async def recall(caller_id: str, query: str) -> list[str]:
    """Relevant things this caller said before. Empty on any failure."""
    if not configured() or not caller_id:
        return []
    try:
        async with asyncio.timeout(RECALL_TIMEOUT_S):
            found = await _get().search.memories(
                q=query,
                container_tag=tag_for(caller_id),
                limit=RECALL_LIMIT,
            )
    except TimeoutError:
        logger.info("memory recall exceeded %.0fms, continuing without it",
                    RECALL_TIMEOUT_S * 1000)
        return []
    except Exception as exc:
        logger.warning("memory recall failed, continuing without it: %s", exc)
        return []

    out = []
    for item in (getattr(found, "results", None) or []):
        text = getattr(item, "memory", None) or getattr(item, "content", None)
        if text:
            out.append(str(text).strip())
    return out


def format_memory_entry(fact: str, category: str = "fact") -> str:
    """Formats fact according to AGENTS.md structured memory style."""
    clean_cat = (category or "fact").strip().lower()
    return f"[{clean_cat}] {fact.strip()}"


async def remember(caller_id: str, text: str, category: str = "fact") -> bool:
    """Store one fact about this caller. Returns whether it was stored."""
    if not configured() or not caller_id or not text.strip():
        return False
    try:
        content = format_memory_entry(text, category) if not text.strip().startswith("[") else text.strip()
        async with asyncio.timeout(WRITE_TIMEOUT_S):
            await _get().documents.add(
                content=content,
                container_tag=tag_for(caller_id),
            )
        return True
    except Exception as exc:
        logger.warning("memory write failed: %s", exc)
        return False
