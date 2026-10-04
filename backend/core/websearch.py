"""Web search via Tavily.

Measured on this account: 475ms warm, 3.1s cold. That is inside the fast lane
but only with filler speech over the gap, so callers hear an acknowledgement
rather than silence.

Uses the SDK rather than Tavily's hosted MCP, which passes the API key as a URL
query parameter. Keys in URLs end up in logs, proxies and error traces in a way
headers do not.
"""

import os

from tavily import AsyncTavilyClient

# Beyond this the caller has been waiting too long for a search to still be
# worth having. Better to say the search failed than to hold the turn open.
TIMEOUT_S = float(os.getenv("TAVILY_TIMEOUT", "6"))
MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))

_client: AsyncTavilyClient | None = None


def configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def _get() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


async def search(query: str) -> str:
    """Search the web and return something short enough to say out loud.

    Full page content is deliberately not returned. This text goes to an LLM
    that is about to read it aloud, and a wall of scraped HTML produces a
    rambling answer and a slow one.
    """
    response = await _get().search(
        query=query,
        max_results=MAX_RESULTS,
        search_depth="basic",          # "advanced" roughly doubles the latency
        include_answer=True,
    )

    if answer := (response.get("answer") or "").strip():
        return answer

    results = response.get("results") or []
    if not results:
        return f"No results for {query!r}."

    return " ".join(
        f"{r.get('title', 'Untitled')}: {(r.get('content') or '')[:200]}"
        for r in results[:MAX_RESULTS]
    )
