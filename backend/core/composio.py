"""Composio endpoint assembly.

Its own module, and deliberately free of any livekit import. This used to live in
backend/worker/main.py, which registers every LiveKit plugin at import time; the debug
backend imported it from there to report connection status, and importing it
from inside a request handler raised "Plugins must be registered on the main
thread". Nothing here needs more than os.
"""

import logging
import os
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("service-desk.composio")

COMPOSIO_MCP_BASE = "https://backend.composio.dev/v3/mcp"
COMPOSIO_API_BASE = "https://backend.composio.dev/api/v3.1"


async def verify_composio_key(api_key: str | None = None) -> dict[str, Any]:
    """Checks if the given or configured Composio API key is valid."""
    key = (api_key or os.getenv("COMPOSIO_API_KEY", "")).strip()
    if not key:
        return {"valid": False, "error": "No API key configured. Set COMPOSIO_API_KEY in .env"}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                f"{COMPOSIO_API_BASE}/auth_configs",
                headers={"x-api-key": key},
            )
            if resp.status_code == 200:
                return {"valid": True, "status": "active", "error": None}
            elif resp.status_code == 401:
                return {
                    "valid": False,
                    "status": "invalid_key",
                    "error": "Invalid API key. Please generate a new key on platform.composio.dev",
                }
            else:
                return {
                    "valid": False,
                    "status": f"http_{resp.status_code}",
                    "error": f"Composio returned HTTP {resp.status_code}: {resp.text[:120]}",
                }
    except Exception as ex:
        return {"valid": False, "status": "network_error", "error": f"Connection error: {ex}"}


def composio_mcp_url(server_id: str | None = None, user_id: str | None = None) -> str:
    """The Composio endpoint, assembled from whichever pieces are configured."""
    url = os.getenv("COMPOSIO_MCP_URL", "").strip()
    if not url:
        sid = server_id or os.getenv("COMPOSIO_MCP_SERVER_ID", "").strip()
        if sid:
            url = f"{COMPOSIO_MCP_BASE}/{sid}"
        else:
            return ""

    uid = user_id or os.getenv("COMPOSIO_USER_ID", "").strip()
    if uid and "user_id=" not in url:
        url = f"{url}{'&' if '?' in url else '?'}user_id={quote(uid, safe='')}"
    return url


def get_allowed_tools_for_toolkits(toolkits: tuple[str, ...] | list[str]) -> list[str] | None:
    """Composio tool allowlist for the active toolkits.

    Thin wrapper: the rule lives in core.connectors.allowed_tools_for, which the
    API also uses, so the worker and the console cannot disagree about which
    tools an agent is allowed to see.
    """
    from backend.core.connectors import allowed_tools_for

    return allowed_tools_for(toolkits)


async def get_auth_config_id(app_slug: str, api_key: str) -> str | None:
    """Look up the default auth config id for an app on Composio.

    If the slug is already an auth_config id (starts with 'ac_'), return it
    directly. Otherwise, query the Composio API to find the default auth config
    for the given app name (e.g. 'gmail', 'slack').
    """
    slug = app_slug.strip()
    if slug.startswith("ac_"):
        return slug

    # Query Composio for auth configs of this app
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COMPOSIO_API_BASE}/auth_configs",
                params={"appName": slug},
                headers={"x-api-key": api_key},
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data if isinstance(data, list) else data.get("items", data.get("auth_configs", []))
                if items:
                    # Prefer the default / first auth config
                    return items[0].get("id") or items[0].get("auth_config_id")
    except Exception as e:
        logger.warning("Failed to look up auth config for %s: %s", slug, e)

    return None


async def initiate_connection(
    app_slug: str,
    *,
    callback_url: str | None = None,
) -> dict:
    """Call Composio's REST API to start an OAuth connection flow.

    Returns a dict with:
      - redirect_url: the URL to send the user to for OAuth consent
      - connected_account_id: the pending account ID
    Or raises RuntimeError on failure.
    """
    api_key = os.getenv("COMPOSIO_API_KEY", "").strip()
    user_id = os.getenv("COMPOSIO_USER_ID", "").strip() or "default"

    if not api_key:
        raise RuntimeError("COMPOSIO_API_KEY is not configured")

    # Resolve the auth_config_id for this app/slug
    auth_config_id = await get_auth_config_id(app_slug, api_key)
    if not auth_config_id:
        raise RuntimeError(
            f"Could not find a Composio auth config for '{app_slug}'. "
            "Make sure the app is enabled in your Composio dashboard."
        )

    body: dict = {
        "auth_config_id": auth_config_id,
        "user_id": user_id,
    }
    if callback_url:
        body["callback_url"] = callback_url

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{COMPOSIO_API_BASE}/connected_accounts/link",
                json=body,
                headers={"x-api-key": api_key},
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                redirect_url = data.get("redirect_url") or data.get("redirectUrl") or data.get("url")
                if redirect_url:
                    return {
                        "redirect_url": redirect_url,
                        "connected_account_id": data.get("connected_account_id") or data.get("connectedAccountId"),
                    }
                # If no redirect_url, the auth might be non-OAuth (API key, etc.)
                logger.warning("Composio link response had no redirect_url: %s", data)
                return {"redirect_url": None, "raw": data}

            logger.error("Composio link failed [%d]: %s", resp.status_code, resp.text)
            raise RuntimeError(f"Composio API returned {resp.status_code}: {resp.text}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"Network error calling Composio: {e}") from e

