"""Connector logic, with no I/O.

Storage moved to the store and the catalog to connector_catalog.py, leaving this
module with the parts that are actually decisions: what Composio's configuration
amounts to, how a connector's status is derived, and where its authorization
link points.

Everything here is a pure function of its arguments. The router fetches rows and
passes them in.
"""

import logging
import os
from typing import Any
from urllib.parse import quote

from backend.core.composio import composio_mcp_url
from backend.core.connector_catalog import CATALOG

logger = logging.getLogger("service-desk.connectors")


def get_composio_status() -> dict[str, Any]:
    """Composio configuration state, read from the environment."""
    api_key = os.getenv("COMPOSIO_API_KEY", "").strip()
    server_id = os.getenv("COMPOSIO_MCP_SERVER_ID", "").strip()
    user_id = os.getenv("COMPOSIO_USER_ID", "").strip()
    mcp_url = composio_mcp_url()

    return {
        "configured": bool(api_key and mcp_url),
        "api_key_set": bool(api_key),
        "server_id": server_id or ("connect" if "connect.composio.dev" in mcp_url else "default"),
        "user_id": user_id or "default",
        "mcp_url": mcp_url,
        "missing": [] if api_key else ["COMPOSIO_API_KEY"],
        "dashboard_url": "https://platform.composio.dev",
    }


def auth_url_for(slug: str, user_id: str) -> str:
    """Where a caller goes to authorize this connector.

    An `ac_` prefix is a Composio auth-config id and keeps its exact case; an
    app name is lowercased. Getting that backwards produces a link that 404s.
    """
    if slug.startswith("ac_"):
        return f"https://platform.composio.dev/integrations/{quote(slug)}?user_id={quote(user_id)}"
    return f"https://platform.composio.dev/apps/{quote(slug.lower())}"


def agents_by_toolkit(agent_overrides: dict[str, dict]) -> dict[str, list[dict[str, str]]]:
    """Which agents have each toolkit enabled, keyed by lowercased slug."""
    out: dict[str, list[dict[str, str]]] = {}
    for agent_id, entry in (agent_overrides or {}).items():
        label = entry.get("label", agent_id)
        for toolkit in entry.get("toolkits", []) or []:
            out.setdefault(str(toolkit).strip().lower(), []).append(
                {"id": agent_id, "label": label}
            )
    return out


def merge(stored: list[dict], agent_overrides: dict[str, dict]) -> list[dict[str, Any]]:
    """The catalog, with stored state and live agent bindings layered on.

    The catalog is the floor: a connector the database has never heard of still
    appears, disconnected, rather than vanishing from the UI.
    """
    composio = get_composio_status()
    user_id = composio["user_id"]
    by_toolkit = agents_by_toolkit(agent_overrides)
    stored_by_id = {(c.get("id") or "").strip().lower(): c for c in stored}

    out = []
    for entry in CATALOG:
        cid = entry["id"].strip().lower()
        row = stored_by_id.get(cid, {})
        merged = {**entry, **{k: v for k, v in row.items() if k != "id"}, "id": cid}

        raw_slug = (merged.get("slug") or cid).strip()
        toolkit = (merged.get("toolkit") or cid).strip()
        active = (
            by_toolkit.get(raw_slug.lower())
            or by_toolkit.get(cid)
            or by_toolkit.get(toolkit.lower())
            or []
        )
        connected = bool(merged.get("connected")) or bool(active)

        if connected:
            status = "connected"
        elif not composio["configured"]:
            status = "setup_required"
        else:
            status = "available"

        out.append({
            **merged,
            "slug": raw_slug,
            "connected": connected,
            "status": status,
            "active_agents": active,
            "auth_url": auth_url_for(raw_slug, user_id),
            "user_id": user_id,
            "composio_configured": composio["configured"],
        })

    # Connectors added at runtime that are not in the shipped catalog.
    for cid, row in stored_by_id.items():
        if any(c["id"] == cid for c in out):
            continue
        raw_slug = (row.get("slug") or cid).strip()
        out.append({
            **row, "id": cid, "slug": raw_slug,
            "connected": bool(row.get("connected")),
            "status": "connected" if row.get("connected") else "available",
            "active_agents": by_toolkit.get(raw_slug.lower(), []),
            "auth_url": auth_url_for(raw_slug, user_id),
            "user_id": user_id,
            "composio_configured": composio["configured"],
        })

    return sorted(out, key=lambda c: c["id"])


def normalise(payload: dict[str, Any]) -> dict[str, Any]:
    """A connector record from user input, with defaults filled in."""
    cid = (payload.get("id") or "").strip().lower()
    if not cid:
        cid = (payload.get("name") or "").strip().lower().replace(" ", "")
    if not cid:
        raise ValueError("Connector 'name' or 'id' is required")

    return {
        "id": cid,
        "name": (payload.get("name") or cid.capitalize()).strip(),
        "slug": (payload.get("slug") or cid).strip().lower(),
        "category": (payload.get("category") or "General").strip(),
        "description": (payload.get("description") or "").strip(),
        "recommended": bool(payload.get("recommended", False)),
        "color": payload.get("color") or "#4F46E5",
        "icon": payload.get("icon") or "app",
        "tools": payload.get("tools") or [],
    }


def toggle_toolkit(toolkits: list[str], slug: str) -> tuple[list[str], bool]:
    """Add or remove a toolkit slug. Returns the new list and whether it is on."""
    clean = slug.strip().lower()
    current = [str(t) for t in (toolkits or [])]
    if any(t.lower() == clean for t in current):
        return [t for t in current if t.lower() != clean], False
    return current + [clean], True


def allowed_tools_for(toolkits, stored: list[dict] | None = None) -> list[str] | None:
    """Composio tool allowlist for the given toolkits.

    An unlisted tool cannot be called at all, so this is a real boundary rather
    than a hint to the model.
    """
    global_allowed = [
        t.strip() for t in os.getenv("COMPOSIO_ALLOWED_TOOLS", "").split(",") if t.strip()
    ]
    if global_allowed:
        return global_allowed
    if not toolkits:
        return None

    wanted = {str(t).lower() for t in toolkits if t}
    pool = list(CATALOG) + list(stored or [])
    allowed: list[str] = []
    for connector in pool:
        cid = (connector.get("id") or "").lower()
        slug = (connector.get("slug") or "").lower()
        toolkit = (connector.get("toolkit") or "").lower()
        if {cid, slug, toolkit} & wanted:
            for tool in connector.get("tools", []) or []:
                allowed.extend([tool, f"{cid.upper()}_{tool.upper()}", f"{cid}_{tool}"])
    return allowed or None
